import os
import re
import json
import argparse
from typing import Any, Dict, List, Tuple, Optional
from collections import Counter

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

# ========== 路径配置（可按需修改） ==========
# INPUT_JSON  = "/path/to/qa_items.json"  # 你的题目JSON，示例见提问
# IMAGE_ROOT  = "/home/mcislab_cj/MME_realworld"  # 图像根目录，和JSON里的相对路径拼接
# MODEL_DIR   = "/home/mcislab_cj/fm9g4bv/FM9G4B-V"  # 大模型路径（FM9G等）
# OUTPUT_JSON = "/path/to/qa_results.json"

# ========== 工具函数 ==========
def open_image_rgb(path: str) -> Image.Image:
    """
    尝试以 RGB 读取图像。
    1) PIL 直接读取 + convert("RGB")
    2) 如遇部分 GeoTIFF/TIFF 读不动，回退到 tifffile 再转 PIL
    """
    try:
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except Exception as e:
        try:
            import tifffile as tiff
            arr = tiff.imread(path)
            # 统一转三通道
            if arr.ndim == 2:
                # 灰度 -> RGB
                arr = (arr[..., None]).repeat(3, axis=-1)
            elif arr.ndim == 3 and arr.shape[2] > 3:
                arr = arr[..., :3]
            img = Image.fromarray(arr)
            if img.mode != "RGB":
                img = img.convert("RGB")
            return img
        except Exception as e2:
            raise RuntimeError(f"Cannot open image as RGB: {path}\nPIL error: {e}\nTIFF fallback error: {e2}")

def extract_option_letter(s: Any) -> str:
    """
    从模型输出里提取 A/B/C/D 的选项字母（返回大写字母或空串）。
    兼容 '(A)', 'A:', '答案A', 'A' 等形式。
    """
    if s is None:
        return ""
    text = str(s).strip().upper()

    # 先匹配规范形式
    m = re.search(r'[\(\[\{<]?\s*([ABCD])\s*[\)\]\}>:\.]?', text)
    if m:
        return m.group(1)

    # 若未命中，宽松包含
    for ch in ["A", "B", "C", "D"]:
        if re.search(rf'\b{ch}\b', text):
            return ch

    # 仍未命中，尝试按关键字回推（可选）：若回答中复述了某个选项文本
    return ""

def build_prompt_variant(question: str, answer_choices: List[str], variant_type: int) -> str:
    """
    构建不同风格的提示词变体，从多个角度提问。
    
    Args:
        variant_type: 提示词变体类型 (0-2)
    """
    choices_text = "\n".join(answer_choices or [])
    
    if variant_type == 0:
        # 标准版本
        prompt = (
            f"{question}\n\n"
            "请在以下选项中选择最合适的一项，并且只输出选项字母（A/B/C/D），不要输出任何解释：\n"
            f"{choices_text}\n\n"
            "只输出一个字母：A 或 B 或 C 或 D。"
        )
    elif variant_type == 1:
        # 强调推理的版本
        prompt = (
            f"{question}\n\n"
            "请仔细分析问题，在以下选项中选择正确答案。"
            "请基于图像内容进行逻辑推理，然后只输出选项字母（A/B/C/D）：\n"
            f"{choices_text}\n\n"
            "答案："
        )
    elif variant_type == 2:
        # 简洁直接的版本
        prompt = (
            f"{question}\n\n"
            "选项：\n"
            f"{choices_text}\n\n"
            "正确答案是："
        )
    else:
        # 强调准确性的版本
        prompt = (
            f"{question}\n\n"
            "这是一个重要的选择题，请确保选择最准确的答案。"
            "请只输出选项字母（A/B/C/D）：\n"
            f"{choices_text}\n\n"
            "最准确的答案是："
        )
    
    return prompt

# ========== 大模型封装 ==========
class MCQModel:
    def __init__(self, model_dir: str, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(
            model_dir, trust_remote_code=True,        
            attn_implementation="eager",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        ).eval().to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    @torch.inference_mode()
    def answer_mcq(self, image: Image.Image, question: str, choices: List[str]) -> str:
        """
        走模型的 chat 接口（常见的多模态范式）。
        """
        prompt = build_prompt_variant(question, choices, 0)  # 使用标准版本
        msgs = [{"role": "user", "content": [image, prompt]}]
        resp = self.model.chat(image=None, msgs=msgs, tokenizer=self.tokenizer)
        return resp or ""

class VotingMCQModel:
    """
    带投票机制的MCQ模型，通过多次生成答案并投票来提高稳定性
    """
    def __init__(self, model_dir: str, num_votes: int = 3, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.num_votes = num_votes
        self.model = AutoModel.from_pretrained(
            model_dir, trust_remote_code=True,        
            attn_implementation="eager",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        ).eval().to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        
    @torch.inference_mode()
    def answer_mcq_with_voting(self, image: Image.Image, question: str, choices: List[str]) -> Dict[str, Any]:
        """
        使用投票机制回答多选题，返回详细投票结果
        
        Returns:
            Dict containing:
                - final_answer: 最终选择的答案
                - votes: 所有投票结果列表
                - vote_distribution: 投票分布统计
                - confidence: 置信度（最高票数/总票数）
        """
        votes = []
        vote_details = []
        
        # 多次生成答案
        for i in range(self.num_votes):
            # 使用不同的提示词变体
            prompt = build_prompt_variant(question, choices, i % 4)
            msgs = [{"role": "user", "content": [image, prompt]}]
            
            try:
                resp = self.model.chat(image=None, msgs=msgs, tokenizer=self.tokenizer)
                pred_letter = extract_option_letter(resp)
                
                votes.append(pred_letter)
                vote_details.append({
                    "vote_id": i,
                    "prompt_variant": i % 4,
                    "raw_response": resp,
                    "extracted_letter": pred_letter
                })
            except Exception as e:
                print(f"Vote {i} failed: {e}")
                votes.append("")  # 投票失败记为空
                vote_details.append({
                    "vote_id": i,
                    "prompt_variant": i % 4,
                    "raw_response": f"ERROR: {e}",
                    "extracted_letter": ""
                })
        
        # 统计投票结果
        valid_votes = [v for v in votes if v]  # 只统计有效的投票（非空）
        if not valid_votes:
            # 所有投票都无效，返回空答案
            final_answer = ""
            confidence = 0.0
        else:
            vote_counter = Counter(valid_votes)
            # 选择得票最多的选项
            final_answer, max_votes = vote_counter.most_common(1)[0]
            confidence = max_votes / len(valid_votes)
            
            # 处理平票情况：如果平票且置信度低于阈值，可能需要额外处理
            if len(vote_counter) > 1 and confidence < 0.6:
                # 可以在这里添加平票处理逻辑，比如选择第一个出现的选项
                # 或者进行额外投票
                pass
        
        # 构建投票分布
        vote_distribution = {letter: 0 for letter in ["A", "B", "C", "D"]}
        for letter in votes:
            if letter in vote_distribution:
                vote_distribution[letter] += 1
        
        return {
            "final_answer": final_answer,
            "votes": votes,
            "vote_details": vote_details,
            "vote_distribution": vote_distribution,
            "confidence": confidence,
            "valid_votes_count": len(valid_votes),
            "total_votes_count": len(votes)
        }

# ========== 主流程 ==========
def run_eval_with_voting(input_json: str, image_root: str, model_dir: str, out_json: str, num_votes: int = 3):
    """
    使用投票机制运行评估
    """
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)

    model = VotingMCQModel(model_dir, num_votes=num_votes)

    total = 0
    correct = 0
    details: List[Dict[str, Any]] = []
    confidence_scores = []

    for item_idx, item in enumerate(data):
        img_rel = item.get("Image", "")
        img_path = img_rel if os.path.isabs(img_rel) else os.path.join(image_root, img_rel)
        question = item.get("Text", "")
        choices  = item.get("Answer choices", []) or []
        gt_raw   = item.get("Ground truth", "")
        qid      = item.get("Question id") or item.get("Question_id") or f"q_{item_idx}"

        print(f"Processing {item_idx + 1}/{len(data)}: {qid}")

        # 读图
        try:
            image = open_image_rgb(img_path)
        except Exception as e:
            print(f"Failed to open image {img_path}: {e}")
            continue

        # 使用投票机制询问模型
        voting_result = model.answer_mcq_with_voting(image, question, choices)
        pred = voting_result["final_answer"]
        gt   = extract_option_letter(gt_raw)

        # 统计
        if gt:
            total += 1
            if pred == gt:
                correct += 1
            confidence_scores.append(voting_result["confidence"])

        # 记录详细结果
        detail_record = {
            "Question_id": qid,
            "Image": img_rel,
            "Image_abs": img_path,
            "Text": question,
            "Answer_choices": choices,
            "GT_letter": gt,
            "Voting_result": voting_result
        }
        details.append(detail_record)

        # 实时输出进度
        if total > 0:
            current_acc = correct / total
            print(f"Current accuracy: {correct}/{total} = {current_acc:.4f}, "
                  f"Avg confidence: {sum(confidence_scores)/len(confidence_scores):.4f}")

    # 计算最终结果
    accuracy = (correct / total) if total > 0 else None
    avg_confidence = (sum(confidence_scores) / len(confidence_scores)) if confidence_scores else 0
    
    result = {
        "accuracy": accuracy,
        "total_questions": total,
        "correct": correct,
        "average_confidence": avg_confidence,
        "voting_config": {
            "num_votes": num_votes,
            "vote_strategy": "majority_vote"
        },
        "detail": details
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"[SAVE] {out_json}")

    # 输出统计信息
    if total > 0:
        print(f"\n=== 最终结果 ===")
        print(f"准确率: {correct}/{total} = {accuracy:.4f}")
        print(f"平均置信度: {avg_confidence:.4f}")
        print(f"投票次数: {num_votes}")
        
        # 分析投票稳定性
        stable_predictions = sum(1 for detail in details 
                               if detail["Voting_result"]["confidence"] >= 0.6)
        print(f"高置信度预测比例: {stable_predictions}/{len(details)} = {stable_predictions/len(details):.4f}")
    else:
        print("未计算准确率（GT 缺失或为空）。")

def run_eval_original(input_json: str, image_root: str, model_dir: str, out_json: str):
    """
    原始的单次推理版本（保持向后兼容）
    """
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)

    model = MCQModel(model_dir)

    total = 0
    correct = 0
    details: List[Dict[str, Any]] = []

    for item in data:
        img_rel = item.get("Image", "")
        img_path = img_rel if os.path.isabs(img_rel) else os.path.join(image_root, img_rel)
        question = item.get("Text", "")
        choices  = item.get("Answer choices", []) or []
        gt_raw   = item.get("Ground truth", "")
        qid      = item.get("Question id") or item.get("Question_id") or None

        # 读图
        image = open_image_rgb(img_path)

        # 询问模型
        raw_answer = model.answer_mcq(image, question, choices)
        pred = extract_option_letter(raw_answer)
        gt   = extract_option_letter(gt_raw)

        # 统计
        if gt:
            total += 1
            if pred == gt:
                correct += 1

        # 记录
        details.append({
            "Question_id": qid,
            "Image": img_rel,
            "Image_abs": img_path,
            "Text": question,
            "Answer_choices": choices,
            "Model_raw": raw_answer,
            "Pred_letter": pred,
            "GT_letter": gt,
        })

    result = {
        "accuracy": (correct / total) if total > 0 else None,
        "total": total,
        "correct": correct,
        "detail": details
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"[SAVE] {out_json}")

    if total > 0:
        print(f"Accuracy: {correct}/{total} = {correct/total:.4f}")
    else:
        print("未计算准确率（GT 缺失或为空）。")

# ========== CLI ==========
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCQ eval with voting mechanism")
    parser.add_argument("--input_json",  type=str, required=True,  help="Path to input QA json")
    parser.add_argument("--image_root",  type=str, required=True,  help="Root dir of images")
    parser.add_argument("--model_dir",   type=str, required=True,  help="FM9G (or compatible) model dir")
    parser.add_argument("--output_json", type=str, required=True,  help="Path to save eval results json")
    parser.add_argument("--use_voting",  default=True,    help="Enable voting mechanism")
    parser.add_argument("--num_votes",   type=int, default=3,      help="Number of votes when using voting mechanism")
    
    args = parser.parse_args()

    if args.use_voting:
        print(f"使用投票机制，投票次数: {args.num_votes}")
        run_eval_with_voting(args.input_json, args.image_root, args.model_dir, args.output_json, args.num_votes)
    else:
        print("使用原始单次推理")
        run_eval_original(args.input_json, args.image_root, args.model_dir, args.output_json)
