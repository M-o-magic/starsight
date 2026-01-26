import os
import re
import json
import argparse
from typing import Any, Dict, List, Tuple, Optional

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

def build_prompt(question: str, answer_choices: List[str]) -> str:
    """
    中文多选提示词：强制只输出 A/B/C/D。
    answer_choices 形如 ["(A) ...", "(B) ...", ...]
    """
    choices_text = "\n".join(answer_choices or [])
    prompt = (
        f"{question}\n\n"
        "请在以下选项中选择最合适的一项，并且只输出选项字母（A/B/C/D），不要输出任何解释：\n"
        f"{choices_text}\n\n"
        "只输出一个字母：A 或 B 或 C 或 D。"
    )
    return prompt

# ========== 大模型封装 ==========
class MCQModel:
    def __init__(self, model_dir: str, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(
            model_dir, trust_remote_code=True, attn_implementation="sdpa",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        ).eval().to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    @torch.inference_mode()
    def answer_mcq(self, image: Image.Image, question: str, choices: List[str]) -> str:
        """
        走模型的 chat 接口（常见的多模态范式）。如果你的模型是别的 API，可在这里改。
        """
        prompt = build_prompt(question, choices)
        msgs = [{"role": "user", "content": [image, prompt]}]
        # 某些实现需要 image=None，仅把图像放在 content 里；这里沿用常见自定义 chat 习惯。
        resp = self.model.chat(image=None, msgs=msgs, tokenizer=self.tokenizer)
        return resp or ""

# ========== 主流程 ==========
def run_eval(input_json: str, image_root: str, model_dir: str, out_json: str):
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

    if total > 0:
        print(f"Accuracy: {correct}/{total} = {correct/total:.4f}")
    else:
        print("未计算准确率（GT 缺失或为空）。")

# ========== CLI（可选） ==========
# def parse_args():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--input_json",  default=INPUT_JSON)
#     ap.add_argument("--image_root",  default=IMAGE_ROOT)
#     ap.add_argument("--model_dir",   default=MODEL_DIR)
#     ap.add_argument("--output_json", default=OUTPUT_JSON)
    # return ap.parse_args()

if __name__ == "__main__":
    # args = parse_args()
    input_json='/data/cj/valid_contest/subset_high/Object_spatial_relationship__Object_spatial_relationship.json'
    image_root='/data/cj/valid_contest/valid_images'
    model_dir='/data/cj/FM9G4B-V'
    output_json='/data/cj/valid_contest/eval_high_Object_spatial_relationship.json'
    run_eval(input_json, image_root, model_dir, output_json)
