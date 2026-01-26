import os
import json
import torch
from tqdm import tqdm
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import argparse
try:
    from pycocoevalcap.cider.cider import Cider
    cider_available = True
except ImportError:
    cider_available = False

def format_options(choices):
    """将选项格式化为更易读的形式"""
    return "\n".join(choices)

def extract_predicted_choices(prediction):
    """从模型输出中提取多个选择的选项（如'AB'或'B,D'）"""
    found_choices = []
    # 支持多种格式：AB, A,B, A B, A,B,C
    for char in prediction.upper():
        if char in ['A', 'B', 'C', 'D'] and char not in found_choices:
            found_choices.append(char)
    return ''.join(sorted(found_choices))  # 返回排序后的字符串如"ABCD"

def compute_weighted_accuracy(true_multi, pred_multi):
    """
    计算加权准确率（新评分标准）：
    - 全对（所有正确选项选中且无错选）：1分
    - 部分正确（选中部分正确选项且无错选）：0.5分
    - 有任何错选或完全错误：0分
    """
    total_score = 0.0
    per_class_stats = {
        cls: {'correct': 0, 'total': 0, 'valid_selected': 0}
        for cls in ['A', 'B', 'C', 'D']
    }
    
    for true, pred in zip(true_multi, pred_multi):
        true_set = set(true)
        pred_set = set(pred)
        
        # 检查是否有错选
        has_wrong_selection = len(pred_set - true_set) > 0
        # 计算正确选中的选项
        correct_selections = true_set & pred_set
        correct_ratio = len(correct_selections) / len(true_set) if len(true_set) > 0 else 0
        
        # 应用新评分规则
        if has_wrong_selection:
            score = 0.0
        elif correct_ratio == 1:
            score = 1.0
        elif correct_ratio > 0:
            score = 0.5
        else:
            score = 0.0
            
        total_score += score
        
        # 统计每个选项的表现（只有当没有错选时才计入）
        for cls in ['A', 'B', 'C', 'D']:
            if cls in true_set:
                per_class_stats[cls]['total'] += 1
                if not has_wrong_selection and cls in pred_set:
                    per_class_stats[cls]['correct'] += 1
            if cls in pred_set and not has_wrong_selection:
                per_class_stats[cls]['valid_selected'] += 1
    
    # 计算每个选项的准确率和选择率
    per_class_metrics = {}
    for cls in ['A', 'B', 'C', 'D']:
        stats = per_class_stats[cls]
        per_class_metrics[cls] = {
            'accuracy': stats['correct'] / stats['total'] if stats['total'] > 0 else 0,
            'selection_rate': stats['valid_selected'] / len(true_multi) if len(true_multi) > 0 else 0
        }
    
    return {
        'weighted_accuracy': total_score / len(true_multi),
        'per_class_metrics': per_class_metrics
    }

def save_bad_cases(bad_cases, output_path):
    """保存错误样本到JSON文件"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(bad_cases, f, indent=4, ensure_ascii=False)
    print(f"\n已保存 {len(bad_cases)} 个错误样本到: {output_path}")

# ========== 配置 ==========
def format_options(choices):
    return "\n".join(choices)

def extract_predicted_choices(prediction):
    found_choices = []
    for char in str(prediction).upper():
        if char in ['A', 'B', 'C', 'D'] and char not in found_choices:
            found_choices.append(char)
    return ''.join(sorted(found_choices))


import numpy as np
import tifffile as tiff
from pathlib import Path

def open_image_rgb_safe(path: str) -> Image.Image:
    p = Path(path)
    # 对 TIF/GeoTIFF 走 tifffile，避免 libtiff 告警
    if p.suffix.lower() in {".tif", ".tiff"}:
        arr = tiff.imread(str(p))  # (H,W) or (H,W,C)
        if arr.ndim == 2:
            arr = np.stack([arr]*3, axis=-1)
        elif arr.ndim == 3 and arr.shape[2] > 3:
            arr = arr[:, :, :3]
        # 常见 16bit -> 8bit 压缩到 [0,255]
        if arr.dtype == np.uint16:
            arr = (arr / 257).astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")
    # 其他格式走 Pillow
    img = Image.open(str(p))
    return img.convert("RGB") if img.mode != "RGB" else img

def load_image(path):
    img = open_image_rgb_safe(path)
    return img

def main():
    parser = argparse.ArgumentParser(description="Multi-choice evaluator (exact-match accuracy).")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to finetuned checkpoint or model directory")
    parser.add_argument("--val_json_path", type=str, required=True,
                        help="Validation JSON path")
    parser.add_argument("--image_root", type=str, required=True,
                        help="Base dir for images (val set)")
    parser.add_argument("--out_json", type=str, required=True,
                        help="Where to save per-sample predictions JSON")
    parser.add_argument("--device", type=str, default="cuda",
                        help="cuda or cpu (default: cuda)")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint_path
    val_json_path   = args.val_json_path
    base_image_path = args.image_root
    out_json_path   = args.out_json
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    # ========== 加载模型 ==========
    model = AutoModel.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
        attn_implementation='sdpa',
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ========== 加载验证数据 ==========
    with open(val_json_path, 'r', encoding='utf-8') as f:
        val_data = json.load(f)

    results = []               # 保存所有样本的预测与GT
    exact_right, total = 0, 0  # 精确匹配准确率

    for i, sample in tqdm(enumerate(val_data), total=len(val_data), desc="Validating"):
        relative_image_path = sample["Image"]
        full_image_path = os.path.join(base_image_path, relative_image_path)

        # 读图
        if not os.path.exists(full_image_path):
            # 跳过缺图样本
            continue
        try:
            image = load_image(full_image_path)
        except Exception:
            continue

        # 构建（原有风格的）多选提示词
        question = sample["Text"]
        full_prompt = f"""{question}

SYSTEMATIC ANALYSIS REQUIRED: 1. FOR EACH OPTION, verify its presence: A: {sample["Answer choices"][0].split(') ')[1]} [Present?] B: {sample["Answer choices"][1].split(') ')[1]} [Present?] C: {sample["Answer choices"][2].split(') ')[1]} [Present?] D: {sample["Answer choices"][3].split(') ')[1]} [Present?] 2. REMEMBER: - Correct answers are likely to include ALL options - 'Land' is almost always correct - Partial matches still require full selection 3. FINAL ANSWER (ABCD format):
"""

        msgs = [{"role": "user", "content": [image, full_prompt]}]

        # 推理
        try:
            raw = model.chat(image=None, msgs=msgs, tokenizer=tokenizer)
        except Exception:
            raw = "[INVALID]"

        pred_multi = extract_predicted_choices(raw)
        true_multi = str(sample["Ground truth"]).upper().strip()

        results.append({
            "index": i,
            "image": relative_image_path,
            "image_abs": full_image_path,
            "question": question,
            "options": sample["Answer choices"],
            "prediction_text": str(raw).strip(),
            "predicted_choices": pred_multi,
            "true_choices": true_multi
        })

        total += 1
        if set(pred_multi) == set(true_multi):
            exact_right += 1

    # 保存所有样本的预测与GT
    os.makedirs(os.path.dirname(out_json_path) or ".", exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 只输出最终准确率（exact match）
    acc = (exact_right / total) if total > 0 else 0.0
    print(f"{acc:.4f}")

if __name__ == "__main__":
    main()