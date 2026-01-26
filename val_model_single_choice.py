import os
import json
import torch
from tqdm import tqdm
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from sklearn.metrics import accuracy_score
import argparse

try:
    from pycocoevalcap.cider.cider import Cider
    cider_available = True
except ImportError:
    cider_available = False

# 顶部增加:
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

def format_options(choices):
    """将选项格式化为更易读的形式"""
    return "\n".join(choices)

def extract_predicted_choice(prediction):
    """从模型输出中提取选择的选项"""
    for char in prediction:
        if char.upper() in ['A', 'B', 'C', 'D']:
            return char.upper()
    return prediction[:1].upper()

def compute_accuracy(references, predictions):
    """计算准确率（选项匹配）"""
    if not references:
        return 0.0
    correct = 0
    for ref, pred in zip(references, predictions):
        if ref == pred:
            correct += 1
    return correct / len(references)

def save_jsonl(records, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

# ========== 配置 ==========
def main():
    parser = argparse.ArgumentParser(description="Land-use multi-choice exact-match evaluator")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Model/ckpt path")
    parser.add_argument("--val_json_path", type=str, required=True, help="Validation JSON path")
    parser.add_argument("--image_root", type=str, required=True, help="Base image directory")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu (default: cuda)")
    parser.add_argument("--save_json", type=str, default=None, help="Optional path to save per-sample results JSON")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint_path
    val_json_path   = args.val_json_path
    base_image_path = args.image_root
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    # 加载模型
    model = AutoModel.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
        attn_implementation='sdpa',
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载验证数据
    with open(val_json_path, 'r', encoding='utf-8') as f:
        val_data = json.load(f)

    pred_choices = []
    true_choices = []
    records = []  # ← 新增：收集每个样本的结果

    for i, sample in tqdm(enumerate(val_data), total=len(val_data), desc="Validating"):
        relative_image_path = sample["Image"]
        full_image_path = os.path.join(base_image_path, relative_image_path)

        # 读图（优先 png，其次原始后缀）
        img_path_try = full_image_path.replace('.tif', '.png').replace('.TIF', '.png')
        try_path = img_path_try if os.path.exists(img_path_try) else full_image_path
        if not os.path.exists(try_path):
            # 缺图则记录并跳过
            records.append({
                "idx": i,
                "image": relative_image_path,
                "image_abs": try_path,
                "error": "image_not_found"
            })
            continue
        try:
            image = open_image_rgb_safe(str(try_path))
        except Exception as e:
            records.append({
                "idx": i,
                "image": relative_image_path,
                "image_abs": try_path,
                "error": f"open_image_fail: {e}"
            })
            continue

        # 构建包含选项的 prompt
        question = sample["Text"]
        options = format_options(sample["Answer choices"])
        full_prompt = f"""{question}

        You are an expert in land use classification. Analyze the image carefully and:
        1. Focus specifically on the area marked with red circle
        2. Compare all given options systematically
        3. Pay special attention to distinguishing features:
           - Helipad: small square/round platform, usually with 'H' marking
           - Airport Runway: long straight strip for airplanes
           - Airplane: entire aircraft shape

        Options:
        {options}

        Provide your answer strictly as a single letter (A, B, C or D).
        Answer:"""

        msgs = [{"role": "user", "content": [image, full_prompt]}]

        # 推理
        try:
            res = model.chat(image=None, msgs=msgs, tokenizer=tokenizer)
        except Exception as e:
            res = "[INVALID]"

        # 预测与GT
        predicted_choice = extract_predicted_choice(res)
        gt_choice = str(sample["Ground truth"]).upper().strip()

        pred_choices.append(predicted_choice)
        true_choices.append(gt_choice)

        # 记录每个样本
        records.append({
            "idx": i,
            "image": relative_image_path,
            "image_abs": str(try_path),
            "question": question,
            "options": sample["Answer choices"],
            "prediction_raw": str(res).strip(),
            "pred_choice": predicted_choice,
            "gt_choice": gt_choice,
            "correct": bool(predicted_choice == gt_choice),
        })

    # 只输出准确率（exact match）
    accuracy = compute_accuracy(true_choices, pred_choices)
    print(f"{accuracy:.4f}")

    # 可选：保存结果 JSON
    if args.save_json:
        save_jsonl(records, args.save_json)

if __name__ == "__main__":
    main()
