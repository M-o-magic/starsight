import os, json, re, torch,argparse
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# ===== 路径配置 =====
parser = argparse.ArgumentParser()
parser.add_argument("--input_json",  type=str, default="/home/mcislab_cj/VRSBench_images/valid/subset_low/en/Object_properties__Object_color.json")
parser.add_argument("--image_root", type=str, default="/home/mcislab_cj/VRSBench_images/valid/images")
parser.add_argument("--model_dir",      type=str, default="/home/mcislab_cj/fm9g4bv/FM9G4B-V")
parser.add_argument("--out_json",        type=str, default="/home/mcislab_cj/VRSBench_images/valid/mme_eval_results_color_crop_fm9g_clean_question.json")
args = parser.parse_args()

# ===== 路径配置（使用命令行参数）=====
mme_json_file = args.input_json
image_root     = args.image_root
model_file     = args.model_dir
save_path      = args.out_json
# ===== 模型加载 =====
print(mme_json_file)
model = AutoModel.from_pretrained(
    model_file, trust_remote_code=True,
    attn_implementation='sdpa', torch_dtype=torch.bfloat16
).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(model_file, trust_remote_code=True)

# ===== 工具函数 =====
import re

_bbox_pat = re.compile(
    r"(?:Bounding\s*box|边界框)\s*[:：]?\s*[\[\【]\s*"
    r"([0-9]+)\s*[,，]\s*([0-9]+)\s*[,，]\s*([0-9]+)\s*[,，]\s*([0-9]+)\s*"
    r"[\]\】]",
    flags=re.IGNORECASE
)


def parse_bbox(text):
    """
    从 Text 中解析 'Bounding box: [x1, y1, x2, y2]'
    返回 (x1, y1, x2, y2) 或 None
    """
    m = _bbox_pat.search(text)
    if not m:
        return None
    return tuple(map(int, m.groups()))

def safe_crop(img: Image.Image, bbox, pad=2):
    """
    对 bbox 裁剪并可选 padding；自动裁到图像边界内
    bbox: (x1, y1, x2, y2)
    """
    w, h = img.size
    x1, y1, x2, y2 = bbox
    if pad > 0:
        x1 -= pad; y1 -= pad; x2 += pad; y2 += pad
    x1 = max(0, min(x1, w-1))
    y1 = max(0, min(y1, h-1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return img  # 框异常就退回整图
    return img.crop((x1, y1, x2, y2))

def extract_choice_letter_list(choices):
    """
    从选项数组里提取顺序字母列表和“字母->原文”的映射
    e.g. ["(A) White airplane", "(B) Red buildings"] ->
         letters = ["A","B"], map = {"A":"(A) White airplane", ...}
    """
    letters, letter2raw = [], {}
    for opt in choices:
        m = re.match(r"\(\s*([A-Za-z])\s*\)\s*", opt)
        if m:
            L = m.group(1).upper()
            letters.append(L)
            letter2raw[L] = opt
    return letters, letter2raw
import re

def clean_question(text):
    """
    去掉坐标提示：
    - 英文：Bounding box: [x1, y1, x2, y2]
    - 中文：边界框：[x1，y1，x2，y2] / 边界框（…）/ 边界框【…】
    """
    pattern = (
        r"(?:Bounding\s*box|边界框)"   # 关键词（英文/中文）
        r"\s*[:：]?\s*"                # 可选冒号（中/英）
        r"[（(【\[]\s*"                # 左括号（中/英）
        r"[\d\s,，]+"                  # 坐标内容：数字/空格/逗号（中/英）
        r"\s*[）)】\]]"                # 右括号（中/英）
        r"(?:\s*[。．\.，,])?"          # 可选收尾标点
    )
    return re.sub(pattern, "", text, flags=re.I).strip()

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
def build_prompt(qitem):
    """
    构建更强约束的多选 prompt（只输出选项字母）
    """
    question = clean_question(qitem['Text'])
    options  = qitem['Answer choices']
    opts_str = "\n".join([opt for opt in options])
    prompt = (
        # f"{question}\n"
        'Determine the color of the object in the images. \n'
        "You are given the CROPPED region corresponding to the bounding box.\n"
        "Choose the single best answer from the options.\n"
        "Output ONLY the option letter (A/B/C/D/E). Do not explain.\n"
        "Options:\n" + opts_str + "\n"
        "Answer (letter only):"
    )
    return prompt

def robust_pick_letter(raw_out, allowed_letters):
    """
    从模型输出中稳健提取一个合法选项字母。
    - 先匹配独立字母（避免误击中单词里的 A/B/C）
    - 若多个，取第一个；若没有，尝试包含判断；仍无就返回 None
    """
    s = raw_out.strip().upper()
    # 1) 独立字母（行首/行尾/空白分隔）
    m = re.search(r"\b([A-E])\b", s)
    if m and m.group(1) in allowed_letters:
        return m.group(1)
    # 2) 退而求其次：优先级顺序扫描
    for L in allowed_letters:
        if re.search(rf"\b{L}\b", s):
            return L
    for L in allowed_letters:
        if L in s:
            return L
    return None

# ===== 数据加载 =====
with open(mme_json_file, "r", encoding="utf-8") as f:
    mme_data = json.load(f)

answer_letters_master = ['A', 'B', 'C', 'D', 'E']

# ===== 评测（裁剪后输入大模型）=====
correct, total = 0, 0
results = []

for qitem in tqdm(mme_data):
    # 路径
    img_path = os.path.join(image_root, qitem['Image'])
    if not os.path.exists(img_path):
        print(f"[MISS] 图像不存在: {img_path}")
        continue

    # 读图
    # image = Image.open(img_path).convert('RGB')
    image = open_image_rgb_safe(img_path)

    # 解析 bbox 并裁剪
    bbox = parse_bbox(qitem['Text'])
    if bbox is not None:
        crop = safe_crop(image, bbox, pad=4)   # pad 可按需调大 (如 8~16)
        input_image = crop
    else:
        # 没解析到就退回整图（也方便你定位失败样本）
        input_image = image
        print(f"[WARN] 未解析到bbox，使用整图：{qitem.get('Question id') or qitem.get('Question_id')}")

    # 构建 prompt（声明我们给的是裁剪图）
    prompt = build_prompt(qitem)
    print(prompt)

    # 选项字母集合（按题目实际给的为准）
    allowed_letters, _ = extract_choice_letter_list(qitem['Answer choices'])
    if not allowed_letters:
        allowed_letters = answer_letters_master  # 兜底

    # VLM 聊天
    msgs = [{'role': 'user', 'content': [input_image, prompt]}]
    with torch.inference_mode():
        raw = model.chat(image=None, msgs=msgs, tokenizer=tokenizer)

    pred_letter = robust_pick_letter(raw, allowed_letters)
    gt_letter   = qitem['Ground truth'].strip().upper()

    # 记录
    qid = qitem.get('Question id') or qitem.get('Question_id') or ""
    results.append({
        "qid": qid,
        "image": qitem['Image'],
        "bbox": bbox if bbox is not None else "",
        "pred": pred_letter,
        "gt": gt_letter,
        "raw_output": raw
    })
    print("pred", pred_letter,
        "gt", gt_letter,
        "raw_output", raw)

    total += 1
    if pred_letter == gt_letter:
        correct += 1

# 汇总
acc = (correct/total) if total > 0 else 0.0
print(f"\nMME多选题（裁剪输入）评测：正确 {correct}/{total}, 准确率={acc:.4f}")

with open(save_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("评测日志已保存：", save_path)
