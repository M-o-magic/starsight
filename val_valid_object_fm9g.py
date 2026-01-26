# -*- coding: utf-8 -*-
"""
基于FM9G的裁剪区域多选判题脚本
- 从题干解析 Bounding box -> 裁剪ROI
- 以 [裁剪图像 + 题干(去坐标提示) + 选项(A/B/C/D)] 喂给FM9G
- 限制只输出一个选项字母
- 汇总准确率并导出详细JSON
"""

import os,argparse
import re
import json
from typing import Any, Dict, List, Tuple, Optional

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer


# ========== 图像读取与裁剪 ==========

def open_image_rgb(path: str) -> Image.Image:
    """
    优先用 PIL 读并转 RGB；失败则回退到 tifffile，再转 PIL->RGB。
    """
    try:
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except Exception as e_pil:
        try:
            import tifffile as tiff
            arr = tiff.imread(path)
            # 统一转三通道
            if arr.ndim == 2:
                arr = arr[..., None].repeat(3, axis=-1)
            elif arr.ndim == 3 and arr.shape[2] > 3:
                arr = arr[..., :3]
            img = Image.fromarray(arr)
            if img.mode != "RGB":
                img = img.convert("RGB")
            return img
        except Exception as e_tif:
            raise RuntimeError(
                f"Cannot open image as RGB: {path}\n"
                f"PIL error: {e_pil}\nTIFF fallback error: {e_tif}"
            )


def parse_bbox(text: str) -> Optional[Tuple[int, int, int, int]]:
    """
    从文本中解析出边界框坐标，支持英文 "Bounding box" 和中文 "边界框"。
    格式示例：
        "Bounding box: [123, 456, 789, 1011]"
        "边界框: [123, 456, 789, 1011]"
        "边界框：[123，456，789，1011]"
    """
    if not text:
        return None
    
    # 英文或中文关键字
    pattern = r"(Bounding\s*box|边界框)\s*[:：]?\s*[\[\【]\s*(\d+)[,\，]\s*(\d+)[,\，]\s*(\d+)[,\，]\s*(\d+)\s*[\]\】]"
    m = re.search(pattern, text, flags=re.I)
    if not m:
        return None
    
    # group(2)-group(5) 才是四个数字
    x1, y1, x2, y2 = map(int, m.groups()[1:])
    return x1, y1, x2, y2


def clamp_bbox_to_image(
    bbox: Tuple[int, int, int, int], img_w: int, img_h: int
) -> Tuple[int, int, int, int]:
    """
    将 bbox 裁剪到图像范围内，并确保 x1<=x2, y1<=y2。
    """
    x1, y1, x2, y2 = bbox
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(0, min(x2, img_w))
    y2 = max(0, min(y2, img_h))
    return x1, y1, x2, y2


def safe_crop(img: Image.Image, bbox: Tuple[int, int, int, int], pad: int = 0) -> Image.Image:
    """
    按 bbox 裁剪并可选 padding，自动裁剪到图像边界。
    bbox: (x1, y1, x2, y2)  —— 右下角是“开区间”坐标更安全，PIL crop 支持
    """
    w, h = img.size
    x1, y1, x2, y2 = bbox
    if pad > 0:
        x1 -= pad
        y1 -= pad
        x2 += pad
        y2 += pad
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        # 兜底：异常时返回原图
        return img
    return img.crop((x1, y1, x2, y2))


def strip_bbox_in_question(text: str) -> str:
    """
    从题干中删除坐标提示，兼容：
    - 英文：Bounding box: [x1, y1, x2, y2]
    - 中文：边界框：[x1，y1，x2，y2]
    支持中英文冒号/逗号、不同括号形式与可选空白。
    """
    if not text:
        return ""
    pattern = (
        r"(?:Bounding\s*box|边界框)"      # 关键词（英文/中文）
        r"\s*[:：]?\s*"                   # 可选冒号（英文/中文）
        r"[（(【\[]\s*"                   # 左括号（中/英）
        r"[\d\s,，]+"                     # 坐标内容：数字/空格/逗号（中/英）
        r"\s*[）)】\]]"                   # 右括号（中/英）
        r"\s*[。．\.，,]?\s*"             # 可选收尾标点与空白
    )
    return re.sub(pattern, "", text, flags=re.I).strip()



# ========== 选项与答案工具 ==========

def strip_choice_prefix(choice: str) -> str:
    """ 去掉 '(A) ' 之类前缀，只保留选项语义文本 """
    return re.sub(r"^\([A-Za-z]\)\s*", "", choice or "").strip()


def parse_choice_letters(choices: List[str]) -> List[str]:
    """
    从 ["(A) ...", "(B) ...", ...] 提取字母序列 ["A","B","C","D"...]；若无则按顺序分配。
    """
    letters = []
    for i, ch in enumerate(choices):
        m = re.match(r"^\(([A-Za-z])\)\s*", (ch or "").strip())
        if m:
            letters.append(m.group(1).upper())
        else:
            letters.append(chr(ord('A') + i))
    return letters


def canonicalize(s: str) -> str:
    """ 文本宽松归一：小写、去空格/下划线/连字符、去结尾小写复数s """
    if s is None:
        return ""
    t = re.sub(r"[\s_\-]+", "", s.strip().lower())
    if t.endswith("s") and len(t) > 3:
        t = t[:-1]
    return t


def label_to_letter(gt_raw: Any, choices: List[str]) -> str:
    """
    将 GT 转换为字母：
    - 若已是 A/B/C/D 则直接用；
    - 否则和选项文本（去前缀）宽松匹配；
    - 兼容 yes/no -> 匹配到对应选项。
    """
    if gt_raw is None:
        return ""
    text = str(gt_raw).strip()

    # 直接字母
    m = re.match(r"^\s*([A-Da-d])\s*$", text)
    if m:
        return m.group(1).upper()

    # 宽松文本对齐
    can_gt = canonicalize(text)
    choices_letters = parse_choice_letters(choices)
    choice_texts = [strip_choice_prefix(c) for c in choices]
    can_texts = [canonicalize(t) for t in choice_texts]

    # yes/no 单独兜底
    if can_gt in {"yes", "no"}:
        for L, t in zip(choices_letters, can_texts):
            if can_gt == canonicalize(t):
                return L

    # 普通对齐
    for L, t in zip(choices_letters, can_texts):
        if can_gt and can_gt == t:
            return L

    # 最后再尝试包含匹配
    for L, t in zip(choices_letters, can_texts):
        if can_gt and (can_gt in t or t in can_gt):
            return L
    return ""


def extract_option_letter_from_output(s: Any) -> str:
    """
    从模型输出里提取 A/B/C/D 的选项字母（返回大写或空串）。
    兼容 '(A)', 'A:', '答案A', '选B', 'Option C', 'D'，以及 '1/2/3/4' -> A/B/C/D。
    """
    if s is None:
        return ""
    text = str(s).strip().upper()

    # 规范形式
    m = re.search(r'[\(\[\{<]?\s*([ABCD])\s*[\)\]\}>:\.]?', text)
    if m:
        return m.group(1)

    # 数字到字母
    m2 = re.search(r'[\s:：\-]*(\d)\b', text)
    if m2:
        num = m2.group(1)
        mapping = {"1": "A", "2": "B", "3": "C", "4": "D"}
        if num in mapping:
            return mapping[num]

    # 宽松包含
    for ch in ["A", "B", "C", "D"]:
        if re.search(rf'\b{ch}\b', text):
            return ch

    return ""


# ========== Prompt 构造 ==========

def build_prompt(question: str, answer_choices: List[str]) -> str:
    """
    提示词：明确“这张图已经是题干中参考框的裁剪区域”，并强制只输出一个字母。
    """
    choices_text = "\n".join(answer_choices or [])
    prompt = (
        "以下是一个遥感多选题。注意：你看到的图像已经是题干中参考框的裁剪区域，请只基于裁剪图判断。\n\n"
        f"{question}\n\n"
        "请在以下选项中选择最合适的一项，并且只输出选项字母（A/B/C/D），不要输出任何解释：\n"
        f"{choices_text}\n\n"
        "只输出一个字母：A 或 B 或 C 或 D。"
    )
    return prompt


# ========== 大模型封装 ==========

class FM9GMCQ:
    def __init__(self, model_dir: str, device_prefer: str = "cuda"):
        self.device = (
            device_prefer if (device_prefer.startswith("cuda") and torch.cuda.is_available())
            else ("cpu" if device_prefer == "cpu" else ("cuda" if torch.cuda.is_available() else "cpu"))
        )
        dtype = torch.bfloat16 if (self.device == "cuda") else torch.float32

        self.model = AutoModel.from_pretrained(
            model_dir,
            trust_remote_code=True,
            attn_implementation="sdpa",
            torch_dtype=dtype
        ).eval().to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    @torch.inference_mode()
    def answer(self, image: Image.Image, question: str, choices: List[str]) -> str:
        """
        通过 chat 接口，让模型在 [裁剪图 + 题干(去坐标) + 选项] 上直接输出 A/B/C/D。
        说明：绝大多数 FM9G 实现允许把图像对象放进 msgs[0]["content"] 列表。
        """
        prompt = build_prompt(question, choices)
        msgs = [{"role": "user", "content": [image, prompt]}]
        # 一些实现写法是 chat(image=None, msgs=..., tokenizer=...)；保持与用户提供实现一致
        resp = self.model.chat(image=None, msgs=msgs, tokenizer=self.tokenizer)
        return resp or ""


# ========== 评测主流程 ==========

def run_eval_on_crops(
    input_json: str,
    image_root: str,
    fm9g_dir: str,
    out_json: str,
    pad: int = 2,
    save_crops_dir: Optional[str] = None,
    device_prefer: str = "cuda"
):
    """
    逐条题目：
      - 打开原图 -> 解析bbox -> 裁剪 -> 问FM9G -> 抽取选项字母
      - 与GT对齐计算准确率
      - 导出详细结果
    """
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("输入 JSON 顶层应为 list[dict]")

    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    if save_crops_dir:
        os.makedirs(save_crops_dir, exist_ok=True)

    model = FM9GMCQ(fm9g_dir, device_prefer=device_prefer)

    total = 0
    correct = 0
    details: List[Dict[str, Any]] = []

    for i, item in enumerate(data):
        qid = item.get("Question id") or item.get("Question_id") or f"q_{i:06d}"
        img_field = (item.get("Image") or "").strip()
        text = item.get("Text", "")
        choices: List[str] = item.get("Answer choices", []) or []
        gt_raw = item.get("Ground truth", "")

        # 路径拼接
        img_path = img_field if os.path.isabs(img_field) else os.path.join(image_root, img_field)
        if not os.path.exists(img_path):
            details.append({
                "Question_id": qid,
                "Image": img_field,
                "Image_abs": img_path,
                "error": "image_not_found"
            })
            continue

        # 读图
        try:
            img = open_image_rgb(img_path)
        except Exception as e_img:
            details.append({
                "Question_id": qid,
                "Image": img_field,
                "Image_abs": img_path,
                "error": f"open_image_fail: {e_img}"
            })
            continue

        # 解析 bbox 并裁剪
        bbox = parse_bbox(text)
        if bbox is None:
            # 若没有 bbox，退化用整图（但仍然说明“已裁剪”不会影响选择）
            crop = img
            bbox_clamped = None
        else:
            w, h = img.size
            bbox_clamped = clamp_bbox_to_image(bbox, w, h)
            crop = safe_crop(img, bbox_clamped, pad=pad)

        # 题干里去掉具体坐标，避免干扰
        q_for_crop = strip_bbox_in_question(text)

        # 模型推理
        try:
            raw_answer = model.answer(crop, q_for_crop, choices)
        except Exception as e_run:
            details.append({
                "Question_id": qid,
                "Image": img_field,
                "Image_abs": img_path,
                "BBox": bbox_clamped,
                "error": f"model_infer_fail: {e_run}"
            })
            continue

        pred_letter = extract_option_letter_from_output(raw_answer)
        gt_letter = label_to_letter(gt_raw, choices)

        # 对齐预测到选项文本（便于检查）
        choice_letters = parse_choice_letters(choices)
        pred_choice_text = ""
        if pred_letter in choice_letters:
            pred_idx = choice_letters.index(pred_letter)
            pred_choice_text = strip_choice_prefix(choices[pred_idx])

        gt_choice_text = ""
        if gt_letter in choice_letters:
            gt_idx = choice_letters.index(gt_letter)
            gt_choice_text = strip_choice_prefix(choices[gt_idx])

        match = (pred_letter and gt_letter and (pred_letter == gt_letter))

        if gt_letter:
            total += 1
            if match:
                correct += 1

        # 可选保存裁剪图
        saved_crop_path = None
        if save_crops_dir:
            base = os.path.splitext(os.path.basename(img_path))[0]
            saved_crop_path = os.path.join(save_crops_dir, f"{base}_{qid}.jpg")
            try:
                crop.save(saved_crop_path, quality=90)
            except Exception:
                saved_crop_path = None

        details.append({
            "Question_id": qid,
            "Image": img_field,
            "Image_abs": img_path,
            "Saved_crop": saved_crop_path,
            "Text": text,
            "Text_used": q_for_crop,
            "BBox": bbox_clamped,
            "Answer_choices": choices,
            "Model_raw": raw_answer,
            "Pred_letter": pred_letter,
            "Pred_choice_text": pred_choice_text,
            "GT_letter": gt_letter,
            "GT_choice_text": gt_choice_text,
            "match": bool(match)
        })

    result = {
        "total": total,
        "correct": correct,
        "accuracy": (correct / total) if total > 0 else None,
        "detail": details
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    if total > 0:
        print(f"[ACC] {correct}/{total} = {correct/total:.4f}")
    else:
        print("[ACC] 未计算：无有效GT。")
    print(f"[SAVE] {out_json}")


# ========== MAIN 调用（按需修改） ==========

# if __name__ == "__main__":
#     # 示例：把下面四个路径替换成你的实际路径
#     INPUT_JSON  = "/home/mcislab_cj/VRSBench_images/valid/subset_low/en/Object_properties__Object_classification.json"
#     IMAGE_ROOT  = "/home/mcislab_cj/VRSBench_images/valid/images"
#     FM9G_DIR    = "/home/mcislab_cj/fm9g4bv/FM9G4B-V"
#     OUTPUT_JSON = "/home/mcislab_cj/VRSBench_images/valid/eval_Object_classification_crop_fm9g_results.json"

#     # 可选：保存裁剪图的目录（不需要就设为 None）
#     SAVE_CROPS_DIR = None  # 如需保存，设为 "/data/cj/valid_contest/crops"

#     # 设备优先选择（"cuda" 或 "cpu"）
#     DEVICE_PREFER = "cuda"

#     run_eval_on_crops(
#         input_json=INPUT_JSON,
#         image_root=IMAGE_ROOT,
#         fm9g_dir=FM9G_DIR,
#         out_json=OUTPUT_JSON,
#         pad=2,
#         save_crops_dir=SAVE_CROPS_DIR,
#         device_prefer=DEVICE_PREFER
#     )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FM9G 裁剪区域多选判题")
    parser.add_argument("--input_json",  type=str, required=True, help="题目 JSON 路径")
    parser.add_argument("--image_root",  type=str, required=True, help="图像根目录")
    parser.add_argument("--model_dir",   type=str, required=True, help="FM9G 模型目录")
    parser.add_argument("--out_json",    type=str, required=True, help="评测结果输出 JSON 路径")
    parser.add_argument("--pad",         type=int, default=2,      help="裁剪时 bbox padding 像素")
    parser.add_argument("--save_crops_dir", type=str, default=None, help="保存裁剪图目录（可选）")
    parser.add_argument("--device",      type=str, default="cuda", choices=["cuda","cpu"], help="优先设备")
    args = parser.parse_args()

    run_eval_on_crops(
        input_json=args.input_json,
        image_root=args.image_root,
        fm9g_dir=args.model_dir,
        out_json=args.out_json,
        pad=args.pad,
        save_crops_dir=args.save_crops_dir,
        device_prefer=args.device
    )