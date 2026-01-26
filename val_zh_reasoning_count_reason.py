# import os
# import re
# import json
# import argparse
# from typing import Any, Dict, List, Tuple, Optional

# import torch
# from PIL import Image
# from transformers import AutoModel, AutoTokenizer

# # ========== 路径配置（可按需修改） ==========
# # INPUT_JSON  = "/path/to/qa_items.json"  # 你的题目JSON，示例见提问
# # IMAGE_ROOT  = "/home/mcislab_cj/MME_realworld"  # 图像根目录，和JSON里的相对路径拼接
# # MODEL_DIR   = "/home/mcislab_cj/fm9g4bv/FM9G4B-V"  # 大模型路径（FM9G等）
# # OUTPUT_JSON = "/path/to/qa_results.json"

# # ========== 工具函数 ==========
# def open_image_rgb(path: str) -> Image.Image:
#     """
#     尝试以 RGB 读取图像。
#     1) PIL 直接读取 + convert("RGB")
#     2) 如遇部分 GeoTIFF/TIFF 读不动，回退到 tifffile 再转 PIL
#     """
#     try:
#         img = Image.open(path)
#         if img.mode != "RGB":
#             img = img.convert("RGB")
#         return img
#     except Exception as e:
#         try:
#             import tifffile as tiff
#             arr = tiff.imread(path)
#             # 统一转三通道
#             if arr.ndim == 2:
#                 # 灰度 -> RGB
#                 arr = (arr[..., None]).repeat(3, axis=-1)
#             elif arr.ndim == 3 and arr.shape[2] > 3:
#                 arr = arr[..., :3]
#             img = Image.fromarray(arr)
#             if img.mode != "RGB":
#                 img = img.convert("RGB")
#             return img
#         except Exception as e2:
#             raise RuntimeError(f"Cannot open image as RGB: {path}\nPIL error: {e}\nTIFF fallback error: {e2}")

# def extract_option_letter(s: Any) -> str:
#     """
#     从模型输出里提取 A/B/C/D 的选项字母（返回大写字母或空串）。
#     兼容 '(A)', 'A:', '答案A', 'A' 等形式。
#     """
#     if s is None:
#         return ""
#     text = str(s).strip().upper()

#     # 先匹配规范形式
#     m = re.search(r'[\(\[\{<]?\s*([ABCD])\s*[\)\]\}>:\.]?', text)
#     if m:
#         return m.group(1)

#     # 若未命中，宽松包含
#     for ch in ["A", "B", "C", "D"]:
#         if re.search(rf'\b{ch}\b', text):
#             return ch

#     # 仍未命中，尝试按关键字回推（可选）：若回答中复述了某个选项文本
#     return ""

# def build_prompt(question: str, answer_choices: List[str]) -> str:
#     """
#     中文多选提示词：强制只输出 A/B/C/D。
#     answer_choices 形如 ["(A) ...", "(B) ...", ...]
#     """
#     choices_text = "\n".join(answer_choices or [])
#     prompt = (
#         f"{question}\n\n"
#         "请根据在以下选项中选择最合适的一项，并且只输出选项字母（A/B/C/D），不要输出任何解释：\n"
#         f"{choices_text}\n\n"
#         "只输出一个字母：A 或 B 或 C 或 D。"
#     )
#     return prompt

# # ========== 大模型封装 ==========
# class MCQModel:
#     def __init__(self, model_dir: str, device: str = "cuda"):
#         self.device = device if torch.cuda.is_available() else "cpu"
#         self.model = AutoModel.from_pretrained(
#             model_dir, trust_remote_code=True, attn_implementation="sdpa",
#             torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
#         ).eval().to(self.device)
#         self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

#     @torch.inference_mode()
#     def answer_mcq(self, image: Image.Image, question: str, choices: List[str]) -> str:
#         """
#         走模型的 chat 接口（常见的多模态范式）。如果你的模型是别的 API，可在这里改。
#         """
#         prompt = build_prompt(question, choices)
#         msgs = [{"role": "user", "content": [image, prompt]}]
#         # 某些实现需要 image=None，仅把图像放在 content 里；这里沿用常见自定义 chat 习惯。
#         resp = self.model.chat(image=None, msgs=msgs, tokenizer=self.tokenizer)
#         return resp or ""

# # ========== 主流程 ==========
# def run_eval(input_json: str, image_root: str, model_dir: str, out_json: str):
#     with open(input_json, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)

#     model = MCQModel(model_dir)

#     total = 0
#     correct = 0
#     details: List[Dict[str, Any]] = []

#     for item in data:
#         img_rel = item.get("Image", "")
#         img_path = img_rel if os.path.isabs(img_rel) else os.path.join(image_root, img_rel)
#         question = item.get("Text", "")
#         choices  = item.get("Answer choices", []) or []
#         gt_raw   = item.get("Ground truth", "")
#         qid      = item.get("Question id") or item.get("Question_id") or None

#         # 读图
#         image = open_image_rgb(img_path)

#         # 询问模型
#         raw_answer = model.answer_mcq(image, question, choices)
#         pred = extract_option_letter(raw_answer)
#         gt   = extract_option_letter(gt_raw)

#         # 统计
#         if gt:
#             total += 1
#             if pred == gt:
#                 correct += 1

#         # 记录
#         details.append({
#             "Question_id": qid,
#             "Image": img_rel,
#             "Image_abs": img_path,
#             "Text": question,
#             "Answer_choices": choices,
#             "Model_raw": raw_answer,
#             "Pred_letter": pred,
#             "GT_letter": gt,
#         })

#     result = {
#         "accuracy": (correct / total) if total > 0 else None,
#         "total": total,
#         "correct": correct,
#         "detail": details
#     }

#     with open(out_json, "w", encoding="utf-8") as f:
#         json.dump(result, f, indent=2, ensure_ascii=False)

#     if total > 0:
#         print(f"Accuracy: {correct}/{total} = {correct/total:.4f}")
#     else:
#         print("未计算准确率（GT 缺失或为空）。")

# # ========== CLI（可选） ==========
# # def parse_args():
# #     ap = argparse.ArgumentParser()
# #     ap.add_argument("--input_json",  default=INPUT_JSON)
# #     ap.add_argument("--image_root",  default=IMAGE_ROOT)
# #     ap.add_argument("--model_dir",   default=MODEL_DIR)
# #     ap.add_argument("--output_json", default=OUTPUT_JSON)
#     # return ap.parse_args()

# if __name__ == "__main__":
#     # args = parse_args()
#     input_json='/data/cj/valid_contest/Counting__Counting_with_complex_reasoning.json'
#     image_root='/data/cj/valid_contest/valid_images'
#     model_dir='/data/cj/FM9G4B-V'
#     # model_dir='/data/cj/valid_contest/vqa_train/checkpoint-3088'
#     output_json='/data/cj/valid_contest/eval_Counting_with_complex.json'
#     run_eval(input_json, image_root, model_dir, output_json)


import os, argparse
import re
import json
from typing import Any, Dict, List, Tuple, Optional, Union

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

# ========== 路径配置（可按需修改） ==========
# INPUT_JSON  = "/path/to/qa_items.json"
# IMAGE_ROOT  = "/home/mcislab_cj/MME_realworld"
# MODEL_DIR   = "/home/mcislab_cj/fm9g4bv/FM9G4B-V"
# OUTPUT_JSON = "/path/to/qa_results.json"

# ========== 工具函数：读图 ==========
def open_image_rgb(path: str) -> Image.Image:
    """
    尝试以 RGB 读取图像：
    1) PIL + convert("RGB")
    2) 失败则用 tifffile 兜底读取，再转 PIL
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
            if arr.ndim == 2:
                arr = (arr[..., None]).repeat(3, axis=-1)
            elif arr.ndim == 3 and arr.shape[2] > 3:
                arr = arr[..., :3]
            img = Image.fromarray(arr)
            if img.mode != "RGB":
                img = img.convert("RGB")
            return img
        except Exception as e2:
            raise RuntimeError(f"Cannot open image as RGB: {path}\nPIL error: {e}\nTIFF fallback error: {e2}")

# ========== 提取选项字母 ==========
def extract_option_letter(s: Any) -> str:
    """
    从输出里提取 A/B/C/D（返回大写字母或空串），兼容 '(A)', 'A:', '答案A', 'A' 等。
    """
    if s is None:
        return ""
    text = str(s).strip().upper()
    m = re.search(r'[\(\[\{<]?\s*([ABCD])\s*[\)\]\}>:\.]?', text)
    if m:
        return m.group(1)
    for ch in ["A", "B", "C", "D"]:
        if re.search(rf'\b{ch}\b', text):
            return ch
    return ""

# ========== Prompt 构造 ==========
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

# ========== 方位词裁剪：词表与解析 ==========
_DIR_ALIASES = {
    "top":     {"top","upper","up","above"},
    "bottom":  {"bottom","lower","down","below"},
    "left":    {"left"},
    "right":   {"right"},
    "center":  {"center","centre","middle","mid","centered","midline","midpoint"},
}
# 复合方位（如 northeast）此处关闭
_COMPASS_COMPOSITES = {}

def _token_to_dirs(tok: str) -> List[str]:
    t = tok.lower().strip()
    for base, alias in _DIR_ALIASES.items():
        if t in alias:
            return [base]
    parts = re.split(r"[^a-z]+", t)
    out = []
    for p in parts:
        if not p:
            continue
        for base, alias in _DIR_ALIASES.items():
            if p in alias:
                out.append(base)
                break
    seen=set(); uniq=[]
    for d in out:
        if d not in seen:
            uniq.append(d); seen.add(d)
    return uniq

def _last_n_word_tokens(text: str, n: int = 9) -> List[str]:
    toks = re.findall(r"[A-Za-z\-]+", text or "")
    return toks if len(toks) <= n else toks[-n:]

def extract_directions_from_tail(text: str, tail_n: int = 9) -> List[str]:
    """
    从句尾窗口（最后 tail_n 个英文词）提取出现最靠后的方位词：
    - 若同时命中垂直(top/bottom)与水平(left/right)，返回二元 [v,h]
    - 若包含 center 与某一方向，返回 [那一方向, 'center']
    - 否则返回最近出现的单一方向
    - 未命中返回 []
    """
    window = _last_n_word_tokens(text, tail_n)
    last_pos = {"top": -1, "bottom": -1, "left": -1, "right": -1, "center": -1}
    for i, tok in enumerate(window):
        for d in _token_to_dirs(tok):
            if d in last_pos:
                last_pos[d] = i
    present = {k for k,v in last_pos.items() if v >= 0}
    vertical = [("top", last_pos["top"]), ("bottom", last_pos["bottom"])]
    horizontal = [("left", last_pos["left"]), ("right", last_pos["right"])]
    v_pick = max([v for v in vertical if v[1] >= 0], key=lambda x: x[1], default=None)
    h_pick = max([h for h in horizontal if h[1] >= 0], key=lambda x: x[1], default=None)
    if v_pick and h_pick: return [v_pick[0], h_pick[0]]
    if last_pos["center"] >= 0 and v_pick: return [v_pick[0], "center"]
    if last_pos["center"] >= 0 and h_pick: return [h_pick[0], "center"]
    if present: return [max(present, key=lambda k: last_pos[k])]
    return []

# ========== 方位词裁剪：九宫格裁剪 ==========
def crop_by_directions_halves(
    img: Image.Image,
    dirs: List[str],
    return_box: bool = False
):
    """
    1/2 裁剪（按上下/左右/中心）：
      - 单方向：left/right/top/bottom -> 对应一半；
      - center：返回居中的“宽高各一半”的区域；
      - 两个方向：四象限（top-left 等），或与 center 的组合（top+center / left+center）。
    """
    w, h = img.size
    Wm, Hm = w // 2, h // 2  # 中线

    if not dirs:
        box = (0, 0, w, h)

    elif len(dirs) == 1:
        d = dirs[0]
        if d == "left":
            box = (0, 0, Wm, h)
        elif d == "right":
            box = (Wm, 0, w, h)
        elif d == "top":
            box = (0, 0, w, Hm)
        elif d == "bottom":
            box = (0, Hm, w, h)
        elif d == "center":
            # 居中 1/2 宽 × 1/2 高
            x1, x2 = w // 4, (3 * w) // 4
            y1, y2 = h // 4, (3 * h) // 4
            box = (x1, y1, x2, y2)
        else:
            box = (0, 0, w, h)

    else:
        a, b = dirs[0], dirs[1]
        s = {a, b}

        # 四象限
        if "top" in s and "left" in s:
            box = (0, 0, Wm, Hm)
        elif "top" in s and "right" in s:
            box = (Wm, 0, w, Hm)
        elif "bottom" in s and "left" in s:
            box = (0, Hm, Wm, h)
        elif "bottom" in s and "right" in s:
            box = (Wm, Hm, w, h)

        # 与 center 的组合：上中 / 下中
        elif ("top" in s or "bottom" in s) and "center" in s:
            x1, x2 = w // 4, (3 * w) // 4  # 水平居中一半
            if "top" in s:
                box = (x1, 0, x2, Hm)
            else:  # bottom
                box = (x1, Hm, x2, h)

        # 与 center 的组合：中左 / 中右
        elif ("left" in s or "right" in s) and "center" in s:
            y1, y2 = h // 4, (3 * h) // 4  # 垂直居中一半
            if "left" in s:
                box = (0, y1, Wm, y2)
            else:  # right
                box = (Wm, y1, w, y2)

        else:
            # 其他组合，回退中心 1/2×1/2
            box = (w // 4, h // 4, (3 * w) // 4, (3 * h) // 4)

    out = img.crop(box)
    return (out, box) if return_box else out

# ========== 方位词裁剪：从问题尾部删除定位短语 ==========
def _alias_group(dir_key: str) -> str:
    al = set(_DIR_ALIASES.get(dir_key, []))
    if dir_key == "top":    al |= {"uppermost"}
    if dir_key == "bottom": al |= {"lowermost"}
    pats = sorted(al, key=len, reverse=True)
    return r"(?:%s)" % "|".join(map(re.escape, pats))

def _compass_group_for_pair(a: str, b: str) -> str:
    return ""

def _cleanup_spaces(s: str) -> str:
    s = re.sub(r"\s{2,}", " ", s)
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    return s.strip()

def _split_head_tail_by_last_n_words(text: str, n: int = 9) -> Tuple[str, str]:
    spans = list(re.finditer(r"[A-Za-z\-]+", text or ""))
    if not spans:
        return text, ""
    start_idx = spans[max(0, len(spans)-n)].start()
    return text[:start_idx], text[start_idx:]

def _build_tail_patterns(dirs: list):
    patterns = []
    preps = r"(?:in|on|at|near|toward|towards|from|along|around|over|by|inside|within)"
    det   = r"(?:the|this|that|these|those)?"
    of_img = r"(?:\s*of\s*the\s*(?:image|picture|photo|scene|satellite\s*image))?"
    nouns = r"(?:\s*(?:area|region|part|section|side|edge|corner|pot|quadrant|half|zone|portion))?"

    if len(dirs) >= 2:
        a, b = dirs[0], dirs[1]
        ag = _alias_group(a); bg = _alias_group(b)
        between = r"(?:[^A-Za-z]+(?:and|or)?\s*){0,3}"
        compass = _compass_group_for_pair(a, b)
        combo = rf"(?:{ag}{between}{bg}|{bg}{between}{ag}"
        combo += rf"|{compass})" if compass else ")"
        pat1 = re.compile(rf"\b{preps}\s+{det}\s*{combo}{nouns}{of_img}\b", re.IGNORECASE)
        pat2 = re.compile(rf"\b{combo}{nouns}{of_img}\b", re.IGNORECASE)
        patterns.extend([pat1, pat2])

    for d in dirs[:2]:
        dg = _alias_group(d)
        pat3 = re.compile(rf"\b{preps}\s+{det}\s*{dg}{nouns}{of_img}\b", re.IGNORECASE)
        pat4 = re.compile(rf"\b{dg}{nouns}{of_img}\b", re.IGNORECASE)
        pat5 = re.compile(rf"\(\s*{dg}\s*(?:-[ ]*{_alias_group('left')}|-[ ]*{_alias_group('right')})?\s*\)", re.IGNORECASE)
        patterns.extend([pat3, pat4, pat5])

    return patterns

def head_before_tail_text(question: str, tail_n: int = 9) -> str:
    if not question:
        return question
    head, _tail = _split_head_tail_by_last_n_words(question, n=tail_n)
    head = _cleanup_spaces(head or question)
    return head if head else question

def remove_position_phrase_from_question_tail_only(question: str, tail_n: int = 9) -> str:
    if not question:
        return question
    head, tail = _split_head_tail_by_last_n_words(question, n=tail_n)
    if not tail.strip():
        return _cleanup_spaces(question)

    dirs = extract_directions_from_tail(question, tail_n=tail_n)
    if not dirs:
        return _cleanup_spaces(question)

    tail_clean = tail
    for pat in _build_tail_patterns(dirs):
        tail_clean = pat.sub("", tail_clean)

    return _cleanup_spaces(head + tail_clean)

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
        prompt = build_prompt(question, choices)
        msgs = [{"role": "user", "content": [image, prompt]}]
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

        # === 方位词 → 裁剪 + 文本清理 ===
        crop_dirs = extract_directions_from_tail(question, tail_n=9)
        used_image = image
        used_question = question
        crop_box = None  # (x1,y1,x2,y2) in pixels of original image

        if crop_dirs:
            # 按九宫格裁剪；并从尾部删除定位短语
            try:
                # used_image, crop_box = crop_by_directions_thirds(image, crop_dirs, return_box=True)
                used_image, crop_box = crop_by_directions_halves(image, crop_dirs, return_box=True)
                used_question = remove_position_phrase_from_question_tail_only(question, tail_n=9)
            except Exception as e:
                # 任意异常则回退到全图与原问题
                used_image = image
                used_question = question
                crop_dirs = []
                crop_box = None

        # 询问模型（使用裁剪后的图像与文本；若未命中方位词则是原图原文）
        raw_answer = model.answer_mcq(used_image, used_question, choices)
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
            "Text_orig": question,
            "Text_used": used_question,
            "Answer_choices": choices,
            "Crop_dirs": crop_dirs,       # e.g., ["top","left"] / ["right"] / []
            "Crop_box": crop_box,         # 像素坐标 (x1,y1,x2,y2)（原图系）
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
        print(f"Results saved to {out_json}")

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
#     return ap.parse_args()

# if __name__ == "__main__":
#     # args = parse_args()
#     input_json  = '/data/cj/valid_contest/Counting__Counting_with_complex_reasoning.json'
#     image_root  = '/data/cj/valid_contest/valid_images'
#     model_dir   = '/data/cj/FM9G4B-V'
#     # model_dir = '/data/cj/valid_contest/vqa_train/checkpoint-3088'
#     output_json = '/data/cj/valid_contest/eval_Counting_with_complex.json'
#     run_eval(input_json, image_root, model_dir, output_json)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCQ eval with optional directional cropping")
    parser.add_argument("--input_json",  type=str, required=True,  help="Path to input QA json")
    parser.add_argument("--image_root",  type=str, required=True,  help="Root dir of images")
    parser.add_argument("--model_dir",   type=str, required=True,  help="FM9G (or compatible) model dir")
    parser.add_argument("--output_json", type=str, required=True,  help="Path to save eval results json")
    args = parser.parse_args()

    run_eval(args.input_json, args.image_root, args.model_dir, args.output_json)