# # -*- coding: utf-8 -*-
# import os, argparse
# import re
# import json
# from typing import Any, Dict, List, Tuple, Optional

# import torch
# from PIL import Image
# from transformers import AutoModel, AutoTokenizer

# # =========================
# # 通用工具
# # =========================
# def open_image_rgb(path: str) -> Image.Image:
#     """以 RGB 打开图像；PIL 失败则回退 tifffile。"""
#     try:
#         img = Image.open(path)
#         if img.mode != "RGB":
#             img = img.convert("RGB")
#         return img
#     except Exception as e:
#         try:
#             import tifffile as tiff
#             arr = tiff.imread(path)
#             if arr.ndim == 2:
#                 arr = (arr[..., None]).repeat(3, axis=-1)
#             elif arr.ndim == 3 and arr.shape[2] > 3:
#                 arr = arr[..., :3]
#             img = Image.fromarray(arr)
#             if img.mode != "RGB":
#                 img = img.convert("RGB")
#             return img
#         except Exception as e2:
#             raise RuntimeError(
#                 f"Cannot open image as RGB: {path}\nPIL error: {e}\nTIFF fallback error: {e2}"
#             )

# def extract_option_letter(s: Any) -> str:
#     """从模型输出里提取 A/B/C/D 的选项字母（返回大写字母或空串）。"""
#     if s is None:
#         return ""
#     text = str(s).strip().upper()
#     m = re.search(r'[\(\[\{<]?\s*([ABCD])\s*[\)\]\}>:\.]?', text)
#     if m:
#         return m.group(1)
#     for ch in ["A", "B", "C", "D"]:
#         if re.search(rf'\b{ch}\b', text):
#             return ch
#     return ""

# def build_prompt(question: str, answer_choices: List[str]) -> str:
#     choices_text = "\n".join(answer_choices or [])
#     prompt = (
#         f"{question}\n\n"
#         "请在以下选项中选择最合适的一项，并且只输出选项字母（A/B/C/D），不要输出任何解释：\n"
#         f"{choices_text}\n\n"
#         "只输出一个字母：A 或 B 或 C 或 D。"
#     )
#     return prompt

# # =========================
# # 多模态选择题模型（FM9G 封装）
# # =========================
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
#         """使用多模态 chat 进行选择题回答。"""
#         prompt = build_prompt(question, choices)
#         msgs = [{"role": "user", "content": [image, prompt]}]
#         try:
#             # 多模态实现习惯：image 走 content，入参 image=None
#             resp = self.model.chat(image=None, msgs=msgs, tokenizer=self.tokenizer)
#         except TypeError:
#             # 少数实现需要把图像放到 chat 的 image 参数里
#             resp = self.model.chat(image=image, msgs=[{"role": "user", "content": prompt}], tokenizer=self.tokenizer)
#         return resp or ""

# # =========================
# # 文本抽取（FM9G 原生模型，仅文本）
# # =========================
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# def load_fm9g(model_dir: str, device: str):
#     model = AutoModel.from_pretrained(
#         model_dir, trust_remote_code=True, attn_implementation="sdpa",
#         torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32
#     ).eval().to(device)
#     tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
#     return model, tok

# EXTRACT_PROMPT = """You are an information extraction model.

# Task:
# From the given English sentence, extract the EXACT TWO NOUN PHRASES being compared or related (A vs B).
# - Return pure JSON with keys: A_np, A_head, B_np, B_head (lowercase strings).
# - Remove color/size/material and other adjectives from *_np (keep core nouns only).
# - *_head should be the syntactic head (single word, lemma if plural).
# - If multiple candidates, choose the pair linked by a relation like “in relation to”, “compared with”, “relative to”, “between ... and ...”.
# - Return only ONE line of JSON and nothing else.

# Example:
# Input: "In the picture, where is the blue-gray airplane runway located in relation to the airplane?"
# Output:
# {"A_np":"airplane runway","A_head":"runway","B_np":"airplane","B_head":"airplane"}

# Now extract for the following input:
# """

# _ADJ_STOP = {
#     "blue","gray","grey","blue-gray","red","green","yellow","white","black","brown",
#     "big","small","large","tiny","little","huge","massive","long","short","wide","narrow",
#     "old","new","wooden","metal","steel","plastic","concrete"
# }

# def _singularize(w: str) -> str:
#     s = w.lower()
#     if s.endswith("ies") and len(s) > 3: return s[:-3] + "y"
#     if s.endswith("sses") or s.endswith("shes") or s.endswith("ches"): return s[:-2]
#     if s.endswith("s") and not s.endswith("ss"): return s[:-1]
#     return s

# def _norm_text(s: str) -> str:
#     s = s.lower()
#     s = re.sub(r"[-_]", " ", s)
#     s = re.sub(r"\s+", " ", s).strip()
#     return s

# def _clean_np(np: str) -> str:
#     """去冠词、颜色/尺寸等常见形容词、连接符，保留核心名词序列。"""
#     toks = re.findall(r"[A-Za-z0-9\-]+", np.lower())
#     toks = [t for t in toks if t not in _ADJ_STOP and t not in {"the","a","an"}]
#     s = " ".join(toks).replace("-", " ").strip()
#     s = re.sub(r"\s{2,}", " ", s)
#     return s

# def _head_of(np: str) -> str:
#     toks = [t for t in re.findall(r"[A-Za-z0-9]+", np.lower())]
#     if not toks: return ""
#     return _singularize(toks[-1])

# def rule_fallback(sentence: str):
#     """基于规则的兜底（覆盖常见‘in relation to’等句式）。"""
#     s = sentence.strip()

#     m_b = re.search(r"(?:in\s+relation\s+to|relative\s+to|compared\s+(?:to|with)|vs\.?|versus)\s+([^?.,;]+)", s, flags=re.I)
#     B_np = _clean_np(m_b.group(1)) if m_b else ""

#     m_a = re.search(r"where\s+is\s+(?:the\s+)?([^?.,;]+?)\s+(?:located|situated|positioned|placed)\b", s, flags=re.I)
#     if not m_a:
#         m_a = re.search(r"where\s+is\s+(?:the\s+)?([^?.,;]+)", s, flags=re.I)
#     A_np = _clean_np(m_a.group(1)) if m_a else ""

#     if not A_np:
#         m = re.search(r"([A-Za-z0-9\-\s]+?\brunway\b)", s, flags=re.I)
#         if m: A_np = _clean_np(m.group(1))
#     if not B_np:
#         m = re.search(r"\b(?:airplane|plane|aircraft)\b", s, flags=re.I)
#         if m: B_np = _clean_np(m.group(0))

#     A_head = _head_of(A_np) if A_np else ""
#     B_head = _head_of(B_np) if B_np else ""
#     return {"A_np": A_np, "A_head": A_head, "B_np": B_np, "B_head": B_head}

# @torch.inference_mode()
# def extract_comparative_nouns(text: str, model, tokenizer) -> dict:
#     """调用 FM9G（纯文本）抽取 A/B 名词短语；失败则回退规则。"""
#     prompt = EXTRACT_PROMPT + text.strip() + "\nJSON:"
#     msgs = [{"role": "user", "content": prompt}]
#     try:
#         out = model.chat(image=None, msgs=msgs, tokenizer=tokenizer)
#         m = re.search(r"\{.*\}", out, flags=re.S)
#         if not m:
#             raise ValueError("no JSON in model output")
#         obj = json.loads(m.group(0))

#         A_np = _clean_np(obj.get("A_np",""))
#         B_np = _clean_np(obj.get("B_np",""))
#         A_head = obj.get("A_head") or _head_of(A_np)
#         B_head = obj.get("B_head") or _head_of(B_np)
#         return {
#             "A_np": A_np,
#             "A_head": _singularize(A_head),
#             "B_np": B_np,
#             "B_head": _singularize(B_head)
#         }
#     except Exception as e:
#         print(f"[WARN] IE model failed, fallback to rules. Err={e}")
#         return rule_fallback(text)

# # =========================
# # “谁先触发/出现” 判定（A vs B）
# # =========================
# def _variants_phrase_first(np: str, head: str):
#     """生成候选，短语优先（完整短语→末词→head）。"""
#     out = []
#     seen = set()
#     if np:
#         base = _norm_text(np)
#         for p, tag in [(base, "phrase"), (base.replace("  ", " "), "phrase")]:
#             if p and p not in seen:
#                 out.append((p, tag)); seen.add(p)
#         last = base.split()[-1] if base else ""
#         if last and last not in seen:
#             out.append((last, "phrase-last")); seen.add(last)
#     if head:
#         h = _singularize(_norm_text(head))
#         if h and h not in seen:
#             out.append((h, "head")); seen.add(h)
#     return out

# def _first_match_with_meta(text_norm: str, cand_with_tags):
#     best = (None, None, None)
#     for patt, tag in cand_with_tags:
#         m = re.search(rf"\b{re.escape(patt)}\b", text_norm)
#         if not m:
#             m = re.search(re.escape(patt), text_norm)
#         if m:
#             idx = m.start()
#             if best[0] is None or idx < best[0]:
#                 best = (idx, patt, tag)
#     return best

# def which_appears_first_verbose(sentence: str, A_np: str, A_head: str, B_np: str, B_head: str):
#     text_norm = _norm_text(sentence)

#     A_cands = _variants_phrase_first(A_np, A_head)
#     B_cands = _variants_phrase_first(B_np, B_head)

#     ia, ma, ta = _first_match_with_meta(text_norm, A_cands)
#     ib, mb, tb = _first_match_with_meta(text_norm, B_cands)

#     if ia is None and ib is None:
#         print("【牵出线】两者均未在句中命中（unknown）")
#         return {'winner': 'unknown',
#                 'A': {'idx': None, 'matched': None, 'type': None},
#                 'B': {'idx': None, 'matched': None, 'type': None}}

#     if ia is None:
#         print(f"【牵出线】B 先触发（A 未命中）。B 命中：'{mb}'（{tb}），位置 {ib}")
#         return {'winner': 'B',
#                 'A': {'idx': None, 'matched': None, 'type': None},
#                 'B': {'idx': ib, 'matched': mb, 'type': tb}}

#     if ib is None:
#         print(f"【牵出线】A 先触发（B 未命中）。A 命中：'{ma}'（{ta}），位置 {ia}")
#         return {'winner': 'A',
#                 'A': {'idx': ia, 'matched': ma, 'type': ta},
#                 'B': {'idx': None, 'matched': None, 'type': None}}

#     if ia < ib:
#         print(f"【牵出线】A 先触发。A 命中：'{ma}'（{ta}）@{ia}；B 命中：'{mb}'（{tb}）@{ib}")
#         winner = 'A'
#     elif ib < ia:
#         print(f"【牵出线】B 先触发。B 命中：'{mb}'（{tb}）@{ib}；A 命中：'{ma}'（{ta}）@{ia}")
#         winner = 'B'
#     else:
#         print(f"【牵出线】两者同时命中（tie）。A：'{ma}'（{ta}）@{ia}；B：'{mb}'（{tb}）@{ib}")
#         winner = 'tie'

#     return {'winner': winner,
#             'A': {'idx': ia, 'matched': ma, 'type': ta},
#             'B': {'idx': ib, 'matched': mb, 'type': tb}}

# # =========================
# # 主流程
# # =========================
# def run_eval(input_json: str, image_root: str, model_dir: str, out_json: str):
#     if not os.path.exists(input_json):
#         raise FileNotFoundError(f"JSON not found: {input_json}")
#     with open(input_json, "r", encoding="utf-8") as f:
#         data = json.load(f)
#     print(f"[INFO] loaded {len(data)} items")

#     os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)

#     # 模型：答题 + 文本抽取
#     mcq = MCQModel(model_dir)
#     raw_model, raw_tok = load_fm9g(model_dir, DEVICE)

#     total = 0
#     correct = 0
#     details: List[Dict[str, Any]] = []

#     for idx, item in enumerate(data, 1):
#         img_rel = item.get("Image", "") or item.get("image_id", "")
#         img_path = img_rel if os.path.isabs(img_rel) else os.path.join(image_root, img_rel)
#         question = item.get("Text", "") or item.get("question", "")
#         choices  = item.get("Answer choices", []) or []
#         gt_raw   = item.get("Ground truth", "") or item.get("ground_truth", "")
#         qid      = item.get("Question id") or item.get("Question_id") or None

#         if not os.path.exists(img_path):
#             print(f"[WARN] ({idx}) image not found: {img_path}")
#             details.append({
#                 "Question_id": qid, "Image": img_rel, "Image_abs": img_path,
#                 "error": "image_not_found", "Text": question
#             })
#             continue

#         image = open_image_rgb(img_path)

#         # A/B 抽取
#         result = extract_comparative_nouns(question, raw_model, raw_tok)
#         A_np, A_head = result['A_np'], result['A_head']
#         B_np, B_head = result['B_np'], result['B_head']

#         info = which_appears_first_verbose(question, A_np, A_head, B_np, B_head)

#         # 不改原题；只在输入时用改写版
#         rewritten_question = None
#         roi = None
#         if info['A']['idx'] is not None and info['B']['idx'] is not None:
#             roi = (A_np or A_head) if int(info['A']['idx']) <= int(info['B']['idx']) else (B_np or B_head)
#             if roi:
#                 rewritten_question = f"Where is the {roi} located in?"

#         q4model = rewritten_question if rewritten_question else question

#         raw_answer = mcq.answer_mcq(image, q4model, choices)
#         pred = extract_option_letter(raw_answer)
#         gt   = extract_option_letter(gt_raw)

#         if gt:
#             total += 1
#             if pred == gt:
#                 correct += 1

#         details.append({
#             "idx": idx,
#             "Question_id": qid,
#             "Image": img_rel,
#             "Image_abs": img_path,
#             "Text": question,                      # 原题
#             "Rewritten_Text": rewritten_question,  # 改写版（如有）
#             "A_np": A_np, "A_head": A_head,
#             "B_np": B_np, "B_head": B_head,
#             "Trigger": info,
#             "Answer_choices": choices,
#             "Model_raw": raw_answer,
#             "Pred_letter": pred,
#             "GT_letter": gt,
#         })

#         if idx % 20 == 0:
#             acc_so_far = f"{correct}/{total} = {(correct/total):.4f}" if total else "N/A"
#             print(f"[INFO] progress {idx}/{len(data)}  acc_so_far={acc_so_far}")

#     result = {
#         "accuracy": (correct / total) if total > 0 else None,
#         "total": total,
#         "correct": correct,
#         "detail": details
#     }

#     with open(out_json, "w", encoding="utf-8") as f:
#         json.dump(result, f, indent=2, ensure_ascii=False)

#     if total > 0:
#         print(f"[FINAL] Accuracy: {correct}/{total} = {correct/total:.4f}")
#     else:
#         print("[FINAL] No GT; accuracy not computed.")

# # ====== 入口 ======
# # if __name__ == "__main__":
# #     # 按你的实际路径修改
# #     input_json  = '/data/cj/valid_contest/Object_spatial_relationship__Object_spatial_relationship.json'
# #     image_root  = '/data/cj/valid_contest/valid_images'
# #     model_dir   = '/data/cj/FM9G4B-V'
# #     output_json = '/data/cj/valid_contest/eval_spatial_relationship.json'
# #     run_eval(input_json, image_root, model_dir, output_json)

# if __name__ == "__main__":
#     # --- 新增命令行参数，提供默认值与原脚本一致 ---
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--input_json",  type=str, default="/data/cj/valid_contest/Object_spatial_relationship__Object_spatial_relationship.json")
#     parser.add_argument("--image_root",  type=str, default="/data/cj/valid_contest/valid_images")
#     parser.add_argument("--model_dir",   type=str, default="/data/cj/FM9G4B-V")
#     parser.add_argument("--out_json", type=str, default="/data/cj/valid_contest/eval_spatial_relationship.json")
#     args = parser.parse_args()

#     run_eval(
#         input_json=args.input_json,
#         image_root=args.image_root,
#         model_dir=args.model_dir,
#         out_json=args.out_json
#     )

# -*- coding: utf-8 -*-
import os, argparse
import re
import json
from typing import Any, Dict, List, Tuple, Optional

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

# =========================
# 通用工具
# =========================
def open_image_rgb(path: str) -> Image.Image:
    """以 RGB 打开图像；PIL 失败则回退 tifffile。"""
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
            raise RuntimeError(
                f"Cannot open image as RGB: {path}\nPIL error: {e}\nTIFF fallback error: {e2}"
            )

def extract_option_letter(s: Any) -> str:
    """从模型输出里提取 A/B/C/D 的选项字母（返回大写字母或空串）。"""
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

def build_prompt(question: str, answer_choices: List[str]) -> str:
    choices_text = "\n".join(answer_choices or [])
    prompt = (
        f"{question}\n\n"
        "请在以下选项中选择最合适的一项，并且只输出选项字母（A/B/C/D），不要输出任何解释：\n"
        f"{choices_text}\n\n"
        "只输出一个字母：A 或 B 或 C 或 D。"
    )
    return prompt

# 选项字母解析/子集过滤
def parse_choice_letters(choices: List[str]) -> List[str]:
    letters = []
    for i, ch in enumerate(choices):
        m = re.match(r"^\(([A-Za-z])\)\s*", (ch or "").strip())
        if m:
            letters.append(m.group(1).upper())
        else:
            letters.append(chr(ord('A') + i))
    return letters

def filter_choices_by_letters(choices: List[str], allowed: List[str]) -> List[str]:
    out = []
    for ch in choices:
        m = re.match(r"^\(([A-Za-z])\)\s*", (ch or "").strip())
        if m and m.group(1).upper() in allowed:
            out.append(ch)
    return out

# =========================
# 方位解析 + 九宫格裁剪（1/3 网格）
# =========================
_DIR_ALIASES = {
    "top":     {"top","upper","up","above","north","northern","uppermost"},
    "bottom":  {"bottom","lower","down","below","south","southern","lowermost"},
    "left":    {"left","west","western","left-hand"},
    "right":   {"right","east","eastern","right-hand"},
    "center":  {"center","centre","middle","mid","central","midline","midpoint"},
}

def parse_dirs_from_text(s: str) -> List[str]:
    """
    从字符串里抽取方位：返回 [] / ['top'] / ['left'] / ['center'] /
    ['top','left'] / ['top','center'] / ['center','left'] 等。
    """
    t = (s or "").lower()
    toks = re.split(r"[^a-z]+", t)
    found = set()
    for tok in toks:
        if not tok: continue
        for base, ali in _DIR_ALIASES.items():
            if tok in ali:
                found.add(base)
    v = "top" if "top" in found else ("bottom" if "bottom" in found else None)
    h = "left" if "left" in found else ("right" if "right" in found else None)
    has_c = "center" in found
    if v and h: return [v, h]
    if has_c and (v or h): return [v or h, "center"]
    if has_c: return ["center"]
    if v: return [v]
    if h: return [h]
    return []

def crop_by_directions_thirds(
    img: Image.Image,
    dirs: List[str],
) -> Image.Image:
    """
    按 1/3 网格做方位裁剪：
      - 单方向：left/right/top/bottom/center -> 对应 1/3 条带或中间块
      - 双方向：九宫格块（含与 center 的组合）
      - 无方向：返回原图
    """
    w, h = img.size
    W1, W2 = w // 3, (2 * w) // 3
    H1, H2 = h // 3, (2 * h) // 3

    if not dirs:
        box = (0, 0, w, h)

    elif len(dirs) == 1:
        d = dirs[0]
        if d == "left":
            box = (0, 0, W1, h)
        elif d == "right":
            box = (W2, 0, w, h)
        elif d == "top":
            box = (0, 0, w, H1)
        elif d == "bottom":
            box = (0, H2, w, h)
        elif d == "center":
            box = (W1, H1, W2, H2)
        else:
            box = (0, 0, w, h)
    else:
        mapping_h = {"left": 0, "center": 1, "right": 2}
        mapping_v = {"top": 0, "center": 1, "bottom": 2}
        s = set(dirs[:2])
        # 纵+横
        if (("top" in s) or ("bottom" in s)) and (("left" in s) or ("right" in s)):
            row = mapping_v["top"] if "top" in s else mapping_v["bottom"]
            col = mapping_h["left"] if "left" in s else mapping_h["right"]
        # 纵+中
        elif (("top" in s) or ("bottom" in s)) and ("center" in s):
            row = mapping_v["top"] if "top" in s else mapping_v["bottom"]
            col = mapping_h["center"]
        # 横+中
        elif (("left" in s) or ("right" in s)) and ("center" in s):
            col = mapping_h["left"] if "left" in s else mapping_h["right"]
            row = mapping_v["center"]
        else:
            # 默认中心
            row = 1; col = 1

        if col == 0: x1, x2 = 0, W1
        elif col == 1: x1, x2 = W1, W2
        else: x1, x2 = W2, w

        if row == 0: y1, y2 = 0, H1
        elif row == 1: y1, y2 = H1, H2
        else: y1, y2 = H2, h

        box = (int(x1), int(y1), int(x2), int(y2))

    return img.crop(box)

def normalize_yesno(s: str) -> Optional[str]:
    if not s: return None
    t = str(s).strip().lower()
    if re.search(r"\by(es)?\b", t) or "yes" in t:
        return "Yes"
    if re.search(r"\bn(o)?\b", t) or "no" in t:
        return "No"
    return None

# =========================
# 多模态选择题模型（FM9G 封装）
# =========================
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
        """使用多模态 chat 进行选择题回答。"""
        prompt = build_prompt(question, choices)
        msgs = [{"role": "user", "content": [image, prompt]}]
        try:
            resp = self.model.chat(image=None, msgs=msgs, tokenizer=self.tokenizer)
        except TypeError:
            resp = self.model.chat(image=image, msgs=[{"role": "user", "content": prompt}], tokenizer=self.tokenizer)
        return resp or ""

    @torch.inference_mode()
    def yesno_on_image(self, image: Image.Image, roi_phrase: str) -> str:
        """
        让大模型只判定：这张裁剪图里是否包含 ROI 目标（是/否）。
        """
        prompt = (
            "You are given a CROPPED region from a remote-sensing image.\n"
            f"Question: Does this region contain the target object: \"{roi_phrase}\"?\n"
            "Answer with ONE WORD only: yes or no."
        )
        msgs = [{"role": "user", "content": [image, prompt]}]
        try:
            out = self.model.chat(image=None, msgs=msgs, tokenizer=self.tokenizer)
        except TypeError:
            out = self.model.chat(image=image, msgs=[{"role": "user", "content": prompt}], tokenizer=self.tokenizer)
        return out or ""

# =========================
# 文本抽取（FM9G 原生模型，仅文本）
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_fm9g(model_dir: str, device: str):
    model = AutoModel.from_pretrained(
        model_dir, trust_remote_code=True, attn_implementation="sdpa",
        torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32
    ).eval().to(device)
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    return model, tok

EXTRACT_PROMPT = """You are an information extraction model.

Task:
From the given English sentence, extract the EXACT TWO NOUN PHRASES being compared or related (A vs B).
- Return pure JSON with keys: A_np, A_head, B_np, B_head (lowercase strings).
- Remove color/size/material and other adjectives from *_np (keep core nouns only).
- *_head should be the syntactic head (single word, lemma if plural).
- If multiple candidates, choose the pair linked by a relation like “in relation to”, “compared with”, “relative to”, “between ... and ...”.
- Return only ONE line of JSON and nothing else.

Example:
Input: "In the picture, where is the blue-gray airplane runway located in relation to the airplane?"
Output:
{"A_np":"airplane runway","A_head":"runway","B_np":"airplane","B_head":"airplane"}

Now extract for the following input:
"""

_ADJ_STOP = {
    "blue","gray","grey","blue-gray","red","green","yellow","white","black","brown",
    "big","small","large","tiny","little","huge","massive","long","short","wide","narrow",
    "old","new","wooden","metal","steel","plastic","concrete"
}

def _singularize(w: str) -> str:
    s = w.lower()
    if s.endswith("ies") and len(s) > 3: return s[:-3] + "y"
    if s.endswith("sses") or s.endswith("shes") or s.endswith("ches"): return s[:-2]
    if s.endswith("s") and not s.endswith("ss"): return s[:-1]
    return s

def _norm_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[-_]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _clean_np(np_text: str) -> str:
    """去冠词、颜色/尺寸等常见形容词、连接符，保留核心名词序列。"""
    toks = re.findall(r"[A-Za-z0-9\-]+", (np_text or "").lower())
    toks = [t for t in toks if t not in _ADJ_STOP and t not in {"the","a","an"}]
    s = " ".join(toks).replace("-", " ").strip()
    s = re.sub(r"\s{2,}", " ", s)
    return s

def _head_of(np_text: str) -> str:
    toks = [t for t in re.findall(r"[A-Za-z0-9]+", (np_text or "").lower())]
    if not toks: return ""
    return _singularize(toks[-1])

def rule_fallback(sentence: str):
    s = sentence.strip()
    m_b = re.search(r"(?:in\s+relation\s+to|relative\s+to|compared\s+(?:to|with)|vs\.?|versus)\s+([^?.,;]+)", s, flags=re.I)
    B_np = _clean_np(m_b.group(1)) if m_b else ""
    m_a = re.search(r"where\s+is\s+(?:the\s+)?([^?.,;]+?)\s+(?:located|situated|positioned|placed)\b", s, flags=re.I)
    if not m_a:
        m_a = re.search(r"where\s+is\s+(?:the\s+)?([^?.,;]+)", s, flags=re.I)
    A_np = _clean_np(m_a.group(1)) if m_a else ""
    if not A_np:
        m = re.search(r"([A-Za-z0-9\-\s]+?\brunway\b)", s, flags=re.I)
        if m: A_np = _clean_np(m.group(1))
    if not B_np:
        m = re.search(r"\b(?:airplane|plane|aircraft)\b", s, flags=re.I)
        if m: B_np = _clean_np(m.group(0))
    A_head = _head_of(A_np) if A_np else ""
    B_head = _head_of(B_np) if B_np else ""
    return {"A_np": A_np, "A_head": A_head, "B_np": B_np, "B_head": B_head}

@torch.inference_mode()
def extract_comparative_nouns(text: str, model, tokenizer) -> dict:
    prompt = EXTRACT_PROMPT + (text or "").strip() + "\nJSON:"
    msgs = [{"role": "user", "content": prompt}]
    try:
        out = model.chat(image=None, msgs=msgs, tokenizer=tokenizer)
        m = re.search(r"\{.*\}", out, flags=re.S)
        if not m:
            raise ValueError("no JSON in model output")
        obj = json.loads(m.group(0))
        A_np = _clean_np(obj.get("A_np",""))
        B_np = _clean_np(obj.get("B_np",""))
        A_head = obj.get("A_head") or _head_of(A_np)
        B_head = obj.get("B_head") or _head_of(B_np)
        return {
            "A_np": A_np,
            "A_head": _singularize(A_head),
            "B_np": B_np,
            "B_head": _singularize(B_head)
        }
    except Exception as e:
        print(f"[WARN] IE model failed, fallback to rules. Err={e}")
        return rule_fallback(text)

# =========================
# “谁先触发/出现” 判定（A vs B）
# =========================
def _variants_phrase_first(np_text: str, head: str):
    out = []
    seen = set()
    if np_text:
        base = _norm_text(np_text)
        for p, tag in [(base, "phrase"), (base.replace("  ", " "), "phrase")]:
            if p and p not in seen:
                out.append((p, tag)); seen.add(p)
        last = base.split()[-1] if base else ""
        if last and last not in seen:
            out.append((last, "phrase-last")); seen.add(last)
    if head:
        h = _singularize(_norm_text(head))
        if h and h not in seen:
            out.append((h, "head")); seen.add(h)
    return out

def _first_match_with_meta(text_norm: str, cand_with_tags):
    best = (None, None, None)
    for patt, tag in cand_with_tags:
        m = re.search(rf"\b{re.escape(patt)}\b", text_norm)
        if not m:
            m = re.search(re.escape(patt), text_norm)
        if m:
            idx = m.start()
            if best[0] is None or idx < best[0]:
                best = (idx, patt, tag)
    return best

def which_appears_first_verbose(sentence: str, A_np: str, A_head: str, B_np: str, B_head: str):
    text_norm = _norm_text(sentence)
    A_cands = _variants_phrase_first(A_np, A_head)
    B_cands = _variants_phrase_first(B_np, B_head)
    ia, ma, ta = _first_match_with_meta(text_norm, A_cands)
    ib, mb, tb = _first_match_with_meta(text_norm, B_cands)

    if ia is None and ib is None:
        return {'winner': 'unknown',
                'A': {'idx': None, 'matched': None, 'type': None},
                'B': {'idx': None, 'matched': None, 'type': None}}
    if ia is None:
        return {'winner': 'B',
                'A': {'idx': None, 'matched': None, 'type': None},
                'B': {'idx': ib, 'matched': mb, 'type': tb}}
    if ib is None:
        return {'winner': 'A',
                'A': {'idx': ia, 'matched': ma, 'type': ta},
                'B': {'idx': None, 'matched': None, 'type': None}}
    winner = 'A' if ia < ib else ('B' if ib < ia else 'tie')
    return {'winner': winner,
            'A': {'idx': ia, 'matched': ma, 'type': ta},
            'B': {'idx': ib, 'matched': mb, 'type': tb}}

# =========================
# 主流程
# =========================
def run_eval(input_json: str, image_root: str, model_dir: str, out_json: str):
    if not os.path.exists(input_json):
        raise FileNotFoundError(f"JSON not found: {input_json}")
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[INFO] loaded {len(data)} items")

    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)

    # 模型：答题 + 文本抽取
    mcq = MCQModel(model_dir)
    raw_model, raw_tok = load_fm9g(model_dir, DEVICE)

    total = 0
    correct = 0
    details: List[Dict[str, Any]] = []

    for idx, item in enumerate(data, 1):
        img_rel = item.get("Image", "") or item.get("image_id", "")
        img_path = img_rel if os.path.isabs(img_rel) else os.path.join(image_root, img_rel)
        question = item.get("Text", "") or item.get("question", "")
        choices  = item.get("Answer choices", []) or []
        gt_raw   = item.get("Ground truth", "") or item.get("ground_truth", "")
        qid      = item.get("Question id") or item.get("Question_id") or None

        if not os.path.exists(img_path):
            print(f"[WARN] ({idx}) image not found: {img_path}")
            details.append({
                "idx": idx,
                "Question_id": qid, "Image": img_rel, "Image_abs": img_path,
                "error": "image_not_found", "Text": question
            })
            continue

        image = open_image_rgb(img_path)

        # A/B 抽取
        result = extract_comparative_nouns(question, raw_model, raw_tok)
        A_np, A_head = result['A_np'], result['A_head']
        B_np, B_head = result['B_np'], result['B_head']

        info = which_appears_first_verbose(question, A_np, A_head, B_np, B_head)

        # ROI 语义短语
        roi_phrase = None
        if info['A']['idx'] is not None and info['B']['idx'] is not None:
            roi_phrase = (A_np or A_head) if int(info['A']['idx']) <= int(info['B']['idx']) else (B_np or B_head)

        # 默认回退问答的问句（若未提取到 ROI 或后续策略需要回退）
        rewritten_question = None
        if roi_phrase:
            rewritten_question = f"Where is the {roi_phrase} located in?"
        q4model = rewritten_question if rewritten_question else question

        # ---------------------------
        # 新增：方位 -> 裁剪 -> ROI 存在性判定
        # ---------------------------
        per_option_probe = {}  # 记录每个选项的裁剪/判定
        positive_letters: List[str] = []

        choice_letters = parse_choice_letters(choices)
        # 针对每个选项：解析方位并裁剪；让模型回答裁剪图是否包含 ROI
        if roi_phrase:
            for ch_text in choices:
                m = re.match(r"^\(([A-Za-z])\)\s*(.+)$", (ch_text or "").strip())
                if not m:
                    continue
                L, text_after = m.group(1).upper(), m.group(2)
                dirs = parse_dirs_from_text(text_after)
                if not dirs:
                    per_option_probe[L] = {"dirs": [], "yesno_raw": None, "yesno": None}
                    continue
                crop_img = crop_by_directions_thirds(image, dirs)
                raw_yn = mcq.yesno_on_image(crop_img, roi_phrase)
                yn = normalize_yesno(raw_yn)
                per_option_probe[L] = {"dirs": dirs, "yesno_raw": raw_yn, "yesno": yn}
                if yn == "Yes":
                    positive_letters.append(L)

        # 决策：唯一命中 / 全否回退 / 多命中再限定选项问一次
        logic_path = "fallback_full_image"
        model_raw = ""
        pred = ""
        if roi_phrase and positive_letters:
            if len(positive_letters) == 1:
                logic_path = "roi_unique_positive"
                pred = positive_letters[0]
            else:
                logic_path = "roi_multi_positive_tiebreak"
                # 只保留这些候选选项再问一次（全图）
                sub_choices = filter_choices_by_letters(choices, positive_letters)
                # 构造一个更明确的 tie-break 提示词，限定只输出这几个字母
                sub_letters = "".join(positive_letters)
                tie_prompt = (
                    f"The target object is \"{roi_phrase}\".\n"
                    f"Based on the FULL image, choose which option BEST describes the location of the target.\n"
                    f"Output ONLY ONE letter among [{sub_letters}]."
                )
                tie_choices = sub_choices
                msgs = [{"role": "user", "content": [image, tie_prompt + "\n" + "\n".join(tie_choices) + "\nAnswer:"]}]
                try:
                    model_raw = mcq.model.chat(image=None, msgs=msgs, tokenizer=mcq.tokenizer)
                except TypeError:
                    model_raw = mcq.model.chat(image=image, msgs=[{"role": "user", "content": tie_prompt}], tokenizer=mcq.tokenizer)
                cand = extract_option_letter(model_raw)
                if cand in positive_letters:
                    pred = cand
                else:
                    # 若未能落在候选里，稳妥起见取候选中的第一个
                    pred = positive_letters[0]
        else:
            # 全否/没 ROI -> 回到原图+原问题
            logic_path = "fallback_full_image"
            model_raw = mcq.answer_mcq(image, q4model, choices)
            pred = extract_option_letter(model_raw)

        gt = extract_option_letter(gt_raw)

        if gt:
            total += 1
            if pred == gt:
                correct += 1

        details.append({
            "idx": idx,
            "Question_id": qid,
            "Image": img_rel,
            "Image_abs": img_path,
            "Text": question,                      # 原题
            "Rewritten_Text": rewritten_question,  # 改写版（如有）
            "ROI_phrase": roi_phrase,
            "A_np": A_np, "A_head": A_head,
            "B_np": B_np, "B_head": B_head,
            "Trigger": info,
            "Answer_choices": choices,
            "Probe_by_option": per_option_probe,   # 新增：每个选项的裁剪判定
            "Logic_path": logic_path,              # 走了哪个分支
            "Model_raw": model_raw,
            "Pred_letter": pred,
            "GT_letter": gt,
        })

        if idx % 20 == 0:
            acc_so_far = f"{correct}/{total} = {(correct/total):.4f}" if total else "N/A"
            print(f"[INFO] progress {idx}/{len(data)}  acc_so_far={acc_so_far}")

    result = {
        "accuracy": (correct / total) if total > 0 else None,
        "total": total,
        "correct": correct,
        "detail": details
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    if total > 0:
        print(f"[FINAL] Accuracy: {correct}/{total} = {correct/total:.4f}")
    else:
        print("[FINAL] No GT; accuracy not computed.")

# ====== 入口 ======
if __name__ == "__main__":
    # --- 新增命令行参数，提供默认值与原脚本一致 ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json",  type=str, default="/data/cj/valid_contest/Object_spatial_relationship__Object_spatial_relationship.json")
    parser.add_argument("--image_root",  type=str, default="/data/cj/valid_contest/valid_images")
    parser.add_argument("--model_dir",   type=str, default="/data/cj/FM9G4B-V")
    parser.add_argument("--out_json",    type=str, default="/data/cj/valid_contest/eval_spatial_relationship.json")
    args = parser.parse_args()

    run_eval(
        input_json=args.input_json,
        image_root=args.image_root,
        model_dir=args.model_dir,
        out_json=args.out_json
    )
