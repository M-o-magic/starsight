import os
import re
import json
from glob import glob
from typing import List, Tuple, Optional, Dict, Any

import numpy as np

# ======================
# 路径与配置
# ======================
INPUT_JSON  = "/data/cj/valid_contest/Land_use_classification__Overall_Land_use_classification.json"   # 题目 JSON（列表或单条）
SKYSENSE_NPZ_ROOT = "/data/cj/valid_contest/valid_images/FAIR1M_sky_out"  # 存放 skysense npz 的根目录
OUTPUT_JSON = "/data/cj/valid_contest/eval_Landuse_overall_fm9g.json"

# 可选：过滤超小类别实例（避免误检的极小碎屑导致“类出现”）
MIN_PIXELS = 50             # 类别像素和的最小阈值（像素）
MIN_RATIO  = 1e-6           # 或者按占比过滤（和 H*W 相比）

# ======================
# FM9G 加载
# ======================
import ast
from transformers import AutoModel, AutoTokenizer
import torch

FM9G_MODEL_PATH = os.environ.get("FM9G_MODEL_PATH", "/data/cj/FM9G4B-V")

_fm9g_model = None
_fm9g_tokenizer = None

def load_fm9g():
    global _fm9g_model, _fm9g_tokenizer
    if _fm9g_model is None or _fm9g_tokenizer is None:
        _fm9g_model = AutoModel.from_pretrained(
            FM9G_MODEL_PATH, trust_remote_code=True, attn_implementation='sdpa',
            torch_dtype=torch.bfloat16
        ).eval().cuda()
        _fm9g_tokenizer = AutoTokenizer.from_pretrained(FM9G_MODEL_PATH, trust_remote_code=True)
    return _fm9g_model, _fm9g_tokenizer

# ======================
# 基础工具 & I/O
# ======================
def norm_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[_\-/]+", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def basename_noext(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]

def prefix_from_image_field(field: str) -> str:
    # FAIR1M/212.tif -> 212
    return basename_noext(field)

def find_npz_for_prefix(root: str, prefix: str) -> Optional[str]:
    """
    在 root 下递归寻找以 prefix 开头的 npz；返回最短匹配路径（稳定）。
    """
    pats = [
        os.path.join(root, "**", f"{prefix}.npz"),
        os.path.join(root, "**", f"{prefix}_*.npz"),
        os.path.join(root, "**", f"{prefix}*.npz"),
    ]
    for pat in pats:
        cand = glob(pat, recursive=True)
        if cand:
            cand.sort(key=lambda x: (len(x), x))
            return cand[0]
    return None

def load_npz(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    mask = data["mask"]                   # (H, W) int (类别索引)
    classes = data["classes"]             # list 或 ndarray
    if not isinstance(classes, (list, tuple)):
        classes = classes.tolist()
    return mask, classes

def present_classes_from_npz(npz_path: str,
                             min_pixels: int = MIN_PIXELS,
                             min_ratio: float = MIN_RATIO) -> List[str]:
    """
    返回图像中“出现”的类别名称列表（去掉背景0，并做面积过滤）。
    """
    mask, classes = load_npz(npz_path)
    H, W = mask.shape
    ids, counts = np.unique(mask, return_counts=True)
    present = []
    for cid, cnt in zip(ids, counts):
        if cid == 0:
            continue  # 背景
        if cnt < min_pixels or cnt < H * W * min_ratio:
            continue  # 过滤太小
        name = classes[int(cid)]
        present.append(str(name))
    return present

# ======================
# Prompt 设计（多选）
# ======================
PROMPT_SYS = (
    "You are an expert land-use analyst for remote-sensing imagery. "
    "You must ONLY use the provided list of detected class names as evidence. "
    "Map them to the answer choices using synonyms when necessary."
)

def build_choice_prompt(present_classes, choice_text):
    """
    极短、偏向「yes」的 prompt：
    - 明确：只依据 npz 列表做判断
    - 倾向召回：如同义/近义可合理映射，或不确定但看起来可能存在 → 回答 yes
    - 输出限制：只回答 yes 或 no
    """
    pcs = ", ".join(sorted(set(map(str, present_classes)))) if present_classes else "(none)"
    return (
        "You will answer about one land-use option using ONLY the detected class list below.\n"
        f"Detected classes: [{pcs}]\n"
        f"Question: Is \"{choice_text}\" present in the image?\n"
        "Rules: Map reasonable synonyms (e.g., helipad≈heliport; buildings+roads≈“Buildings and Roads”; terminal≈airport terminal; land≈ground/field/soil). "
        "If uncertain but plausible from the list, prefer answering 'yes'.\n"
        "Answer with only 'yes' or 'no'."
    )


def fm9g_yes_no(prompt: str) -> bool:
    """
    调FM9G回答yes/no。返回True表示yes，False表示no。
    对输出做鲁棒解析：只看前10个字符里的 'yes'/'no' 关键词。
    """
    model, tokenizer = load_fm9g()
    msgs = [{"role": "user", "content": prompt}]
    out = model.chat(image=None, msgs=msgs, tokenizer=tokenizer)
    s = (out or "").strip().lower()
    s = s[:10]  # 限定前缀，避免“多余解释”
    if "yes" in s and "no" in s:
        # 同时出现，取第一个命中的词
        y = s.find("yes"); n = s.find("no")
        return bool(y != -1 and (n == -1 or y < n))
    if "yes" in s:
        return True
    if "no" in s:
        return False
    # 兜底：再强搜一次完整字符串
    s2 = (out or "").lower()
    if "yes" in s2 and "no" not in s2:
        return True
    if "no" in s2 and "yes" not in s2:
        return False
    # 实在看不出来，保守返回 False（不选）
    return False

# ======================
# 调 FM9G 获取多选答案
# ======================
def fm9g_select_letters(prompt: str) -> str:
    """
    让 FM9G 只输出字母组合（如 'ABCD' 或 ''），并做强解析与清洗。
    """
    model, tokenizer = load_fm9g()
    msgs = [{"role": "user", "content": prompt}]
    out = model.chat(image=None, msgs=msgs, tokenizer=tokenizer).strip()

    # 强行只保留 A-Z
    letters = re.findall(r"[A-Z]", out)
    letters = sorted(set(letters))  # 去重 + 排序，以免乱序/重复
    return "".join(letters)

def accuracy_strict_multilabel(pred: str, gt: str) -> bool:
    """
    严格多选判分：pred 必须与 gt 完全一致（忽略空白；大小写按大写比较）
    """
    p = "".join(re.findall(r"[A-Z]", pred.upper()))
    g = "".join(re.findall(r"[A-Z]", gt.upper()))
    return p == g

# ======================
# 主流程
# ======================
def main():
    # 载入题目
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = [data] if isinstance(data, dict) else data

    results = []
    correct = 0
    total = 0

    for it in items:
        qid = it.get("Question id") or it.get("Question_id") or ""
        img_field = it.get("Image") or ""
        text = it.get("Text", "")
        answer_choices = it.get("Answer choices", [])
        gt_letters = (it.get("Ground truth") or "").strip()

        # 用 image 字段找对应的 npz
        prefix = prefix_from_image_field(img_field)
        npz_path = find_npz_for_prefix(SKYSENSE_NPZ_ROOT, prefix)

        status = "ok"
        reason = ""
        pred_letters = ""
        present = []

        if npz_path is None:
            status = "fail"
            reason = "npz_not_found"
        else:
            try:
                present = present_classes_from_npz(npz_path, MIN_PIXELS, MIN_RATIO)
                # 构造 prompt
                pred_letters = ""
                # 解析出 (A) 文本
                pairs = []
                for ch in answer_choices:
                    m = re.match(r"\(([A-Z])\)\s*(.+)", ch.strip())
                    if m:
                        pairs.append((m.group(1), m.group(2)))

                # 逐项问：只要回答yes就把该字母加入
                for letter, text_label in pairs:
                    q = build_choice_prompt(present, text_label)
                    if fm9g_yes_no(q):
                        pred_letters += letter

                # 规范化（排序去重，防止万一）
                pred_letters = "".join(sorted(set(pred_letters)))

            except Exception as e:
                status = "fail"
                reason = f"exception:{e}"

        match = False
        total += 1
        if status == "ok":
            match = accuracy_strict_multilabel(pred_letters, gt_letters)
            if match:
                correct += 1

        results.append({
            "Question id": qid,
            "image": img_field,
            "npz": npz_path,
            "present_classes": present,
            "answer_choices": answer_choices,
            "gt": gt_letters,
            "pred": pred_letters,
            "match": bool(match),
            "status": status,
            "reason": reason
        })

    acc = (correct / total) if total > 0 else 0.0
    out = {
        "total": total,
        "correct": correct,
        "accuracy": acc,
        "detail": results
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[EVAL] total={total} correct={correct} acc={acc:.3f}")
    print(f"[SAVE] {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
