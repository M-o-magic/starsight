import os
import re
import json
from glob import glob
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import cv2

# ========= 配置 =========
INPUT_JSON = "/data/cj/valid_contest/Counting__Counting_with_changing_detection.json"  # 你的输入JSON（列表）
SKYSENSE_NPZ_ROOT = "/data/cj/valid_contest/cropped_red_regions/HRSCD-4_sky_out"       # 放 Skysense 输出 npz 的根目录
OUTPUT_JSON = "/data/cj/valid_contest/Counting__Changedetection_eval_building_change.json"

# ========= 计数超参（可按需微调）=========
MIN_AREA   = 80     # 过滤过小碎片（像素）
HOLE_DILATE = 3     # 空洞膨胀迭代次数（用于切缝）
NECK_WIDTH  = 3.0   # 断颈阈值（像素）；越大切分越强
CLOSE_KSIZE = 3     # 切分后轻微闭运算平滑
EDGE_MARGIN = 2     # 新增：边缘容差(像素)。离图像边缘 <= EDGE_MARGIN 的实例视为触边，不计数


# ========= 工具 =========
def norm_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[_\-/]+", " ", s)
    # 关键改动：保留中文 \u4e00-\u9fff
    s = re.sub(r"[^a-z0-9 \u4e00-\u9fff]+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_gt_value(answer_choices: List[str], gt_letter: str) -> Optional[int]:
    tag = f"({gt_letter})"
    for ch in answer_choices:
        ch_s = ch.strip()
        if ch_s.startswith(tag):
            m = re.search(r"-?\d+", ch_s)
            if m:
                return int(m.group(0))
            return None
    return None

def basename_noext(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]

def prefix_from_image_field(field: str) -> str:
    # 例: "HRSCD-4/35-2012-0340-6805-LA93-0M50-E080-changing2.png" -> "35-2012-0340-6805-LA93-0M50-E080-changing2"
    return basename_noext(field)

def find_npz_for_prefix(root: str, prefix: str) -> Optional[str]:
    # 在 root 下递归找以 prefix 开头的 npz
    pats = [
        os.path.join(root, "**", f"{prefix}.npz"),
        os.path.join(root, "**", f"{prefix}_*.npz"),
        os.path.join(root, "**", f"{prefix}*.npz"),
    ]
    for pat in pats:
        cand = glob(pat, recursive=True)
        if cand:
            # 返回最短路径或第一个（稳定）
            cand.sort(key=lambda x: (len(x), x))
            return cand[0]
    return None

# ========= npz 读取 =========
def load_npz(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    mask = data["mask"]           # (H, W) int 类别索引
    classes = data["classes"]     # 列表或 ndarray
    if not isinstance(classes, (list, tuple)):
        classes = classes.tolist()
    return mask, classes

def find_building_class_id(classes: List[str]) -> Optional[int]:
    """
    在 classes 里寻找 'building' 类（做了一些朴素的别名兼容）
    """
    targets = {
        "building", "buildings", "house", "houses",
        "residential building", "residential", "structure", "edifice",
        "建筑", "建筑物", "屋", "住宅", "居民楼", "楼", "厂", "构筑物", "房"
    }
    # 先精确 'building'
    for i, name in enumerate(classes):
        if norm_text(name) == "building":
            return i
    # 再模糊
    for i, name in enumerate(classes):
        if norm_text(name) in targets:
            return i
    return None
# ==== Fallback VLM 依赖（仅在需要时才用） ====
from PIL import Image
import torch
from transformers import AutoModel, AutoTokenizer

_VLM_MODEL = None
_VLM_TOKENIZER = None
_VLM_DIR = None     # 由命令行 --vlm_dir 传入（可选）
_IMG_ROOT = None    # 用现有 --img_root 覆盖

def _open_image_rgb(path: str) -> Image.Image:
    img = Image.open(path)
    return img.convert("RGB") if img.mode != "RGB" else img

def _load_vlm_once(vlm_dir: str):
    global _VLM_MODEL, _VLM_TOKENIZER
    if _VLM_MODEL is None or _VLM_TOKENIZER is None:
        _VLM_MODEL = AutoModel.from_pretrained(
            vlm_dir, trust_remote_code=True, attn_implementation='sdpa',
            torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32)
        ).eval().to('cuda' if torch.cuda.is_available() else 'cpu')
        _VLM_TOKENIZER = AutoTokenizer.from_pretrained(vlm_dir, trust_remote_code=True)

def _vlm_build_prompt(question: str, answer_choices: list[str]) -> str:
    choices_text = "\n".join(answer_choices or [])
    return (
        f"{question}\n\n"
        "请在以下选项中选择最合适的一项，并且只输出选项字母（A/B/C/D），不要输出任何解释：\n"
        f"{choices_text}\n\n"
        "只输出一个字母：A 或 B 或 C 或 D。"
    )

def _extract_letter(text: str) -> str:
    if not text: return ""
    t = str(text).strip().upper()
    m = re.search(r'[\(\[\{<]?\s*([ABCD])\s*[\)\]\}>:\.]?', t)
    return m.group(1) if m else ""

def _numbers_in_choices(answer_choices: list[str]) -> list[int]:
    nums = []
    for ch in answer_choices or []:
        m = re.search(r"-?\d+", ch)
        if m:
            nums.append(int(m.group(0)))
    return nums

# ========= 切分算法 =========
def _fill_holes(mask01: np.ndarray) -> np.ndarray:
    m = (mask01 > 0).astype(np.uint8).copy()
    h, w = m.shape
    ff = np.zeros((h + 2, w + 2), np.uint8)
    flood = m.copy()
    cv2.floodFill(flood, ff, (0, 0), 1)
    inv = 1 - flood
    filled = np.clip(m | inv, 0, 1).astype(np.uint8)
    return filled

def split_mask_by_holes_and_necks(
    binary_mask: np.ndarray,
    hole_dilate: int = HOLE_DILATE,
    neck_width: float = NECK_WIDTH,
    min_area: int = MIN_AREA,
    close_ksize: int = CLOSE_KSIZE
) -> List[np.ndarray]:
    m = (binary_mask > 0).astype(np.uint8)
    if m.sum() == 0:
        return []

    # 1) 用“空洞扩张”做切缝
    filled = _fill_holes(m)
    holes = (filled - m).astype(np.uint8)
    if hole_dilate > 0 and holes.any():
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        holes_d = holes.copy()
        for _ in range(hole_dilate):
            holes_d = cv2.dilate(holes_d, k, iterations=1)
            holes_d = np.minimum(holes_d, m)  # 仅在前景内扩张
        m_cut = (m & (1 - holes_d)).astype(np.uint8)
    else:
        m_cut = m

    # 2) 距离变换“断颈”
    if neck_width > 0:
        dist = cv2.distanceTransform((m_cut * 255).astype(np.uint8), cv2.DIST_L2, 3)
        keep = (dist >= float(neck_width)).astype(np.uint8)
        m_cut = (m_cut & keep).astype(np.uint8)

    # 3) 轻微闭运算
    if close_ksize > 0:
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        m_cut = cv2.morphologyEx(m_cut, cv2.MORPH_CLOSE, k2, iterations=1)

    # 4) 连通域 + 面积过滤
    num_labels, labels = cv2.connectedComponents(m_cut.astype(np.uint8))
    comps = []
    for lab in range(1, num_labels):
        comp = (labels == lab).astype(np.uint8)
        if comp.sum() >= int(min_area):
            comps.append(comp)
    return comps

# ========= 触边判定 =========
def is_touching_border(comp: np.ndarray, edge_margin: int = 0) -> bool:
    """
    comp: 0/1 子mask（H,W）
    edge_margin: 边缘容差，>0 时离边缘 <= margin 也算触边
    """
    ys, xs = np.where(comp > 0)
    if xs.size == 0:
        return False
    H, W = comp.shape
    return (
        ys.min() < edge_margin or
        xs.min() < edge_margin or
        ys.max() >= (H - edge_margin) or
        xs.max() >= (W - edge_margin)
    )

def count_instances_for_class(mask: np.ndarray, class_id: int) -> int:
    """
    统计 class_id 的实例数：切分 -> 过滤触边 -> 计数
    """
    bin_all = (mask == class_id).astype(np.uint8)
    submasks = split_mask_by_holes_and_necks(
        bin_all,
        hole_dilate=HOLE_DILATE,
        neck_width=NECK_WIDTH,
        min_area=MIN_AREA,
        close_ksize=CLOSE_KSIZE
    )
    # 过滤触边的实例
    kept = [m for m in submasks if not is_touching_border(m, EDGE_MARGIN)]
    return len(kept)

# ========= 主评测 =========
def evaluate_one(item: Dict[str, Any]) -> Dict[str, Any]:
    qid = item.get("Question id", "")
    img1_field = item.get("Image1", "")
    img2_field = item.get("Image2", "")
    answer_choices = item.get("Answer choices", [])
    gt_letter = item.get("Ground truth", "").strip()

    pref1 = prefix_from_image_field(img1_field)
    pref2 = prefix_from_image_field(img2_field)

    # 根目录根据 HRSCD-4 / HRSCD-5 切换
    if img1_field.split('/')[0] == 'HRSCD-4':
        npz1 = find_npz_for_prefix(SKYSENSE_NPZ_ROOT, pref1)
    else:
        npz1 = find_npz_for_prefix(SKYSENSE_NPZ_ROOT.replace("HRSCD-4", "HRSCD-5"), pref1)

    if img2_field.split('/')[0] == 'HRSCD-4':
        npz2 = find_npz_for_prefix(SKYSENSE_NPZ_ROOT, pref2)
    else:
        npz2 = find_npz_for_prefix(SKYSENSE_NPZ_ROOT.replace("HRSCD-4", "HRSCD-5"), pref2)

    # 解析 GT 差值
    gt_value = parse_gt_value(answer_choices, gt_letter)

    status = "ok"
    reason = ""
    pred_diff = None
    n1 = n2 = None

    if npz1 is None or npz2 is None:
        status = "fail"
        reason = "npz_not_found"
    else:
        try:
            mask1, classes1 = load_npz(npz1)
            mask2, classes2 = load_npz(npz2)
            # building 类 id（两张图可能 class 列表不同，所以分别找）
            bid1 = find_building_class_id(classes1)
            bid2 = find_building_class_id(classes2)
            if bid1 is None or bid2 is None:
                status = "fail"
                reason = "building_class_not_found"
            else:
                n1 = count_instances_for_class(mask1, bid1)
                n2 = count_instances_for_class(mask2, bid2)
                pred_diff = abs(int(n1) - int(n2))
        except Exception as e:
            status = "fail"
            reason = f"exception:{e}"

    # ==== 新增：当 pred_diff 不在选项数字中时，启用兜底大模型 ====
    vlm_used = False
    vlm_pred_letter = None
    vlm_raw = None
    vlm_match = False
    gt_letter_text = (item.get("Ground truth") or "").strip()
    gt_letter_char = _extract_letter(gt_letter_text)  # 期望是 A/B/C/D

    try:
        numeric_choices = _numbers_in_choices(answer_choices)
    except Exception:
        numeric_choices = []

    need_fallback = (
        status == "ok"
        and (pred_diff is not None)
        and (len(numeric_choices) > 0)
        and (pred_diff not in numeric_choices)
        and (_VLM_DIR is not None)            # 只有提供了 --vlm_dir 才会用兜底
    )

    if need_fallback:
        try:
            # 用“原图2 + 原始题干”提问大模型
            img2_field = item.get("Image2", "")
            img2_path = img2_field if os.path.isabs(img2_field) else (
                os.path.join(_IMG_ROOT, img2_field) if _IMG_ROOT else img2_field
            )
            if os.path.exists(img2_path):
                _load_vlm_once(_VLM_DIR)
                image2 = _open_image_rgb(img2_path)
                question_text = item.get("Text", "") or ""
                prompt = _vlm_build_prompt(question_text, answer_choices)
                msgs = [{"role": "user", "content": [image2, prompt]}]
                vlm_raw = _VLM_MODEL.chat(image=None, msgs=msgs, tokenizer=_VLM_TOKENIZER)
                vlm_pred_letter = _extract_letter(vlm_raw)
                vlm_match = bool(vlm_pred_letter and gt_letter_char and vlm_pred_letter == gt_letter_char)
                vlm_used = True
        except Exception as _:
            vlm_used = False



    # 正确性
    # match = False
    # if status == "ok" and (gt_value is not None) and (pred_diff is not None):
    #     match = (int(pred_diff) == int(gt_value))
    # elif gt_value is None:
    #     status = "fail"
    #     reason = reason or "gt_not_numeric"
    # 正确性（合并数值匹配 与 兜底模型匹配）
    final_match = False
    if status == "ok":
        if (gt_value is not None) and (pred_diff is not None) and (int(pred_diff) == int(gt_value)):
            final_match = True
        elif vlm_match:  # 数值不在选项里时触发兜底；若兜底选项与 GT 字母一致，则视为命中
            final_match = True
    else:
        final_match = False

    if gt_value is None:
        status = "fail"
        reason = reason or "gt_not_numeric"


    return {
        "Question id": qid,
        "prefix1": img1_field,
        "prefix2": img2_field,
        "npz1": npz1,
        "npz2": npz2,
        "building_count_image1": n1,
        "building_count_image2": n2,
        "pred_diff": pred_diff,
        "gt_value": gt_value,
        # "match": bool(match),
        "match": bool(final_match),
        "status": status,
        "reason": reason,
        # 新增：兜底信息（便于复盘）
        "fallback_vlm_used": vlm_used,
        "fallback_vlm_pred_letter": vlm_pred_letter,
        "fallback_vlm_raw": vlm_raw,
        "gt_letter": gt_letter_char
    }

def main():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        items = [data]
    else:
        items = data

    results = []
    correct = 0
    total = 0

    for item in items:
        r = evaluate_one(item)
        results.append(r)
        # 这里将 fail 也计入总数（按你的上一版逻辑）
        total += 1
        if r["status"] == "ok" and r["match"]:
            correct += 1

    acc = (correct / total) if total > 0 else 0.0

    out = {
        "total": total,
        "correct": correct,
        "accuracy": acc,
        "detail": results
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[EVAL] total={total} correct={correct} acc={acc:.3f}")
    print(f"[SAVE] {OUTPUT_JSON}")

if __name__ == "__main__":
    # ====== 新增：命令行参数，仅覆盖全局配置，其他代码保持不变 ======
    import argparse
    parser = argparse.ArgumentParser(description="Building-change eval from SkySense npz")
    parser.add_argument("--input_json", type=str, help="输入 JSON 路径（列表或含题项）")
    parser.add_argument("--npz_root", type=str, help="Skysense 输出 npz 的根目录（包含 HRSCD-4/HRSCD-5 结构）")
    parser.add_argument("--output_json", type=str, help="评测结果输出 JSON 路径")
    parser.add_argument("--img_root", type=str, default=None, help="图像根目录（本脚本当前未使用，预留）")
    parser.add_argument("--vlm_dir", type=str, default=None, help="可选：兜底多模态大模型目录（用于数值不匹配时）")

    args = parser.parse_args()
    if args.img_root:
        IMG_ROOT = args.img_root  # 预留，不影响现有逻辑
    # 新增：给兜底用
    _VLM_DIR = args.vlm_dir
    _IMG_ROOT = args.img_root

    if args.input_json:
        INPUT_JSON = args.input_json
    if args.npz_root:
        SKYSENSE_NPZ_ROOT = args.npz_root
    if args.output_json:
        OUTPUT_JSON = args.output_json
    if args.img_root:
        IMG_ROOT = args.img_root  # 预留，不影响现有逻辑

    main()