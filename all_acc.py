import os
import re
import json
import argparse
from typing import Tuple, Optional, List, Any

LETTER_RE = re.compile(r"[A-Z]", re.I)

def _canon_letters(s: Any) -> str:
    """
    提取字符串中的 A-Z 字母，去重、排序后返回（多选精确匹配用）。
    若不存在字母，返回空串。
    """
    if s is None:
        return ""
    letters = LETTER_RE.findall(str(s).upper())
    if not letters:
        return ""
    # 去重但保持稳定排序：统一用集合后再排序
    return "".join(sorted(set(letters)))

def _first_letter(s: Any) -> str:
    """从文本中抓第一个 A-Z 字母；没有则空串。"""
    if s is None:
        return ""
    m = re.search(r"[A-Z]", str(s).upper())
    return m.group(0) if m else ""

def _pick(item: dict, keys: List[str]) -> Optional[Any]:
    for k in keys:
        if k in item:
            return item[k]
    return None

def _extract_pred_gt_from_item(it: dict) -> Tuple[Optional[str], Optional[str]]:
    """
    从一个明细项里尽力抽出 (pred, gt)。
    兼容键名：Pred_letter/GT_letter, pred_choice/Ground truth, pred/gt, predicted_choice/true_choice,
             predicted_choices/true_choices, model_prediction 等。
    支持多选与单选：多选时返回 canon 化后的字符串（如 'ABD'）。
    """
    # 常见键名组合（按优先级）
    pred_cands = [
        "Pred_letter","pred_letter","pred_choice","pred","prediction","pred_choice",
        "predicted_choice","model_prediction","pred_choices","predicted_choices","Model_pred","Model_raw"
    ]
    gt_cands = [
        "GT_letter","gt_letter","Ground truth","ground_truth","true_choice","gt_choice",
        "true_choices","gt"
    ]

    pred_raw = _pick(it, pred_cands)
    gt_raw   = _pick(it, gt_cands)

    if pred_raw is None or gt_raw is None:
        # 某些结果里字段名不同，再尝试 detail 的别名
        return None, None

    # 尝试作为“多选字母集合”解析
    pred_multi = _canon_letters(pred_raw)
    gt_multi   = _canon_letters(gt_raw)

    if pred_multi and gt_multi:
        return pred_multi, gt_multi

    # 退化为“首个字母”
    pred_one = _first_letter(pred_raw)
    gt_one   = _first_letter(gt_raw)
    if pred_one and gt_one:
        return pred_one, gt_one

    return None, None

def _count_from_detail_list(detail: list) -> Tuple[int, int]:
    """
    从 detail 数组中累加 (correct, total)。
    优先用 'match' 布尔，其次用 (pred, gt) 对。
    """
    correct = 0
    total = 0
    for it in detail:
        if not isinstance(it, dict):
            continue
        if "match" in it and isinstance(it["match"], bool):
            total += 1
            correct += int(it["match"])
            continue
        pred, gt = _extract_pred_gt_from_item(it)
        if pred is not None and gt is not None:
            total += 1
            # 单选：一个字母；多选：canon 后精确匹配
            correct += int(pred == gt)
    return correct, total

def _count_from_top_level(obj: Any) -> Tuple[int, int]:
    """
    尝试直接从顶层结构获取 (correct, total)。
    1) dict 且有 total/correct -> 直接用
    2) dict 且有 detail -> 逐项统计
    3) 顶层就是 list -> 逐项统计
    4) 只有 accuracy + detail -> 用 len(detail) 推回
    5) 否则(0,0)
    """
    # 1) 直接有 total/correct
    if isinstance(obj, dict):
        if "total" in obj and "correct" in obj:
            try:
                t = int(obj["total"])
                c = int(obj["correct"])
                if t >= 0 and 0 <= c <= t:
                    return c, t
            except Exception:
                pass

        # 2) 有 detail
        if "detail" in obj and isinstance(obj["detail"], list):
            c, t = _count_from_detail_list(obj["detail"])
            if t > 0:
                return c, t
            # 4) 只有 accuracy + detail
            if "accuracy" in obj:
                try:
                    acc = float(obj["accuracy"])
                    t = len(obj["detail"])
                    if t > 0 and 0.0 <= acc <= 1.0:
                        c = round(acc * t)
                        return c, t
                except Exception:
                    pass

        # 一些结果可能在 "outputs" 里
        if "outputs" in obj and isinstance(obj["outputs"], list):
            c, t = _count_from_detail_list(obj["outputs"])
            if t > 0:
                return c, t

    # 3) 顶层就是 list
    if isinstance(obj, list):
        c, t = _count_from_detail_list(obj)
        if t > 0:
            return c, t

    # 5) 兜底
    return 0, 0

def summarize_file(path: str) -> Tuple[int, int]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load JSON: {path} ({e})")
        return 0, 0
    c, t = _count_from_top_level(obj)
    if t == 0 and isinstance(obj, dict) and "accuracy" in obj:
        # 最末兜底：仅 accuracy，无 detail/total 情况（几乎不会，但保底）
        try:
            acc = float(obj["accuracy"])
            t = int(obj.get("num_samples", 0) or obj.get("size", 0) or 0)
            if t > 0:
                c = round(acc * t)
        except Exception:
            pass
    return c, t

def main():
    ap = argparse.ArgumentParser(description="Summarize per-file and overall accuracy for a folder of JSON results.")
    ap.add_argument("--root", type=str, required=True, help="Directory containing JSON files (recursively).")
    ap.add_argument("--pattern", type=str, default="*.json", help="Glob pattern for file names (default: *.json).")
    ap.add_argument("--quiet", action="store_true", help="Do not print per-file details, only overall.")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    pattern = args.pattern
    # pattern = '"*.json" '  # 统一小写，避免大小写敏感问题
    # root = os.path.abspath('/data/cj/RS_StarSight/src/datasets/valid/output')

    # 收集所有匹配的 json 文件（递归）
    json_files: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if _fnmatch(fn=fn, pattern=pattern):
                json_files.append(os.path.join(dirpath, fn))
    json_files.sort()

    if not json_files:
        print(f"[INFO] No JSON files matched under: {root} (pattern={pattern})")
        return

    overall_c = 0
    overall_t = 0

    # 每个文件统计
    for fp in json_files:
        c, t = summarize_file(fp)
        overall_c += c
        overall_t += t
        if not args.quiet:
            acc_str = f"{(c/t):.4f}" if t > 0 else "N/A"
            rel = os.path.relpath(fp, root)
            print(f"{rel}\tcorrect={c}\ttotal={t}\tacc={acc_str}")

    # 汇总
    print("-" * 60)
    overall_acc = (overall_c / overall_t) if overall_t > 0 else 0.0
    print(f"Files: {len(json_files)}")
    print(f"Overall: correct={overall_c}  total={overall_t}  acc={overall_acc:.4f}")

def _fnmatch(fn: str, pattern: str) -> bool:
    """
    简易通配匹配：支持 '*.json'、'*_results.json' 等。
    用正则编译一次足够；为避免引额外依赖，手写一个轻量匹配。
    """
    # 将 shell 风格 pattern 转正则
    # '.' -> '\.', '*' -> '.*', '?' -> '.', 其余转义
    esc = re.escape(pattern)
    esc = esc.replace(r"\*", ".*").replace(r"\?", ".")
    return re.fullmatch(esc, fn) is not None

if __name__ == "__main__":
    main()
