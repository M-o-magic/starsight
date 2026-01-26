import os
import re
import json
from glob import glob
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import cv2
import tifffile as tiff
from PIL import Image
import argparse  # ← 新增

# ======================
# 路径与配置（默认值，可被命令行覆盖）
# ======================
INPUT_JSON  = "/data/cj/RS_StarSight/src/datasets/valid/data/sub_json/object_motion_state.json"
IMAGE_ROOT  = "/data/cj/valid_contest/valid_images"
OUTPUT_DIR  = "/data/cj/valid_contest/motion_eval_vis"
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "eval_motion_state_from_npz_obb_fm9g.json")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- 标准阈值与常量 ----
MIN_PIXELS            = 20
ON_ROAD_RATIO_THR     = 0.10
NEIGHBOR_EXPAND_SCALE = 1.0
NEIGHBOR_MIN_PIXELS   = 20
DETECT_IOU_THR        = 0.10

# ======================
# FM9G 模型加载（仅文本）
# ======================
import torch
from transformers import AutoModel, AutoTokenizer

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

def fm9g_yesno_from_text(prompt: str) -> str:
    model, tokenizer = load_fm9g()
    msgs = [{"role": "user", "content": prompt + "Answer with a single word: yes or no."}]
    out = model.chat(image=None, msgs=msgs, tokenizer=tokenizer) or ""
    return out.strip()

def normalize_yesno(s: str) -> Optional[str]:
    if not s:
        return None
    raw = s.strip()
    low = raw.lower()

    # 1) 先直接匹配（中英文，都用原始字符串）
    zh_yes = {"是", "对", "正确", "是的"}
    zh_no  = {"否", "不是", "不", "不是的"}
    en_yes = {"y", "yes", "true", "1", "yeah", "yep"}
    en_no  = {"n", "no", "false", "0", "nope"}

    if raw in zh_yes: return "Yes"
    if raw in zh_no:  return "No"
    if low in en_yes: return "Yes"
    if low in en_no:  return "No"

    # 2) 包含式匹配（避免长句/解释性回答漏判）
    if "yes" in low: return "Yes"
    if "no"  in low: return "No"
    if any(k in raw for k in zh_yes): return "Yes"
    if any(k in raw for k in zh_no):  return "No"

    # 3) 英文首 token 兜底（把“yes,” “no.”这类清掉）
    tokens = re.findall(r"[a-zA-Z]+", low)
    if tokens:
        if tokens[0] in en_yes: return "Yes"
        if tokens[0] in en_no:  return "No"

    return None


# ======================
# 基础工具
# ======================
def basename_noext(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]

def parse_bbox_from_text(text: str) -> Optional[Tuple[int, int, int, int]]:
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


def clamp_xyxy(x1, y1, x2, y2, w, h) -> Tuple[int,int,int,int]:
    if x1 > x2: x1, x2 = x2, x1
    if y1 > y2: y1, y2 = y2, y1
    x1 = max(0, min(int(x1), w-1))
    y1 = max(0, min(int(y1), h-1))
    x2 = max(0, min(int(x2), w-1))
    y2 = max(0, min(int(y2), h-1))
    return x1, y1, x2, y2

def extract_choice_map(choices: List[str]) -> Dict[str, str]:
    m: Dict[str, str] = {}
    for ch in choices or []:
        mm = re.match(r"\(([A-Z])\)\s*(.+)", ch.strip(), re.I)
        if mm:
            m[mm.group(1).upper()] = mm.group(2).strip()
    return m

YES_ALIASES = {"yes", "y", "true", "1", "yeah", "yep", "是", "对", "正确", "是的"}
NO_ALIASES  = {"no", "n", "false", "0", "nope", "否", "不是", "不", "不是的"}

def find_letter_for_label(choice_map: Dict[str,str], label_target: str) -> Optional[str]:
    tgt = label_target.strip().lower()
    aliases = YES_ALIASES if tgt == "yes" else NO_ALIASES

    for k, v in choice_map.items():
        vv = v.strip().lower()
        # 完全相等或包含任意别名都算命中
        if vv in aliases or any(a in vv for a in aliases):
            return k

    # 兜底：英文 y/n 首字母
    if tgt == "yes":
        for k, v in choice_map.items():
            if v.strip().lower().startswith("y"):
                return k
    else:
        for k, v in choice_map.items():
            if v.strip().lower().startswith("n"):
                return k
    return None

# ======================
# NPZ 语义信息（环境/邻域）
# ======================
def npz_path_from_image_field(image_field: str) -> Optional[str]:
    image_field = (image_field or "").strip().lstrip("/").replace("\\", "/")
    dataset = image_field.split("/", 1)[0] if "/" in image_field else "FAIR1M"
    prefix = basename_noext(image_field)
    npz_dir = os.path.join(IMAGE_ROOT, f"{dataset}_sky_out")
    exact = os.path.join(npz_dir, f"{prefix}.npz")
    if os.path.exists(exact):
        return exact
    cands = glob(os.path.join(npz_dir, "**", f"{prefix}*.npz"), recursive=True)
    if cands:
        cands.sort(key=lambda p: (len(p), p))
        return cands[0]
    return None

def load_npz(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    mask = data["mask"]
    classes = data["classes"]
    if not isinstance(classes, (list, tuple)):
        classes = classes.tolist()
    return mask, classes

def top_class_in_bbox(mask: np.ndarray, classes: List[str], xyxy: Tuple[int,int,int,int]) -> Tuple[Optional[int], Optional[str], List[Tuple[str,int]]]:
    H, W = mask.shape
    x1, y1, x2, y2 = xyxy
    x1 = max(0, min(x1, W-1)); x2 = max(0, min(x2, W-1))
    y1 = max(0, min(y1, H-1)); y2 = max(0, min(y2, H-1))
    if x2 <= x1 or y2 <= y1: return None, None, []
    patch = mask[y1:y2+1, x1:x2+1]
    ids, cnts = np.unique(patch, return_counts=True)
    items = []
    for cid, cnt in zip(ids, cnts):
        if cid == 0:
            continue
        cname = str(classes[int(cid)])
        items.append((cname, int(cnt)))
    items.sort(key=lambda x: -x[1])
    if not items: return None, None, []
    main_name = items[0][0]
    try:
        cid = next(i for i, nm in enumerate(classes) if str(nm) == main_name)
    except StopIteration:
        cid = None
    return cid, main_name, items

def road_like_ratio_in_bbox(mask: np.ndarray, classes: List[str], xyxy: Tuple[int,int,int,int]) -> float:
    H, W = mask.shape
    x1, y1, x2, y2 = xyxy
    x1 = max(0, min(x1, W-1)); x2 = max(0, min(x2, W-1))
    y1 = max(0, min(y1, H-1)); y2 = max(0, min(y2, H-1))
    if x2 <= x1 or y2 <= y1: return 0.0
    patch = mask[y1:y2+1, x1:x2+1]
    road_ids = set()
    for i, nm in enumerate(classes):
        nm_l = str(nm).lower()
        if "road" in nm_l or "roadway" in nm_l:
            road_ids.add(i)
    if not road_ids: return 0.0
    total = patch.size
    cnt = sum(int((patch == rid).sum()) for rid in road_ids)
    return (cnt / total) if total > 0 else 0.0

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import math

def _xyxy_center(x1, y1, x2, y2):
    return (0.5*(x1+x2), 0.5*(y1+y2))

def _point_in_rect(px: float, py: float, rect_xyxy: Tuple[int,int,int,int]) -> bool:
    x1, y1, x2, y2 = rect_xyxy
    return (px >= x1) and (px <= x2) and (py >= y1) and (py <= y2)

def _aabb_expand_ring(xyxy: Tuple[int,int,int,int], scale: float,
                      W: Optional[int]=None, H: Optional[int]=None) -> Tuple[Tuple[int,int,int,int], Tuple[int,int,int,int]]:
    x1, y1, x2, y2 = xyxy
    bw, bh = (x2 - x1 + 1), (y2 - y1 + 1)
    mx, my = int(round(bw * scale)), int(round(bh * scale))
    X1, Y1 = x1 - mx, y1 - my
    X2, Y2 = x2 + mx, y2 + my
    if W is not None and H is not None:
        X1, Y1 = max(0, X1), max(0, Y1)
        X2, Y2 = min(W-1, X2), min(H-1, Y2)
    return (X1, Y1, X2, Y2), (x1, y1, x2, y2)

def _aabb_iou(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter + 1e-6
    return inter / union

def _norm_name(name: str) -> str:
    return (name or "").strip().lower().replace(" ", "").replace("-", "").replace("_", "")

def _canonical_name(name: str, group_map: Optional[Dict[str,str]]=None) -> str:
    n = _norm_name(name)
    default_groups = {
        "car": "vehicle", "truck": "vehicle", "bus": "vehicle", "van": "vehicle", "motorcycle": "vehicle",
        "vehicle": "vehicle",
        "ship": "ship", "boat": "ship", "vessel": "ship",
        "airplane": "airplane", "plane": "airplane", "aircraft": "airplane",
        "person": "person", "pedestrian": "person",
        "train": "train", "trainengine": "train", "wagon": "train",
    }
    if group_map:
        gmap = {**default_groups, **{_norm_name(k): v for k, v in group_map.items()}}
    else:
        gmap = default_groups
    return gmap.get(n, n)

def has_neighbor_same_class_obb(
    detections: List[Dict[str, Any]],
    target_xyxy: Tuple[int,int,int,int],
    target_cls_name: Optional[str],
    img_wh: Optional[Tuple[int,int]] = None,
    expand_scale: float = 1.0,
    min_count: int = 1,
    min_conf: float = 0.0,
    iou_exclude_thr: float = 0.5,
    class_group_map: Optional[Dict[str,str]] = None,
) -> Tuple[bool, int, List[Dict[str, Any]]]:
    if target_xyxy is None:
        return False, 0, []

    W = img_wh[0] if img_wh else None
    H = img_wh[1] if img_wh else None
    outer_rect, inner_rect = _aabb_expand_ring(target_xyxy, expand_scale, W, H)

    canon_target = _canonical_name(target_cls_name or "", class_group_map)

    neighbors = []
    for det in detections:
        if "xyxy" not in det: 
            continue
        if det.get("conf", 1.0) < min_conf:
            continue

        det_xyxy = tuple(int(v) for v in det["xyxy"])
        if _aabb_iou(det_xyxy, target_xyxy) >= iou_exclude_thr:
            continue

        det_name = det.get("cls_name", "")
        if canon_target:
            if _canonical_name(det_name, class_group_map) != canon_target:
                continue

        cx, cy = _xyxy_center(*det_xyxy)
        in_outer = _point_in_rect(cx, cy, outer_rect)
        in_inner = _point_in_rect(cx, cy, inner_rect)
        if in_outer and (not in_inner):
            neighbors.append({
                "xyxy": list(det_xyxy),
                "cls_name": det_name,
                "conf": float(det.get("conf", 0.0))
            })

    count = len(neighbors)
    return (count >= min_count), count, neighbors

# ======================
# OBB 检测（Ultralytics YOLO）
# ======================
DETECT_WEIGHTS = os.environ.get("DETECT_WEIGHTS", "/home/cj/yolov11/yolo11x-obb.pt")
_use_detector = True
_yolo_model = None
_yolo_names = None

def load_obb_detector():
    global _yolo_model, _yolo_names
    if _yolo_model is not None: return _yolo_model, _yolo_names
    try:
        from ultralytics import YOLO
    except Exception as e:
        print("[WARN] ultralytics 未安装或导入失败，将跳过检测器：", e)
        _yolo_model = None
        return None, None
    if not os.path.exists(DETECT_WEIGHTS):
        print(f"[WARN] 检测权重不存在：{DETECT_WEIGHTS}，将跳过检测器")
        _yolo_model = None
        return None, None
    _yolo_model = YOLO(DETECT_WEIGHTS)
    _yolo_names = getattr(_yolo_model, "names", None) or getattr(getattr(_yolo_model, "model", None), "names", None)
    return _yolo_model, _yolo_names

def aabb_of_polygon(poly: np.ndarray) -> Tuple[int,int,int,int]:
    pts = poly.reshape(-1, 2)
    xmin = int(np.floor(pts[:,0].min()))
    ymin = int(np.floor(pts[:,1].min()))
    xmax = int(np.ceil(pts[:,0].max()))
    ymax = int(np.ceil(pts[:,1].max()))
    return xmin, ymin, xmax, ymax

def iou_xyxy(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter + 1e-6
    return inter / union

def run_obb_detector(image_path: str) -> List[Dict[str, Any]]:
    model, names = load_obb_detector()
    if model is None or (not _use_detector):
        return []
    res = model.predict(image_path, imgsz=1024, conf=0.25, verbose=False)
    out = []
    for r in res:
        if hasattr(r, "obb") and r.obb is not None and r.obb.data is not None:
            polys = getattr(r.obb, "xyxyxyxy", None)
            clss  = getattr(r.obb, "cls", None)
            confs = getattr(r.obb, "conf", None)
            if polys is None:
                boxes = getattr(r.obb, "xyxy", None)
                if boxes is None: continue
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = [int(v) for v in boxes[i].tolist()]
                    cid = int(clss[i].item()) if clss is not None else -1
                    conf = float(confs[i].item()) if confs is not None else 0.0
                    cname = str(names.get(cid, cid)) if isinstance(names, dict) else str(names[cid]) if names else str(cid)
                    out.append({"xyxy": (x1,y1,x2,y2), "poly": None, "cls_id": cid, "cls_name": cname, "conf": conf})
            else:
                for i in range(len(polys)):
                    poly = polys[i].cpu().numpy().reshape(-1)
                    x1, y1, x2, y2 = aabb_of_polygon(poly)
                    cid = int(clss[i].item()) if clss is not None else -1
                    conf = float(confs[i].item()) if confs is not None else 0.0
                    cname = str(names.get(cid, cid)) if isinstance(names, dict) else str(names[cid]) if names else str(cid)
                    out.append({"xyxy": (x1,y1,x2,y2), "poly": poly.tolist(), "cls_id": cid, "cls_name": cname, "conf": conf})
        else:
            boxes = getattr(r, "boxes", None)
            if boxes is None: continue
            xyxy = getattr(boxes, "xyxy", None)
            clss = getattr(boxes, "cls", None)
            confs= getattr(boxes, "conf", None)
            if xyxy is None: continue
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = [int(v) for v in xyxy[i].tolist()]
                cid = int(clss[i].item()) if clss is not None else -1
                conf = float(confs[i].item()) if confs is not None else 0.0
                cname = str(names.get(cid, cid)) if isinstance(names, dict) else str(names[cid]) if names else str(cid)
                out.append({"xyxy": (x1,y1,x2,y2), "poly": None, "cls_id": cid, "cls_name": cname, "conf": conf})
    return out

def match_detection_to_bbox(dets: List[Dict[str,Any]], bbox_xyxy: Tuple[int,int,int,int], iou_thr: float = DETECT_IOU_THR):
    best = None; best_iou = 0.0
    for d in dets:
        iou = iou_xyxy(d["xyxy"], bbox_xyxy)
        if iou > best_iou:
            best_iou = iou; best = d
    if best is not None and best_iou >= iou_thr:
        return best, best_iou
    return None, best_iou

# ======================
# Prompt 构造
# ======================
def _fmt_env_top(env_top: List[Tuple[str,int]], k: int = 6) -> str:
    if not env_top:
        return "no salient classes"
    parts = [f"{nm} ({cnt} px)" for nm, cnt in env_top[:k]]
    return ", ".join(parts)

def _fmt_road_sentence(road_ratio: float, on_road: bool, thr: float) -> str:
    pct = f"{road_ratio*100:.1f}%"
    if on_road:
        return f"Road-like surfaces cover about {pct} of the box (≥ {thr*100:.0f}% threshold), so the region is likely on a road."
    else:
        return f"Road-like surfaces cover about {pct} of the box (< {thr*100:.0f}% threshold), so the region is unlikely to be on a road."

def _fmt_neighbor_sentence(has_same_neighbor: bool, neighbor_pixels: int, expand_scale: float) -> str:
    ring_desc = f"a ring around the box expanded by {expand_scale:.1f}× the box size"
    if has_same_neighbor:
        return f"There are other objects of the same class within {ring_desc} (approx. {neighbor_pixels} px detected)."
    else:
        return f"No other objects of the same class were found within {ring_desc}."

def build_prompt_branch_A(env_top: List[Tuple[str,int]], road_ratio: float,
                          on_road: bool, thr: float) -> str:
    road_sent = _fmt_road_sentence(road_ratio, on_road, thr)
    return (
        "You are given metadata about a region in a remote-sensing image.\n"
        f"{road_sent}\n"
        "No object was detected by the oriented bounding box detector at this location.\n"
        "Given that the region is not a road environment and no object was detected here, "
        "the object is unlikely to be in motion.\n"
        " Determine whether the object within the given reference bounding box is in motion. \n"
        "Answer with a single word: yes or no."
    )

def build_prompt_branch_B(det_name: str, env_top: List[Tuple[str,int]], road_ratio: float,
                          on_road: bool, has_same_neighbor: bool, neighbor_pixels: int,
                          thr: float, expand_scale: float) -> str:
    road_sent = _fmt_road_sentence(road_ratio, on_road, thr)
    neighbor_sent = _fmt_neighbor_sentence(has_same_neighbor, neighbor_pixels, expand_scale)
    return (
        "You are given metadata about a detected target in a remote-sensing image.\n"
        f"Detected target class: {det_name}.\n"
        f"{road_sent}\n"
        f"{neighbor_sent}\n"
        "If the target is on a road/roadway and there are no nearby objects of the same class, "
        "it is very likely to be in motion.\n"
        " Determine whether the object within the given reference bounding box is in motion. \n"
        "Answer with a single word: yes or no."
    )

def build_prompt_branch_C(det_name: str, env_top: List[Tuple[str,int]], road_ratio: float,
                          on_road: bool, thr: float) -> str:
    road_sent = _fmt_road_sentence(road_ratio, on_road, thr)
    return (
        "You are given metadata about a detected target in a remote-sensing image.\n"
        f"Detected target class: {det_name}.\n"
        f"{road_sent}\n"
        "Since the detected target is not on a road-like surface, it is unlikely to be in motion.\n"
        " Determine whether the object within the given reference bounding box is in motion. \n"
        "Answer with a single word: yes or no."
    )

def build_prompt_fallback(target_name: Optional[str], env_top: List[Tuple[str,int]], road_ratio: float,
                          on_road: bool, has_same_neighbor: bool, neighbor_pixels: int,
                          det_hit: bool, det_name: Optional[str],
                          thr: float, expand_scale: float) -> str:
    road_sent = _fmt_road_sentence(road_ratio, on_road, thr)
    neighbor_sent = _fmt_neighbor_sentence(has_same_neighbor, neighbor_pixels, expand_scale)
    det_sent = ("A detector did find a target here" if det_hit else "No detector target was found at this location")
    det_label = f" ({det_name})" if (det_hit and det_name) else ""
    tgt_sent = (f"The mask-majority target class is {target_name}." if target_name else "The mask-majority target class is unknown.")
    return (
        "You are given metadata about a region in a remote-sensing image.\n"
        f"{tgt_sent}\n"
        f"{det_sent}{det_label}.\n"
        f"{road_sent}\n"
        f"{neighbor_sent}\n"
        "Decide whether the object is in motion. Answer with a single word: yes or no."
    )

# ======================
# 主流程
# ======================
def main():
    global INPUT_JSON, IMAGE_ROOT, OUTPUT_DIR, OUTPUT_JSON, FM9G_MODEL_PATH, DETECT_WEIGHTS, _use_detector

    # —— 新增：命令行参数 ——
    ap = argparse.ArgumentParser(description="Motion-state evaluation with NPZ+OBB+FM9G (yes/no)")
    ap.add_argument("--input_json",  type=str, default=INPUT_JSON,  help="Path to input QA JSON")
    ap.add_argument("--image_root",  type=str, default=IMAGE_ROOT,  help="Root directory for images (for relative paths)")
    ap.add_argument("--output_dir",  type=str, default=OUTPUT_DIR,  help="Directory to save outputs")
    ap.add_argument("--output_json", type=str, default=None,        help="Output JSON path (default: <output_dir>/eval_motion_state_from_npz_obb_fm9g.json)")
    ap.add_argument("--fm9g_model_path", type=str, default=FM9G_MODEL_PATH, help="FM9G textual model path")
    ap.add_argument("--detect_weights",  type=str, default=DETECT_WEIGHTS,  help="Ultralytics OBB weights")
    det_group = ap.add_mutually_exclusive_group()
    det_group.add_argument("--use_detector",  dest="use_detector", action="store_true",  help="Enable detector (default)")
    det_group.add_argument("--no_detector",   dest="use_detector", action="store_false", help="Disable detector")
    ap.set_defaults(use_detector=True)
    args = ap.parse_args()

    # 覆盖全局配置
    INPUT_JSON       = args.input_json
    IMAGE_ROOT       = args.image_root
    OUTPUT_DIR       = args.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    OUTPUT_JSON      = args.output_json or os.path.join(OUTPUT_DIR, "eval_motion_state_from_npz_obb_fm9g.json")
    FM9G_MODEL_PATH  = args.fm9g_model_path
    DETECT_WEIGHTS   = args.detect_weights
    _use_detector    = args.use_detector

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = [data] if isinstance(data, dict) else data

    results = []
    correct = 0
    total = 0

    for idx, it in enumerate(items):
        qid       = it.get("Question id") or it.get("Question_id") or f"q_{idx:06d}"
        img_field = (it.get("Image") or "").strip()
        text      = it.get("Text", "")
        choices   = it.get("Answer choices", [])
        gt_letter = (it.get("Ground truth") or "").strip().upper()[:1]

        choice_map = extract_choice_map(choices)
        letter_yes = find_letter_for_label(choice_map, "Yes")
        letter_no  = find_letter_for_label(choice_map, "No")

        status = "ok"; reason = ""
        pred_letter = None
        model_raw = ""
        model_yesno = None

        # 解析 bbox
        bbox = parse_bbox_from_text(text)
        if bbox is None:
            status, reason = "fail", "bbox_parse_fail"

        # 找 npz + 读取 mask/classes
        npz_path = None
        mask = None; classes = None; img_h = None; img_w = None
        if status == "ok":
            npz_path = npz_path_from_image_field(img_field)
            if npz_path is None or not os.path.exists(npz_path):
                status, reason = "fail", "npz_not_found"
            else:
                mask, classes = load_npz(npz_path)
                img_h, img_w = mask.shape

        # clamp bbox
        xyxy = None
        if status == "ok":
            x1, y1, x2, y2 = clamp_xyxy(*bbox, w=img_w, h=img_h)
            xyxy = (x1, y1, x2, y2)
            if x2 <= x1 or y2 <= y1:
                status, reason = "fail", "bbox_invalid_after_clamp"

        # bbox 内主类 & 类别统计
        target_cid = None; target_name = None; bbox_items = []
        if status == "ok":
            target_cid, target_name, bbox_items = top_class_in_bbox(mask, classes, xyxy)
            if target_cid is None:
                status, reason = "fail", "no_object_in_bbox"

        # 加载原图路径
        img_path = None
        if status == "ok":
            img_path = img_field if os.path.isabs(img_field) else os.path.join(IMAGE_ROOT, img_field)
            if not os.path.exists(img_path):
                status, reason = "fail", "image_not_found"

        # 运行 OBB 检测并匹配
        dets = []; det_hit = False; det_match = None; det_iou = 0.0
        if status == "ok" and _use_detector:
            dets = run_obb_detector(img_path)
            if dets:
                det_match, det_iou = match_detection_to_bbox(dets, xyxy, DETECT_IOU_THR)
                det_hit = det_match is not None

        # 环境/邻域
        road_ratio = 0.0; on_road = False
        has_same_neighbor = False; neighbor_pixels = 0
        neighbor_count = 0                 # ← 新增默认，防止未定义
        neighbor_list: List[Dict[str,Any]] = []  # ← 新增默认
        if status == "ok":
            road_ratio = road_like_ratio_in_bbox(mask, classes, xyxy)
            on_road = (road_ratio >= ON_ROAD_RATIO_THR)
            has_same_neighbor, neighbor_count, neighbor_list = has_neighbor_same_class_obb(
                detections=dets,
                target_xyxy=xyxy,
                target_cls_name=(det_match["cls_name"] if det_hit else target_name),
                img_wh=(img_w, img_h),
                expand_scale=NEIGHBOR_EXPAND_SCALE,
                min_count=1,
                min_conf=0.25,
                iou_exclude_thr=0.5,
                class_group_map={
                    "sedan": "vehicle","minivan": "vehicle","lorry": "vehicle",
                    "ship": "ship","boat": "ship","vessel": "ship",
                    "plane": "airplane","aircraft": "airplane",
                }
            )

        # 构造分支 prompt
        prompt = ""
        branch = "fallback"
        if status == "ok":
            if (not det_hit) and (not on_road):
                prompt = build_prompt_branch_A(bbox_items, road_ratio, on_road, ON_ROAD_RATIO_THR); branch = "A"
            elif det_hit and on_road and (not has_same_neighbor):
                prompt = build_prompt_branch_B(det_match["cls_name"], bbox_items, road_ratio, on_road,
                                               has_same_neighbor, neighbor_pixels,
                                               ON_ROAD_RATIO_THR, NEIGHBOR_EXPAND_SCALE); branch = "B"
            elif det_hit and (not on_road):
                prompt = build_prompt_branch_C(det_match["cls_name"], bbox_items, road_ratio, on_road, ON_ROAD_RATIO_THR); branch = "C"
            else:
                det_name = det_match["cls_name"] if det_match else None
                prompt = build_prompt_fallback(target_name, bbox_items, road_ratio, on_road,
                                               has_same_neighbor, neighbor_pixels,
                                               det_hit, det_name, ON_ROAD_RATIO_THR, NEIGHBOR_EXPAND_SCALE); branch = "fallback"

            # 调试输出
            print("="*80)
            print(f"[QID] {qid}")
            print(f"[IMG] {img_field} -> {img_path}")
            print(f"[NPZ] {npz_path}")
            print(f"[BBOX] {xyxy}")
            print(f"[MASK TARGET] {target_name} (cid={target_cid})")
            print(f"[BBOX TOP CLASSES] {bbox_items[:6]}")
            print(f"[STD-1] road_ratio={road_ratio:.3f} -> on_road={on_road} (thr={ON_ROAD_RATIO_THR})")
            print(f"[STD-2] has_same_neighbor={has_same_neighbor} (count={neighbor_count}, expand={NEIGHBOR_EXPAND_SCALE})")
            print(f"[DETECT] num={len(dets)} hit={det_hit} iou={det_iou:.3f} match={det_match['cls_name'] if det_match else None}")
            print(f"[BRANCH] {branch}")
            print(f"[PROMPT]\n{prompt}")

        # 问 FM9G（文本）
        if status == "ok":
            model_raw = fm9g_yesno_from_text(prompt)
            model_yesno = normalize_yesno(model_raw)
            if model_yesno is None:
                status, reason = "fail", "model_no_yesno"
            else:
                pred_letter = (letter_yes if model_yesno == "Yes" else letter_no)
                if pred_letter is None:
                    status, reason = "fail", "choice_align_fail"

        # 打分
        match = (pred_letter == gt_letter) if (pred_letter and gt_letter) else False
        total += 1
        if match: correct += 1

        # 记录
        rec_dets = [{"xyxy": list(d["xyxy"]), "cls_name": d["cls_name"], "conf": d["conf"]} for d in dets[:100]]
        results.append({
            "Question id": qid,
            "image": img_field,
            "image_path": img_path,
            "npz": npz_path,
            "bbox_xyxy": list(xyxy) if xyxy else None,
            "mask_target_class": target_name,
            "bbox_class_counts": [{"name": nm, "pixels": cnt} for nm, cnt in bbox_items],
            "standard1_on_road": {"on_road": on_road, "road_like_ratio": road_ratio, "threshold": ON_ROAD_RATIO_THR},
            "standard2_neighbor": {
                "has_same_class_neighbor": has_same_neighbor,
                "neighbor_count": neighbor_count,
                "neighbors": neighbor_list,
                "expand_scale": NEIGHBOR_EXPAND_SCALE,
                "min_conf": 0.25
            },
            "detector": {
                "weights": DETECT_WEIGHTS,
                "use": _use_detector,
                "hits": det_hit,
                "hit_iou": det_iou,
                "hit_det": ({"xyxy": list(det_match["xyxy"]), "cls_name": det_match["cls_name"], "conf": det_match["conf"]}
                            if det_match else None),
                "detections": rec_dets
            },
            "prompt_branch": branch,
            "prompt": prompt,
            "model_raw": model_raw,
            "model_yesno": model_yesno,
            "choices": choice_map,
            "pred_letter": pred_letter,
            "gt_letter": gt_letter,
            "match": bool(match),
            "status": status,
            "reason": reason
        })

    acc = (correct / total) if total > 0 else 0.0
    out = {"total": total, "correct": correct, "accuracy": acc, "detail": results}
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("-"*80)
    print(f"[EVAL] total={total} correct={correct} acc={acc:.4f}")
    print(f"[SAVE] {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
