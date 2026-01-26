import os, re, json, math, argparse
from PIL import Image
import torch
from ultralytics import YOLO
from transformers import AutoModel, AutoTokenizer
COARSE_CLASSES = ["airplane", "ship", "vehicle", "court", "road"]
FINE_CLASSES = [
    "A220","A321","A330","A350","ARJ21",
    "Baseball Field","Basketball Court",
    "Boeing737","Boeing747","Boeing777","Boeing787",
    "Bridge","Bus","C919","Cargo Truck",
    "Dry Cargo Ship","Dump Truck","Engineering Ship",
    "Excavator","Fishing Boat","Football Field",
    "Intersection","Liquid Cargo Ship","Motorboat",
    "other-airplane","other-ship","other-vehicle",
    "Passenger Ship","Roundabout","Small Car","Tennis Court",
    "Tractor","Trailer","Truck Tractor","Tugboat","Van","Warship"
]
FINE_TO_COARSE = {
    "A220":"airplane","A321":"airplane","A330":"airplane","A350":"airplane","ARJ21":"airplane",
    "Boeing737":"airplane","Boeing747":"airplane","Boeing777":"airplane","Boeing787":"airplane",
    "C919":"airplane","other-airplane":"airplane",
    "Dry Cargo Ship":"ship","Engineering Ship":"ship","Fishing Boat":"ship",
    "Liquid Cargo Ship":"ship","Motorboat":"ship","Passenger Ship":"ship",
    "Tugboat":"ship","Warship":"ship","other-ship":"ship",
    "Bus":"vehicle","Cargo Truck":"vehicle","Dump Truck":"vehicle",
    "Excavator":"vehicle","Small Car":"vehicle","Tractor":"vehicle",
    "Trailer":"vehicle","Truck Tractor":"vehicle","Van":"vehicle","other-vehicle":"vehicle",
    "Baseball Field":"court","Basketball Court":"court","Football Field":"court","Tennis Court":"court",
    "Bridge":"road","Roundabout":"road","Intersection":"road"
}
SYNONYM_MAP = {
    "aircraft":"airplane","plane":"airplane","planes":"airplane","airplanes":"airplane",
    "ship":"ship","ships":"ship","boat":"ship","boats":"ship","vessel":"ship","vessels":"ship",
    "vehicle":"vehicle","vehicles":"vehicle","car":"vehicle","cars":"vehicle","truck":"vehicle","trucks":"vehicle","bus":"vehicle","buses":"vehicle",
    "court":"court","courts":"court","field":"court","fields":"court","stadium":"court", 
    "road":"road","roads":"road","bridge":"road","bridges":"road","roundabout":"road","roundabouts":"road","intersection":"road","intersections":"road",
    "boeing 737":"Boeing737","boeing737":"Boeing737",
    "boeing 747":"Boeing747","boeing747":"Boeing747",
    "boeing 777":"Boeing777","boeing777":"Boeing777",
    "boeing 787":"Boeing787","boeing787":"Boeing787",
    "a321":"A321","a330":"A330","a350":"A350","a220":"A220","arj21":"ARJ21","c919":"C919",
    "small car":"Small Car","small cars":"Small Car",
    "basketball court":"Basketball Court","football field":"Football Field","soccer fields":"Football Field","baseball field":"Baseball Field","tennis court":"Tennis Court",
    "dry cargo ship":"Dry Cargo Ship","liquid cargo ship":"Liquid Cargo Ship","passenger ship":"Passenger Ship",
    "tugboat":"Tugboat","warship":"Warship","motorboat":"Motorboat",
    "cargo truck":"Cargo Truck","dump truck":"Dump Truck","truck tractor":"Truck Tractor","van":"Van","tractor":"Tractor","trailer":"Trailer",
    "excavator":"Excavator","bus":"Bus",
    "roundabout":"Roundabout","intersection":"Intersection","bridge":"Bridge"
}

def norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[-_/]", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def canonical(s: str) -> str:
    n = norm(s)
    if n in SYNONYM_MAP:
        return SYNONYM_MAP[n]
    if n.endswith("s") and n[:-1] in SYNONYM_MAP:
        return SYNONYM_MAP[n[:-1]]
    return n

def extract_target_phrase_llm(model, tokenizer, image, question) -> str:
    prompt = (
        "Extract and output ONLY the object phrase being asked about in the question, "
        "including its adjectives, color, quantity modifier and reference words if any. "
        "Do not add any other words.\n"
        "Here are some examples:\n"
        "Q: How many large white airplanes are parked near the terminal building?\nA: large white airplanes\n"
        "Q: Count the number of small vehicles located in the parking lot.\nA: small vehicles\n"
        f"Q: {question}\nA:"
    )
    try:
        msgs = [{'role': 'user', 'content': [image, prompt]}]
        res = model.chat(image=None, msgs=msgs, tokenizer=tokenizer)
        return str(res).strip().split("\n")[0]
    except Exception:
        return ""

def extract_target_phrase_fallback(q: str) -> str:
    ql = q.lower()
    m = re.search(r"how many (.+?) (?:are|is|in|on|within)", ql)
    if m: return m.group(1).strip()
    m = re.search(r"what .*? (?:is|are) the (.+?)\?", ql)
    if m: return m.group(1).strip()
    return q.strip()

def best_match_coarse(target: str):
    t = canonical(target)
    for i, c in enumerate(COARSE_CLASSES):
        if canonical(c) == t:
            return i, c
    for i, c in enumerate(COARSE_CLASSES):
        if canonical(c) in t or t in canonical(c):
            return i, c
    if t in FINE_TO_COARSE.values():
        cc = t
        return COARSE_CLASSES.index(cc), cc
    for fine in FINE_CLASSES:
        if canonical(fine) == t:
            cc = FINE_TO_COARSE.get(fine)
            if cc:
                return COARSE_CLASSES.index(cc), cc
    return None

def best_match_fine(target: str):
    t = canonical(target)
    for i, c in enumerate(FINE_CLASSES):
        if canonical(c) == t:
            return i, c
    return None

def choose_option_from_count(answer_choices, count_pred: int):
    nums = []
    for s in answer_choices:
        m = re.findall(r"-?\d+", s)
        nums.append(int(m[0]) if m else None)
    for idx, n in enumerate(nums):
        if n is not None and n == count_pred:
            return idx
    best_idx, best_d = None, 1e9
    for idx, n in enumerate(nums):
        if n is None: 
            continue
        d = abs(n - count_pred)
        if d < best_d:
            best_d = d
            best_idx = idx
    return best_idx


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

def load_image(image_field: str):
    """相对路径用 IMG_ROOT 拼；绝对路径直接用。"""
    p = image_field
    if not os.path.isabs(p):
        p = os.path.join(IMG_ROOT, p)
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return p, open_image_rgb_safe(p)

def count_detections_obb(result_list, target_cls_id: int):
    total = 0
    for r in result_list:
        if getattr(r, "obb", None) is None or r.obb is None or r.obb.cls is None:
            continue
        idxs = torch.where(r.obb.cls == target_cls_id)[0]
        total += int(len(idxs))
    return total

def summarize_fine_detections(res_list, class_names):
    counts = {c: 0 for c in class_names}
    for r in res_list:
        if getattr(r, "obb", None) is None or r.obb is None or r.obb.cls is None:
            continue
        cls_ids = r.obb.cls.detach().cpu().numpy().tolist()
        for cid in cls_ids:
            if 0 <= int(cid) < len(class_names):
                counts[class_names[int(cid)]] += 1
    counts = {k: v for k, v in counts.items() if v > 0}
    if counts:
        lines = [f"- {k}: {v}" for k, v in sorted(counts.items())]
        summary = "Detected objects (fine-grained summary):\n" + "\n".join(lines)
    else:
        summary = "Detected objects (fine-grained summary):\n- None"
    return counts, summary

def build_llm_fallback_prompt(question, answer_choices, detection_summary_text):
    opts_text = "\n".join(answer_choices)
    prompt = (
        f"{detection_summary_text}\n\n"
        f"[User Question]\n{question}\n\n"
        "Please choose the best answer ONLY by outputting its option letter (A/B/C/D etc.), no explanation.\n"
        "Answer choices:\n"
        f"{opts_text}\n"
        "Output: "
    )
    return prompt

def extract_option_letter(llm_text, valid_letters=("A","B","C","D","E","F")):
    t = str(llm_text).upper()
    m = re.search(r"\(([A-Z])\)", t)
    if m and m.group(1) in valid_letters:
        return m.group(1)
    for ch in valid_letters:
        if re.search(rf"\b{ch}\b", t):
            return ch
    return None

def main():
    # -------- 新增：命令行参数，并写回到全局变量 --------
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_in", type=str, default=None)
    parser.add_argument("--img_root", type=str, default=None)
    parser.add_argument("--out_json", type=str, default=None)
    parser.add_argument("--coarse_model", type=str, default=None)
    parser.add_argument("--fine_model", type=str, default=None)
    parser.add_argument("--fm9g_path", type=str, default=None)
    args = parser.parse_args()

    # # global JSON_IN, IMG_ROOT, OUT_JSON, MODEL_COARSE_PATH, MODEL_FINE_PATH, FM9G_PATH
    JSON_IN = args.json_in
    IMG_ROOT = args.img_root
    OUT_JSON = args.out_json
    MODEL_COARSE_PATH = args.coarse_model
    MODEL_FINE_PATH = args.fine_model
    FM9G_PATH = args.fm9g_path

    # global JSON_IN, IMG_ROOT, OUT_JSON, MODEL_COARSE_PATH, MODEL_FINE_PATH, FM9G_PATH
    # JSON_IN = '/home/mcislab_cj/VRSBench_images/valid/subset_low/en/Counting__Overall_counting.json'
    # IMG_ROOT = '/home/mcislab_cj/VRSBench_images/valid/images'
    # OUT_JSON = '/home/mcislab_cj/VRSBench_images/valid/10_13_tunegood/output_json/eval_en_1029_overall_counting.json'
    # MODEL_COARSE_PATH = '/home/mcislab_cj/VRSBench_images/valid/best_for_large_fair1m.pt'
    # MODEL_FINE_PATH = '/home/mcislab_cj/yolov11_copy/fair1m5_large_yoloxobb/train2/weights/best_for_small_fair1m.pt'
    # FM9G_PATH = '/home/mcislab_cj/fm9g4bv/FM9G4B-V'
    # ---------------------------------------------------

    # 1) 读数据
    with open(JSON_IN, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 2) 加载模型（一次）
    yolo_coarse = YOLO(MODEL_COARSE_PATH)  # 5类 OBB
    yolo_fine   = YOLO(MODEL_FINE_PATH)    # 细粒度 OBB

    fm9g = AutoModel.from_pretrained(
        FM9G_PATH, trust_remote_code=True,
        attn_implementation='eager', torch_dtype=torch.bfloat16
    ).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(FM9G_PATH, trust_remote_code=True)

    outputs = []
    right, wrong = 0, 0

    for item in data:
        try:
            # 【微调】此处改为使用 load_image(item["Image"])，使 --img_root 生效
            img_path, pil_img = load_image(os.path.join(IMG_ROOT,item["Image"]))
        except Exception as e:
            print(f"[MISS IMG] {item.get('Image')}: {e}")
            continue

        text = item.get("Text", "")
        choices = item.get("Answer choices", [])
        gt = str(item.get("Ground truth", "")).strip().upper()

        # 3) LLM抽取目标短语
        target_phrase = extract_target_phrase_llm(fm9g, tokenizer, pil_img, text) or extract_target_phrase_fallback(text)

        # 4) 匹配类别并检测
        match_c = best_match_coarse(target_phrase)
        match_f = best_match_fine(target_phrase)

        model_used = None
        cls_name = None
        cls_id = None
        det_count = None

        if match_f:
            cls_id, cls_name = match_f
            model_used = "fine(36)"
            res = yolo_fine.predict(img_path, save=True, conf=0.5)
            det_count = count_detections_obb(res, cls_id)
        elif match_c:
            cls_id, cls_name = match_c
            model_used = "coarse(5)"
            res = yolo_coarse.predict(img_path, save=True, conf=0.5)
            det_count = count_detections_obb(res, cls_id)
        else:
            print(f"[FALLBACK] 类别匹配失败，改用细粒度检测摘要 + LLM 选择：{target_phrase}")
            res_fallback = yolo_fine.predict(img_path, save=True, conf=0.5)
            det_counts, det_summary = summarize_fine_detections(res_fallback, FINE_CLASSES)
            fb_prompt = build_llm_fallback_prompt(text, choices, det_summary)
            msgs = [{'role': 'user', 'content': [fb_prompt]}]
            try:
                llm_out = fm9g.chat(image=None, msgs=msgs, tokenizer=tokenizer)
            except Exception as e:
                print(f"[LLM Fallback ERROR] {item.get('Image')}: {e}")
                llm_out = ""
            pred_choice = extract_option_letter(llm_out, valid_letters=tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
            correct = (pred_choice == str(gt).upper())
            right += int(correct); wrong += int(not correct)
            outputs.append({
                "Image": item.get("Image"),
                "Text": text,
                "target_phrase": target_phrase,
                "model_used": "fallback_fine_summary",
                "fine_counts": det_counts,
                "detection_summary": det_summary,
                "Answer choices": choices,
                "pred_choice": pred_choice,
                "Ground truth": gt,
                "llm_raw_output": str(llm_out)
            })
            print(f"[FALLBACK RESULT] pred={pred_choice}  gt={gt}  {'✓' if correct else '✗'}")
            continue

        # 5) 将数量映射到选项
        pred_option_idx = choose_option_from_count(choices, det_count)
        pred_choice = None
        if pred_option_idx is not None:
            m = re.search(r"\(([A-Z])\)", choices[pred_option_idx])
            pred_choice = m.group(1) if m else None

        correct = (pred_choice == gt)
        right += int(correct); wrong += int(not correct)

        outputs.append({
            "Image": item.get("Image"),
            "Text": text,
            "target_phrase": target_phrase,
            "model_used": model_used,
            "class_name": cls_name,
            "count": det_count,
            "Answer choices": choices,
            "pred_choice": pred_choice,
            "Ground truth": gt
        })

        print(f"[{os.path.basename(img_path)}] target=<{target_phrase}>  model={model_used}/{cls_name}  count={det_count}  pred={pred_choice}  gt={gt}  {'✓' if correct else '✗'}")

    # 6) 写结果
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)

    total = right + wrong
    acc = right / total if total else 0.0
    print(f"\n== DONE ==")
    print(f"Total: {total} | Right: {right} | Wrong: {wrong} | Acc: {acc:.4f}")
    print(f"Saved: {OUT_JSON}")

if __name__ == "__main__":
    main()

# python /home/mcislab_cj/VRSBench_images/valid/10_13_tunegood/final_code/overall_counting.py   --json_in /home/mcislab_cj/VRSBench_images/valid/subset_low/en/Counting__Overall_counting.json   --img_root /home/mcislab_cj/VRSBench_images/valid/images   --out_json /home/mcislab_cj/VRSBench_images/valid/10_13_tunegood/output_json/eval_en_1029_overall_counting.json  --coarse_model /home/mcislab_cj/VRSBench_images/valid/best_for_large_fair1m.pt   --fine_model /home/mcislab_cj/yolov11_copy/fair1m5_large_yoloxobb/train2/weights/best_for_small_fair1m.pt   --fm9g_path /home/mcislab_cj/fm9g4bv/FM9G4B-V 
