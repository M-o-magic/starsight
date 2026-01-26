import os
import json
import cv2
import numpy as np
import tifffile as tiff
from tqdm import tqdm
from PIL import Image

from PIL import Image
import tifffile as tiff
import numpy as np

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

def load_image(image_path):
    ext = os.path.splitext(image_path)[1].lower()
    if ext in [".tif", ".tiff"]:
        img = tiff.imread(image_path)
    else:
        # img = np.array(Image.open(image_path).convert("RGB"))
        img = np.array(open_image_rgb_safe(image_path))
    return img


# ================= 原有函数（保持不变） =================
def extract_red_region(image_path, save_path):
    try:
        img=load_image(image_path)
        # img = tiff.imread(image_path)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[2] > 3:
            img = img[:, :, :3]

        # 归一化到uint8
        if img.dtype != np.uint8:
            img = img.astype(np.float32)
            denom = (img.max() - img.min())
            if denom < 1e-8:
                # 全常量图像：直接转为零图，避免NaN
                img = np.zeros_like(img, dtype=np.uint8)
            else:
                img = (img - img.min()) / (denom + 1e-5) * 255
                img = img.astype(np.uint8)
    except Exception as e:
        print(f"❌ 无法读取图像: {image_path}，错误信息：{e}")
        return False

    img_rgb = img
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # 更鲁棒的红色提取（HSV 空间）
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    lower1 = np.array([0, 70, 50])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([170, 70, 50])
    upper2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(img_hsv, lower1, upper1)
    mask2 = cv2.inRange(img_hsv, lower2, upper2)
    mask_red = cv2.bitwise_or(mask1, mask2)

    # 膨胀并提取最大连通区域
    kernel = np.ones((3, 3), np.uint8)
    mask_dilated = cv2.dilate(mask_red, kernel, iterations=2)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_dilated, connectivity=8)

    if num_labels <= 1:
        print(f"⚠️ 无红色区域: {image_path}")
        return False

    max_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    x, y, w, h = stats[max_label, cv2.CC_STAT_LEFT:cv2.CC_STAT_LEFT+4]
    cropped = img_bgr[y:y+h, x:x+w]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, cropped)
    return True


# ================= 新的命令行入口（仅新增这部分） =================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="从JSON收集图像，提取红色区域并裁剪保存")
    parser.add_argument("--json", required=True, help="输入的 JSON 文件路径")
    parser.add_argument("--image_root", required=True,
                        help="图像根目录（JSON里的相对路径会拼在这里）")
    parser.add_argument("--output_root", required=True, help="输出裁剪图像的根目录")
    args = parser.parse_args()

    json_file = args.json
    image_root = args.image_root
    output_root = args.output_root
    os.makedirs(output_root, exist_ok=True)

    # 读 JSON
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 收集所有可能的图像字段（兼容 HRSCD/FAIR1M/Land_use）
    all_paths = set()
    for item in data:
        # 兼容三种键名：Image / Image1 / Image2
        if "Image" in item and item["Image"]:
            all_paths.add(item["Image"])
        if "Image1" in item and item["Image1"]:
            all_paths.add(item["Image1"])
        if "Image2" in item and item["Image2"]:
            all_paths.add(item["Image2"])

    # JSON所在目录，作为兜底相对路径根（如果在 image_root 下找不到）
    json_dir = os.path.dirname(os.path.abspath(json_file))

    success = 0
    for rel_path in tqdm(sorted(all_paths)):
        # 1) 优先用 image_root 解析
        cand_in_root = os.path.join(image_root, rel_path)
        # 2) 若 rel_path 本身是绝对路径，直接用
        cand_abs = rel_path if os.path.isabs(rel_path) else None
        # 3) 再兜底：相对 JSON 文件所在目录
        cand_in_json_dir = os.path.join(json_dir, rel_path)

        if cand_abs and os.path.exists(cand_abs):
            input_path = cand_abs
        elif os.path.exists(cand_in_root):
            input_path = cand_in_root
        elif os.path.exists(cand_in_json_dir):
            input_path = cand_in_json_dir
        else:
            print(f"❌ 找不到图像文件：{rel_path}")
            continue

        output_path = os.path.join(output_root, rel_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if extract_red_region(input_path, output_path):
            success += 1

    print(f"\n✅ 所有图像处理完成：共处理 {len(all_paths)} 张，成功提取红色区域 {success} 张")
    print("📁 输出目录：", output_root)
