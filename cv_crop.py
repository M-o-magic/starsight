import os
import json
import cv2
import numpy as np
import tifffile as tiff
from tqdm import tqdm
from PIL import Image

# ========== 手动设置部分 ==========
flag = 2  # 0 表示 HRSCD，1 表示 FAIR1M，2 表示 Land_use

if flag == 0:
    json_file = "/home/mcislab_cj/VRSBench_images/valid/subset_high/en/Counting__Counting_with_changing_detection.json"
    image_root = "/home/mcislab_cj/VRSBench_images/valid/images"
    output_root = "/home/mcislab_cj/VRSBench_images/valid/cropped_red_regions/Counting__Counting_with_changing_detection"
elif flag == 1:
    json_file = "/home/mcislab_cj/VRSBench_images/valid/subset_low/en/Counting__Regional_counting.json"
    image_root = "/home/mcislab_cj/VRSBench_images/valid/images"
    output_root = "/home/mcislab_cj/VRSBench_images/valid/cropped_red_regions/Counting__Regional_counting"
else:
    json_file = "/home/mcislab_cj/VRSBench_images/valid/subset_low/en/Land_use_classification__Regional_Land_use_classification.json"
    image_root = "/home/mcislab_cj/VRSBench_images/valid/images"
    output_root = "/home/mcislab_cj/VRSBench_images/valid/cropped_red_regions/Land_use_classification__Regional_Land_use_classification"
os.makedirs(output_root, exist_ok=True)


def extract_red_region(image_path, save_path):
    try:
        img = tiff.imread(image_path)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[2] > 3:
            img = img[:, :, :3]

        # 归一化到uint8
        if img.dtype != np.uint8:
            img = img.astype(np.float32)
            img = (img - img.min()) / (img.max() - img.min() + 1e-5) * 255
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


# ========== 主处理逻辑 ==========
with open(json_file, 'r') as f:
    data = json.load(f)

# 获取所有图像路径
all_paths = set()
for item in data:
    if flag == 0:
        all_paths.add(item["Image1"])
        all_paths.add(item["Image2"])
    else:
        all_paths.add(item["Image"])

success = 0
for rel_path in tqdm(sorted(all_paths)):
    input_path = os.path.join(image_root, rel_path)
    output_path = os.path.join(output_root, rel_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if extract_red_region(input_path, output_path):
        success += 1

print(f"\n✅ 所有图像处理完成：共处理 {len(all_paths)} 张，成功提取红色区域 {success} 张")
print("📁 输出目录：", output_root)
