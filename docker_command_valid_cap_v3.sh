#!/usr/bin/env bash
set -euo pipefail

# ========= 可配置变量（宿主机路径） =========
export HOST_SRC=/data/cj/RS_StarSight/src             # 代码
export HOST_TOOL=/data/cj/RS_StarSight/tool           # 工具/权重/提示词
export HOST_FM9G=$HOST_TOOL/FM9G4B-V                  # base 模型目录
export HOST_OUT=/data/cj/RS_StarSight/src/datasets/valid/caption/out  # 输出目录（确保可写）

# ★ 宿主机的图片目录
export HOST_IMGS=/data/cj/valid_contest/valid_images

# ========= 任务输入参数 =========
export JSON_FILE=/data/cj/valid_contest/zh/Image_caption__Overall_image_caption_with_details.json
export OUTPUT_SUBDIR=caption_qwen05_zh

# ========= 镜像 =========
export IMAGE_TAG=rs-eval:1.0

# ========= 启动容器 =========
sudo docker run --rm -it --gpus all \
  -e JSON_FILE="/data/cj/valid_contest/zh/valid.json" \
  -e OUTPUT_DIR="/app/src/datasets/valid/output/$OUTPUT_SUBDIR" \
  -e IMAGE_PATH_ROOT="/mnt/images" \
  -v "$JSON_FILE":/data/cj/valid_contest/zh/valid.json \
  -v "$HOST_SRC":/app/src \
  -v "$HOST_TOOL":/data/cj/RS_StarSight/tool \
  -v "$HOST_FM9G":/data/cj/FM9G4B-V \
  -v "$HOST_FM9G":/home/mcislab_cj/fm9g4bv/FM9G4B-V\
  -v "$HOST_IMGS":/mnt/images \
  -v "$HOST_OUT":/app/src/datasets/valid/output \
  --entrypoint bash "$IMAGE_TAG" -lc '
    set -e
    ln -sf /usr/bin/python3 /usr/local/bin/python
    bash /app/src/datasets/valid/caption/run_caption_valid_v2.sh
  '
