#!/usr/bin/env bash
set -euo pipefail

# ========= 可配置变量 =========
export HOST_SRC=/data/cj/RS_StarSight/src             # 代码目录
export HOST_VALID=/data/cj/valid_contest              # 评测数据根目录
export HOST_TOOL=/data/cj/RS_StarSight/tool           # 工具目录
export HOST_FM9G=$HOST_TOOL/FM9G4B-V                  # 模型目录
export HOST_OUT=/path_to/rs_out                           # 输出目录
export IMAGE_TAG=rs-eval:1.0                          # 镜像名

# 评测输入与输出子目录
export JSON_FILE=$HOST_VALID/zh/Image_caption__Overall_image_caption_with_details_zh.json
export OUTPUT_SUBDIR=caption_qwen05_zh

# ========= 启动容器 =========
sudo docker run --rm -it --gpus all \
  -e JSON_FILE=/data/cj/valid_contest/zh/Image_caption__Overall_image_caption_with_details_zh.json \
  -e OUTPUT_DIR=/app/src/datasets/valid/output/$OUTPUT_SUBDIR \
  -v "$HOST_SRC":/app/src \
  -v "$HOST_VALID":/data/cj/valid_contest \
  -v "$HOST_TOOL":/data/cj/RS_StarSight/tool \
  -v "$HOST_FM9G":/data/cj/FM9G4B-V \
  -v "$HOST_OUT":/app/src/datasets/valid/output \
  --entrypoint bash "$IMAGE_TAG" -lc '
    set -e
    ln -sf /usr/bin/python3 /usr/local/bin/python
    bash /app/src/datasets/valid/caption/run_caption_valid.sh
  '
