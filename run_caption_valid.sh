#!/usr/bin/env bash
# run_caption_valid.sh — portable runner for host & docker

set -euo pipefail

# -------- 1) 自动识别运行环境：容器里优先用 /app/src，宿主机用本地路径 --------
if [[ -d "/app/src" ]]; then
  # inside container (we mounted /data/.../src -> /app/src)
  SRC_ROOT="/app/src"
else
  # on host
  SRC_ROOT="/data/cj/RS_StarSight/src"
fi

TOOL_ROOT="${TOOL_ROOT:-/data/cj/RS_StarSight/tool}"
DATA_ROOT="${DATA_ROOT:-/data/cj/valid_contest}"

# -------- 2) 可覆盖的参数（支持环境变量覆盖；无则用默认） --------
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT="${SCRIPT:-${SRC_ROOT}/datasets/valid/caption/valid_caption.py}"

MODEL_PATH="${MODEL_PATH:-${TOOL_ROOT}/mllm}"
MODEL_PATH_QWEN="${MODEL_PATH_QWEN:-${TOOL_ROOT}/Qwen2.5-0.5B-Instruct}"
PROMPT_FILE="${PROMPT_FILE:-${TOOL_ROOT}/prompt2.txt}"
TRANS_PROMPT="${TRANS_PROMPT:-${TOOL_ROOT}/prompt_en-zh.txt}"

JSON_FILE="${JSON_FILE:-${DATA_ROOT}/en/Image_caption__Overall_image_caption_with_details.json}"
IMAGE_PATH_ROOT="${IMAGE_PATH_ROOT:-${DATA_ROOT}/valid_images}"

# 输出目录：容器里我们把它映射到了主机 /tmp/rs_out（见 docker run）
OUTPUT_DIR_DEFAULT="${SRC_ROOT}/datasets/valid/output/caption"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_DIR_DEFAULT}"

# -------- 3) 保障环境 --------
mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="${SRC_ROOT}:${PYTHONPATH:-}"

# -------- 4) 自检（可按需注释掉） --------
echo "[INFO] SRC_ROOT=$SRC_ROOT"
echo "[INFO] TOOL_ROOT=$TOOL_ROOT"
echo "[INFO] DATA_ROOT=$DATA_ROOT"
echo "[INFO] OUTPUT_DIR=$OUTPUT_DIR"
echo "[INFO] Running: $PYTHON_BIN $SCRIPT"

# -------- 5) 执行 --------
exec "$PYTHON_BIN" "$SCRIPT" \
  --model_path        "$MODEL_PATH" \
  --prompt_file       "$PROMPT_FILE" \
  --model_path_qwen   "$MODEL_PATH_QWEN" \
  --trans_prompt      "$TRANS_PROMPT" \
  --json_file         "$JSON_FILE" \
  --image_path_root   "$IMAGE_PATH_ROOT" \

