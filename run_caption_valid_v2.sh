#!/usr/bin/env bash
# run_caption_valid.sh — portable runner for host & docker (REQUIRES IMAGE_PATH_ROOT)

set -euo pipefail

# -------- 1) 自动识别运行环境 --------
if [[ -d "/app/src" ]]; then
  SRC_ROOT="/app/src"                        # inside container
else
  SRC_ROOT="/data/cj/RS_StarSight/src"       # on host
fi

TOOL_ROOT="${TOOL_ROOT:-/data/cj/RS_StarSight/tool}"

# -------- 2) 必填参数：IMAGE_PATH_ROOT --------
: "${IMAGE_PATH_ROOT:?[ERROR] IMAGE_PATH_ROOT is required. Please export IMAGE_PATH_ROOT=/abs/path/to/images}"

# 可覆盖参数（可选）
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT="${SCRIPT:-${SRC_ROOT}/datasets/valid/caption/valid_caption.py}"

MODEL_PATH="${MODEL_PATH:-${TOOL_ROOT}/mllm}"
MODEL_PATH_QWEN="${MODEL_PATH_QWEN:-${TOOL_ROOT}/Qwen2.5-0.5B-Instruct}"
PROMPT_FILE="${PROMPT_FILE:-${TOOL_ROOT}/prompt2.txt}"
TRANS_PROMPT="${TRANS_PROMPT:-${TOOL_ROOT}/prompt_en-zh.txt}"

# 允许外部指定 JSON 文件；没有就给个英文默认
JSON_FILE="${JSON_FILE:-/data/cj/valid_contest/en/Image_caption__Overall_image_caption_with_details.json}"

# 输出目录（允许覆盖）
OUTPUT_DIR_DEFAULT="${SRC_ROOT}/datasets/valid/output/caption"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_DIR_DEFAULT}"

# -------- 3) 保障环境 --------
mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="${SRC_ROOT}:${PYTHONPATH:-}"

# -------- 4) 自检 --------
echo "[INFO] SRC_ROOT=$SRC_ROOT"
echo "[INFO] TOOL_ROOT=$TOOL_ROOT"
echo "[INFO] OUTPUT_DIR=$OUTPUT_DIR"
echo "[INFO] IMAGE_PATH_ROOT=$IMAGE_PATH_ROOT"
echo "[INFO] JSON_FILE=$JSON_FILE"

# -------- 5) 执行 --------
exec "$PYTHON_BIN" "$SCRIPT" \
  --model_path        "$MODEL_PATH" \
  --prompt_file       "$PROMPT_FILE" \
  --model_path_qwen   "$MODEL_PATH_QWEN" \
  --trans_prompt      "$TRANS_PROMPT" \
  --json_file         "$JSON_FILE" \
  --image_path_root   "$IMAGE_PATH_ROOT"
