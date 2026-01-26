sudo docker run --rm -it --gpus all \
  -e JSON_FILE=/data/cj/valid_contest/zh/Image_caption__Overall_image_caption_with_details_zh.json \
  -e OUTPUT_DIR=/app/src/datasets/valid/output/caption_qwen05_zh \
  -v /data/cj/RS_StarSight/src:/app/src \
  -v /data/cj/valid_contest:/data/cj/valid_contest \
  -v /data/cj/RS_StarSight/tool:/data/cj/RS_StarSight/tool \
  -v /data/cj/RS_StarSight/tool/FM9G4B-V:/data/cj/FM9G4B-V \
  -v /tmp/rs_out:/app/src/datasets/valid/output \
  --entrypoint bash rs-eval:1.0 -lc '
    set -e; ln -sf /usr/bin/python3 /usr/local/bin/python
    bash /app/src/datasets/valid/caption/run_caption_valid.sh
  '
