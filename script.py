from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from statistics import mean
import json

def read_json(json_path):
    with open(json_path, "r") as f:
        datas = json.load(f)
    return datas

def save_json(datas, json_path):
    with open(json_path, "w") as f:
        json.dump(datas, f, indent=4)

def read_txt(txt_path):
    with open(txt_path, "r") as f:
        datas = f.read()
    return datas

def write_txt(datas, txt_path):
    with open(txt_path, "w") as f:
        f.write(datas)


def evalute_BLUE(ref_list, hyp_list, output_path, avg):
    smooth = SmoothingFunction().method1
    # BLEU-1
    weights1 = (1, 0, 0, 0)
    bleu1 = corpus_bleu(ref_list, hyp_list, weights=weights1, smoothing_function=smooth)
    # BLEU-2
    weights2 = (0.5, 0.5, 0, 0)
    bleu2 = corpus_bleu(ref_list, hyp_list, weights=weights2, smoothing_function=smooth)
    # BLEU-4
    weights4 = (0.25, 0.25, 0.25, 0.25)
    bleu4 = corpus_bleu(ref_list, hyp_list, weights=weights4, smoothing_function=smooth)
    # 平均
    bleu_avg = (bleu1 + bleu2 + bleu4) / 3
    avg.append(bleu_avg)
    print("="*40)
    print(f"BLEU-1 Score: {bleu1:.4f}")
    print(f"BLEU-2 Score: {bleu2:.4f}")
    print(f"BLEU-4 Score: {bleu4:.4f}")
    print(f"BLEU 平均值: {bleu_avg:.4f}")

    write_data = f"BLEU-1 Score: {bleu1:.4f}" + "\n" + f"BLEU-2 Score: {bleu2:.4f}" + "\n" + f"BLEU-4 Score: {bleu4:.4f}" + "\n" + f"BLEU 平均值: {bleu_avg:.4f}\n" + f"总平均值: {mean(avg):.4f}\n\n"
    with open(output_path, "a") as f:
        f.write(write_data)


def evalute_BLUE_SIMPLE(ref_list, hyp_list):
    smooth = SmoothingFunction().method1
    # BLEU-1
    weights1 = (1, 0, 0, 0)
    bleu1 = corpus_bleu(ref_list, hyp_list, weights=weights1, smoothing_function=smooth)
    # BLEU-2
    weights2 = (0.5, 0.5, 0, 0)
    bleu2 = corpus_bleu(ref_list, hyp_list, weights=weights2, smoothing_function=smooth)
    # BLEU-4
    weights4 = (0.25, 0.25, 0.25, 0.25)
    bleu4 = corpus_bleu(ref_list, hyp_list, weights=weights4, smoothing_function=smooth)
    # 平均
    bleu_avg = (bleu1 + bleu2 + bleu4) / 3
    # avg.append(bleu_avg)
    print("="*40)
    print(f"BLEU-1 Score: {bleu1:.4f}")
    print(f"BLEU-2 Score: {bleu2:.4f}")
    print(f"BLEU-4 Score: {bleu4:.4f}")
    print(f"BLEU 平均值: {bleu_avg:.4f}")

    # write_data = f"BLEU-1 Score: {bleu1:.4f}" + "\n" + f"BLEU-2 Score: {bleu2:.4f}" + "\n" + f"BLEU-4 Score: {bleu4:.4f}" + "\n" + f"BLEU 平均值: {bleu_avg:.4f}\n" + f"总平均值: {mean(avg):.4f}\n\n"
    # with open(output_path, "a") as f:
    #     f.write(write_data)



#     sudo docker run --rm -it --gpus all   -v /data/cj/RS_StarSight/src:/app/src   -v /data/cj/valid_contest/valid_images:/data/cj/valid_contest/valid_images:ro -v  /home/cj/miniconda3/envs/fm9g4bv/lib/python3.10/site-packages/detectron2/data/transforms/transform.py:/opt/patches/transform.py:ro  -v /data/cj/RS_StarSight/tool:/data/cj/RS_StarSight/tool:ro   -v /tmp/rs_out:/app/src/datasets/valid/output   -v /data/cj/RS_StarSight/tool/FM9G4B-V:/data/cj/FM9G4B-V:ro   --entrypoint bash rs-eval:1.0 -lc '
#     set -e           
                                                 
#     # 统一 python 命令名
#     ln -sf /usr/bin/python3 /usr/local/bin/python
                                  
#     # 让脚本里写死的 /data/... 路径在容器里也能访问到你的本地 src
#     mkdir -p /data/cj/RS_StarSight
#     ln -sfn /app/src /data/cj/RS_StarSight/src
                                                                                    
#     # 关键：把本地源码 + SkySense-O 包根目录加入 Python 模块搜索路径
#     # /data/cj/RS_StarSight/tool/SkySense-O/skysense_o 为包目录，因此应把它的父目录 "SkySense-O" 加入 PYTHONPATH
#     export PYTHONPATH="/app/src:/data/cj/RS_StarSight/tool/SkySense-O:${PYTHONPATH}"
                                                                       
#     # 可选：快速自检（能看到导入是否成功）
#     python - <<PY                                             
# import sys                                                           
# print("[PYTHONPATH]", sys.path)
# import skysense_o
# print("[OK] skysense_o loaded from:", skysense_o.__file__)
# PY                                   
                                                               
#     # 修补脚本并运行
#     cp /app/src/datasets/valid/run_valid_eval.sh /tmp/run_valid_eval.sh
#     sed -E -i '"'"'s#rm -rf "\$SUB_JSON_DIR" "\$RED_CROP_DIR" "\$SKY_OUT_DIR" "\$MOTION_CROP_DIR" "\$OUT_DIR"#rm -rf "\$SUB_JSON_DIR" "\$RED_CROP_DIR" "\$SKY_OUT_DIR" "\$MOTION_CROP_DIR"; mkdir -p "\$OUT_DIR"; find "\$OUT_DIR" -mindepth 1 -maxdepth 1 -exec rm -rf {} +#'"'"' /tmp/run_valid_eval.sh
#     chmod +x /tmp/run_valid_eval.sh
#     /tmp/run_valid_eval.sh
#   '

# sudo docker run --rm -it --gpus all \
#   -v /data/cj/RS_StarSight/src:/app/src \
#   -v /data/cj/valid_contest/valid_images:/data/cj/valid_contest/valid_images:ro \
#   -v /data/cj/RS_StarSight/tool:/data/cj/RS_StarSight/tool:ro \
#   -v /home/cj/miniconda3/envs/fm9g4bv/lib/python3.10/site-packages/detectron2/data/transforms/transform.py:/usr/local/lib/python3.10/dist-packages/detectron2/data/transforms/transform.py:ro \
#   -v /tmp/rs_out:/app/src/datasets/valid/output \
#   -v /data/cj/RS_StarSight/tool/FM9G4B-V:/data/cj/FM9G4B-V:ro \
#   --entrypoint bash rs-eval:1.0 -lc '
#     set -e

#     # 统一 python 命令名
#     ln -sf /usr/bin/python3 /usr/local/bin/python

#     # 让脚本里写死的 /data/... 路径在容器里也能访问到你的本地 src
#     mkdir -p /data/cj/RS_StarSight
#     ln -sfn /app/src /data/cj/RS_StarSight/src

#     # 关键：把本地源码 + SkySense-O 包根目录加入 Python 模块搜索路径
#     # /data/cj/RS_StarSight/tool/SkySense-O/skysense_o 为包目录，因此应把它的父目录 "SkySense-O" 加入 PYTHONPATH
#     export PYTHONPATH="/app/src:/data/cj/RS_StarSight/tool/SkySense-O:${PYTHONPATH}"

#     # 可选：快速自检（能看到导入是否成功）
#     python - <<PY
# import sys
# print("[PYTHONPATH]", sys.path)
# import skysense_o
# print("[OK] skysense_o loaded from:", skysense_o.__file__)
# PY

#     # 修补脚本并运行
#     cp /app/src/datasets/valid/run_valid_eval.sh /tmp/run_valid_eval.sh
#     sed -E -i '"'"'s#rm -rf "\$SUB_JSON_DIR" "\$RED_CROP_DIR" "\$SKY_OUT_DIR" "\$MOTION_CROP_DIR" "\$OUT_DIR"#rm -rf "\$SUB_JSON_DIR" "\$RED_CROP_DIR" "\$SKY_OUT_DIR" "\$MOTION_CROP_DIR"; mkdir -p "\$OUT_DIR"; find "\$OUT_DIR" -mindepth 1 -maxdepth 1 -exec rm -rf {} +#'"'"' /tmp/run_valid_eval.sh
#     chmod +x /tmp/run_valid_eval.sh
#     /tmp/run_valid_eval.sh
#   '

#   '


# sudo docker run --rm -it --gpus all   -v /data/cj/RS_StarSight/src:/app/src   -v /data/cj/valid_contest/valid_images:/data/cj/valid_contest/valid_images   -v /data/cj/RS_StarSight/tool:/data/cj/RS_StarSight/tool   -v /home/cj/miniconda3/envs/fm9g4bv/lib/python3.10/site-packages/detectron2/data/transforms/transform.py:/usr/local/lib/python3.10/dist-packages/detectron2/data/transforms/transform.py   -v /tmp/rs_out:/app/src/datasets/valid/output   -v /data/cj/RS_StarSight/tool/FM9G4B-V:/data/cj/FM9G4B-V   --entrypoint bash rs-eval:1.0 -lc '
#     set -e

#     ln -sf /usr/bin/python3 /usr/local/bin/python
#     mkdir -p /data/cj/RS_StarSight
#     ln -sfn /app/src /data/cj/RS_StarSight/src

#     # 关键：声明 SkySense 根目录，并放到 PYTHONPATH
#     export SKY_ROOT="/data/cj/RS_StarSight/tool/SkySense-O"
#     export PYTHONPATH="/app/src:${SKY_ROOT}:${PYTHONPATH}"
#     ln -sfn /data/cj/RS_StarSight/tool/SkySense-O /data/cj/SkySense-O
#     # 自检（在 SkySense 根目录下做 import，确保相对路径能命中）
#     cd "$SKY_ROOT"
#     python - <<PY
# import os, sys
# print("[CWD]", os.getcwd())
# print("[PYTHONPATH]", sys.path)
# from skysense_o.data.datasets import register_isaid
# print("[OK] import succeeded")
# PY

#     # 修补脚本并运行（仍然在 SKY_ROOT 下执行，保证相对路径文件可见）
#     cp /app/src/datasets/valid/run_valid_eval.sh /tmp/run_valid_eval.sh
#     sed -E -i '"'"'s#rm -rf "\$SUB_JSON_DIR" "\$RED_CROP_DIR" "\$SKY_OUT_DIR" "\$MOTION_CROP_DIR" "\$OUT_DIR"#rm -rf "\$SUB_JSON_DIR" "\$RED_CROP_DIR" "\$SKY_OUT_DIR" "\$MOTION_CROP_DIR"; mkdir -p "\$OUT_DIR"; find "\$OUT_DIR" -mindepth 1 -maxdepth 1 -exec rm -rf {} +#'"'"' /tmp/run_valid_eval.sh
#     chmod +x /tmp/run_valid_eval.sh

#     # 在 SKY_ROOT 目录下跑整套流程（run_valid_eval.sh 里用的绝对路径依旧有效）
#     bash /tmp/run_valid_eval.sh
#   '