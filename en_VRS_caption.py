import json
import os
import sys
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from script import *

if __name__ == "__main__":
    # 从命令行读取 model_path
    model_path = "/home/cyl/output_llm_version/output_RSGPT_RSICD_type_num/checkpoint-899"
    prompt_file = "/data/cyl/project/RSLLM/code/prompt/prompt1_VRSBench.txt"

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model.to(device)
    print("evalute BLEU score...")
    avg = []
    json_data = read_json(f"/data/cyl/project/RSLLM/dataset/VRSBench/VRSBench_EVAL_Cap.json")
    image_path_root = "/home/mcislab_cj/VRSBench_images/Images_val"
    prompt = read_txt(prompt_file)

    hyp_list = []
    ref_list = []
    for i, sample in tqdm(enumerate(json_data), total=len(json_data), desc="Validating"):
        image = Image.open(os.path.join(image_path_root, sample["image_id"])).convert("RGB")
        msgs = [{"role": "user", "content": [image, prompt]}]
        res = model.chat(
            image=None,
            msgs=msgs,
            tokenizer=tokenizer
        )
        idx = res.find("image")
        text = res[:idx + len("image")] + " from GoogleEarth" + res[idx + len("image"):]
        hyp_list.append(text.strip().split())
        ref_list.append([sample['ground_truth'].strip().split()])
    evalute_BLUE_SIMPLE(ref_list, hyp_list)