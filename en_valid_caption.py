import json
import os
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import jieba
import re

from script import *
from use_qwen_trans import *

if __name__ == "__main__":


    model_path = "/data/cyl/project/RSLLM/mllm"
    # output_file_name = args.output_file_name
    prompt_file = "/data/cyl/project/RSLLM/code/prompt/prompt2.txt"

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True
    )
    model.to(device)




    
    model_path_qwen = "/data/cyl/project/MLLM/models/Qwen2.5-0.5B-Instruct"
    model_qwen = AutoModelForCausalLM.from_pretrained(
        model_path_qwen,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer_qwen = AutoTokenizer.from_pretrained(model_path_qwen)
    model_qwen.to(device)

    with open("/data/cyl/project/RSLLM/code/output_for_caption/prompt_en-zh.txt", "r") as f:
        txt_data = f.readlines()
    en_txt = txt_data[0]
    zh_txt = txt_data[1]
    prompt_1 = "You are a professional translator specialized in remote sensing scene interpretation. Now, I need you to translate the English paragraph I provided into descriptive Chinese, following the exact narrative structure, sentence length, and stylistic tone of the example below."
    prompt_2 = "Your task: Please translate my next English paragraph in the same way as the example — Follow the paragraph-by-paragraph correspondence. Maintain geographic layout descriptions (e.g., “从左上到右下”、“中下部分”...) Preserve complex reasoning and inference at the end of the paragraph. Use professional, vivid, and analytical language that is common in remote sensing image analysis reports. Do not add any extra interpretation. Just translate accurately and stylistically aligned with the example above. Do not add any extra interpretation. Just translate accurately and stylistically aligned with the example above.\nInput English paragraph:"
    prompt_trans = f"{prompt_1}\nExample Input (English): {en_txt}\nExample Output (Chinese Translation): {zh_txt}\n{prompt_2}"






    json_data = read_json("/data/cyl/project/RSLLM/dataset/valid/zh_caption.json")
    image_path_root = "/data/cyl/project/RSLLM/dataset/valid/images"
    prompt = read_txt(prompt_file)

    print("evalute BLUE score...")
    hyp_list = []
    ref_list = []
    result_all = []
    en_zh_flag = 0
    for i, sample in tqdm(enumerate(json_data), total=len(json_data), desc="Validating"):
        if re.search(r'[\u4e00-\u9fff]', sample['Ground truth']) is not None:
            en_zh_flag = 1
        else:
            en_zh_flag = 0
        image = Image.open(os.path.join(image_path_root, sample["Image"])).convert("RGB")
        msgs = [{"role": "user", "content": [image, prompt]}]
        res = model.chat(
            image=None,
            msgs=msgs,
            tokenizer=tokenizer
        )
        if en_zh_flag:
            prompt_trans_input = f"{prompt_trans}\n{res}"
            output_result = use_qwen(model_qwen, tokenizer_qwen, prompt_trans_input)
            hyp_list.append(list(jieba.cut(output_result.strip())))
            ref_list.append([list(jieba.cut(sample['Ground truth'].strip()))])
        else:
            hyp_list.append(res.strip().split())
            ref_list.append([sample['Ground truth'].strip().split()])

    avg = []
    evalute_BLUE_SIMPLE(ref_list, hyp_list)