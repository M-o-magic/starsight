import json
import os
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import jieba

from script import *
from use_qwen_trans import *

if __name__ == "__main__":

    model_path = "/data/cyl/project/RSLLM/mllm"
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

    cap_flag = 0
    prompt_zh = read_txt("/data/cyl/project/RSLLM/code/output_for_caption/prompt_valid_zh.txt")
    prompt_check_cap = f"""{prompt_zh}


这个输入是否是需要大模型进行描述图片的任务，如果是的话，请直接输出是，如果不是，请直接输出否。"""
    msg_check_cap = [{"role": "user", "content": [prompt_check_cap]}]
    res_check_cap_ = model.chat(
        image=None,
        msgs=msg_check_cap,
        tokenizer=tokenizer
    )
    print(res_check_cap_)
    res_check_cap = res_check_cap_.split("，")[0]
    if res_check_cap == '是' or res_check_cap == '是的' or prompt_zh == '请详细描述这幅图像。':
        cap_flag = 1
    
    image_path_root = "/data/cyl/project/RSLLM/dataset/valid/images"
    if cap_flag == 1:
        with open("/data/cyl/project/RSLLM/code/prompt/prompt_en-zh.txt", "r") as f:
            txt_data = f.readlines()
        en_txt = txt_data[0]
        zh_txt = txt_data[1]
        prompt_1 = "You are a professional translator specialized in remote sensing scene interpretation. Now, I need you to translate the English paragraph I provided into descriptive Chinese, following the exact narrative structure, sentence length, and stylistic tone of the example below."
        prompt_2 = "Your task: Please translate my next English paragraph in the same way as the example — Follow the paragraph-by-paragraph correspondence. Maintain geographic layout descriptions (e.g., “从左上到右下”、“中下部分”...) Preserve complex reasoning and inference at the end of the paragraph. Use professional, vivid, and analytical language that is common in remote sensing image analysis reports. Do not add any extra interpretation. Just translate accurately and stylistically aligned with the example above. Do not add any extra interpretation. Just translate accurately and stylistically aligned with the example above.\nInput English paragraph:"
        prompt_trans = f"{prompt_1}\nExample Input (English): {en_txt}\nExample Output (Chinese Translation): {zh_txt}\n{prompt_2}"
        prompt = read_txt("/data/cyl/project/RSLLM/code/output_for_caption/prompt2.txt")
        json_data = read_json("/data/cyl/project/RSLLM/dataset/valid/en-zh_test.json")

        hyp_list = []
        ref_list = []
        for i, sample in tqdm(enumerate(json_data), total=len(json_data), desc="Validating"):
            image = Image.open(os.path.join(image_path_root, sample["Image"])).convert("RGB")
            msgs = [{"role": "user", "content": [image, prompt]}]
            res = model.chat(
                image=None,
                msgs=msgs,
                tokenizer=tokenizer
            )
            prompt_trans_input = f"{prompt_trans}\n{res}"
            output_result = use_qwen(model_qwen, tokenizer_qwen, prompt_trans_input)
            hyp_list.append(list(jieba.cut(output_result.strip())))
            ref_list.append([list(jieba.cut(sample['Ground truth'].strip()))])
        evalute_BLUE_SIMPLE(ref_list, hyp_list)
        