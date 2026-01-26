import json
import os
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import jieba

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

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Current split index')
    parser.add_argument('--output_file_name', type=str, required=True, help='Current split index')
    parser.add_argument('--image_path_root', type=str, required=True, help='Current split index')
    parser.add_argument('--json_data_path', type=str, required=True, help='Current split index')
    parser.add_argument('--prompt_file', type=str, required=True, help='Current split index')
    args = parser.parse_args()
    model_path = args.model_path
    output_file_name = args.output_file_name
    prompt_file = args.prompt_file

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True
    )
    model.to(device)

    json_data = read_json(args.json_data_path)
    image_path_root = args.image_path_root
    prompt = read_txt(prompt_file)

    print("evalute BLUE score...")
    hyp_list = []
    ref_list = []
    results = []
    result_all = []
    for i, sample in tqdm(enumerate(json_data), total=len(json_data), desc="Validating"):
        image = Image.open(os.path.join(image_path_root, sample["Image"])).convert("RGB")
        msgs = [{"role": "user", "content": [image, prompt]}]
        res = model.chat(
            image=None,
            msgs=msgs,
            tokenizer=tokenizer
        )
        results.append([os.path.join(image_path_root, sample["Image"]), res, sample['Ground truth']])
        hyp_list.append(res.strip().split())
        ref_list.append([sample['Ground truth'].strip().split()])
        # break
    save_json(results, output_file_name)
    output_path = output_file_name
    avg = []
    evalute_BLUE(ref_list, hyp_list, output_path, avg)
    # for i, sample in tqdm(enumerate(json_data), total=len(json_data), desc="Validating"):
    #     image = Image.open(os.path.join(image_path_root, sample["Image"])).convert("RGB")
    #     msgs = [{"role": "user", "content": [image, prompt]}]
    #     res = model.chat(
    #         image=None,
    #         msgs=msgs,
    #         tokenizer=tokenizer
    #     )
    #     results.append([res, sample['Ground truth']])

    #     # 中文分词处理
    #     hyp_list.append(list(jieba.cut(res.strip())))
    #     ref_list.append([list(jieba.cut(sample['Ground truth'].strip()))])

    # save_json(results, "/data/cyl/project/RSLLM/dataset/valid/result/caption_prompt2.json")
    # output_path = output_file_name
    # evalute_BLUE(ref_list, hyp_list, output_path)