from transformers import AutoModelForCausalLM, AutoTokenizer

def use_qwen(model, tokenizer, prompt):

# model_name = "/data/cyl/project/MLLM/models/Qwen2.5-0.5B-Instruct"

    # model = AutoModelForCausalLM.from_pretrained(
    #     model_path,
    #     torch_dtype="auto",
    #     device_map="auto",
    #     trust_remote_code=True
    # )
    # tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

if __name__ == "__main__":
    model_path = "/data/cyl/project/MLLM/models/Qwen2.5-0.5B-Instruct"
    res = use_qwen(model_path, "hello")
    print(res)