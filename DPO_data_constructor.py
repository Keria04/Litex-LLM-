import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
import torch
import json
from peft import LoraConfig, PeftModel
from utils import *

# ===== 配置 =====
MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"
LORA_PATH = "results/checkpoint-233"
ORIGINAL_TRAIN_PATH = "dataset/train_litex_merged.json"
OUTPUT_DPO_PATH = "dataset/dpo_train_from_math23k_and_gsm8k.json"

NUM_SAMPLES_PER_PROMPT = 4  # 每个 prompt 生成多少个响应
MAX_NEW_TOKENS = 512

# ===== 加载原始训练数据 =====
original_data = load_json_datadict(ORIGINAL_TRAIN_PATH)

# ===== 加载 SFT 微调后的模型和 tokenizer =====
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
print("正在加载LoRA适配器...")
model = PeftModel.from_pretrained(model, LORA_PATH)
print("正在合并LoRA权重...")
model = model.merge_and_unload()
model.eval()
print("模型加载完成！")
# ===== 构造 DPO 数据 =====
dpo_pairs = []

original_data = original_data.map(preprocess_function)
original_data = original_data["train"]
for example in tqdm.tqdm(original_data, desc="Generating DPO pairs"):
    user_input = example["user_input"]          # 这是 prompt
    ground_truth = example["full_litex"]        # 这是 chosen（标准答案）
    title = example["title"]
    description = example["description"]
    messages = [{"role": "user", "content": user_input}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    # 生成多个候选响应
    generated_responses = []
    for _ in range(NUM_SAMPLES_PER_PROMPT):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        generated_responses.append(generated_text)
    generated_responses = list(dict.fromkeys(generated_responses))
    # 用 judge 函数筛选 rejected（错误的）
    for resp in generated_responses:
        row = {
            "title": title,
            "description": description,
            "solution": resp,
        }
        if not judge_litex_correctness(row)["correctness"]:  # 如果 judge 返回 False
            dpo_pairs.append({
                "prompt": user_input,
                "chosen": ground_truth,
                "rejected": resp
            })

# ===== 保存 DPO 数据集 =====
print(f"共构造 {len(dpo_pairs)} 个 DPO 偏好对")
with open(OUTPUT_DPO_PATH, "w", encoding="utf-8") as f:
    json.dump(dpo_pairs, f, ensure_ascii=False, indent=2)

print(f"DPO 数据集已保存至: {OUTPUT_DPO_PATH}")
if __name__ == "__main__":
    pass