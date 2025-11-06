import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
import torch
import json
from peft import PeftModel
from utils import load_json_datadict, preprocess_function

# ===== 配置 =====
MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"
LORA_PATH = "results/checkpoint-233" #需要放入 base 模型的最优 lora
ORIGINAL_TRAIN_PATH = "dataset/train_litex_merged.json"
OUTPUT_CANDIDATES_PATH = "dataset/generated_candidates.json"

NUM_SAMPLES_PER_PROMPT = 4
MAX_NEW_TOKENS = 512

# ===== 加载原始训练数据 =====
original_data = load_json_datadict(ORIGINAL_TRAIN_PATH)
original_data = original_data.map(preprocess_function)
original_data = original_data["train"]

# ===== 加载模型和 tokenizer =====
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

# ===== 生成候选响应（不判断正确性）=====
all_candidates = []

for example in tqdm.tqdm(original_data, desc="Generating candidates"):
    user_input = example["user_input"]
    ground_truth = example["full_litex"]
    title = example["title"]
    description = example["description"]

    messages = [{"role": "user", "content": user_input}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

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

    # 保存原始信息 + 所有生成结果
    all_candidates.append({
        "title": title,
        "description": description,
        "prompt": user_input,
        "chosen": ground_truth,
        "candidates": generated_responses  # list of generated responses
    })

# ===== 保存生成结果 =====
print(f"共生成 {len(all_candidates)} 条样本的候选响应")
with open(OUTPUT_CANDIDATES_PATH, "w", encoding="utf-8") as f:
    json.dump(all_candidates, f, ensure_ascii=False, indent=2)

print(f"候选响应已保存至: {OUTPUT_CANDIDATES_PATH}")