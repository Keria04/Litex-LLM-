import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from peft import PeftModel
from utils import load_json_datadict, preprocess_function
from tqdm import tqdm

# ===== 配置 =====
MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"
LORA_PATH = "results/checkpoint-233"
ORIGINAL_TRAIN_PATH = "dataset/train_litex_merged.json"
OUTPUT_CANDIDATES_PATH = "dataset/generated_candidates.json"

NUM_SAMPLES_PER_PROMPT = 1
MAX_NEW_TOKENS = 256
BATCH_SIZE = 4

original_data = load_json_datadict(ORIGINAL_TRAIN_PATH)
original_data = original_data.map(preprocess_function)
original_data = original_data["train"]

prompts = []
metadata = []  # 保存 title, description, ground_truth 等
print("加载 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
for example in original_data:
    user_input = example["user_input"]
    messages = [{"role": "user", "content": user_input}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompts.append(prompt_text)
    metadata.append({
        "title": example["title"],
        "description": example["description"],
        "prompt": user_input,
        "chosen": example["full_litex"]
    })

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
print("加载 LoRA 适配器...")
model = PeftModel.from_pretrained(model, LORA_PATH)
print("合并 LoRA 权重...")
model = model.merge_and_unload()
model.eval()
print("模型加载完成！")

# ===== 批量生成函数 =====
def generate_batch(prompts_batch):
    inputs = tokenizer(
        prompts_batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
        add_special_tokens=False,
        return_length=True  # 返回每个样本的实际 token 数
    ).to(model.device)

    actual_input_lengths = inputs.pop("length")  # shape: [batch_size]

    all_responses = [[] for _ in range(len(prompts_batch))]

    for _ in range(NUM_SAMPLES_PER_PROMPT):
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        for i, output in enumerate(outputs):
            gen_tokens = output[actual_input_lengths[i]:]
            gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            all_responses[i].append(gen_text)

    return all_responses

# ===== 批处理生成 =====
all_candidates = []

for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc="Generating in batches"):
    batch_prompts = prompts[i:i + BATCH_SIZE]
    batch_meta = metadata[i:i + BATCH_SIZE]

    generated_batch = generate_batch(batch_prompts)

    for meta, responses in zip(batch_meta, generated_batch):
        unique_responses = list(dict.fromkeys(responses))
        all_candidates.append({
            "title": meta["title"],
            "description": meta["description"],
            "prompt": meta["prompt"],
            "chosen": meta["chosen"],
            "candidates": unique_responses
        })

# ===== 保存结果 =====
print(f"共生成 {len(all_candidates)} 条样本的候选响应")
with open(OUTPUT_CANDIDATES_PATH, "w", encoding="utf-8") as f:
    json.dump(all_candidates, f, ensure_ascii=False, indent=2)

print(f"候选响应已保存至: {OUTPUT_CANDIDATES_PATH}")