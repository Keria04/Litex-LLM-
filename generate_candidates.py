import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
import torch
import json
from peft import PeftModel
from utils import load_json_datadict, preprocess_function
from torch.utils.data import DataLoader
# ===== 配置 =====
MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"
LORA_PATH = "results/checkpoint-233"
ORIGINAL_TRAIN_PATH = "dataset/train_litex_merged.json"
OUTPUT_CANDIDATES_PATH = "dataset/generated_candidates.json"  # 保存原始生成结果

NUM_SAMPLES_PER_PROMPT = 1
MAX_NEW_TOKENS = 512
BATCH_SIZE = 4
def collate_fn(batch):
    """批处理拼接 prompt"""
    prompts = []
    for example in batch:
        messages = [{"role": "user", "content": example["user_input"]}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt_text)
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(model.device)
    return inputs, batch

# ===== 加载原始训练数据 =====
original_data = load_json_datadict(ORIGINAL_TRAIN_PATH)
original_data = original_data.map(preprocess_function)
original_data = original_data["train"]
dataloader = DataLoader(
    original_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)
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

for inputs, batch in tqdm.tqdm(dataloader, desc="Generating candidates (batch mode)"):

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=NUM_SAMPLES_PER_PROMPT,  # 一次生成多个候选
        )

    # outputs.shape = [batch_size * NUM_SAMPLES_PER_PROMPT, seq_len]
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # 逐样本切片
    for i, example in enumerate(batch):
        start = i * NUM_SAMPLES_PER_PROMPT
        end = (i + 1) * NUM_SAMPLES_PER_PROMPT
        responses = decoded[start:end]

        # 去重
        responses = list(dict.fromkeys(responses))

        all_candidates.append({
            "title": example["title"],
            "description": example["description"],
            "prompt": example["user_input"],
            "chosen": example["full_litex"],
            "candidates": responses
        })

# ===== 保存生成结果 =====
print(f"共生成 {len(all_candidates)} 条样本的候选响应")
with open(OUTPUT_CANDIDATES_PATH, "w", encoding="utf-8") as f:
    json.dump(all_candidates, f, ensure_ascii=False, indent=2)

print(f"候选响应已保存至: {OUTPUT_CANDIDATES_PATH}")