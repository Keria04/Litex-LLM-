# train_dpo_with_eval.py
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments
)
from peft import PeftModel
from trl import DPOTrainer
from datasets import Dataset
import torch
import json
import os
import numpy as np
from datetime import datetime
from utils import judge_litex_correctness, load_json_datadict
from transformers.trainer_callback import TrainerCallback

# ===== 配置 =====
BASE_MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"
SFT_LORA_PATH = "results/checkpoint-233"  #放当前最优的 lora
DPO_DATASET_PATH = "dataset/dpo_train_from_math23k_and_gsm8k.json"
EVAL_DATASET_PATH = "dataset/test_litex_merged.json"  # 用于评估的测试集
OUTPUT_DIR = "./dpo_results"

# ===== 自定义评估回调 =====
class DPOEvalCallback(TrainerCallback):
    def __init__(self, eval_samples, tokenizer, num_samples=10):
        self.eval_samples = eval_samples
        self.tokenizer = tokenizer
        self.num_samples = num_samples

    def on_evaluate(self, args, state, control, model, **kwargs):
        if not state.is_local_process_zero:
            return

        print(f"\n{'='*50}")
        print(f"Step {state.global_step}: Running custom correctness evaluation...")

        model.eval()
        correctness_list = []

        with torch.no_grad():
            for i, sample in enumerate(self.eval_samples):
                if i >= self.num_samples:
                    break

                # 构建输入
                user_input = sample["user_input"]
                messages = [{"role": "user", "content": user_input}]
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self.tokenizer(text, return_tensors="pt").to(model.device)

                # 生成
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                generated = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True
                )

                # 评估
                row = {
                    "title": sample.get("title", ""),
                    "description": sample.get("description", ""),
                    "solution": generated,
                }
                try:
                    result = judge_litex_correctness(row)
                    is_correct = result["correctness"]
                except Exception as e:
                    is_correct = False
                    print(f"Judge error on sample {i}: {e}")

                correctness_list.append(is_correct)

        avg_correctness = np.mean(correctness_list) if correctness_list else 0.0
        print(f"Step {state.global_step} | Correctness: {avg_correctness:.4f} ({len(correctness_list)} samples)")
        print(f"{'='*50}\n")
        log_entry = {
            "step": state.global_step,
            "correctness": float(avg_correctness),
            "num_samples": len(correctness_list),
            "timestamp": datetime.now().isoformat()
        }
        log_path = os.path.join(args.output_dir, "dpo_eval_log.jsonl")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

# ===== 加载 DPO 数据集 =====
def load_dpo_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Dataset.from_list(data)

# ===== 主流程 =====
def main():
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载 base model + SFT LoRA
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    print("Loading SFT LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, SFT_LORA_PATH)
    model.train()  # DPO 需要训练模式

    # 加载 DPO 训练数据
    train_dataset = load_dpo_dataset(DPO_DATASET_PATH)
    print(f"Loaded {len(train_dataset)} DPO pairs.")

    # 加载评估数据（用于回调）
    eval_data_raw = load_json_datadict(EVAL_DATASET_PATH)
    from utils import preprocess_function  # ← 替换为你的实际模块
    eval_dataset = eval_data_raw.map(preprocess_function)["train"]

    # 训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=1e-5,
        logging_steps=10,
        save_strategy="steps",
        save_steps=233,
        eval_strategy="steps",
        eval_steps=233,
        save_total_limit=2,
        fp16=True,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        report_to="none",
    )

    # DPO Trainer
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,  # 使用 implicit reference（SFT 模型初始状态）
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        beta=0.1,
        max_prompt_length=1024,
        max_length=2048,
        loss_type="sigmoid",
    )

    # 添加评估回调
    eval_callback = DPOEvalCallback(
        eval_samples=eval_dataset,
        tokenizer=tokenizer,
        num_samples=5
    )
    dpo_trainer.add_callback(eval_callback)

    # 开始训练
    print("Starting DPO training with SFT-initialized LoRA...")
    dpo_trainer.train()

    # 保存最终模型（LoRA 权重）
    dpo_trainer.save_model(os.path.join(OUTPUT_DIR, "final_lora"))

    print("DPO training completed!")

if __name__ == "__main__":
    main()