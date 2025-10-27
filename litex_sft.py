from trl import SFTTrainer
from datasets import load_dataset
from peft import LoraConfig
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM, TrainerCallback
import os
import torch
import json
import numpy as np
from datetime import datetime
from utils import *


MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"
train_dataset = load_json_datadict("dataset/train_gsm8k_litex_cleaned.json")
test_dataset = load_json_datadict("dataset/test_gsm8k_litex_cleaned.json")


def preprocess_function(example):
    description = example["description"]
    full_litex = example["full litex"]
    claim, prove = split_by_last_prove(full_litex)
    user_input = f"""You are given a mathematical problem stated in natural language.  Your task is to translate it into a complete Litex formal solution, which includes both a `claim:` section stating the formal proposition and a `prove:` section providing a step-by-step logical derivation.

    Show each reasoning step clearly in the proof, and ensure the conclusion in the `claim:` is fully justified by the `prove:` section.
    ### Problem
    {description}"""
    data = {"messages": [{"role": "user", "content": user_input},
                {"role": "assistant", "content": full_litex}],
            "user_input": user_input,
            "question": claim,
            "full_litex": full_litex,}
    return data

train_dataset = train_dataset.map(preprocess_function)
train_dataset = train_dataset["train"]
test_dataset = test_dataset.map(preprocess_function)
test_dataset = test_dataset["train"]


print("训练集大小：", len(train_dataset))
print("评估集大小：", len(test_dataset))

# 显式加载模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map=None,
    trust_remote_code=True
)

# 配置 LoRA
peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

# 自定义回调函数用于生成测试
class EvaluationCallback(TrainerCallback):
    def __init__(self, eval_samples, tokenizer, output_dir="./evaluation_logs"):
        self.eval_samples = eval_samples
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def on_evaluate(self, args, state, control, model, **kwargs):
        """在每次评估后进行生成测试"""
        if state.is_local_process_zero:  # 只在主进程中执行
            print("\n" + "="*50)
            print(f"Step {state.global_step}: 开始生成测试...")
            
            # 生成测试结果
            results = []
            grammar_success_records = []
            model.eval()
            
            with torch.no_grad():
                for i, sample in enumerate(self.eval_samples):
                    # 构建输入
                    if i > 5:
                        break
                    question = sample["question"]
                    user_input = sample["user_input"]
                    # 生成回答
                    messages = [{"role": "user", "content": user_input}]
                    text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    
                    inputs = self.tokenizer(text, return_tensors="pt").to(model.device)
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=512,
                            do_sample=True,
                            temperature=0.7,
                            pad_token_id=self.tokenizer.eos_token_id
                        )
                    
                    generated = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                    full_litex = generated
                    grammar_correctness = judge_litex_grammar_correctness(full_litex)
                    result = {
                        "step": state.global_step,
                        "sample_id": i,
                        "claim": question,
                        "full_litex": full_litex,
                        "grammar_correctness": grammar_correctness,
                        "timestamp": datetime.now().isoformat()
                    }
                    results.append(result)
                    grammar_success_records.append(grammar_correctness)
                    
                    # print(f"\n样本 {i+1}:")
                    # print(f"问题: {result['prompt']}")
                    # print(f"生成: {result['generated']}")
                    # print("-" * 30)
            results.append({
                "num sample": len(grammar_success_records),
                "overall_grammar_correctness": np.mean(grammar_success_records)
            })
            # 保存结果到文件
            log_file = os.path.join(self.output_dir, f"generation_step_{state.global_step}.json")
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            print(f"评估结果已保存到: {log_file}")
            print("="*50 + "\n")

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=1,
    warmup_steps=233,
    learning_rate=1e-4,
    weight_decay=0.01,
    fp16=True,
    logging_steps=1,
    eval_strategy="steps",
    eval_steps=233,
    save_strategy="steps", 
    save_steps=233,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    gradient_checkpointing=True,
    dataloader_pin_memory=False,
    remove_unused_columns=False,
    ddp_find_unused_parameters=False,
    # report_to=None,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    peft_config=peft_config,
    args=training_args,
    callbacks=[EvaluationCallback(test_dataset, tokenizer)],
)


# 训练前进行一次初始评估
print("进行初始评估...")
trainer.evaluate()

# 开始训练
trainer.train()

# 训练结束后进行最终评估
print("\n进行最终评估...")
final_metrics = trainer.evaluate()
print("最终评估结果:", final_metrics)