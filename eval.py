from trl import SFTTrainer
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM, TrainerCallback
import os
import torch
import json
import numpy as np
from tqdm import tqdm
from datetime import datetime
from utils import *

MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"
LORA_PATH = "results/checkpoint-233"
test_dataset = load_json_datadict("dataset/test_litex.json")


def preprocess_function(example):
    description = example["natural task"]
    full_litex = example["full litex"]
    title = example["name"]
    claim, prove = split_by_last_prove(full_litex)
    user_input = f"""You are given a mathematical problem stated in natural language.  Your task is to translate it into a complete Litex formal solution, which includes both a `claim:` section stating the formal proposition and a `prove:` section providing a step-by-step logical derivation.

    Show each reasoning step clearly in the proof, and ensure the conclusion in the `claim:` is fully justified by the `prove:` section.
    ### Problem
    {description}"""
    data = {"messages": [{"role": "user", "content": user_input},
                         {"role": "assistant", "content": full_litex}],
            "user_input": user_input,
            "question": claim,
            "full_litex": full_litex,
            "title": title,
            "description": description, }
    return data

test_dataset = test_dataset.map(preprocess_function)
test_dataset = test_dataset["train"]

print("评估集大小：", len(test_dataset))


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("正在加载基础模型...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",  # 自动分配设备
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

print("正在加载LoRA适配器...")
model = PeftModel.from_pretrained(model, LORA_PATH)
print("正在合并LoRA权重...")
model = model.merge_and_unload()

model.eval()

print("模型加载完成！")

def generate_response(user_input, max_new_tokens=512, temperature=0.7, do_sample=True):
    """生成模型响应 - 与训练时回调函数保持一致"""
    messages = [{"role": "user", "content": user_input}]
    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return generated_text

def evaluate_model(test_data, num_samples=None, save_results=True, output_dir="./evaluation_logs"):
    """评估模型性能 - 与训练时回调函数保持一致"""
    os.makedirs(output_dir, exist_ok=True)
    
    if num_samples is not None:
        test_data = test_data.select(range(min(num_samples, len(test_data))))
    
    results = []
    grammar_success_records = []
    sementic_success_records = []
    success_records = []
    print(f"开始评估 {len(test_data)} 个样本...")
    print("="*50)
    
    model.eval()
    
    with torch.no_grad():
        for i, example in tqdm(enumerate(test_data)):
            print(f"处理样本 {i+1}/{len(test_data)}")
            description = example["description"]
            question = example["question"]
            user_input = example["user_input"]
            expected_answer = example["full_litex"]
            title = example["title"]
            try:
                generated = generate_response(user_input)
                full_litex = generated
                row = {
                    "title": title,
                    "description": description,
                    "solution": full_litex,
                }
                correctness = judge_litex_correctness(row)
                grammar_correctness = correctness["grammar_correctness"]
                sementic_correctness = correctness["sementic_correctness"]
                result = {
                    "sample_id": i,
                    "claim": question,
                    "user_input": user_input,
                    "expected_answer": expected_answer,
                    "answer": full_litex,
                    "grammar_correctness": grammar_correctness,
                    "sementic_correctness": sementic_correctness,
                    "correctness": correctness["correctness"],
                    "timestamp": datetime.now().isoformat()
                }
                
                results.append(result)
                grammar_success_records.append(grammar_correctness)
                sementic_success_records.append(sementic_correctness)
                success_records.append(correctness["correctness"])
            except Exception as e:
                print(f"处理样本 {i} 时出错: {e}")
                result = {
                    "sample_id": i,
                    "claim": question,
                    "user_input": user_input,
                    "expected_answer": expected_answer,
                    "generated": "",
                    "full_litex": "",
                    "grammar_correctness": False,
                    "sementic_correctness": False,
                    "correctness": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                results.append(result)
                grammar_success_records.append(False)
                sementic_success_records.append(False)
                success_records.append(False)
                continue

    overall_stats = {
        "num_samples": len(grammar_success_records),
        "correct_count": int(np.sum(success_records)) if success_records else 0,
        "total_count": len(grammar_success_records),
        "evaluation_timestamp": datetime.now().isoformat()
    }
    results.append(overall_stats)
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(output_dir, f"evaluation_results_{timestamp}.json")
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"结果已保存到: {results_file}")
    
    print("="*50)
    return results, overall_stats





if __name__ == "__main__":
    NUM_EVAL_SAMPLES = None 
    OUTPUT_DIR = "./evaluation_logs"
    
    print("开始模型评估...")
    print(f"评估样本数: {NUM_EVAL_SAMPLES}")
    
    results, overall_stats = evaluate_model(
        test_dataset, 
        num_samples=NUM_EVAL_SAMPLES,
        save_results=True,
        output_dir=OUTPUT_DIR
    )
    
    print("\n评估完成!")
    