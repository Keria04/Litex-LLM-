import json
import config
import sys

MODEL_PATH = config.MODEL_PATH
input_dataset_path = config.INPUT_PATH
output_file_path = config.OUTPUT_PATH
LORA_PATH = config.LORA_PATH
USER_INPUT_PROMPT = config.USER_INPUT_PROMPT  # problem will be appended here

# 若任一配置为空则终止程序
_missing = [
    name for name, val in (
        ("MODEL_PATH", MODEL_PATH),
        ("INPUT_DATASET_PATH", input_dataset_path),
        ("OUTPUT_FILE_PATH", output_file_path),
        ("LORA_PATH", LORA_PATH),
        ("USER_INPUT_PROMPT", USER_INPUT_PROMPT),
    ) if not val
]
if _missing:
    print(f"缺少配置: {', '.join(_missing)}，程序终止。")
    sys.exit(1)

with open(input_dataset_path, 'r', encoding='utf-8') as f:
    input_dataset = [json.loads(line) for line in f if line.strip()]

print("输入数据集大小：", len(input_dataset))

# ----------------预处理结束-----------------


# from trl import SFTTrainer
# from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM, TrainerCallback
# import os
import torch
# import numpy as np
# from tqdm import tqdm
# from datetime import datetime
from utils import *

# 照抄 eval

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
    """
    生成模型响应 - 与训练时回调函数保持一致
    返回的是 full_litex
    """
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

def getclaim(full_litex):
    """
    从 full_litex 中提取 claim 部分
    """
    # 查找 claim: 的位置
    claim_index = full_litex.find("claim:")
    if claim_index == -1:
        return ""  # 如果没有找到 claim，返回空字符串
    
    # 从 claim: 开始截取，直到下一个 prove: 或字符串结尾
    prove_index = full_litex.find("prove:", claim_index)
    if prove_index == -1:
        claim_text = full_litex[claim_index + len("claim:"):].strip()
    else:
        claim_text = full_litex[claim_index + len("claim:"):prove_index].strip()
    
    return claim_text

def generate_outputs(input_dataset, output_file_path):
    """生成输出并保存到文件"""
    results = []
    success_count = 0
    incorrect_ids = []  # 记录编译不通过的编号
    for i, sample in enumerate(input_dataset):
        problem = sample["nl_problem"] # 从输入数据集中获取自然语言问题
        user_input = USER_INPUT_PROMPT + problem
        
        print(f"正在处理样本 {i+1}/{len(input_dataset)}...")
        full_litex = generate_response(user_input)
        compile_ok = judge_litex_grammar_correctness(full_litex)
        print(f"生成的 full_litex 编译是否通过：{compile_ok}")
        if compile_ok:
            success_count += 1
        else:
            # 收集不正确的编号（使用输入样本中的 id 字段）
            incorrect_ids.append(sample.get("id"))
        
        # 要求的输出格式构建
        result = {
            "id": sample["id"],
            "nl_problem": problem,
            "formal_type": "Litex",
            "header": "",
            "formal_statement": getclaim(full_litex),
            "formal_code": full_litex
        }
        results.append(result)
    
    # 保存结果到文件
    with open(output_file_path, 'w', encoding='utf-8') as f_out:
        for res in results:
            f_out.write(json.dumps(res, ensure_ascii=False) + '\n')
    
    # 统计与汇总输出
    total = len(input_dataset)
    success_rate = (success_count / total) if total else 0.0
    print("\n===== 编译统计 =====")
    print(f"编译成功数：{success_count}/{total}")
    print(f"编译成功率：{success_rate:.2%}")
    if incorrect_ids:
        print("编译不通过的编号：", incorrect_ids)
    else:
        print("全部样本均编译通过。")

    print(f"\n生成结果已保存到 {output_file_path}")


if __name__ == "__main__":
    generate_outputs(input_dataset, output_file_path)


    print ("\n完成")