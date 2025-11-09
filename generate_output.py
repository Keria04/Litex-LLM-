import json
import config
import sys
import os
import argparse
from datetime import datetime
from contextlib import nullcontext
try:
    import torch  # 用于 no_grad / cuda 释放
except ImportError:
    torch = None  # 延迟在实际调用时再报错

################################################################################
# 批量 LoRA 生成 & 评估脚本
# 原逻辑：单一 LoRA -> 生成 -> 保存
# 现扩展：多个 LoRA -> 逐个 (重新加载基础模型 + 合并 LoRA) -> 生成 -> 保存 -> 评估
# 备注：不改动核心推理函数 generate_response 的内部逻辑
################################################################################

MODEL_PATH = config.MODEL_PATH
INPUT_PATH = config.INPUT_PATH
LORA_PATHS = config.LORA_PATHS
OUTPUT_DIR = config.OUTPUT_DIR
EVAL_LOG_FILE = config.EVAL_LOG_FILE
USER_INPUT_PROMPT = config.USER_INPUT_PROMPT  # problem will be appended here


def load_input_dataset(path: str):
    if not os.path.isfile(path):
        print(f"输入数据集不存在: {path}")
        sys.exit(1)
    with open(path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]
    print(f"输入数据集大小: {len(data)}  (路径: {path})")
    return data


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def derive_output_path(input_path: str, lora_path: str, output_dir: str | None):
    """根据输入文件与 LoRA 目录生成输出文件名。
    规则: <output_dir>/<input_base>__<lora_dir_name>.jsonl
    """
    base = os.path.splitext(os.path.basename(input_path))[0]
    lora_tag = os.path.basename(os.path.normpath(lora_path))
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(input_path))
    ensure_dir(output_dir)
    return os.path.join(output_dir, f"{base}__{lora_tag}.jsonl")


def derive_eval_json_path(output_jsonl_path: str):
    return output_jsonl_path + ".eval.json"


def load_base_model():
    """加载基础模型（不合并 LoRA）。每个 LoRA 重新调用，避免权重累积。"""
    print("正在加载基础模型...")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    return tokenizer, model


def merge_lora(model, lora_path: str):
    from peft import PeftModel
    print(f"正在加载 LoRA 适配器: {lora_path}")
    model = PeftModel.from_pretrained(model, lora_path)
    print("正在合并 LoRA 权重 (merge_and_unload)...")
    model = model.merge_and_unload()
    model.eval()
    print("LoRA 合并完成。")
    return model


# from trl import SFTTrainer
# from datasets import load_dataset
from utils import *  # 评估需要的 get_output_score / judge 函数

# ================= 保留原始推理核心：generate_response =================
# 注意：tokenizer 与 model 将在批处理循环中动态设置为全局引用
tokenizer = None  # type: ignore
model = None      # type: ignore

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
    guard = torch.no_grad() if torch is not None else nullcontext()
    with guard:
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
    """生成输出并保存到文件 (单次 LoRA 任务)。"""
    results = []
    success_count = 0
    incorrect_ids = []  # 记录编译不通过的编号
    for i, sample in enumerate(input_dataset):
        problem = sample["nl_problem"]  # 从输入数据集中获取自然语言问题
        user_input = USER_INPUT_PROMPT + problem

        print(f"正在处理样本 {i+1}/{len(input_dataset)}...")
        full_litex = generate_response(user_input)
        compile_ok = judge_litex_grammar_correctness(full_litex)
        print(f"生成的 full_litex 编译是否通过：{compile_ok}")
        if compile_ok:
            success_count += 1
        else:
            incorrect_ids.append(sample.get("id"))

        result = {
            "id": sample["id"],
            "nl_problem": problem,
            "formal_type": "Litex",
            "header": "",
            "formal_statement": getclaim(full_litex),
            "formal_code": full_litex
        }
        results.append(result)

    with open(output_file_path, 'w', encoding='utf-8') as f_out:
        for res in results:
            f_out.write(json.dumps(res, ensure_ascii=False) + '\n')

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
    return {
        "compile_success": success_count,
        "total": total,
        "success_rate": success_rate,
        "failed_ids": incorrect_ids,
    }


def evaluate_output(output_file: str, eval_json_path: str | None = None):
    print(f"开始评估输出文件: {output_file}")
    result = get_output_score(output_file)
    if eval_json_path:
        with open(eval_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"评估详细结果已保存: {eval_json_path}")
    return result


def run_batch(input_path: str, lora_paths: list[str], output_dir: str | None, log_file: str, skip_generate: bool):
    # 加载输入数据集
    dataset_cache = None
    summary = []  # 保存每个 LoRA 的评分、输出文件等信息

    with open(log_file, 'w', encoding='utf-8') as log_f:
        log_f.write(f"# 批量 LoRA 生成 & 评估日志\n")
        log_f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_f.write(f"基础模型: {MODEL_PATH}\n")
        log_f.write(f"输入数据集: {input_path}\n")
        log_f.write(f"LoRA 个数: {len(lora_paths)}\n\n")

        for idx, lora_path in enumerate(lora_paths, start=1):
            log_f.write(f"=== [{idx}/{len(lora_paths)}] LoRA: {lora_path} ===\n")
            print(f"\n===== 处理 LoRA {idx}/{len(lora_paths)}: {lora_path} =====")

            output_file = derive_output_path(input_path, lora_path, output_dir)
            eval_json_path = derive_eval_json_path(output_file)

            if not skip_generate or not os.path.isfile(output_file):
                if dataset_cache is None:
                    dataset_cache = load_input_dataset(input_path)
                # 每个 LoRA 重新加载基础模型，避免权重累积
                global tokenizer, model
                tokenizer, model = load_base_model()
                model = merge_lora(model, lora_path)
                gen_stats = generate_outputs(dataset_cache, output_file)
                # 释放 GPU 以防 OOM
                try:
                    import torch
                    del model
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                log_f.write(f"生成完成: {output_file}\n")
                log_f.write(f"编译成功: {gen_stats['compile_success']}/{gen_stats['total']} ({gen_stats['success_rate']:.2%})\n")
            else:
                log_f.write(f"跳过生成 (已存在): {output_file}\n")

            # 评估
            eval_result = evaluate_output(output_file, eval_json_path)
            score = eval_result.get("score", 0.0)
            log_f.write(f"评估得分: {score}\n")
            log_f.write(f"评估结果文件: {eval_json_path}\n\n")

            summary.append({
                "lora": lora_path,
                "output_file": output_file,
                "eval_json": eval_json_path,
                "score": score,
            })

        # 汇总
        if summary:
            best = max(summary, key=lambda x: x.get("score", -1))
            log_f.write("=== 汇总 Summary ===\n")
            for item in summary:
                log_f.write(f"LoRA: {item['lora']} | Score: {item['score']} | Output: {item['output_file']}\n")
            log_f.write("\n最佳 LoRA: {} (Score: {})\n".format(best['lora'], best['score']))
        else:
            log_f.write("无可用结果。\n")

    print(f"\n全部完成。日志已保存: {log_file}")
    if summary:
        print(f"最佳 LoRA: {best['lora']} (Score: {best['score']})")
    return summary


if __name__ == "__main__":
    # 从 config 读取配置
    print("=" * 60)
    print("批量 LoRA 生成与评估")
    print("=" * 60)
    
    # 校验配置
    missing = [
        name for name, val in (
            ("MODEL_PATH", MODEL_PATH),
            ("USER_INPUT_PROMPT", USER_INPUT_PROMPT),
            ("INPUT_PATH", INPUT_PATH),
            ("LORA_PATHS", LORA_PATHS),
            ("OUTPUT_DIR", OUTPUT_DIR),
        ) if not val
    ]
    if missing:
        print(f"❌ 缺少配置: {', '.join(missing)}，程序终止。")
        print(f"请在 config.py 中设置这些配置项。")
        sys.exit(1)
    
    if not isinstance(LORA_PATHS, list) or len(LORA_PATHS) == 0:
        print(f"❌ LORA_PATHS 必须是非空列表，程序终止。")
        print(f"请在 config.py 中设置 LORA_PATHS = ['path/to/lora1', 'path/to/lora2', ...]")
        sys.exit(1)

    # 显示配置信息
    print(f"📂 基础模型: {MODEL_PATH}")
    print(f"📂 输入数据集: {INPUT_PATH}")
    print(f"📂 输出目录: {OUTPUT_DIR}")
    print(f"📝 评估日志: {EVAL_LOG_FILE}")
    print(f"🔧 LoRA 数量: {len(LORA_PATHS)}")
    for i, lora in enumerate(LORA_PATHS, 1):
        print(f"   {i}. {lora}")
    print("=" * 60)
    print()

    # 设置日志文件完整路径
    ensure_dir(OUTPUT_DIR)
    log_file = os.path.join(OUTPUT_DIR, EVAL_LOG_FILE)

    # 执行批量处理
    run_batch(
        input_path=INPUT_PATH,
        lora_paths=LORA_PATHS,
        output_dir=OUTPUT_DIR,
        log_file=log_file,
        skip_generate=False,  # 从 config 读取时默认总是生成
    )

    print("\n✅ 全部完成！")