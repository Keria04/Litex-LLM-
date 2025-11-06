import json
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils import judge_litex_correctness

INPUT_CANDIDATES_PATH = "dataset/generated_candidates.json"
OUTPUT_DPO_PATH = "dataset/dpo_train_from_math23k_and_gsm8k.json"

# 全局函数：用于被子进程调用
def judge_single_response(args):
    title, description, solution, prompt, chosen = args
    row = {
        "title": title,
        "description": description,
        "solution": solution,
    }
    is_correct = judge_litex_correctness(row)["correctness"]
    if not is_correct:
        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": solution
        }
    else:
        return None  # 正确的响应不构成 rejected，返回 None

def main():
    with open(INPUT_CANDIDATES_PATH, "r", encoding="utf-8") as f:
        all_candidates = json.load(f)

    # 构建任务列表：每个 candidate 一个任务
    tasks = []
    for item in all_candidates:
        prompt = item["prompt"]
        chosen = item["chosen"]
        title = item["title"]
        description = item["description"]
        for resp in item["candidates"]:
            tasks.append((title, description, resp, prompt, chosen))

    print(f"共 {len(tasks)} 个候选响应待判断...")

    # 获取 CPU 核心数，合理设置进程数（避免过多）
    num_workers = min(os.cpu_count(), 16)  # 你也可以手动设为 8、12 等

    dpo_pairs = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有任务
        future_to_task = {executor.submit(judge_single_response, task): task for task in tasks}

        # 使用 tqdm 显示进度
        for future in tqdm(as_completed(future_to_task), total=len(tasks), desc="Judging responses (parallel)"):
            result = future.result()
            if result is not None:
                dpo_pairs.append(result)

    print(f"共构造 {len(dpo_pairs)} 个 DPO 偏好对")
    with open(OUTPUT_DPO_PATH, "w", encoding="utf-8") as f:
        json.dump(dpo_pairs, f, ensure_ascii=False, indent=2)

    print(f"DPO 数据集已保存至: {OUTPUT_DPO_PATH}")

if __name__ == "__main__":
    main()