MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"

# ========== 批量 LoRA 生成与评估配置 ==========
# 输入数据集路径
INPUT_PATH = "working_dir/input.jsonl"

# 要批量处理的 LoRA 路径列表
# 示例: ["results/checkpoint-233", "results/checkpoint-500", "results/checkpoint-888"]
LORA_PATHS = [
    "results/checkpoint-233",
    # "results/checkpoint-500",
    # "results/checkpoint-888",
]

# 输出文件夹 (生成的 JSONL 和评估结果都会保存在这里)
OUTPUT_DIR = "working_dir"

# 评估日志文件名 (会保存在 OUTPUT_DIR 中)
EVAL_LOG_FILE = "eval_result.log"

# ========== 用户提示词模板 ==========
USER_INPUT_PROMPT = """You are given a mathematical problem stated in natural language.  Your task is to translate it into a complete Litex formal solution, which includes both a `claim:` section stating the formal proposition and a `prove:` section providing a step-by-step logical derivation.

    Show each reasoning step clearly in the proof, and ensure the conclusion in the `claim:` is fully justified by the `prove:` section.
    ### Problem
    """# problem will be appended here

# ========== 向后兼容的单 LoRA 配置 (已弃用,建议使用 LORA_PATHS) ==========
LORA_PATH = LORA_PATHS[0] if LORA_PATHS else ""
OUTPUT_PATH = "working_dir/output.jsonl"  # 单 LoRA 模式的输出路径(已弃用)