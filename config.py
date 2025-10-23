MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"
LORA_PATH = "results/checkpoint-233"

# for generate output only
INPUT_PATH = "working_dir/input.jsonl"
OUTPUT_PATH = "working_dir/output.jsonl"

USER_INPUT_PROMPT = """You are given a mathematical problem stated in natural language.  Your task is to translate it into a complete Litex formal solution, which includes both a `claim:` section stating the formal proposition and a `prove:` section providing a step-by-step logical derivation.

    Show each reasoning step clearly in the proof, and ensure the conclusion in the `claim:` is fully justified by the `prove:` section.
    ### Problem
    """# problem will be appended here