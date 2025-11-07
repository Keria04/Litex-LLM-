import pylitex
from datasets import load_dataset, Dataset, DatasetDict
import json
import re
import subprocess
from openai import OpenAI,APIStatusError, APIError
import tqdm
API_KEY = "sk-12390044e20d48bd92896b6cd801dc44"
def judge_litex_grammar_correctness(message):
# msg = """claim:
#     forall a, b, c R:
#         a + c = b + c
#         =>:
#             a = b
#     prove:
#         a + c - c = b + c - c
#         a = b + c - c
# """
    result = pylitex.run(message)
    return result["success"]

def load_json_datadict(json_path):
    """
    从 JSON 文件加载 Hugging Face 的 DatasetDict。

    该函数读取 UTF-8 编码的 JSON 文件，文件应为记录列表（字典组成的列表）。
    使用 Dataset.from_list 构建 datasets.Dataset，并封装为只含 'train' 切分的 DatasetDict。

    参数:
        json_path (str | os.PathLike): 指向包含对象列表的 JSON 文件路径。
    返回:
        datasets.DatasetDict: 仅包含一个切分：
            - 'train': 由 JSON 记录构建的 datasets.Dataset。
    说明:
        - 依赖 datasets 库（pip install datasets）。
        - JSON 内容必须是映射（dict）的列表，键将成为列名。
        - 所有记录应具有兼容的模式；值需可 JSON 序列化。
        - 整个 JSON 将一次性读入内存。
    可能抛出:
        FileNotFoundError: 当 json_path 不存在。
        json.JSONDecodeError: 当文件内容不是合法 JSON。
        TypeError | ValueError: 当 JSON 结构与 Dataset.from_list 不兼容。
    示例:
        >>> dd = load_json_datadict("data.json")
        >>> dd["train"]  # 访问 'train' 切分
        Dataset({
          features: ...
          num_rows: ...
        })
    """

    with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

    dataset = Dataset.from_list(data)
    dataset_dict = DatasetDict({
        'train': dataset
    })
    return dataset_dict

def indent_concat_four(str1, str2, str3, str4, indent_spaces=4):
    """
    将四个可能包含换行的字符串按层级缩进拼接，每行保持对应缩进级别
    
    参数:
        str1: 第一个字符串（无缩进）
        str2: 第二个字符串（一级缩进）
        str3: 第三个字符串（二级缩进）
        str4: 第四个字符串（三级缩进）
        indent_spaces: 每级缩进的空格数，默认4个空格
    
    返回:
        拼接后的字符串，每行按对应层级缩进
    """
    # 计算各级缩进
    indent1 = ' ' * indent_spaces
    indent2 = ' ' * (indent_spaces * 2)
    indent3 = ' ' * (indent_spaces * 3)

    if len(str1) > 0: # has difinition
        lines1 = [f"{indent1}{line}" for line in str1.split('\n')]
        lines1 = ["prove:"] + lines1
        lines2 = [f"{indent2}{line}" for line in str2.split('\n')]
        lines2 = [f"{indent1}claim:"] + lines2
        lines3 = [f"{indent3}{line}" for line in str3.split('\n')]
        lines3 = [f"{indent2}prove:"] + lines3
    
    # 处理每个字符串的每行，添加对应缩进
    lines1 = [line for line in str1.split('\n')]
    lines2 = [f"{indent1}{line}" for line in str2.split('\n')]
    lines3 = [f"{indent2}{line}" for line in str3.split('\n')]
    lines4 = [f"{indent3}{line}" for line in str4.split('\n')]
    
    # 合并所有行并拼接
    all_lines = lines1 + lines2 + lines3 + lines4
    return '\n'.join(all_lines)


def split_by_last_prove(s):
    """
    将输入字符串按照最后一次出现的"prove:"进行分割
    
    参数:
        s: 输入的包含证明内容的字符串
        
    返回:
        一个元组，包含两部分内容:
        - 第一部分：最后一个"prove:"之前的所有内容
        - 第二部分：最后一个"prove:"及其之后的所有内容
    """
    # 查找最后一次出现"prove:"的位置
    last_prove_index = s.rfind('prove:')
    
    if last_prove_index == -1:
        # 如果没有找到"prove:"，返回整个字符串和空字符串
        return (s, '')
    
    # 分割为两部分
    part1 = s[:last_prove_index]
    part2 = s[last_prove_index:]
    
    return (part1, part2)
def extract_document_content(tex: str) -> str | None:
    """
    Return the content between \\begin{claim} and \\begin{proof}.

    :param tex: The full LaTeX claim as a string.
    :return: The content inside the document environment, or None if not found.
    """
    pattern = re.compile(r"\\begin\{claim\}(.*?)\\begin\{proof\}", re.DOTALL)
    m = pattern.search(tex)
    if not m:
        return None
    return m.group(1).strip()
def convert_litex_latex(litex_code: str) -> dict:
    """
    Convert a Litex file to LaTeX format using the Litex Core.
    :param litex_code: The Litex code as a string.
    :return: The LaTeX formatted string.
    """
    try:
        result = pylitex.convert_to_latex(litex_code.replace("\r\n", "\n"))
        claim_content = extract_document_content(result["message"])
        if claim_content is not None:
            return {"success": True, "message": claim_content}
        else:
            return {
                "success": False,
                "message": "No claim environment found in the LaTeX output.",
            }
    except subprocess.CalledProcessError as e:
        return {"success": False, "message": e.stderr}
    except FileNotFoundError:
        return {
            "success": False,
            "message": "Litex command not found. Please ensure Litex is installed and in your PATH.",
        }
def generate_prompt(row: dict[str, str]) -> list[dict[str, str]] | None:
    """
    Generate a prompt to verify if the LaTeX code solves the given topic.

    :param row: A dictionary containing 'description' and 'solution' keys.
    :return: A list of message dictionaries in the format [{"role": "...", "content": "..."}, ...], or None if conversion fails..
    :raises ValueError: If Litex conversion fails.
    """
    topic = row["description"]
    litex_code = row["solution"]
    try:
        latex_code_converter_result = convert_litex_latex(litex_code)
        if not latex_code_converter_result["success"]:
            return None
    except Exception as e:
        return None
    latex_code = latex_code_converter_result["message"]
    prompt = [{"role": "system", "content": "You are a knowledgeable assistant skilled in evaluating LaTeX code for mathematical and logical correctness. You should follow the user's instructions carefully and provide accurate assessments based on the provided LaTeX code and topic. you should answer \"Yes\" or \"No\" only."}]
    prompt.append({"role": "user", "content": "Consider this restrict: You should answer \"Yes\" if the LaTeX code is clearly and unambiguously attempting to describe or solve the given topic."})
    prompt.append({"role": "user", "content": "Consider this restrict: You should answer \"Yes\" if the LaTeX code is using different symbol to describe the vars in the topic, like \"x\", \"y\", \"z\" or other math symbol but still represent the same calculation relationship between numbers and vars."})
    prompt.append({"role": "user", "content": "Consider this restrict: You should answer \"Yes\" if the LaTeX code is translating the conceptions only for those basic math conceptions."})
    prompt.append({"role": "user", "content": "Consider this restrict: You should answer \"Yes\" if the LaTeX code is directly providing the final answer or solution to the problem, even it was problem itself for those obvious math problems. "})
    prompt.append({"role": "user", "content": "Consider this restrict: You should answer \"Yes\" if the LaTeX code is solving for a variable or simplifying an expression for those easy math algebra problems. "})
    prompt.append({"role": "user", "content": "Consider this restrict: You should answer \"Yes\" if the LaTeX code is transforming the polynomial to another form for those polynomial transformation or simplification problems."})
    prompt.append({"role": "user", "content": "Consider this restrict: You must answer \"No\" if the same answer shown both before and after the $\\Rightarrow$ symbol."})

    prompt.append({"role": "user", "content": f"Here is the topic and the LaTeX code:\nTopic:\n{topic}\n\nLaTeX code:\n```\n{latex_code}\n```"})
    prompt.append({"role": "user", "content": "Is the LaTeX code describe the topic? Answer \"Yes\" or \"No\" only."})

    return prompt
def get_agent_list() -> list[str]:
    return ["qwen-max", "qwen-plus", "deepseek-v3.2-exp"]
def ask_agent(info: tuple[str, list[dict[str, str]]]) -> str | None:
    (model, prompt) = info
    client = OpenAI(
        api_key=API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    try:
        completion = client.chat.completions.create(
            model=model,
            messages = prompt,
            timeout=30,
        )
        return completion.choices[0].message.content

    except APIStatusError as e:
        if e.status_code == 400 and "balance" in str(e).lower():
            print("❌ Token 配额已耗尽！请前往 DashScope 控制台充值或等待重置。")
        elif e.status_code == 429:
            print("⚠️ 请求过于频繁，触发限流。")
        else:
            print(f"API 状态错误: {e.status_code} - {e.message}")
        return None

    except APIError as e:
        print(f"OpenAI API 错误: {e}")
        return None

    except Exception as e:
        print(f"未知错误: {e}")
        return None
def judge_litex_semantic_correctness(row: dict[str, str]):
    """
        Verify if the Litex code solves the given topic using multiple LLMs.
        :param row: A dictionary containing 'title', 'description', 'solution', and 'expect' keys.
        :return: A dictionary with the original data and the verification results.
        Example input:
        test_data = {
            "title": "Problem Title",
            "description": "Mathematical problem description",
            "solution": "claim:\n    forall a, b R:\n        a + b = b + a\n    prove:\n        a + b = b + a",
            }
    """
    prompt = generate_prompt(row)
    if prompt is None:
        return {
            "title": row["title"],
            "description": row["description"],
            "solution": row["solution"],
            "actual": "No",
        }

    else:
        results = []
        for i in range(2):  # Two rounds of voting
            for model in get_agent_list():
                result = ask_agent((model, prompt))
                results.append(result)

        answers = results # type: ignore
        answer = "Yes" if "Yes" in answers else "No"

        return {
            "title": row["title"],
            "description": row["description"],
            "solution": row["solution"],
            "actual": answer,
        }
def judge_litex_correctness(row: dict[str, str]):
    """
    综合验证 Litex 代码的语义正确性与语法正确性。
    :param row: 包含 'title', 'description', 'solution' 的字典。
    :return: 包含各项验证结果的字典。
    """
    title = row.get("title", "")
    description = row.get("description", "")
    solution = row.get("solution", "")

    # ---------- 1. 语法验证 ----------
    grammar_result = judge_litex_grammar_correctness(solution)
    grammar_correctness = bool(grammar_result)  # 确保是 True / False

    # ---------- 2. 语义验证 ----------
    semantic_result = judge_litex_semantic_correctness(row)
    # 语义函数返回的字典中 actual 为 "Yes"/"No"
    semantic_correctness = semantic_result.get("actual", "").strip().lower() == "yes"

    # ---------- 3. 综合判断 ----------
    correctness = grammar_correctness and semantic_correctness

    # ---------- 4. 返回 ----------
    return {
        "title": title,
        "description": description,
        "solution": solution,
        "semantic_correctness": semantic_correctness,
        "grammar_correctness": grammar_correctness,
        "correctness": correctness,
    }
def get_output_score(file_path: str) -> dict:
    """
        对 JSONL 文件中所有样本执行 Litex 正确性评估，并计算总得分。
        每个样本得分规则：
            - 语义正确且语法正确 => 1分
            - 否则 => 0分
        返回：
            - 总分百分制（float）
            - 每个样本的详细结果列表
        """
    results = []
    total = 0
    correct = 0

    # 先读取所有行，方便 tqdm 知道总数
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]

    # tqdm 包裹循环
    for line in tqdm.tqdm(lines, desc="Evaluating Litex correctness", unit="sample"):
        data = json.loads(line)

        # 构造输入字典
        row = {
            "title": data.get("id", ""),
            "description": data.get("nl_problem", ""),
            "solution": data.get("formal_code", ""),
        }

        # 调用验证函数
        result = judge_litex_correctness(row)

        # 计算单样本得分
        score = 1 if result["correctness"] else 0
        correct += score
        total += 1

        results.append({
            **result,
            "score": score
        })

    # 计算百分制分数
    accuracy = round((correct / total) * 100, 2) if total > 0 else 0.0

    print(f"\n✅ Evaluated {total} samples.")
    print(f"🎯 Total Score: {correct}/{total} = {accuracy}%")

    return {
        "score": accuracy,
        "results": results
    }
def preprocess_function(example):
    description = example["description"]
    full_litex = example["full litex"]
    title = example["title"]
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