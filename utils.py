import pylitex
from datasets import load_dataset, Dataset, DatasetDict
import json

def judge_litex_correctness(message):
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

data = load_json_datadict("train_litex.json")
data = data["train"][0]
first, second = split_by_last_prove(data["full litex"])
print(first)
print(second)
print(first+second)