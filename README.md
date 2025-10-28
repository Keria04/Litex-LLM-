# Litex LLM 

可运行版本：

autodl镜像

```
PyTorch  2.8.0
Python  3.12(ubuntu22.04)
CUDA  12.8
```

显存要求：

> 使用 Qwen/Qwen2.5-7B-Instruct
> 66G+

## Environment Setup

```
bash setup_env.sh
```

## Training

```
bash run.sh
```

## Evaluation

evaluate on miniF2F

```
python eval.py
```

练习赛的数据是practice_data.jsonl，可以用eval.py中的gnerate_response来调用训练好的模型

## Construct DPO Pairs

```
python DPO_data_constructor.py
```
## 运行debug

网络问题：

使用autodl的镜像源加速可能会出现下载一般炸掉情况，需要换为下面的镜像源（不配置加速）

```bash
export HF_ENDPOINT=https://hf-mirror.com
```