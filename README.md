# Litex LLM 

可运行版本：

autodl镜像

```
PyTorch  2.8.0
Python  3.12(ubuntu22.04)
CUDA  12.8
```

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

一不小心往项目里拉了💩，比如DStore之类的，小达你看不惯可以清一下。

config.yaml是计算加速的一些硬件参数，可能要根据实际的环境修改一下。
