### 1 多分类任务指南
#### 1.1 训练模型

使用CPU/GPU训练，默认为GPU训练。使用CPU训练只需将设备参数配置改为`--device cpu`，可以使用`--device gpu:0`指定GPU卡号：
```shell
python algo/multi_class/train.py --model_name_or_path ernie-3.0-medium-zh --data_dir ./data/bgqx_code --output_dir ./checkpoint/bgqx_code --device gpu --learning_rate 3e-5 --early_stopping_patience 4 --max_seq_length 256 --per_device_eval_batch_size 32 
--per_device_train_batch_size 32 --num_train_epochs 10 --do_train --do_eval --metric_for_best_model accuracy --load_best_model_at_end --evaluation_strategy epoch --save_strategy epoch --save_total_limit 1 
```

主要的配置的参数为：
- `model_name_or_path`: 内置模型名，或者模型参数配置目录路径。默认为`ernie-3.0-base-zh`。
- `data_dir`: 训练数据集路径，数据格式要求详见[数据标注](#数据标注)。
- `output_dir`: 模型参数、训练日志和静态图导出的保存目录。
- `max_seq_length`: 最大句子长度，超过该长度的文本将被截断，不足的以Pad补全。提示文本不会被截断。
- `num_train_epochs`: 训练轮次，使用早停法时可以选择100
- `early_stopping_patience`: 在设定的早停训练轮次内，模型在开发集上表现不再上升，训练终止；默认为4。
- `learning_rate`: 预训练语言模型参数基础学习率大小，将与learning rate scheduler产生的值相乘作为当前学习率。
- `do_train`: 是否进行训练。
- `do_eval`: 是否进行评估。
- `device`: 使用的设备，默认为`gpu`。
- `per_device_train_batch_size`: 每次训练每张卡上的样本数量。可根据实际GPU显存适当调小/调大此配置。
- `per_device_eval_batch_size`: 每次评估每张卡上的样本数量。可根据实际GPU显存适当调小/调大此配置。


#### 1.2 训练评估与模型优化

文本分类预测过程中常会遇到诸如"模型为什么会预测出错误的结果"，"如何提升模型的表现"等问题。[Analysis模块](./analysis) 提供了**模型评估、可解释性分析、数据优化**等功能，旨在帮助开发者更好地分析文本分类模型预测结果和对模型效果进行优化。

<div align="center">
    <img src="https://user-images.githubusercontent.com/63761690/195241942-70068989-df17-4f53-9f71-c189d8c5c88d.png" width="600">
</div>

**模型评估：** 训练后的模型我们可以使用 [Analysis模块](./analysis) 对每个类别分别进行评估，并输出预测错误样本（bad case），默认在GPU环境下使用，在CPU环境下修改参数配置为`--device "cpu"`:

```shell
python analysis/multi_class/evaluate.py --device "gpu" --max_seq_length 256 --batch_size 32 --bad_case_file "bad_case.txt" --dataset_dir "./data/bgqx_code" --params_path "./checkpoint/bgqx_code"
```

预测错误的样本保存在bad_case.txt文件中：

####  1.3 模型可解释性分析

我们可以运行代码，得到支持样本模型预测结果的训练数据：
```shell
python analysis/interpret.py --device "gpu" --dataset_dir "data/bgqx_code" --params_path "checkpoint/bgqx_code" --cache_file "checkpoint/bgqx_code" --max_seq_length 256 --batch_size 32 --top_k 5 --train_file "train.txt" --interpret_input_file "bad_case.txt" --interpret_result_file "sent_interpret.txt"
```

可支持配置的参数：

* `device`: 选用什么设备进行训练，可可选择cpu、gpu、xpu、npu；默认为"gpu"。
* `dataset_dir`：必须，本地数据集路径，数据集路径中应包含dev.txt和label.txt文件;默认为None。
* `params_path`：保存训练模型的目录；默认为"../checkpoint/"。
* `max_seq_length`：分词器tokenizer使用的最大序列长度，ERNIE模型最大不能超过2048。请根据文本长度选择，通常推荐128、256或512，若出现显存不足，请适当调低这一参数；默认为128。
* `batch_size`：批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为32。
* `seed`：随机种子，默认为3。
* `top_k`：筛选支持训练证据数量；默认为3。
* `train_file`：本地数据集中训练集文件名；默认为"train.txt"。
* `interpret_input_file`：本地数据集中待分析文件名；默认为"bad_case.txt"。
* `interpret_result_file`：保存句子级别可解释性结果文件名；默认为"sent_interpret.txt"


### 2 多标签任务指南

#### 2.1 训练模型

使用CPU/GPU训练，默认为GPU训练。使用CPU训练只需将设备参数配置改为`--device cpu`,使用GPU训练只需将设备参数配置改为`--device gpu:0`指定GPU卡号：
```shell
python algo/multi_label/train.py --dataset_dir "data/divorce" --save_dir "checkpoint/divorce" --device "gpu" --max_seq_length 128 --batch_size 32 --epochs 100
```
#### 2.2 模型评估 
训练后的模型我们可以使用 [Analysis模块](./analysis) 对每个类别分别进行评估，并输出预测错误样本（bad case），默认在GPU环境下使用，在CPU环境下修改参数配置为`--device "cpu"`:

```shell
python analysis/multi_label/evaluate.py --device "gpu" --max_seq_length 128 --batch_size 32 --bad_case_file "bad_case.txt" --dataset_dir "data/divorce" --params_path "./checkpoint/divorce"
```