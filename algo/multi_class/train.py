# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import os
from dataclasses import dataclass, field
import sys
sys.path.append(os.getcwd())
import paddle
from paddle.metric import Accuracy
from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.datasets import load_dataset
from paddlenlp.trainer import PdArgumentParser, EarlyStoppingCallback, Trainer
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils.multi_class.util import preprocess_function, read_local_dataset
from domain.trainingArgumentsExt import TrainingArgumentsExt

SUPPORTED_MODELS = [
    "ernie-1.0-large-zh-cw",
    "ernie-3.0-xbase-zh",
    "ernie-3.0-base-zh",
    "ernie-3.0-medium-zh",
    "ernie-3.0-micro-zh",
    "ernie-3.0-mini-zh",
    "ernie-3.0-nano-zh",
    "ernie-2.0-base-en",
    "ernie-2.0-large-en",
    "ernie-m-base",
    "ernie-m-large",
]


# yapf: disable
@dataclass
class DataArguments:
    data_dir: str = field(default="./data/", metadata={"help": "Path to a dataset which includes train.txt, dev.txt, test.txt, label.txt and data.txt (optional)."})
    max_seq_length: int = field(default=256, metadata={"help": "Maximum number of tokens for the model"})
    early_stopping_patience: int = field(default=4, metadata={"help": "Stop training when the specified metric worsens for early_stopping_patience evaluation calls"})

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="ernie-3.0-base-zh", metadata={"help": "Build-in pretrained model name or the path to local model."})
# yapf: enable


def parse_arg(args, model_args, data_args, training_args):
    # ModelArguments 属性替换
    if 'model_name_or_path' in args:
        model_args.model_name_or_path = args['model_name_or_path']

    # DataArguments 属性替换
    if 'data_dir' in args:
        data_args.data_dir = args['data_dir']
    if 'max_seq_length' in args:
        data_args.max_seq_length = args['max_seq_length']
    if 'early_stopping_patience' in args:
        data_args.early_stopping_patience = args['early_stopping_patience']

    # TrainingArguments 属性替换
    if 'output_dir' in args:
        training_args.output_dir = args['output_dir']
    if 'device' in args:
        training_args.device = args['device']
    if 'learning_rate' in args:
        training_args.learning_rate = args['learning_rate']

    if 'per_device_eval_batch_size' in args:
        training_args.per_device_eval_batch_size = args['per_device_eval_batch_size']
    if 'per_device_train_batch_size' in args:
        training_args.per_device_train_batch_size = args['per_device_train_batch_size']
    if 'num_train_epochs' in args:
        training_args.num_train_epochs = args['num_train_epochs']

    if 'do_train' in args:
        training_args.do_train = args['do_train']
    if 'do_eval' in args:
        training_args.do_eval = args['do_eval']

    if 'metric_for_best_model' in args:
        training_args.metric_for_best_model = args['metric_for_best_model']
    if 'load_best_model_at_end' in args:
        training_args.load_best_model_at_end = args['load_best_model_at_end']
    if 'evaluation_strategy' in args:
        training_args.evaluation_strategy = args['evaluation_strategy']
    if 'save_strategy' in args:
        training_args.save_strategy = args['save_strategy']
    if 'save_total_limit' in args:
        training_args.save_total_limit = args['save_total_limit']


def do_train(**args):
    """
    Training a binary or multi classification model
    """
    # TrainingArgumentsExt(输入参数类) 拷贝自 paddlenlp.trainer.TrainingArguments
    # parser = PdArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    parser = PdArgumentParser((ModelArguments, DataArguments, TrainingArgumentsExt))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 解析api调用参数
    parse_arg(args, model_args, data_args, training_args)

    # 打印Model、Data参数
    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")

    # check输入参数
    print(not os.path.isdir(model_args.model_name_or_path))
    print(model_args.model_name_or_path not in SUPPORTED_MODELS)
    if not os.path.isdir(model_args.model_name_or_path) and model_args.model_name_or_path not in SUPPORTED_MODELS:
        raise ValueError(
            f"{model_args.model_name_or_path} is not a supported model type. Either use a local model path or select a model from {SUPPORTED_MODELS}"
        )

    paddle.set_device(training_args.device)

    # load and preprocess dataset
    label_list = {}
    with open(os.path.join(data_args.data_dir, "label.txt"), "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            l = line.strip()
            label_list[l] = i

    train_ds = load_dataset(
        read_local_dataset, path=os.path.join(data_args.data_dir, "train.txt"), label_list=label_list, lazy=False
    )
    dev_ds = load_dataset(
        read_local_dataset, path=os.path.join(data_args.data_dir, "dev.txt"), label_list=label_list, lazy=False
    )

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    trans_func = functools.partial(preprocess_function, tokenizer=tokenizer, max_seq_length=data_args.max_seq_length)
    train_ds = train_ds.map(trans_func)
    dev_ds = dev_ds.map(trans_func)

    # Define model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path, num_classes=len(label_list)
    )

    # Define the metric function.
    def compute_metrics(eval_preds):
        metric = Accuracy()
        correct = metric.compute(paddle.to_tensor(eval_preds.predictions), paddle.to_tensor(eval_preds.label_ids))
        metric.update(correct)
        acc = metric.accumulate()
        return {"accuracy": acc}

    # Define the early-stopping callback.
    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=data_args.early_stopping_patience, early_stopping_threshold=0.0)
    ]

    # Define loss function
    criterion = paddle.nn.loss.CrossEntropyLoss()

    # Define Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        criterion=criterion,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        callbacks=callbacks,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    if training_args.do_train:
        # modified by zengtao.2023.01.26
        # train_result = trainer.train(resume_from_checkpoint=None)
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        metrics = train_result.metrics
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()


if __name__ == "__main__":
    do_train()
