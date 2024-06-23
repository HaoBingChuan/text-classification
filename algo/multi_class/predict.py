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
import os

import argparse
import functools
from concurrent.futures import as_completed
import paddle
import paddle.nn.functional as F
from paddle.io import DataLoader, BatchSampler
from paddle.fluid.dataloader import Dataset
from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils.multi_class.util import preprocess_function, read_list
from analysis.interpret import find_sim_data
from analysis.interpret import find_key_word
from trustai.interpretation import FeatureSimilarityModel
from config.logConfig import logger
from config.base import DEVICE
from utils.threadManager import pooling

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument('--device', default=DEVICE, help="Select which device to train model, defaults to gpu.")
parser.add_argument("--dataset_dir", default='./data', type=str,
                    help="Local dataset directory should include data.txt and label.txt")
parser.add_argument("--output_file", default="output.txt", type=str, help="Save prediction result")
parser.add_argument("--params_path", default="./checkpoint/", type=str,
                    help="The path to model parameters to be loaded.")
parser.add_argument("--max_seq_length", default=1024, type=int,
                    help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", default=1, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument("--data_file", type=str, default="data.txt", help="Unlabeled data file name")
parser.add_argument("--label_file", type=str, default="label.txt", help="Label file name")
args = parser.parse_args()

# set device
paddle.set_device(args.device)


@paddle.no_grad()
def predict(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    train_ds: Dataset,
    label_list: dict,
    texts: list,
    feature_sim: FeatureSimilarityModel = None,
    confidence_level=False,
):
    """
    Args:
        model: 模型结构
        tokenizer: 模型
        train_ds: 训练数据集合
        label_list: 分类名称列表
        texts: 预测文本
        feature_sim: 相似特征样本
    Returns:
    """

    # 处理输入预测数据
    predict_data_ds = load_dataset(
        read_list, data_list=list(texts), lazy=False, is_test=True
    )
    trans_func = functools.partial(
        preprocess_function,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        is_test=True,
    )
    predict_data_ds = predict_data_ds.map(trans_func)

    # batchify dataset
    collate_fn = DataCollatorWithPadding(tokenizer)
    data_batch_sampler = BatchSampler(
        predict_data_ds, batch_size=args.batch_size, shuffle=False
    )
    predict_data_loader = DataLoader(
        dataset=predict_data_ds, batch_sampler=data_batch_sampler, collate_fn=collate_fn
    )

    # 反转label_list,便于预测结果输出
    label_reverse = {v: k for k, v in label_list.items()}

    tag_list = []
    confidence_levels = []
    model.eval()
    for batch in predict_data_loader:
        logits = model(**batch)
        probs = F.softmax(logits, axis=1)
        idx = paddle.argmax(probs, axis=1).numpy()
        idx = idx.tolist()
        for i, p in zip(idx, probs.numpy()):
            confidence_levels.append(p[i])
        tag = [label_reverse[i] for i in idx]
        tag_list.append(tag[0])

    if confidence_level:
        rs_data = []
        for txt, tag, confidence_level_score in list(
            zip(texts, tag_list, confidence_levels)
        ):
            dic = {
                "text": txt,
                "predict": [{"label": tag, "score": float(confidence_level_score)}],
            }
            rs_data.append(dic)
        logger.info("预测返回结果: {}".format(rs_data))
        return rs_data
    else:
        # 多线程不受no_grad注解的约束
        if feature_sim:
            # 语句级别可解释
            async_trust = [
                pooling.submit(
                    find_sim_data, tokenizer, train_ds, predict_data_ds, feature_sim
                )
            ]
        else:
            # 单词级别可解释
            async_trust = [
                pooling.submit(find_key_word, model, tokenizer, predict_data_ds)
            ]

        # 异步结果获取
        for task in as_completed(async_trust):
            trust_lst = task.result()

        rs_data = []
        for txt, tag, trust in list(zip(texts, tag_list, trust_lst)):
            if feature_sim:
                dic = {"text": txt, "label": tag, "sim": trust}
            else:
                dic = {
                    "text": txt,
                    "label": tag,
                    "words": Visualization(trust).output_record(),
                }
            rs_data.append(dic)

        logger.info("预测返回结果: {}".format(rs_data))
        return rs_data


class Visualization(object):
    def __init__(self, word_res):
        self.words = word_res.words
        self.key_words = set(word_res.rationale_tokens)
        word_attributions = word_res.word_attributions
        _max = max(word_attributions)
        _min = min(word_attributions)
        self.word_attributions = [
            (word_imp - _min) / (_max - _min) for word_imp in word_attributions
        ]

    def _background_color(self, importance):
        importance = max(-1, min(1, importance))
        if importance > 0:
            hue = 120
            sat = 75
            lig = 100 - int(30 * importance)
        else:
            hue = 0
            sat = 75
            lig = 100 - int(-40 * importance)
        return "hsl({}, {}%, {}%)".format(hue, sat, lig)

    def output_record(self):
        res = []
        for word, importance in zip(
            self.words, self.word_attributions[: len(self.words)]
        ):
            color = self._background_color(importance)
            # unwrapped_tag = '<mark style="background-color: {color}; opacity:1.0; \
            #             line-height:1.75"><font color="black"> {word}\
            #             </font></mark>' \
            #             .format(color=color, word=word)
            tag_dic = {
                "background_color": color,
                "word": word,
                "key_word": 1 if word in self.key_words else 0,
            }
            res.append(tag_dic)
        return res
