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
import os.path

import numpy as np
import pandas as pd
from paddlenlp.data import DataCollatorWithPadding
from infra.mysql.mysqlPool import MysqlPool
from sklearn import model_selection


class LocalDataCollatorWithPadding(DataCollatorWithPadding):
    """
    Convert the  result of DataCollatorWithPadding from dict dictionary to a list
    """

    def __call__(self, features):
        batch = super().__call__(features)
        batch = list(batch.values())
        return batch


def preprocess_function(examples, tokenizer, max_seq_length, is_test=False):
    """
    Builds model inputs from a sequence for sequence classification tasks
    by concatenating and adding special tokens.

    Args:
        examples(obj:`list[str]`): List of input data, containing text and label if it have label.
        tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from :class:`~paddlenlp.transformers.PretrainedTokenizer`
            which contains most of the methods. Users should refer to the superclass for more information regarding methods.
        max_seq_length(obj:`int`): The maximum total input sequence length after tokenization.
            Sequences longer than this will be truncated, sequences shorter will be padded.
        label_nums(obj:`int`): The number of the labels.
    Returns:
        result(obj:`dict`): The preprocessed data including input_ids, token_type_ids, labels.
    """
    result = tokenizer(text=examples["text"], max_seq_len=max_seq_length)
    if not is_test:
        result["labels"] = np.array([examples["label"]], dtype="int64")
    return result


def read_sim_dataset(path):
    """
    Read dataset file
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            items = line.strip().split("\t")
            if items[0] == "Text":
                continue
            if len(items) == 3:
                yield {"text": items[0], "label": items[1], "archiveId": items[2]}
            elif len(items) == 2:
                yield {"text": items[0], "label": items[1], "archiveId": ""}
            elif len(items) == 1:
                yield {"text": items[0], "label": "", "archiveId": ""}
            else:
                raise ValueError("{} should be in fixed format.".format(path))


def read_local_dataset(path, label_list=None, is_test=False):
    """
    Read dataset
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if is_test:
                items = line.strip().split("\t")
                sentence = "".join(items)
                yield {"text": sentence}
            else:
                items = line.strip().split("\t")
                sentence = "".join(items[:-1])
                tag = items[-1]
                yield {"text": sentence, "label": label_list[tag]}


def read_list(data_list=None, label_list=None, is_test=False):
    for item in data_list:
        yield {"text": item}


def export_data(output_file: str, data_df: pd.DataFrame):
    text = data_df[["text"]]
    label = data_df["label"]
    # x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1, random_state=1234)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(text, label, test_size=0.1)
    # 输出训练集
    with open(os.path.join(output_file, "train.txt"), "w+", encoding="utf-8") as f:
        # 标题不需要输出
        # f.write("text" + "\t" + "label" + "\n")
        for idx, row in x_train.iterrows():
            tag = y_train[idx]
            f.write(row["text"] + "\t" + tag + "\n")

    # 输出测试集
    with open(os.path.join(output_file, "dev.txt"), "w+", encoding="utf-8") as f:
        # 标题不需要输出
        # f.write("text" + "\t" + "label" + "\n")
        for idx, row in x_test.iterrows():
            tag = y_test[idx]
            f.write(row["text"] + "\t" + tag + "\n")


def read_context_from_db(url: str, sql: str):
    pool = MysqlPool(url)
    data = pool.fetch_all(sql)
    df = pd.DataFrame(data)
    return df


class LocalDataCollatorWithPadding(DataCollatorWithPadding):
    """
    Convert the  result of DataCollatorWithPadding from dict dictionary to a list
    """

    def __call__(self, features):
        batch = super().__call__(features)
        batch = list(batch.values())
        return batch