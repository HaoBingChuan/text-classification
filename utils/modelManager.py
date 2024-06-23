import os
import time
import functools
from paddle.io import BatchSampler, DataLoader
from paddle.fluid.dataloader import Dataset
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer
from paddlenlp.datasets import load_dataset
from trustai.interpretation import FeatureSimilarityModel
from utils.multi_class.util import (
    read_sim_dataset,
    LocalDataCollatorWithPadding,
    preprocess_function,
)
from config.logConfig import logger


class ModelPair(object):

    def __init__(
        self,
        model: AutoModelForSequenceClassification,
        tok: AutoTokenizer,
        train_ds: Dataset,
        label_list: dict,
        feature_sim: FeatureSimilarityModel,
    ):
        self.model = model
        self.tok = tok
        self.train_ds = train_ds
        self.label_list = label_list
        self.feature_sim = feature_sim


class ModelManager(object):
    dic_model = {}
    model_path = None
    data_path = None

    def __init__(self, model_path: str, data_path: str, codes: list):
        self.model_path = model_path
        self.data_path = data_path
        for item in codes:
            self.reload(item)

    def clear_cache(self, code: str):
        os.remove(self.model_path + code + "/cache_feature.pdparams")

    def reload(self, code: str, max_seq_length=256):

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path + code
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_path + code)

        # 标签分类读取
        # key->value
        label_list = {}
        label_path = os.path.join(self.data_path, code, "label.txt")
        with open(label_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                tag = line.strip()
                label_list[tag] = i

        # 训练数据读取(可解释性使用)
        train_path = os.path.join(self.data_path, code, "train.txt")
        train_ds = load_dataset(
            read_sim_dataset, path=train_path, label_list=label_list, lazy=False
        )

        # 处理训练数据，生成特征模型(is_test=True)
        trans_func = functools.partial(
            preprocess_function,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            is_test=True,
        )
        collate_fn = LocalDataCollatorWithPadding(tokenizer)
        train_ds = train_ds.map(trans_func)
        train_batch_sampler = BatchSampler(train_ds, batch_size=16, shuffle=False)
        train_data_loader = DataLoader(
            dataset=train_ds, batch_sampler=train_batch_sampler, collate_fn=collate_fn
        )
        t = time.time()
        feature_sim = FeatureSimilarityModel(
            model,
            train_data_loader,
            classifier_layer_name="classifier",
            cached_train_feature=self.model_path + code + "/cache_feature.pdparams",
        )
        logger.info(f"feature_sim coast:{time.time() - t:.8f}s")
        # feature_sim = None
        model_pair = ModelPair(model, tokenizer, train_ds, label_list, feature_sim)
        self.dic_model[code] = model_pair

    def find_model(self, code: str):
        if code in self.dic_model:
            return self.dic_model[code]
        return None
