import os, sys

sys.path.append(os.getcwd())
import functools
import paddle
import paddle.nn.functional as F
from paddle.io import BatchSampler, DataLoader
from utils.multi_label.utils import preprocess_function
from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer
from paddlenlp.utils.log import logger
from trustai.interpretation import FeatureSimilarityModel
from utils.threadManager import pooling
from analysis.interpret import find_sim_data
from analysis.interpret import find_key_word
from paddle.fluid.dataloader import Dataset
from concurrent.futures import as_completed
from utils.multi_class.util import read_list
from pprint import pprint
from utils.modelManager import ModelManager, ModelPair
from config.base import DEVICE

"""预测模型参数配置"""
MAX_SEQ_LENGTH = 1024
BATCH_SIZE = 1

# set device
paddle.set_device(DEVICE)


@paddle.no_grad()
def multi_label_predict(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    train_ds: Dataset = None,
    label_list: dict = None,
    texts: list = None,
    feature_sim: FeatureSimilarityModel = None,
    confidence_level=False,
):
    """
    预测多标签数据所属类别
    Args:
        model: 待加载模型,
        tokenizer: Token向量化,
        train_ds: 训练数据集,
        label_list: 全部标签类别,
        texts: 待预测文本数据,
        feature_sim: 特征相似度模型
    Returns:
        list: 多标签预测结果,包含句子和单词级别可解释信息
    """
    label_list = [label for label, _ in label_list.items()]
    # 处理输入预测数据
    data_ds = load_dataset(read_list, data_list=list(texts), lazy=False, is_test=True)

    trans_func = functools.partial(
        preprocess_function,
        tokenizer=tokenizer,
        max_seq_length=MAX_SEQ_LENGTH,
        label_nums=len(label_list),
        is_test=True,
    )
    predict_data_ds = data_ds.map(trans_func)
    # batchify dataset
    collate_fn = DataCollatorWithPadding(tokenizer)
    data_batch_sampler = BatchSampler(
        predict_data_ds, batch_size=BATCH_SIZE, shuffle=False
    )

    data_data_loader = DataLoader(
        dataset=predict_data_ds, batch_sampler=data_batch_sampler, collate_fn=collate_fn
    )

    tag_list = []
    confidence_levels = []
    model.eval()
    for batch in data_data_loader:
        logits = model(**batch)
        probs = F.sigmoid(logits).numpy()
        for prob in probs:
            labels = []
            confidence_level_list = []
            for i, p in enumerate(prob):
                if p > 0.5:
                    labels.append(i)
                    confidence_level_list.append(p)
            tag_list.append(",".join([label_list[r] for r in labels]))
            confidence_levels.append(confidence_level_list)

    if confidence_level:
        rs_data = []
        for txt, tag, confidence_level_score in list(
            zip(texts, tag_list, confidence_levels)
        ):
            predict = []
            for label, score in zip(label_list, confidence_level_score):
                predict.append({"label": label, "score": float(score)})
            dic = {
                "text": txt,
                "predict": predict,
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
            trust_list = task.result()

        rs_data = []
        for txt, tag, trust in list(zip(texts, tag_list, trust_list)):
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


if __name__ == "__main__":
    data_dic = {
        "data": {
            "texts": [
                "2、被告每月支付小孩抚养费1200元",
                "被告于2016年3月将车牌号为皖B×××××出售了2.7万元，被告通过原告偿还了齐荷花人民币2.6万元，原、被告尚欠齐荷花2万元。",
            ],
            "trust": "word",
        }
    }
    manager = ModelManager("./checkpoint/", "./data", ["divorce"])
    model_pair: ModelPair = manager.find_model("divorce")
    texts = data_dic["data"]["texts"]
    trust = data_dic["trust"] if "trust" in data_dic else "sent"
    data = multi_label_predict(
        model=model_pair.model,
        tokenizer=model_pair.tok,
        label_list=model_pair.label_list,
        texts=texts,
        train_ds=model_pair.train_ds,
        feature_sim=model_pair.feature_sim,
        # confidence_level=True,
    )
    pprint(data)
