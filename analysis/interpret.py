import argparse
import os
import time
import random
import numpy as np
import jieba
import paddle
from paddle.fluid.dataloader import Dataset
from paddle.io import DataLoader, BatchSampler
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer
from trustai.interpretation import get_word_offset
from trustai.interpretation import FeatureSimilarityModel
from utils.multi_class.util import LocalDataCollatorWithPadding
from config.logConfig import logger
from config.base import DEVICE

# 可选 "ig","lime","grad" ,可以根据实际任务效果选择解释器
INTERPRETER = "ig"

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument('--device', default=DEVICE, help="Select which device to train model, defaults to gpu.")
parser.add_argument("--dataset_dir", default='./data', type=str, help="The dataset directory should include train.txt,dev.txt and test.txt files.")
parser.add_argument("--params_path", default="../checkpoint/", type=str, help="The path to model parameters to be loaded.")
parser.add_argument("--max_seq_length", default=256, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", default=1, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument("--seed", type=int, default=3, help="random seed for initialization")
parser.add_argument("--top_k", type=int, default=5, help="Top K important training data.")
parser.add_argument("--train_file", type=str, default="train.txt", help="Train dataset file name")
parser.add_argument("--interpret_input_file", type=str, default="bad_case.txt", help="interpretation file name")
parser.add_argument("--interpret_result_file", type=str, default="sent_interpret.txt", help="interpreted file name")
args = parser.parse_args()
# yapf: enable


def set_seed(seed):
    """
    Set random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def find_sim_data(
    tokenizer: AutoTokenizer,
    train_ds: Dataset,
    predict_ds: Dataset,
    feature_sim: FeatureSimilarityModel,
):
    set_seed(args.seed)
    paddle.set_device(args.device)

    # 处理预测数据集(外部已经执行map)
    # predict_ds = predict_ds.map(trans_func)
    predict_batch_sampler = BatchSampler(
        predict_ds, batch_size=args.batch_size, shuffle=False
    )
    collate_fn = LocalDataCollatorWithPadding(tokenizer)
    predict_data_loader = DataLoader(
        dataset=predict_ds, batch_sampler=predict_batch_sampler, collate_fn=collate_fn
    )

    sim_lst = []
    for batch in predict_data_loader:
        sim_batch = []
        analysis = feature_sim(batch, sample_num=args.top_k)
        # 这个循环仅处理结构约束(无实际意义)
        for item in analysis:
            for i, (idx, score) in enumerate(zip(item.pos_indexes, item.pos_scores)):
                sim_dic = {}
                sim = train_ds.data[idx]
                logger.info(
                    "support idx: {} text: {} \t label: {} \t score: {:.5f} \n".format(
                        i + 1, sim["text"], sim["label"], score
                    )
                )
                sim_dic["text"] = sim["text"]
                sim_dic["label"] = sim["label"]
                sim_dic["archiveId"] = sim["archiveId"]
                sim_dic["score"] = str(score)

                sim_batch.append(sim_dic)
        sim_lst.append(sim_batch)
    return sim_lst


def find_key_word(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    predict_ds: Dataset,
):
    paddle.set_device(args.device)
    t = time.time()

    # 处理预测数据集(外部已经执行map)
    # predict_ds = predict_ds.map(trans_func)
    predict_batch_sampler = BatchSampler(
        predict_ds, batch_size=args.batch_size, shuffle=False
    )
    collate_fn = LocalDataCollatorWithPadding(tokenizer)
    predict_data_loader = DataLoader(
        dataset=predict_ds, batch_sampler=predict_batch_sampler, collate_fn=collate_fn
    )

    # Init an interpreter
    if INTERPRETER == "ig":
        from trustai.interpretation.token_level import IntGradInterpreter

        interpreter = IntGradInterpreter(model)
    elif INTERPRETER == "lime":
        from trustai.interpretation.token_level import LIMEInterpreter

        interpreter = LIMEInterpreter(
            model,
            unk_id=tokenizer.convert_tokens_to_ids("[UNK]"),
            pad_id=tokenizer.convert_tokens_to_ids("[PAD]"),
        )
    else:
        from trustai.interpretation.token_level import GradShapInterpreter

        interpreter = GradShapInterpreter(model)

    # Use interpreter to get the importance scores for all data
    logger.info("Start token level interpretion, it will take some time...")
    analysis_result = []
    for batch in predict_data_loader:
        analysis_result += interpreter(tuple(batch))

    # Add CLS and SEP tags to both original text and standard splited tokens
    contexts = []
    words = []
    for i in range(len(predict_ds)):
        text = predict_ds.data[i]["text"]
        contexts.append("[CLS]" + text + "[SEP]")
        words.append(["[CLS]"] + list(jieba.cut(text)) + ["[SEP]"])

    # Get the offset map of tokenized tokens and standard splited tokens
    logger.info("Start word level alignment, it will take some time...")
    ori_offset_maps = []
    word_offset_maps = []
    for i in range(len(contexts)):
        ori_offset_maps.append(tokenizer.get_offset_mapping(contexts[i]))
        word_offset_maps.append(get_word_offset(contexts[i], words[i]))

    align_rs = interpreter.alignment(
        analysis_result,
        contexts,
        words,
        word_offset_maps,
        ori_offset_maps,
        special_tokens=["[CLS]", "[SEP]"],
        rationale_num=args.top_k,
    )

    logger.info(f"find_key_word coast:{time.time() - t:.8f}s")
    return align_rs
