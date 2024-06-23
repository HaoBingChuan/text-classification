import sys
import os

sys.path.append(os.getcwd())
from utils.multi_class.util import *
from config.base import *
from algo.multi_class.train import do_train as multi_class_train
from algo.multi_label.train import train as multi_label_train
from infra.redis.redisPool import RedisPool
from config.base import DEVICE
from infra.bizException import BizException


def pre_train(cls: str):
    """
    数据库中读取训练数据
    Args:
        cls:
    Returns:

    """
    sql = "select text, label from classification_context where is_del =0 and pre_type='{}'"
    data_df = read_context_from_db(CONTEXT_DB_URL, sql.format(cls))
    export_data("./data/" + cls, data_df)


def retrain(cls: str, model_type: str):
    """
    ERNIE 模型训练
    Args:
        cls: 训练类别：bgqx_code\control_code\divorce
    Returns:

    """
    if model_type == "multi_class":
        args_dic = dict()
        args_dic["model_name_or_path"] = "ernie-3.0-medium-zh"
        args_dic["data_dir"] = "./data/" + cls
        args_dic["output_dir"] = "./checkpoint/" + cls
        args_dic["device"] = DEVICE
        args_dic["learning_rate"] = 3e-5
        args_dic["early_stopping_patience"] = 4
        args_dic["max_seq_length"] = 256
        args_dic["per_device_eval_batch_size"] = 32
        args_dic["per_device_train_batch_size"] = 32
        args_dic["num_train_epochs"] = 10
        args_dic["do_train"] = True
        args_dic["do_eval"] = True
        args_dic["metric_for_best_model"] = "accuracy"
        args_dic["load_best_model_at_end"] = True
        args_dic["evaluation_strategy"] = "epoch"
        args_dic["save_strategy"] = "epoch"
        args_dic["save_total_limit"] = 1
        multi_class_train(**args_dic)
    elif model_type == "multi_label":
        args_dic = dict()
        args_dic["device"] = DEVICE
        args_dic["dataset_dir"] = "./data/" + cls
        args_dic["save_dir"] = "./checkpoint/" + cls
        args_dic["max_seq_length"] = 128
        args_dic["model_name"] = "ernie-3.0-medium-zh"
        args_dic["batch_size"] = 32
        args_dic["learning_rate"] = 3e-5
        args_dic["epochs"] = 10
        args_dic["early_stop"] = "early_stop"
        args_dic["early_stop_nums"] = 3
        args_dic["logging_steps"] = 5
        args_dic["weight_decay"] = 0.0
        args_dic["warmup"] = "warmup"
        args_dic["init_from_ckpt"] = None
        args_dic["seed"] = 3
        args_dic["train_file"] = "train.txt"
        args_dic["dev_file"] = "dev.txt"
        args_dic["label_file"] = "label.txt"
        multi_label_train(**args_dic)


def post_train(cls: str):
    pass


def train_flow(cls: str, model_type: str, key: str, redis_pool: RedisPool):
    """
    训练主流程
    Args:
        cls:
        key:
        redis_pool:
    Returns:
    """
    # pre_train(cls)
    retrain(cls, model_type)
    post_train(cls)
    conn = redis_pool.get_conn()
    conn.delete(key)


if __name__ == "__main__":
    cls_code = "divorce"
    retrain(cls=cls_code,model_type="multi_label")
