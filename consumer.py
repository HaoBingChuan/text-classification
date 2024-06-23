# -*- coding: utf-8 -*-

import re
import json
import time
from infra.rabbitmq.rabbitmqClient import RabbitClient
from config.logConfig import logger
from httpx import HTTPTransport
import httpx
from domain.archiveDomain import ArchiveDomain
from utils.threadManager import pooling
from concurrent.futures import as_completed
from producer import send_archives_exchange
from infra.bizException import BizException

from config.base import (
    TIMEOUT,
    RETRIES,
    MULTI_CLASS_URL,
    MULTI_LABEL_URL,
    KEYWORD_SIMILAR_URL,
    NER_URL,
    SUMMARY_URL,
    MOUNT_FOLDER,
)


def post_request(url, param):
    # 定义一个HTTPTransport变量，用于设置重试次数
    transport = HTTPTransport(retries=RETRIES)
    # 定义一个httpx的Client变量，设置超时时间和传输方式
    client = httpx.Client(timeout=TIMEOUT, transport=transport)
    response = client.post(url, json=param)
    text = json.loads(response.text)
    return text


# 定义一个函数来清理文本
def clean_text(text: str):
    text = text.replace("\t", "").replace("\n", "").replace(" ", "")
    # 使用正则表达式去除所有英文字母和数字
    return re.sub(r"[a-zA-Z0-9]", "", text)


def text_summary(text: str, sentence_length: int = 1024, num: int = 30):
    """
    文本摘要
    TODO:需要再考虑num数值大小或更换文本摘要方法
    """
    if len(text) >= sentence_length:
        body = {"data": {"text": text, "num": num}}
        text_summary_info = post_request(url=SUMMARY_URL, param=body)
        summary_result = "".join(text_summary_info["data"])
        return (
            summary_result
            if len(summary_result) <= sentence_length
            else summary_result[:sentence_length]
        )
    else:
        return text


def archive_appraisal(data: ArchiveDomain):
    """
    档案鉴定
    """
    archive_id = data.archive_id
    tm = data.tm
    zrz = data.zrz
    nd = data.nd
    trust = data.trust
    body = {
        "data": {
            "texts": [
                f"{tm}|{zrz}|{nd}",
            ],
            "model_type": "multi_class",
            "trust": trust,
        }
    }
    category_info = post_request(url=MULTI_CLASS_URL, param=body)
    if category_info["code"] == 200:
        assert len(category_info["data"]) == 1
        return category_info["data"][0]
    else:
        logger.error(f"error：{archive_id} 档案鉴定错误!")
        return None


def archive_classification(data: ArchiveDomain):
    """
    档案分类,判断档案多标签类别
    TODO:多标签分类，目前使用threshold = 0.5 来控制，需要考虑兜底方案(按照相似度排序?)
    """
    request = data.request  # 请求数据

    result = {}
    # 获取文本所属类别
    for each in request:
        attached_file_id = each["attached_file_id"]
        text_path = MOUNT_FOLDER + each["text_path"]
        with open(text_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if content:
            content = clean_text(content)
            content = text_summary(content)
            body = {
                "data": {
                    "texts": [f"{content}"],
                    "model_type": "multi_label",
                    "trust": "sent",
                }
            }
            category_rs = post_request(url=MULTI_LABEL_URL, param=body)
            if category_rs["code"] == 200:
                assert len(category_rs["data"]) == 1
                archive_category = category_rs["data"][0]["label"]
                result[attached_file_id] = archive_category
            else:
                result[attached_file_id] = ""
                logger.error(
                    f"error：attached_file_id={attached_file_id} 多标签类别预测错误!"
                )
        else:
            result[attached_file_id] = ""
            logger.error(f"error：attached_file_id={attached_file_id} 文本为空!")
    return result


def archive_rule(data: ArchiveDomain):
    """
    获取规则预定义关键词
    """
    rule_path = MOUNT_FOLDER + data.rule_path
    with open(rule_path, "r", encoding="utf-8") as f:
        rule_data = json.load(f)
    pre_archive_rule = {}
    for first_label, second_label_info in rule_data["rule"].items():
        for second_label, words in second_label_info.items():
            label = second_label
            pre_archive_rule[label] = words
    logger.info(f"预定义关键词类别种类：{pre_archive_rule.keys()}")
    return pre_archive_rule


def match_similar_word(label: str, keywords: list, text: str):
    """
    匹配相似词汇
    """
    body = {
        "data": {
            "keywords": keywords,
            "text": text,
        }
    }
    similar_word_rs = post_request(url=KEYWORD_SIMILAR_URL, param=body)
    result = []
    for each in similar_word_rs["data"]["match_result"]:
        keyword = each["keyword"]
        keyword_category = label
        match_result = [
            {
                "similar_word": w,
                "similar_score": v["similarity_score"],
                "count": v["count"],
            }
            for w, v in each["similar_result"].items()
        ]
        result.append(
            {
                "keyword": keyword,
                "keyword_category": keyword_category,
                "match_result": match_result,
            }
        )
    return result


def archive_keyword_match(
    data: ArchiveDomain, pre_archive_rule: dict, archive_category_info: dict
) -> dict:
    """
    档案业务关键词匹配模块

    :param data: 请求数据
    :param pre_archive_rule: 档案预定义规则
    :param archive_category_info: 档案多标签类别信息
    """
    request = data.request  # 请求数据

    result = {}
    for each in request:

        # 获取文本内容
        text_path = MOUNT_FOLDER + each["text_path"]
        with open(text_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        content = clean_text(content)

        # 获取文本文件id
        attached_file_id = each["attached_file_id"]
        archive_labels = archive_category_info[attached_file_id]
        logger.info(
            f"attached_file_id={attached_file_id} 档案多标签类别：{archive_labels}"
        )
        if archive_labels:
            # 多线程匹配相似词汇
            futures = []
            for label in archive_labels.strip(",").split(","):
                if label in pre_archive_rule:
                    label = label
                    label_keywords = pre_archive_rule[label]
                    text = content
                    futures.append(
                        pooling.submit(match_similar_word, label, label_keywords, text)
                    )
                else:
                    logger.error(f"error：无此类别 {label} 预定义关键词!")

            similar_word_match = []
            for task in as_completed(futures):  # 异步结果获取
                similar_word_match.extend(task.result())
            result[attached_file_id] = similar_word_match
        else:
            result[attached_file_id] = []
            logger.error(
                f"error：attached_file_id={attached_file_id} 档案多标签类别为空!"
            )
    return result


def entity_recognition(data: ArchiveDomain) -> dict:
    """
    实体识别
    """
    request = data.request  # 请求数据
    entity_type = data.entity_type
    result = {}
    for each in request:
        attached_file_id = each["attached_file_id"]
        text_path = MOUNT_FOLDER + each["text_path"]
        with open(text_path, "r") as f:
            content = f.read().strip()
        if content:
            content = clean_text(content)
            body = {"data": {"text": f"{content}", "entity_type": entity_type}}
            entity_recognition_rs = post_request(url=NER_URL, param=body)
            if entity_recognition_rs["code"] == 200:
                result[attached_file_id] = entity_recognition_rs["data"]
            else:
                result[attached_file_id] = []
                logger.error(
                    f"attached_file_id={attached_file_id},error：实体识别错误!"
                )
        else:
            result[attached_file_id] = []
            logger.error(f"attached_file_id={attached_file_id},error：文本为空!")
    return result


def receive_archives_topic(queue):
    logger.info("RabbitMQ topic={} 消费程序启动中...".format(queue))
    RabbitClient().receive(
        queue=queue, durable=True, callback=archives_process, prefetch_count=1
    )


def send_keyword_similarity(data):
    """
    消息结果发送队列
    """
    send_archives_exchange(
        data,
        "exc.nlp.archives",
        "direct",
        "classification.feedback",
    )


def archives_process(ch, method, properties, body):
    # TODO:消息异常，队列堵塞

    processed = False  # 标记消息为已处理
    attempt = 0
    while not processed and attempt < 3:
        attempt += 1
        try:
            input_json = str(body, "utf-8")
            logger.info("消息队列输入数据 = {}".format(input_json))
            # 消息队列数据转换为自定义对象及实体校验
            input_data = ArchiveDomain.dict2entity(json.loads(input_json))
            input_data.check_param()

            process_message(input_data)
            processed = True

        except BizException as be:
            logger.error("archives_process error, ex={}".format(be))
            ch.basic_ack(delivery_tag=method.delivery_tag)  # 确认消息
            break

        except Exception as ex:
            logger.error("archives_process error, ex={}".format(ex))
            if attempt == 3:
                # 如果尝试了3次还是失败，则消息被丢弃
                logger.error("消息处理失败超过3次，删除消息")
                ch.basic_ack(delivery_tag=method.delivery_tag)  # 确认消息
                break
            time.sleep(2)  # 稍作等待后再次尝试

    if processed:
        ch.basic_ack(delivery_tag=method.delivery_tag)  # 确认消息已被消费


def process_message(data: ArchiveDomain) -> dict:
    """
    消息处理主函数
    """
    # 1.  预定义档案规则(即业务类别关键词)
    pre_archive_rule = archive_rule(data)

    # 2.  档案鉴定
    archive_appraisal_rs = archive_appraisal(data)

    # 3.  1)判断档案多标签类别 2)多线程匹配相似词汇
    archive_category_info = archive_classification(data)
    match_similar_word_rs = archive_keyword_match(
        data, pre_archive_rule, archive_category_info
    )

    # 4.  实体识别
    entity_recognition_rs = entity_recognition(data)

    request = data.request  # 请求数据
    archive_analysis = []
    for each in request:
        attached_file_id = each["attached_file_id"]
        archive_analysis.append(
            {
                "attached_file_id": attached_file_id,
                "similar_word_match": match_similar_word_rs[attached_file_id],
                "entity_recognition": entity_recognition_rs[attached_file_id],
            }
        )
    result = {
        "archive_agent_code": data.archive_agent_code,
        "fond_code": data.fond_code,
        "lib_id": data.lib_id,
        "archive_id": data.archive_id,
        "appraisal_id": data.appraisal_id,
        "response": {
            "archive_appraisal": archive_appraisal_rs,
            "archive_analysis": archive_analysis,
        },
    }
    send_keyword_similarity(result)


if __name__ == "__main__":

    # 消息队列消费
    receive_archives_topic("nlp.archives.classification")
