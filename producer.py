# -*- coding: utf-8 -*-

import json
from infra.rabbitmq.rabbitmqClient import RabbitClient
from config.logConfig import logger


def send_archives_exchange(
    msg: dict, exchange: str, exchange_type: str, routing_key: str
):
    RabbitClient().send(exchange, exchange_type, routing_key, json.dumps(msg), True, 2)
    logger.info("消息发送成功, msg={}".format(msg))
    return True


if __name__ == "__main__":

    test = {
        "archive_agent_code": "999001",
        "fond_code": "0",
        "lib_id": 3,
        "archive_id": 11308,
        "appraisal_id": 184,
        "tm": "刘杰测试档案",
        "zrz": "1",
        "nd": "2024",
        "trust": "sent",
        "rule_path": "/adms/appraisal/sensitiveWord.json",
        "request": [
            {
                "attached_file_id": 1217,
                "text_path": "/adms/appraisal/20240428/1217.txt",
            },
            {
                "attached_file_id": 1218,
                "text_path": "/adms/appraisal/20240428/1218.txt",
            },
            {
                "attached_file_id": 640,
                "text_path": "/adms/appraisal/20240508/640.txt",
            },
        ],
    }

    multi_label_param = {
        "archive_agent_code": "999001",
        "archive_id": 18,
        "appraisal_id": 1,
        "fond_code": "0",
        "lib_id": 2,
        "nd": "2024",
        "trust": "sent",
        "request": [
            {"attached_file_id": 2, "text_path": "/adms/appraisal/20240421/2.txt"},
            {"attached_file_id": 3, "text_path": "/adms/appraisal/20240421/3.txt"},
            {"attached_file_id": 4, "text_path": "/adms/appraisal/20240421/4.txt"},
            {"attached_file_id": 5, "text_path": "/adms/appraisal/20240421/5.txt"},
            {"attached_file_id": 6, "text_path": "/adms/appraisal/20240421/6.txt"},
            {"attached_file_id": 7, "text_path": "/adms/appraisal/20240421/7.txt"},
            {"attached_file_id": 8, "text_path": "/adms/appraisal/20240421/8.txt"},
        ],
        "rule_path": "/adms/appraisal/keyword_test.json",
        "tm": "test",
        "zrz": "test",
    }
    # 消息发送
    send_archives_exchange(test, "exc.nlp.archives", "direct", "classification")
