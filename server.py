import json
from flask import Flask, request, jsonify
from algo.multi_class.predict import predict as paddle_predict
from utils.modelManager import ModelManager
from utils.threadManager import pooling
from config.logConfig import logger
from producer import send_archives_exchange
from retraining import train_flow
from algo.multi_label.predict import multi_label_predict
from paddlenlp import Taskflow
from domain.classificationDomain import ClassificationDomain
from domain.retrainingDomain import RetrainingDomain
from infra.redis.redisPool import *
from infra.bizException import BizException

app = Flask(__name__)

# 模型预加载
manager = ModelManager(
    "./checkpoint/",
    "./data",
    [
        "divorce",
        "control_code",
        "bgqx_code",
    ],
)
ner = Taskflow("ner", entity_only=True, device_id=DEVICE_ID)
summary = Taskflow("text_summarization", max_target_length=1024, device_id=DEVICE_ID)
similarity = Taskflow("text_similarity", device_id=DEVICE_ID)
corrector = Taskflow("text_correction", device_id=DEVICE_ID)
# Redis线程池
redis_pool = RedisPool()


@app.errorhandler(BizException)
def server_exception(ex: BizException):
    return jsonify(get_error(message=ex.message))


@app.before_request
def before_request():
    logger.info("before_request")


@app.route("/hello/beat/")
def beat():
    return "Hello classification-helper"


@app.route("/api/cls/predict/<cls>", methods=["POST"])
def predict(cls):
    """
    自动鉴定-预测接口
    Args:
        cls:类别(control_code,bgqx_code,divorce)
    Returns:
    """
    req_data = request.get_data(as_text=True)
    logger.info("predict 请求参数: cls={}, param ={} ".format(cls, req_data))
    try:
        input_data = ClassificationDomain.dict2entity(json.loads(req_data))
        input_data.check_param()
    except Exception as ex:
        logger.error("json格式错误, 请检查参数格式！ ex={}".format(ex))
        return jsonify(get_error(message="json格式错误, 请检查参数格式!"))

    texts = input_data.texts
    trust = input_data.trust
    model_type = input_data.model_type
    model_pair = manager.find_model(cls)

    if model_pair:
        if model_type == "multi_label":
            data = multi_label_predict(
                model_pair.model,
                model_pair.tok,
                model_pair.train_ds,
                model_pair.label_list,
                texts,
                None if "word" == trust else model_pair.feature_sim,
            )
            rs_data = {"data": data, "code": 200, "message": "处理成功"}
            return jsonify(rs_data)
        elif model_type == "multi_class":
            data = paddle_predict(
                model_pair.model,
                model_pair.tok,
                model_pair.train_ds,
                model_pair.label_list,
                texts,
                None if "word" == trust else model_pair.feature_sim,
            )
            rs_data = {"data": data, "code": 200, "message": "处理成功"}
            return jsonify(rs_data)
    else:
        return jsonify(get_error(message="模型不存在!"))


@app.route("/api/confidence-level/predict/<cls>", methods=["POST"])
def confidence_level(cls):
    """
    自动鉴定-置信度预测接口
    Args:
        cls:类别(control_code,bgqx_code,divorce)
    Returns:
    """
    req_data = request.get_data(as_text=True)
    logger.info("predict 请求参数: cls={}, param ={} ".format(cls, req_data))
    try:
        input_data = ClassificationDomain.dict2entity(json.loads(req_data))
    except Exception as ex:
        logger.error("json格式错误, 请检查参数格式！ ex={}".format(ex))
        return jsonify(get_error(message="json格式错误, 请检查参数格式!"))
    input_data.check_param()
    texts = input_data.texts
    trust = input_data.trust
    model_type = input_data.model_type
    confidence_level = input_data.confidence_level
    model_pair = manager.find_model(cls)

    if model_pair:
        if model_type == "multi_label":
            data = multi_label_predict(
                model_pair.model,
                model_pair.tok,
                model_pair.train_ds,
                model_pair.label_list,
                texts,
                None if "word" == trust else model_pair.feature_sim,
                confidence_level,
            )
            rs_data = {"data": data, "code": 200, "message": "处理成功"}
            return jsonify(rs_data)
        elif model_type == "multi_class":
            data = paddle_predict(
                model_pair.model,
                model_pair.tok,
                model_pair.train_ds,
                model_pair.label_list,
                texts,
                None if "word" == trust else model_pair.feature_sim,
                confidence_level,
            )
            rs_data = {"data": data, "code": 200, "message": "处理成功"}
            return jsonify(rs_data)
    else:
        return jsonify(get_error(message="模型不存在!"))


@app.route("/api/cls/feedback/<cls>", methods=["POST"])
def feedback(cls):
    """
    自动鉴定-结果反馈接口
    Args:
        cls:类别(control_code,bgqx_code)
    Returns:
    """
    req_data = request.get_data(as_text=True)
    logger.info("feedback 请求参数: cls={}, param ={} ".format(cls, req_data))
    input_json = json.loads(req_data)
    if "data" in input_json:
        data_dic = input_json["data"]
        data_dic["pre_type"] = cls
        data_dic["is_feedback"] = 1
        rs = send_archives_exchange(data_dic)
        rs_data = {"data": rs, "code": 200, "message": "处理成功"}
        return jsonify(rs_data)
    return jsonify({"code": -1, "message": "data不能为空!"})


@app.route("/api/cls/retraining/<cls>", methods=["POST"])
def retraining(cls):
    """
    重新训练
    请求数据样例：{"data":{"cls":"divorce"}}
    """
    req_data = request.get_data(as_text=True)
    logger.info("predict 请求参数: cls={}, param ={} ".format(cls, req_data))
    input_data = RetrainingDomain.dict2entity(json.loads(req_data))
    input_data.check_param()
    model_type = input_data.model_type
    # Redis避免重复提交
    redis_conn = redis_pool.get_conn()
    key = "training_" + cls
    success = redis_conn.setnx(key, 1)
    if success:
        # Redis过期时间设置为24小时
        # redis_conn.expire(key, 24 * 3600)
        pooling.submit(train_flow, cls, model_type, key, redis_pool)
        rs_data = {"code": 200, "message": "处理成功", "cls": cls}
        return jsonify(rs_data)
    else:
        rs_data = {"code": 200, "message": "重复提交", "cls": cls}
        return jsonify(rs_data)


@app.route("/api/cls/reload/<cls>", methods=["POST"])
def reload(cls):
    """
    模型重新导入
    Args:
        cls:类别(control_code,bgqx_code)
    Returns:
    """
    manager.clear_cache(cls)
    manager.reload(cls)
    rs_data = {"code": 200, "message": "处理成功"}
    return jsonify(rs_data)


@app.route("/api/ner", methods=["POST"])
def entity_recognition():
    """
    命名实体识别
    请求示例:
    {
        "data": {
            "text": "《孤女》是2010年九州出版社出版的小说，作者是余兼羽"
        }
    }
    """
    req_data = request.get_data(as_text=True)
    input_json = json.loads(req_data)
    if "data" in input_json:
        data_dic = input_json["data"]
        if "text" in data_dic:
            text = data_dic["text"]
            if text:
                rs_data = ner(text)
                return jsonify(rs_data)
            else:
                return jsonify({"code": -1, "message": "text 不能为空!"})
    return jsonify({"code": -1, "message": "处理异常"})


@app.route("/api/text_similarity", methods=["POST"])
def text_similarity():
    """
    文本相似度计算
    请求示例:
    {
        "data": {
            "text": [["春天适合种什么花？","春天适合种什么菜？"],["谁有狂三这张高清的","这张高清图，谁有"]]
        }
    }
    """
    req_data = request.get_data(as_text=True)
    input_json = json.loads(req_data)
    if "data" in input_json:
        data_dic = input_json["data"]
        if "text" in data_dic:
            text = data_dic["text"]
            if text:
                rs_data = similarity(text)
                return jsonify(rs_data)
            else:
                return jsonify({"code": -1, "message": "text 不能为空!"})
    return jsonify({"code": -1, "message": "处理异常"})


@app.route("/api/text_summary", methods=["POST"])
def text_summary():
    """
    文本摘要提取
    请求示例:
    {
        "data": {
            "text": "2022年，中国房地产进入转型阵痛期，传统“高杠杆、快周转”的模式难以为继，万科甚至直接喊话，中国房地产进入“黑铁时代”"
        }
    }
    """
    req_data = request.get_data(as_text=True)
    input_json = json.loads(req_data)
    if "data" in input_json:
        data_dic = input_json["data"]
        if "text" in data_dic:
            text = data_dic["text"]
            if text:
                rs_data = summary(text)
                return jsonify(rs_data)
            else:
                return jsonify({"code": -1, "message": "text 不能为空!"})
    return jsonify({"code": -1, "message": "处理异常"})


@app.route("/api/text_correct", methods=["POST"])
def text_correct():
    """
    文本纠错接口
    测试样例：
    {"text":"我从北京南做高铁到南京南"}
    """
    req_data = request.get_data(as_text=True)
    logger.info("text_correct 请求数据: param ={} ".format(req_data))
    input_json = json.loads(req_data)
    text = input_json["text"]
    rs_data = corrector(text)
    return jsonify(rs_data[0])


def get_error(code=-1, message=""):
    return {"code": code, "message": message}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8090)

    # server = pywsgi.WSGIServer(('0.0.0.0', 8090), app)
    # server.serve_forever()
