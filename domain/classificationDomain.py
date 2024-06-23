from infra.bizException import BizException


class ClassificationDomain(object):
    """
    classification实体对象
    # trust枚举值:
    # sent:语句级别可解释性(默认值)
    # word:单词级别可解释性
    # model_type:选择模型类型(multi_label or multi_class)
    """

    def __init__(
        self,
        data: str = None,
        texts: str = None,
        trust: str = "sent",
        model_type: str = None,
        confidence_level: bool = False,
    ):
        self.data = data
        self.texts = texts
        self.trust = trust
        self.model_type = model_type
        self.confidence_level = confidence_level

    @staticmethod
    def dict2entity(dic):
        instance = ClassificationDomain()
        if "data" in dic:
            instance.data = dic["data"]
            if "texts" in dic["data"]:
                instance.texts = dic["data"]["texts"]
            if "trust" in dic["data"]:
                instance.trust = dic["data"]["trust"]
            if "model_type" in dic["data"]:
                instance.model_type = dic["data"]["model_type"]
            if "confidence_level" in dic["data"]:
                str2bool = {"True": True, "False": False}
                conf_level = dic["data"]["confidence_level"]
                instance.confidence_level = (
                    str2bool[conf_level] if conf_level in str2bool else conf_level
                )
        return instance

    def check_param(self):
        if not self.data:
            raise BizException("data 不能为空!")
        if not self.texts:
            raise BizException("texts 不能为空!")
        if self.trust not in ["sent", "word"]:
            raise BizException("trust 参数错误!")
        if self.model_type not in ["multi_label", "multi_class"]:
            raise BizException("model_type 参数错误!")
        if self.confidence_level not in [True, False]:
            raise BizException("confidence_level 参数错误!")
