from infra.bizException import BizException


class RetrainingDomain(object):
    """
    retraining实体对象
    """

    def __init__(
        self,
        data: str = None,
        model_type: str = None,
    ):
        self.data = data
        self.model_type = model_type

    @staticmethod
    def dict2entity(dic):
        instance = RetrainingDomain()
        if "data" in dic:
            instance.data = dic["data"]
            if "model_type" in dic["data"]:
                instance.model_type = dic["data"]["model_type"]
        return instance

    def check_param(self):
        if not self.data:
            raise BizException("data 不能为空!")
        if self.model_type not in ["multi_class", "multi_label"]:
            raise BizException("model_type参数类型不对!")
