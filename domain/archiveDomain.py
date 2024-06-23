from infra.bizException import BizException


class ArchiveDomain(object):
    """
    业务请求数据实体对象
    """

    def __init__(
        self,
        archive_agent_code: str = None,
        fond_code: str = None,
        lib_id: int = None,
        archive_id: int = None,
        appraisal_id: int = None,
        tm: str = None,
        zrz: str = None,
        nd: str = None,
        trust: str = None,
        entity_type: list = ["PERSON"],
        rule_path: str = None,
        request: list = [],
    ):
        self.archive_agent_code = archive_agent_code
        self.fond_code = fond_code
        self.lib_id = lib_id
        self.archive_id = archive_id
        self.appraisal_id = appraisal_id
        self.tm = tm
        self.zrz = zrz
        self.nd = nd
        self.trust = trust
        self.entity_type = entity_type
        self.rule_path = rule_path
        self.request = request

    @staticmethod
    def dict2entity(dic):
        instance = ArchiveDomain()
        if "archive_agent_code" in dic:
            instance.archive_agent_code = dic["archive_agent_code"]
        if "fond_code" in dic:
            instance.fond_code = dic["fond_code"]
        if "lib_id" in dic:
            instance.lib_id = dic["lib_id"]
        if "archive_id" in dic:
            instance.archive_id = dic["archive_id"]
        if "appraisal_id" in dic:
            instance.appraisal_id = dic["appraisal_id"]
        if "tm" in dic:
            instance.tm = dic["tm"]
        if "zrz" in dic:
            instance.zrz = dic["zrz"]
        if "nd" in dic:
            instance.nd = dic["nd"]
        if "trust" in dic:
            instance.trust = dic["trust"]
        if "entity_type" in dic:
            instance.entity_type = dic["entity_type"]
        if "rule_path" in dic:
            instance.rule_path = dic["rule_path"]
        if "request" in dic:
            instance.request = dic["request"]
        return instance

    def check_param(self):
        if not self.archive_agent_code:
            raise BizException("archive_agent_code 不能为空!")

        if not self.fond_code:
            raise BizException("fond_code 不能为空!")

        if not self.lib_id:
            raise BizException("lib_id 不能为空!")

        if not self.archive_id:
            raise BizException("archive_id 不能为空!")

        if not self.appraisal_id:
            raise BizException("appraisal_id 不能为空!")

        if not self.tm:
            raise BizException("tm 不能为空!")

        if not self.zrz:
            raise BizException("zrz 不能为空!")

        if not self.nd:
            raise BizException("nd 不能为空!")

        if not self.trust:
            raise BizException("trust 不能为空!")

        if not self.entity_type:
            raise BizException("entity_type 不能为空!")

        if not self.rule_path:
            raise BizException("rule_path 不能为空!")

        if not self.request:
            raise BizException("request 不能为空!")
