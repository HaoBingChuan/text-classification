class ContextDomain(object):
    """
    classification_context实体对象
    """
    def __init__(self,
                 agent_code: str = None,
                 fond_code: str = None,
                 lib_id: str = None,
                 archive_id: str = None,
                 text: str = None,
                 label: str = None,
                 pre_type: str = None,
                 is_feedback: int = 0,
                 pre_label: str = None):
        self.agent_code = agent_code
        self.fond_code = fond_code
        self.lib_id = lib_id
        self.archive_id = archive_id
        self.text = text
        self.label = label
        self.pre_type = pre_type
        self.is_feedback = is_feedback
        self.pre_label = pre_label
        

    @staticmethod
    def dict2entity(dic):
        instance = ContextDomain()
        if 'agent_code' in dic:
            instance.agent_code = dic['agent_code']
        if 'fond_code' in dic:
            instance.fond_code = dic['fond_code']
        if 'lib_id' in dic:
            instance.lib_id = dic['lib_id']
        if 'archive_id' in dic:
            instance.archive_id = dic['archive_id']
        if 'text' in dic:
            instance.text = dic['text']
        if 'label' in dic:
            instance.label = dic['label']
        if 'pre_type' in dic:
            instance.pre_type = dic['pre_type']
        if 'is_feedback' in dic:
            instance.is_feedback = dic['is_feedback']
        if 'pre_label' in dic:
            instance.pre_label = dic['pre_label']
        return instance

