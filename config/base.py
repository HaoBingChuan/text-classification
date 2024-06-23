# 工程名称，logger配置使用
PROJECT_NAME = "classification-helper"

RABBIT_MQ_PORT = 5673
RABBIT_MQ_IP = "172.25.67.164"
RABBIT_USER = "adms"
RABBIT_PASSWORD = "adms"
RABBIT_VIRTUAL_HOST = "/nlp"

REDIS_IP = "172.25.78.26"
REDIS_PORT = 6379
REDIS_PASSWORD = "asdc_ml"

# 档案数据库URL
CONTEXT_DB_URL = "mysql://root:Root123@@172.25.67.197:3306/zt_test"

# Kingbase数据库连接
POSTGRESQL_HOST = "172.25.67.120"
POSTGRESQL_PORT = "54321"
POSTGRESQL_USER = "system"
POSTGRESQL_PASSWORD = "654321"
POSTGRESQL_DATABASE = "archive_db"

# 网络请求设置
TIMEOUT = 100
RETRIES = 5

# 多分类API\多标签API\关键词相似度API
MULTI_CLASS_URL = "http://172.25.67.120:8090/api/cls/predict/control_code"
MULTI_LABEL_URL = "http://172.25.67.120:8090/api/cls/predict/divorce"
KEYWORD_SIMILAR_URL = "http://172.25.67.120:8070/api/keyword_similarity"
NER_URL = "http://172.25.67.120:8070/api/ner"
SUMMARY_URL = "http://172.25.67.120:8070/api/text_summary"

# -1 && 0 && 1
DEVICE_ID = 1
# cpu && gpu:0 && gpu:1
DEVICE = "gpu:1"


MOUNT_FOLDER = "/home/haobingchuan/nas1"  # 挂载文件夹
# MOUNT_FOLDER = "/home/haobingchuan/classification-helper"
