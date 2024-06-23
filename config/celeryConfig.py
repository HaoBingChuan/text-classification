from __future__ import absolute_import
from celery.schedules import crontab, timedelta

broker_url = 'redis://:asdc_ml@172.25.78.26:6379/2'
result_backend = 'redis://:asdc_ml@172.25.78.26:6379/3'

task_serializer = 'json'
result_serializer = 'json'  # 读取任务结果一般性能要求不高，所以使用了可读性更好的JSON
accept_content = ["json"]  # 指定任务接受的内容类型
timezone = 'Asia/Shanghai'

result_expires = 60 * 60 * 24  # 任务过期时间，不建议直接写86400，应该让这样的magic数字表述更明显

worker_hijack_root_logger = False
worker_max_tasks_per_child = 100 # to prevent from mem-leaking
worker_concurrency = 1
# task_ignore_result = True

# include = ['cronTasks', 'task1', 'task2']

include = ['tasks']

beat_schedule = {
    'task1': {
        'task': 'tasks.add1',
        'args': (10, 10),
        'schedule': timedelta(seconds=10)
    },
    'task2': {
        'task': 'tasks.add2',
        'args': (60, 60),
        'schedule': crontab(minute='*')
    }
}

# beat_schedule = {
#     'task1': {
#         'task': 'task1.celery_run',
#         'schedule': timedelta(seconds=10)
#     },
#     'task2': {
#         'task': 'task2.celery_run',
#         'schedule': timedelta(seconds=20)
#     }
# }
