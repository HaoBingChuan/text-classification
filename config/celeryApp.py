from __future__ import absolute_import, unicode_literals
from celery import Celery
from config import celeryConfig

app = Celery("tasks.classification")
app.config_from_object(celeryConfig)
