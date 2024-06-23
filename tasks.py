from __future__ import absolute_import, unicode_literals
import time
from config.celeryApp import app
from config.logConfig import logger


@app.task
def add1(x, y):
    logger.info('tasks.add_1 函数执行开始.....')
    time.sleep(5)
    logger.info('tasks.add_1 函数执行结束.....')
    return x + y


@app.task
def add2(x, y):
    logger.info('tasks.add_2 函数执行开始.....')
    time.sleep(10)
    logger.info('tasks.add_2 函数执行结束.....')
    return x + y
