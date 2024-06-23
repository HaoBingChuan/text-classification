import redis
from config.base import *


class RedisPool(object):
    def __init__(self, db=0):
        self.pool = redis.ConnectionPool(host=REDIS_IP, password=REDIS_PASSWORD, port=REDIS_PORT,
                                         db=db, max_connections=50)

    def get_conn(self):
        return redis.Redis(connection_pool=self.pool, decode_responses=True)
