import pymysql
from dbutils.pooled_db import PooledDB
from urllib.parse import urlparse


class MysqlPool(object):
    def __init__(self, url):
        self.url = url
        if not url:
            return
        url = urlparse(url)
        self.pool = PooledDB(
            creator=pymysql,
            maxconnections=10,  # 连接池的最大连接数
            maxcached=10,
            maxshared=10,
            blocking=True,
            setsession=[],
            host=url.hostname,
            port=url.port or 3306,
            user=url.username,
            password=url.password,
            database=url.path.strip().strip('/'),
            charset='utf8mb4',
        )

    # def __new__(cls, *args, **kw):
    #     if not hasattr(cls, '_instance'):
    #         cls._instance = object.__new__(cls)
    #     return cls._instance

    def connect(self):
        conn = self.pool.connection()
        cursor = conn.cursor(cursor=pymysql.cursors.DictCursor)
        return conn, cursor

    def connect_close(self, conn, cursor):
        cursor.close()
        conn.close()

    def fetch_all(self, sql, args=None):
        conn, cursor = self.connect()
        try:
            if args is None:
                cursor.execute(sql)
            else:
                cursor.execute(sql, args)
            record_list = cursor.fetchall()
        finally:
            self.connect_close(conn, cursor)
        return record_list

    def fetch_one(self, sql, args=None):
        conn, cursor = self.connect()
        try:
            cursor.execute(sql, args)
            result = cursor.fetchone()
        finally:
            self.connect_close(conn, cursor)
        return result

    def insert(self, sql, args):
        conn, cursor = self.connect()
        try:
            row = cursor.execute(sql, args)
            conn.commit()
        finally:
            self.connect_close(conn, cursor)
        return row
