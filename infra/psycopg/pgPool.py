from config.base import *
from psycopg2 import pool


class PGPool(object):
    def __init__(self):
        try:
            self.connectPool = pool.SimpleConnectionPool(10, 10, host=POSTGRESQL_HOST, port=POSTGRESQL_PORT,
                                                         user=POSTGRESQL_USER, password=POSTGRESQL_PASSWORD,
                                                         database=POSTGRESQL_DATABASE, keepalives=1,
                                                         keepalives_idle=30, keepalives_interval=10,
                                                         keepalives_count=5)
        except Exception as e:
            print(e)

    def get_connect(self):
        conn = self.connectPool.getconn()
        cursor = conn.cursor()
        return conn, cursor

    def close_connect(self, conn, cursor):
        cursor.close()
        self.connectPool.putconn(conn)

    def close_all(self):
        self.connectPool.closeall()

    # 执行增删改
    def execute(self, sql, value=None):
        conn, cursor = self.get_connect()
        try:
            res = cursor.execute(sql, value)
            conn.commit()
            self.close_connect(conn, cursor)
            return res
        except Exception as e:
            conn.rollback()
            raise e

    def select_one(self, sql):
        conn, cursor = self.get_connect()
        cursor.execute(sql)
        result = cursor.fetchone()
        self.close_connect(conn, cursor)
        return result

    def select_all(self, sql):
        conn, cursor = self.get_connect()
        cursor.execute(sql)
        result = cursor.fetchall()
        self.close_connect(conn, cursor)
        return result


if __name__ == '__main__':
    pool = PGPool()
    pool.execute(sql="insert into test1 values(2,1,'zt','13671386828','beijing','beijing','shangdi','deshi-3-5')")
    print('Program Ended.')

