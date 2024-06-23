from utils.multi_class.util import *
from config.base import *


def process_data():
    sql = "select concat(IFNULL(tm, ''),'|', IFNULL(zrz, ''), '|', IFNULL(nd, ''), '年度') as text, bgqx as label " \
          "from guoliang_old"
    data_df = read_context_from_db(CONTEXT_DB_URL, sql)
    export_data('./data/other/bgqx', data_df)


if __name__ == '__main__':
    process_data()
