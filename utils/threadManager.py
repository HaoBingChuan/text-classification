from concurrent.futures import ThreadPoolExecutor

"""
系统级线程池定义(获取相似特征样本)
"""
pooling = ThreadPoolExecutor(max_workers=50, thread_name_prefix='Classification')
