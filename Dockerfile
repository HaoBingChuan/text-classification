# temp stage
FROM python:3.9-slim as builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN sed -i "s@http://\(deb\|security\).debian.org@https://mirrors.aliyun.com@g" /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends gcc

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    pip --default-timeout=100 install -r requirements.txt --index-url https://mirror.baidu.com/pypi/simple/ \
    --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple/\nhttps://mirrors.aliyun.com/pypi/simple/\nhttps://pypi.douban.com/simple/

# final stage
FROM python:3.9-slim

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

ENV SERVER flask
ENV LOG_LEVEL INFO
ENV PATH="/opt/venv/bin:$PATH"

ARG DEBIAN_FRONTEND=noninteractive
# timezone设置
ENV TZ=Asia/Shanghai
RUN apt-get update \
    && apt-get -y install -y tzdata \
    && ln -fs /usr/share/zoneinfo/${TZ} /etc/localtime \
    && echo ${TZ} > /etc/timezone \
    && dpkg-reconfigure --frontend noninteractive tzdata

# install libgomp1(open-cv依赖库)
RUN apt-get update && apt-get -y install libgomp1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 暴露端口
EXPOSE 8090

COPY . /app
# RUN chmod +x /app/run.sh
# CMD ["sh", "-c", "export PATH=${PATH}:$PATH; /app/run.sh"]
CMD ["python", "/app/server.py"]
