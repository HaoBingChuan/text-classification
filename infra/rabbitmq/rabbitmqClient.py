import pika
from config.base import *


class RabbitClient(object):

    def __init__(
        self,
        host=RABBIT_MQ_IP,
        port=RABBIT_MQ_PORT,
        user=RABBIT_USER,
        password=RABBIT_PASSWORD,
        virtual_host=RABBIT_VIRTUAL_HOST,
    ):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.virtual_host = virtual_host
        self._set_connection()
        self._set_chanel()

    def _set_connection(self):
        credentials = pika.PlainCredentials(self.user, self.password)
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=self.host,
                heartbeat=0,
                port=self.port,
                virtual_host=self.virtual_host,
                credentials=credentials,
            )
        )

    def _set_chanel(self):
        self.channel = self.connection.channel()

    def _exchange_declare(self, exchange, exchange_type, durable):
        self.channel.exchange_declare(
            exchange=exchange, exchange_type=exchange_type, durable=durable
        )

    def _queue_declare(self, queue, durable):
        self.channel.queue_declare(queue=queue, durable=durable)

    # delivery_mode：1表示非持久化消息，2表示持久化消息
    def _basic_publish(self, exchange, routing_key, mq_msg, delivery_mode):
        self.channel.basic_publish(
            exchange=exchange,
            routing_key=routing_key,
            body=mq_msg,
            properties=pika.BasicProperties(delivery_mode=delivery_mode),
        )

    def send(
        self, exchange, exchange_type, routing_key, mq_msg, durable, delivery_mode
    ):
        """
        Args:
            exchange:
            exchange_type:
            routing_key:
            mq_msg:
            durable:
            delivery_mode: 1表示非持久化消息，2表示持久化消息
        Returns:
        """
        self._exchange_declare(exchange, exchange_type, durable)
        self._basic_publish(exchange, routing_key, mq_msg, delivery_mode)
        self.connection.close()

    def receive(self, queue, durable, callback, prefetch_count):
        self._queue_declare(queue, durable)
        self.channel.basic_qos(prefetch_count=prefetch_count)
        self.channel.basic_consume(
            queue=queue, on_message_callback=callback, auto_ack=False
        )
        self.channel.start_consuming()
