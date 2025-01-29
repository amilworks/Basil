# app/celery_app.py

from celery import Celery

celery_app = Celery(
    "hf_analysis",
    broker="amqp://user:password@rabbitmq:5672//",
    backend="redis://redis:6379/0"
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)
