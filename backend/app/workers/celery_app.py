from celery import Celery
from ..config import settings

celery_app = Celery(
    "deepclean_worker",
    broker='memory://',
    backend='cache+memory://',
    include=["app.workers.tasks"]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)
