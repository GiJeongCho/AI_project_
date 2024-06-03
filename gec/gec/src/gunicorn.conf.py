# Reference Docs : https://docs.gunicorn.org/en/stable/settings.html
import os

port = os.environ.get("PORT", 80)
bind = f"0.0.0.0:{port}"
workers = os.environ.get("WORKERS", 2)
worker_class = "uvicorn.workers.UvicornWorker"
# reload = True
# preload_app = True
# internal timeout 120 for route request
timeout = os.environ.get("TIMEOUT", 300)
# log to stdout & stderr
accesslog = "-"
errorlog = "-"
# restart after request - https://velog.io/@allbegray/fastapi-uvicorn-gunicorn-%EA%B0%9C%EB%B0%9C-%EC%9D%B4%EC%8A%88
max_requests = os.environ.get("MAX_REQUESTS", 1000)
max_requests_jitter = os.environ.get("MAX_REQUESTS_JITTER", 100)
