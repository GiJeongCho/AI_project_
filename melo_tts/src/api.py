import uvicorn
from fastapi import FastAPI
from src.v1.router import router_v1

app = FastAPI(
    docs_url="/v1/tts/en-us/docs",
    redoc_url="/v1/tts/en-us/redoc",
    openapi_url="/v1/tts/en-us/openapi.json"
)

# 라우터를 등록합니다.
app.include_router(router_v1)


