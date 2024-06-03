from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from src.v1.correct_text_utils.exceptions import get_error_type
from src.v1.router import router_v1

app = FastAPI(
    docs_url="/v1/gec/docs",
    redoc_url="/v1/gec/redoc",
    openapi_url="/v1/gec/openapi.json",
)
app.include_router(router_v1)

# 욕설에 감지에 대한 error response입니다. chatGPT의 감지 결과에 따릅니다.
@app.exception_handler(ValueError)
async def value_error_exception_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"message": str(exc), "type":get_error_type(str(exc))},
    )