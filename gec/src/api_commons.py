import datetime
import io
import json
import os
import re
import time
import traceback
import uuid
from collections.abc import Coroutine
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import requests
from fastapi import BackgroundTasks, Request, Response, status
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from fastapi.security.api_key import APIKey, APIKeyHeader  # noqa: F401
from pydantic import BaseModel
from starlette.background import BackgroundTask
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException

INTERNAL_HUB_PORT = os.getenv("INTERNAL_HUB_PORT", 5000)
if INTERNAL_HUB_PORT:
    INTERNAL_HUB_PORT = f":{INTERNAL_HUB_PORT}"
AUTHORIZE_API = os.getenv(
    "AUTHORIZE_API",
    f"http://hub_container{INTERNAL_HUB_PORT}/authorize",
)
LOG_API = os.getenv("LOG_API", f"http://hub_container{INTERNAL_HUB_PORT}/log")
GET_TOKEN_API = os.getenv("GET_TOKEN_API", f"http://hub_container{INTERNAL_HUB_PORT}/token")


async def run(request: Request, api_key: str, func: Callable, *args, **kwargs):
    return await auth_run_log(request, api_key, func, *args, **kwargs)


async def auth_run_log(request: Request, api_key: str, func: Callable, *args, **kwargs):
    print("auth_run_log", now())
    try:
        authorized, response = authorize(request)
        print(authorized, response.status_code if response else "")
        if authorized:
            start_time = time.time()
            content: Union[dict, str, bytes, Response, Coroutine] = func(
                *args, **kwargs
            )
            if isinstance(content, Coroutine):
                content = await content
            print("content type :", type(content))
            processed_time = float(format(time.time() - start_time, ".4f"))
            if isinstance(content, Response):
                response = content
                content = response.body
            elif isinstance(content, BaseModel):
                content = content.dict()
                response = JSONResponse(
                    content=content,
                    status_code=status.HTTP_200_OK,
                )
            elif isinstance(content, dict):
                response = JSONResponse(content=content, status_code=status.HTTP_200_OK)
            elif isinstance(content, io.BytesIO):
                response = Response(
                    content=content.read(),
                    media_type="audio/x-wav",
                    status_code=status.HTTP_200_OK,
                )
                content.seek(0)
            else:
                response = Response(content=content, status_code=status.HTTP_200_OK)
        else:
            content = response.body
            processed_time = None
        response.background = BackgroundTask(
            log_request,
            request=request,
            response_data=content,
            status_code=response.status_code,
            processed_time=processed_time,
        )
        return response
    except Exception as err:
        print(traceback.format_exc())
        # case when user defined err
        if isinstance(err, HTTPException):
            content = {"detail": err.detail}
            return JSONResponse(
                content=content,
                status_code=err.status_code,
                background=BackgroundTask(
                    log_request,
                    request=request,
                    response_data=content,
                    status_code=err.status_code,
                ),
            )
        if hasattr(err, "status_code"):
            content = {"message": str(err)}
            return JSONResponse(
                content=content,
                status_code=err.status_code,
                background=BackgroundTask(
                    log_request,
                    request=request,
                    response_data=content,
                    status_code=err.status_code,
                ),
            )
        return Response(
            content="Internal Server Error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            background=BackgroundTask(
                log_request,
                request=request,
                response_data=traceback.format_exc(),
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            ),
        )


async def request_to_params(request: Request) -> Dict[str, Dict[str, Any]]:
    request_form = {**await request.form()}
    file_list = []
    for key, value in request_form.items():
        if isinstance(value, UploadFile):
            file_list.append(key)
    request_files = {key: request_form.pop(key).file for key in file_list}
    # reset file pointer
    for key, value in request_files.items():
        value.seek(0)
    request_params = {**request.query_params}

    return {"files": request_files, "data": request_form, "params": request_params}


def authorize(request: Request) -> Tuple[bool, Response]:
    api_path = request.url.path
    response = requests.get(
        AUTHORIZE_API, headers=request.headers, params={"api_path": api_path}
    )
    if response.status_code == status.HTTP_200_OK:
        return True, None
    else:
        return (
            False,
            Response(content=response.content, status_code=response.status_code),
        )


async def log_request_by_response(
    request: Request, response: JSONResponse
) -> requests.Response:
    return await log_request(
        request,
        response_data=json.loads(response.body),
        status_code=response.status_code,
    )


async def log_request(
    request: Request,
    response_data: Union[Dict[str, Any], bytes],
    status_code: int,
    processed_time: float = None,
) -> requests.Response:
    params = await request_to_params(request)
    request_data = {
        **params["data"],
        "api_path": request.url.path,
        "status_code": status_code,
        "processed_time": processed_time,
    }
    if isinstance(response_data, io.BytesIO):
        return requests.post(
            LOG_API,
            files={
                **params["files"],
                "response_data": ("tts.wav", response_data.read()),
            },
            data=request_data,
            params=params["params"],
        )
    else:
        if isinstance(response_data, dict):
            response_data = json.dumps(response_data)
        return requests.post(
            LOG_API,
            files=params["files"],
            data={**request_data, "response_data": response_data},
            params=params["params"],
        )


def get_server_auth_token(retry: int = 3, timeout: float = 1) -> str:
    with requests.sessions.Session() as session:
        adapter = requests.adapters.HTTPAdapter(max_retries=retry)
        session.mount(
            "https://" if GET_TOKEN_API.startswith("https://") else "http://", adapter
        )
        response = session.get(GET_TOKEN_API, timeout=timeout)

    return json.loads(response.content)["token"]


def send_to_route(
    api_name: str, api_method: str = "POST", *args, **kwargs
) -> requests.Response:
    token = get_server_auth_token()

    if "headers" not in kwargs:
        kwargs["headers"] = {}
    kwargs["headers"]["x-server-auth-token"] = token
    return requests.request(
        method=api_method, url=f"http:/{api_name}", *args, **kwargs
    )


class ResponseErrorMessage(BaseModel):
    detail: str


class CustomAPIKeyHeader(APIKeyHeader):
    """Wrapper class to make 403 -> 401"""

    def __init__(self, *args, **kwargs):
        super(CustomAPIKeyHeader, self).__init__(*args, **kwargs)

    async def __call__(self, request: Request) -> Optional[str]:
        api_key: str = request.headers.get(self.model.name)
        if not api_key and not request.headers.get("x-server-auth-token"):
            if self.auto_error:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="API Key is required",
                )
        return api_key


api_key_header = CustomAPIKeyHeader(name="X-API-Key", auto_error=True)


def now(as_datetime: bool = False) -> Union[datetime.datetime, str]:
    """현재 시간값의 string"""
    current_time = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
    if as_datetime:
        return current_time
    return current_time.strftime("%Y-%m-%d %H:%M:%S")


def get_random_id(length: int = 8) -> str:
    """get random id by uuid"""
    return str(uuid.uuid4()).replace("-", "")[:length]


def get_elapsed_time(start_time: float) -> float:
    """get elapsed time"""
    return float(format(time.time() - start_time, ".4f"))


def normalize_string(text: str) -> str:
    """Normalize string

    Make string mathces follows:
    - All small letters
    - Only contains a-z, "'" that is in word and space
    - Words are splitted by single space

    Args:
      text: A string you want to normalize.

    Returns:
      Normalized string.
      String will be normalized as "<word_1> <word_2> ... <word_n>"
    """
    text = text.replace("’", "'")
    text = re.sub(r"\s+", " ", text)
    text = re.sub("[^a-z' ]", "", text.lower())
    text = re.sub(r"\s+", " ", text)
    # replace unrelated quote
    text = re.sub("^'|'$|'(?= )|(?<= )'", "", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


# Fix swagger ui bug when list is input for Formdata
# https://stackoverflow.com/questions/74064316/value-is-not-a-valid-email-address-when-sending-multiple-email-addresses-using
def split_formdata_if_not(_iter: List[str]):
    if len(_iter) == 1:
        return _iter[0].split(",")
    else:
        return _iter

def add_log_to_background(request: Request, response: Response, processed_time: float):
    background_task: BackgroundTasks = BackgroundTasks(
        [
            BackgroundTask(
                log_request,
                request,
                response.body,
                response.status_code,
                processed_time,
            )
        ]
    )
    if response.background is not None:
        background_task.add_task(
            response.background.func,
            *response.background.args,
            **response.background.kwargs,
        )
    response.background = background_task


class LogRoute(APIRoute):
    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()

        async def log_route_handler(request: Request) -> Response:
            start = time.time()
            response: Response = await original_route_handler(request)
            processed_time = round(time.time() - start, 4)
            add_log_to_background(request, response, processed_time)
            return response

        return log_route_handler
