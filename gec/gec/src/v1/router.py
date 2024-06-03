from typing import Annotated

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from src.api_commons import APIKey, LogRoute, ResponseErrorMessage, api_key_header, run
from src.v1.correct_text import main
from src.v1.correct_text_utils.depends import make_dependable
from src.v1.schemas import GECRequest, GECResponse

router_v1 = APIRouter(
    prefix="/v1",
    tags=["GEC"],
    responses={
        status.HTTP_401_UNAUTHORIZED: {"model": ResponseErrorMessage},
        status.HTTP_403_FORBIDDEN: {"model": ResponseErrorMessage},
    },
    dependencies=[Depends(api_key_header)],
    route_class=LogRoute,
)

@router_v1.post(
    "/gec", response_model=GECResponse, summary="Grammatical Error Correction (문법교정)"
)
async def _gec(
    request : Request,
    body: Annotated[GECRequest, Depends(make_dependable(GECRequest))],
    api_key: APIKey = Depends(api_key_header),
):
    return await run(
        request,
        api_key,
        main,
        body
    )