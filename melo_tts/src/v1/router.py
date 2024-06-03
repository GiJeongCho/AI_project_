import json
from pathlib import Path
from typing import Dict
import re
from fastapi import APIRouter, HTTPException, status, Request, Depends # status 추가
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from src.v1.main import TTS
from src.v1.utils.chunk import ChunkMaker

from typing import Optional
from api_commons import APIKey, api_key_header, run, authorize
# tts_router = APIRouter()

router_v1 = APIRouter(
    prefix="/v1",
    tags=["TTS"],
    responses={
        status.HTTP_200_OK: {"description": "Successful Response"},
        status.HTTP_401_UNAUTHORIZED: {"description": "Unauthorized"},
        status.HTTP_403_FORBIDDEN: {"description": "Forbidden"},
        status.HTTP_404_NOT_FOUND: {"description": "Not found"}
    },
)

chunk_maker = ChunkMaker()  # Initialize once if possible and reuse

# 서버 코드에서 ChunkMaker를 호출할 때:

def load_models():
    models = {}
    # with open(r"resources/model_data.json", "r") as f:
    
    with open(r"src/v1/resources/model_data_s3.json", "r") as f:
        data = json.load(f)
    for lang, genders in data.items():
        for gender, speakers in genders.items():
            for speaker_name, paths in speakers.items():
                models.setdefault(speaker_name, {}).update({
                    lang: TTS(
                        language=lang,
                        device="auto",
                        use_hf=False,
                        config_path=paths["config_path"],
                        ckpt_path=paths["ckpt_path"]
                    )
                })
                print(f"Model for {speaker_name} in {lang} loaded successfully.")
    return models

models = load_models()

class SynthesizeRequest(BaseModel):
    speaker_name: str
    text: str
    speed: float = Field(default=1.0, ge=0.2, le=2.0)

async def process_synthesis(req: SynthesizeRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="No text provided. Please provide text for synthesis.")

    if not re.match(r'^[a-zA-Z0-9 \.,!\?\'\"-]*$', req.text):
        raise HTTPException(status_code=400, detail="Text contains unsupported characters.")

    try:
        if req.speaker_name in models:
            for language, model in models[req.speaker_name].items():
                if language == "EN":
                    response = chunk_maker.make_chunked_sentence(req.text)
                    chunks_text = ', '.join([chunk.chunk for chunk in response.chunks]) + '.'
                    audio_buffer = model.tts_to_bytes(chunks_text, speaker_id=0, speed=req.speed)
                else:
                    audio_buffer = model.tts_to_bytes(req.text, speaker_id=0, speed=req.speed)
                return StreamingResponse(audio_buffer, media_type="audio/wav")
        else:
            raise HTTPException(status_code=404, detail="Speaker not found")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router_v1.post("/tts/en-us")
async def synthesize(request: Request, req: SynthesizeRequest, api_key: APIKey = Depends(api_key_header)):
    # API 인증
    authorized, response = authorize(request)
    if not authorized:
        return response  # 인증 실패시 반환
    return await run(request, api_key, process_synthesis, req)


