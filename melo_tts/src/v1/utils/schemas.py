from __future__ import annotations

from enum import Enum

from flair.data import Token
from pydantic import BaseModel


class ChunkingModelType(str, Enum):
    small = "small"
    large = "large"


class ChunkToken(BaseModel):
    start_index: int
    end_index: int
    chunk_token: str

    @classmethod
    def make_token_dict(cls, token: Token, sentence_start_index) -> ChunkToken:
        start_index = token.start_position + sentence_start_index
        end_index = token.end_position + sentence_start_index
        chunk_token = token.text

        return ChunkToken(
            start_index=start_index, end_index=end_index, chunk_token=chunk_token
        )


class Chunk(BaseModel):
    start_index: int
    end_index: int
    chunk: str
    chunk_tokens: list[ChunkToken]

    @classmethod
    def make_chunk_dict(cls, chunk_tokens: list[ChunkToken], sentence: str) -> Chunk:
        start_index = chunk_tokens[0].start_index
        end_index = chunk_tokens[-1].end_index
        chunk = sentence[start_index:end_index]

        return Chunk(
            start_index=start_index,
            end_index=end_index,
            chunk=chunk,
            chunk_tokens=chunk_tokens,
        )


class ResponseChunking(BaseModel):
    sentence: str
    chunks: list[Chunk]

    class Config:
        schema_extra = {
            "example": {
                "sentence": "The birds in the sky fly high.",
                "chunks": [
                    {
                        "start_index": 0,
                        "end_index": 9,
                        "chunk": "The birds",
                        "chunk_tokens": [
                            {"start_index": 0, "end_index": 3, "chunk_token": "The"},
                            {"start_index": 4, "end_index": 9, "chunk_token": "birds"},
                        ],
                    },
                    {
                        "start_index": 10,
                        "end_index": 20,
                        "chunk": "in the sky",
                        "chunk_tokens": [
                            {"start_index": 10, "end_index": 12, "chunk_token": "in"},
                            {"start_index": 13, "end_index": 16, "chunk_token": "the"},
                            {"start_index": 17, "end_index": 20, "chunk_token": "sky"},
                        ],
                    },
                    {
                        "start_index": 21,
                        "end_index": 30,
                        "chunk": "fly high.",
                        "chunk_tokens": [
                            {"start_index": 21, "end_index": 24, "chunk_token": "fly"},
                            {"start_index": 25, "end_index": 29, "chunk_token": "high"},
                            {
                                "start_index": 29,
                                "end_index": 30,
                                "chunk_token": ".",
                            },
                        ],
                    },
                ],
            }
        }
