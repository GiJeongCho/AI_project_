from __future__ import annotations

import json
import re
from typing import Annotated

from fastapi import Form
from pydantic import BaseModel, Field, model_validator, validator
from src.v1.correct_text_utils.exceptions import RegexRequestValidationError
from src.v1.correct_text_utils.params import TEXT_MAX_LENGTH, TEXT_MIN_LENGTH
from src.v1.correct_text_utils.parser import unicode_preprocess


class GECRequest(BaseModel):
    text: Annotated[
        str,
        Field(
            Form(
                ...,
                description="문법 교정을 실행할 텍스트</p>"
                "<p>알파벳과 문장부호만 허용됩니다.</p>"
                "<p>최소 한개 이상의 알파벳을 포함해야 합니다.",
                min_length=TEXT_MIN_LENGTH,
                max_length=TEXT_MAX_LENGTH,
            ),
        ),
    ]
    feedback: Annotated[
        bool,
        Field(
            Form(
                False,
                description="피드백을 받을지 선택합니다.</p>"
                "<p>피드백 내용은 skill name, comment, example</p>"
            )
        )
    ]
    raise_error: Annotated[
        bool,
        Field(
            Form(
                True,
                description="True일 경우, 허용되지 않는 문자가 포함되어 있으면 422 Validation Error가 발생합니다.</p>"
                "<p>False일 경우, 허용되지 않는 문자가 포함되어 있으면 해당 부분을 삭제하고 정상 반환합니다.",
            ),
        ),
    ]

    @classmethod
    @property
    def _contains_alphabet_regex(cls):
        return r"[a-zA-Z]"

    @classmethod
    @property
    def _not_allowed_regex(cls) -> str:
        return r"[^\x20-\x7E\s¡¿ßàáâãäæçèéêëìíîïñòóôõöùúûüāćōœşū]+([^a-zA-Z]+[^\x20-\x7E\s¡¿ßàáâãäæçèéêëìíîïñòóôõöùúûüāćōœşū]+)*[^\sa-zA-Z]*"

    @validator("text")
    def validate_text(cls, text: str) -> str:
        if not re.search(cls._contains_alphabet_regex, text):
            raise ValueError("text must contain at least one alphabet")
        return text

    @model_validator(mode="after")
    def validate_model(cls, request: GECRequest) -> GECRequest:
        request.text = unicode_preprocess(request.text)
        if not request.raise_error:
            request.text = re.sub(cls._not_allowed_regex, "", request.text)
            if not re.search(cls._contains_alphabet_regex, request.text):
                raise ValueError("text does not contain alphabet after substitution")

        if invalid_matches := list(re.finditer(cls._not_allowed_regex, request.text)):
            raise RegexRequestValidationError(invalid_matches=invalid_matches)
        return request


class Replacement(BaseModel):
    value: str
    types: list


class Match(BaseModel):
    replacements: list[Replacement]
    length: int
    offset: int


class GECResponse(BaseModel):
    """response model for GEC"""

    matches: list[Match]
    text: str
    correct_text: str

    class Config:
        json_schema_extra = {
            "example": {
                "matches": [
                    {
                        "replacements": [
                            {
                                "value": "went", "types": [
                                {
                                    "type":"R:VERB:INFL", 
                                    "category":"Grammar", 
                                    "feedback": {
                                        "skill_name_en": "Replacing Verb Inflection",
                                        "skill_name_kr": "동사 형태 오류",
                                        "comment": "goed의 동사 시제 형태가 올바르지 않습니다. 시제에 맞는 동사 형태 went으로 변경되었습니다.",
                                        "comment_word_info": [
                                                {
                                                    "word": "goed",
                                                    "start_index": 0,
                                                    "end_index": 4
                                                },
                                                {
                                                    "word": "went",
                                                    "start_index": 40,
                                                    "end_index": 44
                                                },
                                            ],
                                        "example_1": {
                                            "before": "The boys start their adventure on stoping them.",
                                            "after": "The boys start their adventure on stopping them.",
                                            "before_word_info": [
                                                {
                                                    "word": "stoping",
                                                    "start_index": 34,
                                                    "end_index": 41
                                                }
                                            ],
                                            "after_word_info": [
                                                {
                                                    "word": "stopping",
                                                    "start_index": 34,
                                                    "end_index": 42
                                                }
                                            ]
                                        },
                                        "example_2": {
                                            "before": "She eated lunch an hour ago.",
                                            "after": "She ate lunch an hour ago.",
                                            "before_word_info": [
                                                {
                                                    "word": "eated",
                                                    "start_index": 4,
                                                    "end_index": 9
                                                }
                                            ],
                                            "after_word_info": [
                                                {
                                                    "word": "ate",
                                                    "start_index": 4,
                                                    "end_index": 7
                                                }
                                            ]
                                        }
                                    }
                                }
                            ]
                        }
                        ],
                        "length": 4,
                        "offset": 2,
                    }
                ],
                "text": "I goed to school with my friend.",
                "correct_text": "I went to school with my friend."
            }
        }
