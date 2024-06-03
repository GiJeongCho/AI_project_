import re

from fastapi.exceptions import RequestValidationError
from pydantic_core import InitErrorDetails


def get_error_type(exc):
        if "ValueError: The text appears to contain an inappropriate sentence." in str(exc):
            _type = "HateSpeech"
        else:
            _type = "Other"
        return _type

class RegexRequestValidationError(RequestValidationError):
    def __init__(self, invalid_matches: list[re.Match], *args, **kwargs):
        self._regex_matches = invalid_matches
        super().__init__(errors=self._generate_errors(), *args, **kwargs)

    def _generate_errors(self) -> list[InitErrorDetails]:
        return [
            InitErrorDetails(
                type="string_pattern_mismatch",
                loc=["body", "text"],
                msg="string has invalid characters",
                input=self._regex_matches[0].string,
                ctx={
                    "invalid_characters": [
                        {
                            "start": match.start(),
                            "end": match.end(),
                            "text": match.string[match.start() : match.end()],
                        }
                        for match in self._regex_matches
                    ]
                },
            )
        ]
