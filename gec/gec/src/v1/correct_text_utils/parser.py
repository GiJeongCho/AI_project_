import json
import re
from typing import Any

import errant
from errant import Annotator
from errant.edit import Edit
from src.v1.correct_text_utils.edit_tag import correct_type

f = open("resources/change_unicode.json", "r")
unicode_dict = json.load(f)

def unicode_preprocess(original_text:str) -> str:
    for chr in original_text:
        if chr in unicode_dict.keys():
            original_text = re.sub(chr, unicode_dict[chr], original_text)
    return original_text

class Parser:
    annotator: Annotator

    def __new__(cls, *args, **kawargs):
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls)
            cls._instance.annotator = errant.load("en")
        return cls._instance

    def __init__(self, original_text: str, corrected_text: str, use_feedback: bool):
        self.original_text = original_text
        self.corrected_text = corrected_text
        self.use_feedback = use_feedback
        self.words_of_origin_sentence = original_text.split()

    def parse_corrections(self) -> dict[str, str]:
        if self.no_correction(self.original_text, self.corrected_text):
            matches = []
        else:
            matches = self.get_matches(self.original_text, self.corrected_text)

        result = self.result_form(matches)
        return result

    def get_matches(
        self, original_text: str, corrected_text: str
    ) -> list[dict[str, Any]]:
        edits = self.get_edits(original_text, corrected_text)
        offsets = self.get_offsets(self.words_of_origin_sentence)
        matches = self.make_get_matches_result(offsets, edits)
        return matches

    def get_edits(self, origin_text: str, correc_text: str) -> list[Edit]:
        orig = self.annotator.parse(origin_text)
        cor = self.annotator.parse(correc_text)
        edits = self.annotator.annotate(orig, cor)
        return edits

    def get_offsets(self, words_of_origin_sentence: list[str]) -> list[int]:
        offsets = []
        offset = 0

        for word in words_of_origin_sentence:
            pattern = re.sub(r"(\W)", r"\\\1", word)
            search_result = re.search(pattern, self.original_text[offset:])
            if search_result is None:
                raise TypeError(
                    f'search result "{word}" not found in : {self.original_text[offset:]}'
                )

            offset += search_result.start()
            offsets.append(offset)
            offset += len(word)

        offsets.append(len(self.original_text.rstrip()))  # end

        return offsets

    def make_get_matches_result(
        self, offsets: list[int], edits: list[Edit]
    ) -> list[dict[str, Any]]:
        result = []
        for edit in edits:
            match_dict = self.make_replacements(offsets, edit)
            result.append(match_dict)
        return result

    def make_replacements(self, offsets: list[int], edit: Edit) -> dict[str, Any]:
        value = self.refine_c_str(edit, offsets)
        types = self.get_types(edit)
        offset_of_edit, length_of_edit = self.refine_offset_and_length(edit, offsets)

        replacements = self.replacements_form(
            value, types, offset_of_edit, length_of_edit
        )

        return replacements

    def get_types(self, edit: Edit) -> list[dict[str, str]]:
        types = correct_type(edit.o_str, edit.c_str, self.annotator, self.use_feedback)
        return types

    def no_correction(self, original_text: str, corrected_text: str) -> bool:
        return original_text == corrected_text

    # return start_of_text_case case or in multi sentence case.
    def is_start_of_sentence(self, offset_of_edit: int) -> bool:
        return offset_of_edit == 0 or self.original_text[offset_of_edit - 1] == "\n"

    # one sentence case or multi sentence case.
    def is_end_of_sentence(self, offsets: list[int], edit: Edit) -> bool:
        return (
            len(self.original_text) == offsets[edit.o_end]
            or self.original_text[-1] == "\n"
        )

    def refine_c_str(self, edit: Edit, offsets: list[int]) -> str:
        is_end_of_sent = self.is_end_of_sentence(offsets, edit)

        # add case "I + () + home." -> "I + (go ) + home."
        if edit.o_start == edit.o_end and not is_end_of_sent:
            value = edit.c_str + " "
        # add case "I go()" -> "I go (home.)"
        else:
            value = edit.c_str

        return value

    def refine_offset_and_length(
        self,
        edit: Edit,
        offsets: list[int],
    ) -> tuple[int, int]:
        is_end_of_sent = self.is_end_of_sentence(offsets, edit)
        is_start_of_sent = self.is_start_of_sentence(offsets[edit.o_start])
        offset_of_edit = offsets[edit.o_start]

        # Only think about origin_text
        # Missing case
        if offsets[edit.o_end] == offsets[edit.o_start]:
            length_of_edit = 0
        # Other case
        elif is_end_of_sent:
            length_of_edit = offsets[edit.o_end] - offsets[edit.o_start]

        else:
            length_of_edit = offsets[edit.o_end] - offsets[edit.o_start] - 1

        # Unnecessary case
        # 그냥 지우기만 하면 I love it. -> I  it.이 됨 그래서 " love"를 ""로 바꾸자
        if edit.c_str == "":
            if is_end_of_sent:
                if not is_start_of_sent:
                    offset_of_edit -= 1
                    length_of_edit += 1
            else:
                length_of_edit += 1

        return offset_of_edit, length_of_edit

    def result_form(self, matches: list[dict[str, Any]]) -> dict[str, Any]:
        result = {
            "matches": matches,
            "text": self.original_text,
            "correct_text": self.corrected_text,
        }
        return result

    def replacements_form(
        self,
        value: str,
        types: list[dict[str, str]],
        offset_of_edit: int,
        length_of_edit: int,
    ) -> dict[str, Any]:
        replacements = {
            "replacements": [
                {
                    "value": value,
                    "types": types,
                }
            ],
            "offset": offset_of_edit,
            "length": length_of_edit,
        }

        return replacements

