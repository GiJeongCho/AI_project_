import json
import os
import re

# for type_hint
from errant import Annotator
from errant.edit import Edit
from src.v1.correct_text_utils.calculator import compare_text_lengths, lcs
from src.v1.correct_text_utils.feedback import make_feedback
from src.v1.correct_text_utils.spacy_tokenizer import tokenize

type_dict = json.load(open(f"resources/tag_type_AIDT.json", "r", encoding="utf-8"))
### Description of this code page ###
# Modify the type according to the state after tokenization.
# original situation: [good. -> god,] below case is returned as [R:OTHER]
# For wrong part I think it needs to be compared after tokenization.
# correct the situation : [good . -> god ,] below case is returned as [R:NOUN, R:PUNCT]
def correct_type(
    orig_edit_str: str, cor_edit_str: str, annotator: Annotator, use_feedback: bool
) -> list[dict[str, str]]:
    orig_str = ""
    cor_str = ""

    # Tokenize the wrong part.
    if orig_edit_str.strip():
        orig_str = " ".join(tokenize(orig_edit_str))
    if cor_edit_str.strip():
        cor_str = " ".join(tokenize(cor_edit_str))

    # Details are in parser.py
    edits = get_edits(orig_str, cor_str, annotator)

    types = get_edit_types(orig_edit_str, cor_edit_str, edits, use_feedback)
    return types


# For each edit of edits, Make the category dictionary like this. {"R:VERB":"Grammar", "R:PUNCT":"MECHANICS"}
def get_edit_types(
    orig_edit_str: str,
    cor_edit_str: str,
    edits: list[Edit],
    use_feedback: bool,
) -> list[dict[str, str]]:
    types = set()
    is_r_verb_usage = False
    for edit in edits:
        # Check R:VERB is usage or grammar
        is_r_verb_usage = Determine_r_verb_is_usage(edit)

        # Add basic type
        types.add(edit.type)

        # Check WO case
        types = check_R_WO_is_one_word(edit, orig_edit_str, cor_edit_str, types)

        # Check Mechanics Part
        types = check_specific_PUNCT(edit, types)
        types = check_PUNCT(edit, types)
        types = check_R_ORTH(orig_edit_str, cor_edit_str, types, edit)

        # Check Determiner Part
        types = check_DET(edit, types)

    types = check_LAST(orig_edit_str, cor_edit_str, types)
    return refine_category(types, is_r_verb_usage, orig_edit_str, cor_edit_str, use_feedback)


# Determines whether the category is Usage or not
# If edit.o_str and edit.c_str are not similar, it is Usage. [write -> read]
# The default is Grammar. [will -> would, make -> maked]
def Determine_r_verb_is_usage(edit) -> bool:
    is_r_verb_usage = False

    if edit.o_str.lower() in aux_list and edit.c_str.lower() in aux_list:
        return is_r_verb_usage

    if edit.type == "R:VERB":
        # Use lcs algorithm to calculate how it is similar
        longer, shorter = compare_text_lengths(edit.o_str, edit.c_str)
        lcs_threshold = 0.3
        if lcs(shorter, longer) <= lcs_threshold:
            is_r_verb_usage = True
    return is_r_verb_usage


# Specific case below
# [Hi". -> Hi."] this case must be [tagging : "R:OTHER"] but [tagging : "R:WO"] beacuse [Hi " .] and [Hi . "] are compared.
def check_R_WO_is_one_word(
    edit: Edit, orig_edit_str: str, cor_edit_str: str, types: set[str]
) -> set[str]:
    if edit.type == "R:WO":
        if is_one_word(orig_edit_str) or is_one_word(cor_edit_str):
            types.add("R:OTHER")
            types.remove("R:WO")
    return types


def check_specific_PUNCT(edit: Edit, types: set[str]) -> set[str]:
    # year, you -> year. You | tagging : "R:PUNCT"만 잡히는 현상, 이 케이스에서만 문제 생김, 다른건 잘 됨

    if edit.type in ["R:PUNCT", "U:PUNCT", "M:PUNCT"] and (
        len(edit.o_str.split()) >= 2 or len(edit.c_str.split()) >= 2
    ):
        o_str_en = re.sub(not_english_number, "", edit.o_str)
        c_str_en = re.sub(not_english_number, "", edit.c_str)
        if o_str_en != c_str_en and o_str_en.lower() == c_str_en.lower():
            types.add("R:ORTH")

    return types


def check_PUNCT(edit: Edit, types: set[str]) -> set[str]:
    # flight -> fly,
    # Got. -> good
    if len(edit.o_str) == 0 or len(edit.c_str) == 0:
        return types

    if re.search(not_english_number_and_space, edit.o_str) and not re.search(
        not_english_number_and_space, edit.c_str
    ):
        types.add("U:PUNCT")
        if (
            re.sub(not_english_number, "", edit.o_str.lower())
            == re.sub(not_english_number, "", edit.c_str.lower())
            and edit.type != "U:PUNCT"
        ):
            types.remove(edit.type)

    elif re.search(not_english_number_and_space, edit.c_str) and not re.search(
        not_english_number_and_space, edit.o_str
    ):
        types.add("M:PUNCT")

        if (
            re.sub(not_english_number, "", edit.o_str.lower())
            == re.sub(not_english_number, "", edit.c_str.lower())
            and edit.type != "M:PUNCT"
        ):
            types.remove(edit.type)

    return types


def check_R_ORTH(
    orig_edit_str: str, cor_edit_str: str, types: set[str], edit: Edit
) -> set[str]:
    # Check the capitalization.
    # The case of [good -> Good .] is tagged ["R:OTHER"]. The correct tag is ["R:ORTH", "M:PUNCT"].
    # In this case

    if is_not_only_punct(edit):
        if is_same_only_small_english_and_number(orig_edit_str, cor_edit_str):
            if is_not_same_only_original_english_and_number(
                orig_edit_str, cor_edit_str
            ):
                if "R:OTHER" == edit.type and edit.type in types:
                    types.remove("R:OTHER")
                types.add("R:ORTH")

    return types


def check_DET(edit: Edit, types: set[str]) -> set[str]:
    if edit.o_str in ["a", "an", "the"] and edit.c_str == "":
        types = {"U:DET"}
    elif edit.c_str in ["a", "an", "the"] and edit.o_str == "":
        types = {"M:DET"}
    elif edit.o_str in ["a", "an", "the"] and edit.c_str in ["a", "an", "the"]:
        types = {"R:DET"}
    return types


def check_LAST(orig_edit_str: str, cor_edit_str: str, types: set[str]) -> set[str]:
    # 체크하려는 original edit을 전체적으로 확인
    # I 'd -> I'd | tagging : "" 이 생기는 문제
    # 공백 및 소문자로 변환시에 같다면 R:ORTH
    # 그게 아니라면 R:OTHER
    if orig_edit_str != cor_edit_str:
        if (
            re.sub(space, "", orig_edit_str) == re.sub(space, "", cor_edit_str)
            or orig_edit_str.lower() == cor_edit_str.lower()
            or re.sub(space, "", orig_edit_str.lower())
            == re.sub(space, "", cor_edit_str.lower())
        ):
            types = {"R:ORTH"}
        elif types == set():
            types = {"R:OTHER"}
        return types
    return set()


def refine_category(types: set[str], is_r_verb_usage: bool, orig_edit_str: str, cor_edit_str: str, use_feedback: bool) -> list[dict[str, str]]:
    tag_list = []
    for _type in types:
        _dict = {}
        if is_r_verb_usage and _type == "R:VERB":
            _dict["type"] = "R:VERB"
            _dict["category"] = "Usage"
            if use_feedback:
                _dict["feedback"] = make_feedback("R:VERB", orig_edit_str, cor_edit_str)
        else:
            _dict["type"] = _type
            _dict["category"] = type_dict["category"][_type]
            if use_feedback:
                _dict["feedback"] = make_feedback(_type, orig_edit_str, cor_edit_str)
        tag_list.append(_dict)

    # For a few tags we divide it into important part and unimportant part.
    if len(tag_list) > 1:
        subordinate_dict = [
            _dict for _dict in tag_list if list(_dict.values())[0] in type_dict["less_important"]
        ]
        important_dict = [
            _dict for _dict in tag_list if list(_dict.values())[0] not in type_dict["less_important"]
        ]
        important_dict.extend(subordinate_dict)
        tag_list = important_dict

    return tag_list


def is_one_word(text: str) -> bool:
    return len(text.split()) == 1


def is_not_only_punct(edit: Edit) -> bool:
    # [. -> ,]
    return (
        re.search(english_number, edit.o_str) is not None
        and re.search(english_number, edit.c_str) is not None
    )


def is_same_only_small_english_and_number(
    orig_edit_str: str, cor_edit_str: str
) -> bool:
    # After leaving only the English part and converting it to lowercase, Check whether two sentences are the same.
    return (
        str(re.sub(not_english_number, "", orig_edit_str)).lower()
        == str(re.sub(not_english_number, "", cor_edit_str)).lower()
    )


def is_not_same_only_original_english_and_number(
    orig_edit_str: str, cor_edit_str: str
) -> bool:
    return re.sub(not_english_number, "", orig_edit_str) != re.sub(
        not_english_number, "", cor_edit_str
    )


# Utils
aux_list = [
    "do",
    "did",
    "will",
    "would",
    "can",
    "could",
    "shell",
    "should",
    "must",
    "should",
    "have",
    "has",
    "had",
]
english_number = r"[a-zA-Z0-9]"
not_english_number_and_space = r"[^a-zA-Z0-9\s\t\n\r\v\f]"
not_english_number = r"[^a-zA-Z0-9]"
space = r"[\s\t\n\r\v\f]"


def get_edits(origin_text: str, correc_text: str, annotator: Annotator) -> list[Edit]:
    orig = annotator.parse(origin_text)
    cor = annotator.parse(correc_text)
    edits = annotator.annotate(orig, cor)
    return edits
