import copy
import json
import re

f = open("src/v1/resources/feedback.json", "r", encoding="utf-8")
feedback_dict = json.load(f)

def make_feedback(tag_type: str, orig_edit_str: str, cor_edit_str: str):

    _feedback_dict = copy.deepcopy(feedback_dict)
    comment = _feedback_dict[tag_type]["comment"]
    comment_word_info = []
    # 설정해놓은 피드백에서 구체적인 틀린 text, 고쳐진 text로 변경합니다.
    if "wrong_word" in comment:
        comment = re.sub("wrong_word", orig_edit_str, comment)
    if "added_word" in comment:
        comment = re.sub("added_word", cor_edit_str, comment)
    
    # 피드백에서 변경한 단어의 index를 반환해줍니다. 글자를 편하게 찾으라는 의미
    if cor_edit_str in comment:
        escaped_pattern = re.escape(cor_edit_str)
        matches = re.finditer(escaped_pattern, comment)
        _info = [{"word":cor_edit_str, "start_index":x.start(), "end_index":x.end() } for x in matches]
        comment_word_info.append(_info[0])
    if orig_edit_str in comment:
        escaped_pattern = re.escape(orig_edit_str)
        matches = re.finditer(escaped_pattern, comment)
        _info = [{"word":orig_edit_str, "start_index":x.start(), "end_index":x.end() } for x in matches]
        comment_word_info.append(_info[0])

    _feedback_dict[tag_type]["comment"] = comment
    _feedback_dict[tag_type]["comment_info"] = comment_word_info
    new_order = ["skill_name_en", "skill_name_kr", "comment", "comment_info", "example_1", "example_2"]
    sort_feedback_dict = {key: _feedback_dict[tag_type][key] for key in new_order}
    return sort_feedback_dict