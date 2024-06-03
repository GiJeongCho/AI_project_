import json
import os


def test_type_dict_json():
    assert os.path.exists("resources/type_dict.json")
    type_dict = json.load(open("resources/type_dict.json", "r"))
    assert all(
        value in {"Grammar", "Usage", "Punctuation", "Mechanics", "Other"}
        for value in type_dict.values()
    )
