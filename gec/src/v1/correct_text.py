import asyncio
import os
import re
from functools import lru_cache
from typing import Any

import stanza
from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch
from fastapi import FastAPI
import json
from src.v1.correct_text_utils.calculator import lcs
from src.v1.correct_text_utils.parser import Parser
from src.v1.schemas import GECRequest
from stanza.models.common.doc import Document
from transformers import PreTrainedTokenizerFast, AutoTokenizer, T5ForConditionalGeneration , T5Config
import warnings
warnings.filterwarnings('ignore')

import errant
import spacy

# 예외 처리와 로깅을 추가하여 언어 모델 로드 과정을 검증
try:
    nlp = spacy.load("en_core_web_sm")
    annotator = errant.load("en", nlp=nlp)
except Exception as e:
    logging.error(f"Failed to load language model or ERRANT: {e}")
    raise


nlp = stanza.Pipeline(lang="en", processors="tokenize", use_gpu=torch.cuda.is_available(), download_method=None)
error_messages = open("resources/chatgpt_error_messages.txt", "r").readlines()

app = FastAPI()


model_path = r"/model/trained_all_v10_grammarlycoedit-xl/best_model"

model_save_path = model_path
tokenizer_save_path = model_path


tokenizer = AutoTokenizer.from_pretrained(tokenizer_save_path,force_download=False)
model = T5ForConditionalGeneration.from_pretrained(model_save_path,force_download=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # 모델을 평가 모드로 설정

async def main(request: GECRequest) -> dict[str, Any]:
    original_text = space_preprocess(request.text)
    use_feedback = request.feedback
    corrected_text = await correct_text(original_text, 150)  # 글자 수 제한 해서 끊는 부분.
    print(f"original_text : {original_text}")
    print(f"corrected_text : {corrected_text}")
    parse = Parser(original_text, corrected_text, use_feedback)
    result = parse.parse_corrections()
    return result

async def correct_text(original_text: str, segment_length: int) -> str:
    # Space preprocess only for chatGPT inference.
    # When calculating offset, unconverted text is used.
    sentences = split_text_into_unit(original_text, unit_size=segment_length)
    corrected_text = await create_asyncio_gpt_task(sentences)
    return corrected_text

async def create_asyncio_gpt_task(sentences: list[str]) -> str:
    tasks = []

    for sentence in sentences:
        task = asyncio.create_task(get_gpt_result(sentence))
        tasks.append(task)

    await asyncio.gather(*tasks)
    corrected_text_list = [task.result() for task in tasks]
    return " ".join(corrected_text_list)

async def get_gpt_result(text: str) -> str:
    result1, result2 = await correct_sentence(text)
    proper_result = check_inference(text, result1, result2)
    return proper_result

@app.post("/correct/")
async def correct_sentence(text: str) -> tuple[str, str]:
    global model, tokenizer, device
    prompt = "Fix the grammar and Write in a more neutral way: "
    with torch.no_grad():
        all_text = prompt + text
        inputs = tokenizer.encode(all_text, return_tensors="pt", truncation=True).to(device)
        outputs = model.generate(inputs, max_length=256)
        corrected_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_sentence, corrected_sentence

def check_inference(text: str, result1: str, result2: str) -> str:
    if not result1 or not result2:
        return text

    text = check_text(text)

    result1 += "\n"
    result2 += "\n"
    for error_message in error_messages:
        if error_message.strip() in result1:
            return text.strip()
        if error_message.strip() in result2:
            return text.strip()

    result_1 = sentence_preprocessor(text, result1)
    result_2 = sentence_preprocessor(text, result2)

    lcs_score_1 = min(
        round(lcs(result_1.lower(), text.lower())),
        round(lcs(text.lower(), result1.lower())),
    )
    lcs_score_2 = min(
        round(lcs(result_2.lower(), text.lower())),
        round(lcs(text.lower(), result2.lower())),
    )

    result_dict = {lcs_score_1: result_1, lcs_score_2: result_2}
    lcs_score = max(lcs_score_1, lcs_score_2)

    if text in result1 or text in result2:
        if lcs_score < 0.7:
            return text

    highest_result = result_dict[lcs_score]
    if lcs_score > 0.5:
        text = highest_result

    if text[-1] not in ["?", ".", ",", "!", "'", '"']:
        text += "."

    return text

def sentence_preprocessor(input_text: str, output_text: str) -> str:
    output_text = output_text.strip()

    if output_text == "":
        return output_text

    if has_no_quotes(output_text):
        return output_text

    if not has_start_end_quotes(input_text) and has_start_end_quotes(output_text):
        if check_double_quotates(output_text) and count_double_quotes(output_text) > 0:
            output_text = output_text[1:-1]
        elif check_quotate(output_text) and count_single_quotes(output_text) > 0:
            output_text = output_text[1:-1]

    if count_single_quotes(output_text) > 0:
        if count_plural_apostrophy(output_text) > 0:
            pass
        elif not check_quotate(output_text):
            if output_text[-1] == "'":
                if not check_backward_quotates(output_text):
                    output_text = output_text[:-1]
            if output_text[0] == "'":
                if not check_forward_quotates(output_text):
                    output_text = output_text[1:]

    if count_double_quotes(output_text) > 0:
        if not check_double_quotates(output_text):
            if output_text[-1] == '"':
                if not check_backward_double_quotates(output_text):
                    output_text = output_text[:-1]
            if output_text[0] == '"':
                if not check_forward_double_quotates(output_text):
                    output_text = output_text[1:]

    return output_text

def check_text(sentence: str) -> str:
    sentence = sentence.strip()
    sentence_split_list = sentence.split(" ")

    if sentence_split_list[0] == "Corrected:":
        sentence = " ".join(sentence_split_list[1:])
    elif sentence_split_list[0] == "Corrected" and sentence_split_list[1] == "text:":
        sentence = " ".join(sentence_split_list[2:])
    elif sentence_split_list[0] == "Corrected" and sentence_split_list[1] == "Text:":
        sentence = " ".join(sentence_split_list[2:])
    elif sentence_split_list[0] == "Text" and sentence_split_list[1] == "correction:":
        sentence = " ".join(sentence_split_list[2:])

    return sentence

def split_text_into_unit(original_text: str, unit_size: int) -> list[str]:
    target_segment_index_list = binding_segment_index(
        segment_text_to_index(original_text), unit_size
    )
    target_segment_text_list = [
        original_text[start:end] for start, end in target_segment_index_list
    ]

    return target_segment_text_list

def binding_segment_index(
        sentence_index_list: list[tuple[int, int]], unit: int
) -> list[tuple[int, int]]:
    length_list = [end - start for start, end in sentence_index_list]

    result = []
    current_sum = 0
    current_group = []
    for length in length_list:
        if current_sum + length < unit:
            current_group.append(length)
            current_sum += length
        else:
            if len(current_group) > 0:
                result.append(current_group)
            current_group = [length]
            current_sum = length

    if current_group:
        result.append(current_group)

    segment_index_result = []
    index_end = 0
    for length_group in result:
        index_start = index_end
        index_end += len(length_group)
        segment_start = sentence_index_list[index_start][0]
        segment_end = sentence_index_list[index_end - 1][-1]
        segment_index_result.append((segment_start, segment_end))

    return segment_index_result

def segment_text_to_index(original_text: str) -> list[tuple[int, int]]:
    doc = nlp(original_text)
    if not isinstance(doc, Document):
        raise TypeError(f"doc is not Document type : {doc}")

    sentence_indexs = [
        (inform.to_dict()[0]["start_char"], inform.to_dict()[-1]["end_char"])
        for inform in doc.sentences
    ]

    return sentence_indexs

def space_preprocess(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\r", "", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\t{2,}", "\t", text)
    return text

def check_double_quotates(text: str) -> bool:
    return count_double_quotes(text) % 2 == 0

def check_quotate(text: str) -> bool:
    return count_single_quotes(text) % 2 == 0

def has_start_end_quotes(text: str) -> bool:
    return text[0] == text[-1] and text[0] in ['"', "'"]

def has_no_quotes(text: str) -> bool:
    return re.search(r"['\"]", text) is None

@lru_cache(maxsize=2)
def apostrophy_count(text: str) -> int:
    return len(re.findall(r"(s\'|\'s|n\'t|\'re|\'m|\'ll|\'d|'ve)(\s|$)", text))

def count_single_quotes(text: str) -> int:
    return len(re.findall("'", text)) - apostrophy_count(text)

@lru_cache(maxsize=2)
def count_double_quotes(text: str) -> int:
    return len(re.findall('"', text))

def count_plural_apostrophy(text: str) -> int:
    return len(re.findall(r"s\'(\s|$)", text))

def check_forward_double_quotates(text: str) -> bool:
    return re.findall(r"\" | \"", text)[:1] == ['" ']

def check_backward_double_quotates(text: str) -> bool:
    return re.findall(r"\" | \"", text)[-1:] == [' "']

def check_forward_quotates(text: str) -> bool:
    return re.findall(r"\' | \'", text)[:1] == ["' "]

def check_backward_quotates(text: str) -> bool:
    return re.findall(r"\' | \'", text)[-1:] == [" '"]
