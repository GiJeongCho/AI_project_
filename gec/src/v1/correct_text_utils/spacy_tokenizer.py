import string

import nltk
import spacy
from spacy.lang.char_classes import (
    ALPHA,
    ALPHA_LOWER,
    ALPHA_UPPER,
    CONCAT_QUOTES,
    HYPHENS,
    LIST_ELLIPSES,
    LIST_ICONS,
)
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer
from spacy.util import compile_infix_regex

default_nlp = English()


# from : https://stackoverflow.com/questions/50330455/how-to-detokenize-spacy-text-without-doc-context
class Detokenizer:
    """This class is an attempt to detokenize spaCy tokenized sentence"""

    def __init__(self, nlp=None):
        if nlp is None:
            self.nlp = spacy.load("en_core_web_sm")
        else:
            self.nlp = default_nlp

    def __call__(self, tokens: list):
        """Call this method to get list of detokenized words"""
        while self._connect_next_token_pair(tokens):
            pass
        return " ".join(tokens)

    def get_sentence(self, tokens: list) -> str:
        """call this method to get detokenized sentence"""
        return " ".join(self(tokens))

    def _connect_next_token_pair(self, tokens: list):
        i = self._find_first_pair(tokens)
        if i == -1:
            return False
        tokens[i] = tokens[i] + tokens[i + 1]
        tokens.pop(i + 1)
        return True

    def _find_first_pair(self, tokens):
        if len(tokens) <= 1:
            return -1
        for i in range(len(tokens) - 1):
            if self._would_spaCy_join(tokens, i):
                return i
        return -1

    def _would_spaCy_join(self, tokens, index):
        """
        Check whether the sum of lengths of spaCy tokenized words is equal to the length of joined and then spaCy tokenized words...

        In other words, we say we should join only if the join is reversible.
        eg.:
            for the text ["The","man","."]
            we would joins "man" with "."
            but wouldn't join "The" with "man."
        """

        left_part = tokens[index]
        right_part = tokens[index + 1]
        length_before_join = len(self.nlp(left_part)) + len(self.nlp(right_part))
        length_after_join = len(self.nlp(left_part + right_part))
        if self.nlp(left_part)[-1].text in string.punctuation:
            return False
        # edge cases
        elif right_part in ["'m", "'ve"]:
            return True
        elif left_part + right_part == "I'ma":
            return False
        return length_before_join == length_after_join


# from : lm-critic
def get_tokenizer_gec(nlp):
    infixes = (
        LIST_ELLIPSES
        + LIST_ICONS
        + [
            r"(?<=[0-9])[+\-\*^](?=[0-9-])",
            r"(?<=[{al}{q}])\.(?=[{au}{q}])".format(
                al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
            ),
            r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
            # r"(?<=[{a}])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS),
            r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
            # bracket between characters
            r"\b\(\b",
            r"\b\)\b",
        ]
    )
    infix_re = compile_infix_regex(infixes)
    return Tokenizer(
        nlp.vocab,
        prefix_search=nlp.tokenizer.prefix_search,
        suffix_search=nlp.tokenizer.suffix_search,
        infix_finditer=infix_re.finditer,
        token_match=nlp.tokenizer.token_match,
        rules=nlp.Defaults.tokenizer_exceptions,
    )


def get_tokenizer_bea19(nlp):
    infixes = (
        LIST_ELLIPSES
        + LIST_ICONS
        + [
            r"(?<=[0-9])[+\-\*^](?=[0-9-])",
            r"(?<=[{al}{q}])\.(?=[{au}{q}])".format(
                al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
            ),
            r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
            r"(?<=[{a}])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS),
            r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
        ]
    )
    infix_re = compile_infix_regex(infixes)
    return Tokenizer(
        nlp.vocab,
        prefix_search=nlp.tokenizer.prefix_search,
        suffix_search=nlp.tokenizer.suffix_search,
        infix_finditer=infix_re.finditer,
        token_match=nlp.tokenizer.token_match,
        rules=nlp.Defaults.tokenizer_exceptions,
    )


tokenizer_gec = get_tokenizer_gec(default_nlp)
default_nlp.tokenizer = tokenizer_gec
detokenizer_gec = Detokenizer(default_nlp)
# tokenizer_bea19 = get_tokenizer_bea19(nlp)


# spacy 단점

# 1. detokenize가 완전한 역방향을 보장하지 못하는 케이스 존재
# 1-1. 원문장에 연속 3개 이상 스페이스 있을시
# ex) I   have some apples.
# 1-2. 괄호
# ex) I am(was)

# nltk 단점
# 1. dont의 경우 dont로 토큰화 함 -> 에러를 잡지 못함
# 2. puctuation들을 잘못 잡음 (" => ``, detokenize 시 온점에 띄어쓰기)


def tokenize(text):
    tokens = spacy_tokenize_gec(text)
    # add for the case when ending punctuations are not tokenized
    end_char = tokens[-1][-1]
    if end_char in ["!", "?", "."] and len(tokens[-1]) != 1:
        tokens[-1] = tokens[-1][:-1]
        tokens.append(end_char)
    return tokens


detokenizer = nltk.TreebankWordDetokenizer()


def detokenize(tokens):
    second_tokens = detokenizer_gec(tokens).split()
    return detokenizer.detokenize(second_tokens)


def spacy_tokenize_gec(text):
    default_nlp.tokenizer = tokenizer_gec
    return [str(w) for w in default_nlp(text)]


# def spacy_tokenize_bea19(text):
#     nlp.tokenizer = tokenizer_bea19
#     return [str(w) for w in nlp(text)]
