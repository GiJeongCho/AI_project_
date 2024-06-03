import os

from flair.data import Sentence, Token
from flair.models import SequenceTagger
from segtok.segmenter import split_single
from v1.utils.schemas import Chunk, ChunkToken, ResponseChunking


class ChunkMaker:
    def __init__(self):
        # self.tagger_large = SequenceTagger.load("/home/edutem/joge/tts_melo/src2/v1/model/chunk_large.pt")
        self.tagger_large = SequenceTagger.load(r"/model/chunk_large.pt")

    def sentence_tokenizer(self, sentence):
        return [x for x in split_single(sentence) if len(x.strip()) > 0]

    def chunk_tag_predict(self, sentence):
        flair_tagged_sentence = Sentence(sentence)
        self.tagger_large.predict(flair_tagged_sentence)
        return flair_tagged_sentence

    def is_chunk_part(self, token: Token, index: int):
        return token.tag == "1" and index != 0

    def is_empty(self, sentence):
        return len(sentence.strip()) == 0

    def sentence_index_extractor(self, tokenized_sentence, index_sentence):
        sentence_length = len(tokenized_sentence)
        sentence_token_start_index = index_sentence.find(tokenized_sentence)
        index_sentence = index_sentence[sentence_length + sentence_token_start_index :]

        return index_sentence

    def chunks_maker(
        self,
        sentence,
        flair_tagged_sentence,
        chunks,
        chunk_tokens,
        sentence_start_index,
    ):
        for index, token in enumerate(flair_tagged_sentence.tokens):
            token_information = ChunkToken.make_token_dict(token, sentence_start_index)
            if self.is_chunk_part(token, index):
                chunks.append(Chunk.make_chunk_dict(chunk_tokens, sentence))
                chunk_tokens.clear()
            chunk_tokens.append(token_information)

        chunks.append(Chunk.make_chunk_dict(chunk_tokens, sentence))
        chunk_tokens.clear()

        return chunks

    def make_chunked_sentence(self, sentence):
        chunks: list[Chunk] = []
        chunk_tokens: list[ChunkToken] = []
        index_sentence = sentence
        sentence_start_index = 0

        if self.is_empty(sentence):
            return ResponseChunking(sentence="", chunks=[])

        for tokenized_sentence in self.sentence_tokenizer(sentence):
            sentence_length = len(tokenized_sentence)
            sentence_start_index += index_sentence.find(tokenized_sentence)

            index_sentence = self.sentence_index_extractor(
                tokenized_sentence, index_sentence
            )

            flair_tagged_sentence = self.chunk_tag_predict(tokenized_sentence)

            chunks = self.chunks_maker(
                sentence,
                flair_tagged_sentence,
                chunks,
                chunk_tokens,
                sentence_start_index,
            )

            sentence_start_index += sentence_length

        return ResponseChunking(sentence=sentence, chunks=chunks)


chunk_maker = ChunkMaker()
