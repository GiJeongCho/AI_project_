import io
import json
import os
# import re

# import librosa
import numpy as np
import soundfile
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio
from v1.melo import commons, utils
from v1.melo.download_utils import load_or_download_config, load_or_download_model
from v1.melo.mel_processing import spectrogram_torch, spectrogram_torch_conv
from v1.melo.models import SynthesizerTrn
from v1.melo.split_utils import split_sentence
from scipy.io.wavfile import write as wav_write
# from tqdm import tqdm

# cpu에서만 쓸거니까 gpu는 필요 없음
# speed 범위 알아보고 0.2~2사이로 조정
#  __init__부분 확인하기


class TTS(nn.Module):
    def __init__(self, 
                language,
                device='cpu',
                use_hf=True,
                config_path=None,
                ckpt_path=None):
        super().__init__()
        device = 'cpu'
        self.language = language.upper()  # 언어 코드를 대문자로 통일
 

        hps = load_or_download_config(language, use_hf=use_hf, config_path=config_path)

        num_languages = hps.num_languages
        num_tones = hps.num_tones
        symbols = hps.symbols

        model = SynthesizerTrn(
            len(symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            num_tones=num_tones,
            num_languages=num_languages,
            **hps.model,
        ).to(device)

        model.eval()
        self.model = model
        self.symbol_to_id = {s: i for i, s in enumerate(symbols)}
        self.hps = hps
        self.device = device
    
        # load state_dict
        checkpoint_dict = load_or_download_model(language, device, use_hf=use_hf, ckpt_path=ckpt_path)
        self.model.load_state_dict(checkpoint_dict['model'], strict=True)
        
        language = language.split('_')[0]
        self.language = 'ZH_MIX_EN' if language == 'ZH' else language # we support a ZH_MIX_EN model

    def tts_to_bytes(self, text, speaker_id, sr=None, speed=1.0, format='wav'):
        language = self.language
        sentences = self.split_sentences_into_pieces(text, language)
        audio_list = []
        for sentence in sentences:
            bert, ja_bert, phones, tones, lang_ids = utils.get_text_for_tts_infer(sentence, language, self.hps, self.device, self.symbol_to_id)
            with torch.no_grad():

                speakers = torch.LongTensor([speaker_id]).to(self.device)
                audio = self.model.infer(
                    phones.to(self.device).unsqueeze(0),
                    torch.LongTensor([phones.size(0)]).to(self.device),
                    speakers,
                    tones.to(self.device).unsqueeze(0),
                    lang_ids.to(self.device).unsqueeze(0),
                    bert.to(self.device).unsqueeze(0),
                    ja_bert.to(self.device).unsqueeze(0),
                    sdp_ratio=0.2,
                    noise_scale=0.6,
                    noise_scale_w=0.8,
                    length_scale=1. / speed,
                )[0][0, 0].data.cpu().numpy()
            audio_list.append(audio)

        audio = self.audio_numpy_concat(audio_list, sr=self.hps.data.sampling_rate if sr is None else sr, speed=speed)
        
        # print("Max amplitude in audio:", np.max(np.abs(audio)))
        # print("Min amplitude in audio:", np.min(np.abs(audio)))
        audio_buffer = io.BytesIO()
        # Scale the audio data for int16 format and write to a buffer
        scaled_audio = (audio * 32767).astype(np.int16)  # Correct scaling factor
        sf.write(audio_buffer, scaled_audio, self.hps.data.sampling_rate, format="wav")
        audio_buffer.seek(0)  # Rewind to the start of the BytesIO buffer

        return audio_buffer

    @staticmethod
    def audio_numpy_concat(segment_data_list, sr, speed=1.):
        audio_segments = []
        for segment_data in segment_data_list:
            audio_segments += segment_data.reshape(-1).tolist()
            audio_segments += [0] * int((sr * 0.05) / speed)
        audio_segments = np.array(audio_segments).astype(np.float32)
        return audio_segments

    @staticmethod
    def split_sentences_into_pieces(text, language, quiet=False):
        texts = split_sentence(text, language_str=language)
        if not quiet:
            # print(" > Text split to sentences.")
            # print('\n'.join(texts))
            # print(" > ===========================")
            return texts

