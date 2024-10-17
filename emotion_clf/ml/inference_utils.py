import logging
import os
import time
from typing import List, Optional

import numpy as np
import torch
import torchaudio
import librosa

import emotion_clf.ml as ml
import pdb

TAG_MAP= {"0": "action", "1": "adventure", "2": "advertising", "3": "background", "4": "ballad", "5": "calm",
        "6": "children", "7": "christmas", "8": "commercial", "9": "cool", "10": "corporate", "11": "dark",
        "12": "deep", "13": "documentary","14": "drama", "15": "dramatic", "16": "dream", "17": "emotional",
        "18": "energetic", "19": "epic", "20": "fast", "21": "film", "22": "fun", "23": "funny", "24": "game",
        "25": "groovy", "26": "happy", "27": "heavy", "28": "holiday", "29": "hopeful", "30": "inspiring",
        "31": "love", "32": "meditative", "33": "melancholic", "34": "melodic", "35": "motivational",
        "36": "movie", "37": "nature", "38": "party", "39": "positive", "40": "powerful", "41": "relaxing",
        "42": "retro", "43": "romantic", "44": "sad", "45": "sexy", "46": "slow", "47": "soft", "48": "soundscape",
        "49": "space", "50": "sport", "51": "summer", "52": "trailer", "53": "travel", "54": "upbeat", "55": "uplifting"
    }


def inference_process(file):
    if type(file) == str:
        data, sr = torchaudio.load(file)
        data = data.detach().mean(0)  # to mono, shape=torch.Size([length])
    else:  ## if is waveform
        data = file if file.device == torch.device("cpu") else file.detach().to("cpu")
        
    # data = data.detach().mean(0)  # to mono
    #data = torchaudio.transforms.Resample(sr, 44100, dtype=data.dtype)(data)
    data, _ = librosa.effects.trim(data)
    data = data.to("cuda")
    preprocessor = ml.loading.load_preprocessor({"name": "melspectrogram",
                                                    "params": {
                                                        "sample_rate": 44100,
                                                        "n_fft": 2048,
                                                        "f_min": 0.0,
                                                        "f_max": 16000,
                                                        "n_mels": 128
                                                }}, "cuda")
    melspec = preprocessor(data)

    # from AudioFolder
    memmap = melspec
    length = memmap.shape[-1]
    pos = np.random.random_sample()
    idx = int(pos * (length - 224))
    audio = memmap[:, idx:idx + 224]
    return audio[None,:]  # 1, 128, 224