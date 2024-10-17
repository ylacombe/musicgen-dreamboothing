"""Compute mel spectrograms"""

import torch
import torchaudio
import pdb

def get_preprocessor(preproc_params):
    return MelSpectrogram(**preproc_params)


class MelSpectrogram(torch.nn.Module):
    def __init__(self,
                 sample_rate=16000,
                 n_fft=512,
                 f_min=0.0,
                 f_max=8000.0,
                 n_mels=96):
        super().__init__()
        self.spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels)

        self.to_db = torchaudio.transforms.AmplitudeToDB()

    def forward(self, inp):
        out = self.spec(inp)
        out = self.to_db(out)
        return out[:, 1:]
