"""Model with 3 by 3 convolutions"""
from typing import Any, Dict

import torch
from torch import Tensor
from torch.nn import Module

from emotion_clf.models.modules import FC, Collapse, Conv, Output
import pdb

def get_model(model_params: Dict[str, Any]) -> Module:
    return Convs(**model_params)


class Convs(Module):
    """Model with 3 by 3 convolutions"""
    def __init__(self,
                 n_mels: int = 128,
                 input_length: int = 128,
                 n_class: int = 56,
                 dropout: float = 0.2) -> None:
        super().__init__()

        self.dropout = torch.nn.Dropout(dropout)
        self.conv1 = Conv(1, 64, stride=2)
        self.conv2 = Conv(64, 64, stride=1)
        self.conv3 = Conv(64, 128, stride=1)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.filter1 = Conv(128, 256, stride=2, dropout=dropout)
        self.filter2 = Conv(256, 256, stride=2, dropout=dropout)
        self.filter3 = Conv(256, 256, stride=1, dropout=dropout)
        self.filter4 = Conv(256, 256, stride=2, dropout=dropout)
        self.collapse = Collapse(256,
                                 512, (n_mels // 32, input_length // 32),
                                 dropout=dropout)
        self.fco = FC(512, 1024, dropout=0.5)
        self.output = Output(1024, n_class, dropout=0.5)

    def forward(self, inp: Tensor) -> Tensor:
        out = inp.unsqueeze(1)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.maxpool(out)
        out = self.filter1(out)
        out = self.filter2(out)
        out = self.filter3(out)
        out = self.filter4(out)
        out = self.collapse(out)
        out = self.fco(out)
        out = self.output(out)
        return out
