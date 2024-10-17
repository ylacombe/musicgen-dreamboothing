"""Losses"""
from typing import Optional
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import _Loss


def _calc_weights(probas_path: Optional[str]):
    pos_weight = None
    weight = None
    if probas_path is not None:
        probas = torch.from_numpy(np.load(probas_path))
        pos_weight = torch.tensor(1.) / probas
        weight = torch.tensor(2.) / (torch.tensor(1.) + pos_weight)
    return (pos_weight, weight)


class WeightedBCEWithLogitsLoss(BCEWithLogitsLoss):
    def __init__(self,
                 probas_path: Optional[str] = "probas_train.npy",
                 reduction: str = "mean") -> None:

        probas_path = os.path.dirname(os.path.dirname(__file__)) + "/" + probas_path
        probas_path = "./emotion_clf/probas_train.npy"
        pos_weight, weight = _calc_weights(probas_path)
        super().__init__(pos_weight=pos_weight,
                         reduction=reduction,
                         weight=weight)


class BinaryFocalLossWithLogits(_Loss):
    def __init__(self,
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 reduction: str = 'mean') -> None:
        super().__init__()
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return binary_focal_loss_with_logits(input, target, self.alpha,
                                             self.gamma, self.reduction)


def binary_focal_loss_with_logits(input: Tensor,
                                  target: Tensor,
                                  alpha: float = 0.25,
                                  gamma: float = 2.0,
                                  reduction: str = 'none') -> Tensor:
    if not isinstance(input, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(input)}")

    if not len(input.shape) >= 2:
        raise ValueError(
            f"Invalid input shape, we expect BxCx*. Got: {input.shape}")

    if input.size(0) != target.size(0):
        raise ValueError(f"Expected input batch_size ({input.size(0)})"
                         f"match target batch_size ({target.size(0)}).")

    probs_pos = torch.sigmoid(input)
    probs_neg = torch.sigmoid(-input)
    loss_tmp = -alpha * torch.pow(probs_neg, gamma) * target * F.logsigmoid(
        input) - (1. - alpha) * torch.pow(probs_pos, gamma) * (
            torch.tensor(1.) - target) * F.logsigmoid(-input)

    if reduction == 'none':
        loss = loss_tmp
    elif reduction == 'mean':
        loss = torch.mean(loss_tmp)
    elif reduction == 'sum':
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")
    return loss
