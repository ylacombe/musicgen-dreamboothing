"""Metrics"""
from typing import Dict, List, Tuple

import numpy as np
import sklearn.metrics
import torch
from torch import Tensor
from scipy import linalg
import emotion_clf.ml.losses as ml_losses


def extract_first_metric(metrics: Dict[str, np.ndarray]) -> float:
    return float(list(metrics.values())[0].mean())


def _avg_sigmoid(outs: Tensor) -> Tensor:
    return torch.mean(torch.nn.Sigmoid()(outs), 2)


def averaged_sigmoid(outputs: List[Tensor],
                     targets: List[Tensor]) -> Tuple[np.ndarray, np.ndarray]:
    probas = torch.cat(list(map(_avg_sigmoid, outputs))).cpu().numpy()
    targs = torch.cat(targets).cpu().numpy()
    return probas, targs


def roc_auc(targets: np.ndarray, probas: np.ndarray) -> np.ndarray:
    return sklearn.metrics.roc_auc_score(targets > 0.5, probas, average=None)


def pr_auc(targets: np.ndarray, probas: np.ndarray) -> np.ndarray:
    return sklearn.metrics.average_precision_score(targets > 0.5,
                                                   probas,
                                                   average=None)


def bce_loss(targets: np.ndarray, probas: np.ndarray) -> np.ndarray:
    loss = torch.nn.BCELoss(reduction='none')
    targs = torch.Tensor(targets)
    probs = torch.Tensor(probas)
    return torch.mean(loss(probs, targs), 0).cpu().numpy()


def focal_loss(targets: np.ndarray, probas: np.ndarray) -> np.ndarray:
    loss = ml_losses.BinaryFocalLossWithLogits(reduction='none')
    targs = torch.Tensor(targets)
    outs = torch.logit(torch.Tensor(probas))
    return torch.mean(loss(outs, targs), 0).cpu().numpy()


def weighted_bce_loss(targets: np.ndarray, probas: np.ndarray) -> np.ndarray:
    loss = ml_losses.WeightedBCEWithLogitsLoss(reduction="none")
    targs = torch.Tensor(targets)
    outs = torch.logit(torch.Tensor(probas))
    return torch.mean(loss(outs, targs), 0).cpu().numpy()


def mu_sigma_from_activations(act):
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def ensure_tensor(x, device=None):
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    return x.pin_memory().to(device, non_blocking=True) if device else x


def frechet_distance(
    mu_x: torch.Tensor,
    sigma_x: torch.Tensor,
    mu_y: torch.Tensor,
    sigma_y: torch.Tensor,
    device=None,
) -> torch.Tensor:
    mu_x = ensure_tensor(mu_x, device)
    sigma_x = ensure_tensor(sigma_x, device)
    mu_y = ensure_tensor(mu_y, device)
    sigma_y = ensure_tensor(sigma_y, device)
    a = (mu_x - mu_y).square().sum(dim=-1)
    b = sigma_x.trace() + sigma_y.trace()
    c = torch.linalg.eigvals(sigma_x @ sigma_y).sqrt().real.sum(dim=-1)
    return a + b - 2 * c
