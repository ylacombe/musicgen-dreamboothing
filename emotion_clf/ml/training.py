"""Train the model"""
from typing import Tuple

import torch
from torch import Tensor
from tqdm import tqdm

import emotion_clf.ml.utils as mlutils
from emotion_clf.ml.experiment import Experiment
import pdb

def train_one_epoch(exp: Experiment, split: str):
    """Train one epoch or experiment on given split"""

    exp.model.train()

    loss_avg = mlutils.RunningAverage()

    outs = []
    targs = []

    with tqdm(total=len(exp.dls[split]), desc="- " + split,
              ncols=100) as tqbar:
        for data_batch, target_batch in exp.dls[split]:
            data_batch = data_batch.to(exp.device)
            target_batch = target_batch.to(exp.device)

            with torch.no_grad():
                if exp.preprocessor is not None:
                    data_batch = exp.preprocessor(data_batch)
                stats = exp.params.get("stats")
                if stats is not None:
                    data_batch = (data_batch - stats["mean"]) / stats["std"]

                if exp.mixup is not None:
                    data_batch, targets_a, targets_b, lam = mixup_data(
                        data_batch, target_batch, exp.mixup, exp.device)
                    target_batch = lam * targets_a + (1. - lam) * targets_b

            with torch.cuda.amp.autocast(enabled=exp.device == "cuda"):
                output_batch = exp.model(data_batch)
                if exp.mixup is None:
                    loss = exp.loss(output_batch.squeeze(-1), target_batch)
                else:
                    loss = mixup_criterion(exp.loss, output_batch.squeeze(-1),
                                           targets_a, targets_b, lam)

            exp.optimizer.zero_grad()

            exp.scaler.scale(loss).backward()
            if exp.gradient_clipping is not None:
                exp.scaler.unscale_(exp.optimizer)
                torch.nn.utils.clip_grad_norm_(exp.model.parameters(),
                                               exp.gradient_clipping)

            exp.scaler.step(exp.optimizer)
            exp.scaler.update()

            if len(exp.metrics) > 0:
                outs.append(output_batch.detach().float())
                targs.append(target_batch.detach())

            loss_avg.update(loss.item())
            tqbar.set_postfix(loss=f"{loss_avg():.4g}")
            tqbar.update()

        summary = {}
        if len(exp.metrics) > 0:
            eval_outputs, eval_targets = exp.eval_activation(outs, targs)
            summary.update({
                met: exp.metrics[met](eval_targets, eval_outputs)
                for met in exp.metrics
            })

    return summary


def mixup_data(x: Tensor, y: Tensor, alpha: float,
               device: str) -> Tuple[Tensor, Tensor, Tensor, float]:
    """Mixup data"""

    if alpha > 0:
        lam = torch.distributions.beta.Beta(alpha, alpha).sample().item()
    else:
        lam = 1.

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1. - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a.to(device), y_b.to(device), lam


def mixup_criterion(criterion: torch.nn.Module, pred: Tensor, y_a: Tensor,
                    y_b: Tensor, lam: float) -> Tensor:
    """Mixup loss"""

    return lam * criterion(pred, y_a) + (1. - lam) * criterion(pred, y_b)


def update_swa_bn(exp: Experiment, split: str):
    """Update batchnorm running means for swa"""

    model = exp.swa["model"]

    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta:
        module.momentum = None
        module.num_batches_tracked *= 0

    with tqdm(total=len(exp.dls[split]), desc="- updating bn",
              ncols=100) as tqbar:
        for data_batch, _ in exp.dls[split]:
            data_batch = data_batch.to(exp.device)
            with torch.no_grad():
                if exp.preprocessor is not None:
                    data_batch = exp.preprocessor(data_batch)
                stats = exp.params.get("stats")
                if stats is not None:
                    data_batch = (data_batch - stats["mean"]) / stats["std"]

                with torch.cuda.amp.autocast():
                    _ = model(data_batch)

            tqbar.update()

    for bn_module in momenta:
        bn_module.momentum = momenta[bn_module]

    model.train(was_training)
