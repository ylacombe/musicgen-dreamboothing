"""Evaluates the model"""
from typing import Dict, Tuple

import torch
import numpy as np
from tqdm import tqdm

import emotion_clf.ml.utils as ml_utils
from emotion_clf.ml.experiment import Experiment
import pdb

def evaluate(
    exp: Experiment,
    split: str,
    calculate_metrics: bool = True,
    use_swa: bool = False
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Evaluate model of experiment on given split.

    Parameters
    ----------
    exp: Name of the experiment
    split: Name of the split on which to evaluate
    calculate_metrics: whether to calculate metrics specified in parameters
    use_swa: whether to evaluate averaged model stored in exp.swa

    Returns
    -------
    summary: calculated metrics
    eval_targets: list of ground_truth batches
    eval_outputs: list of prediction batches
    """

    if use_swa:
        model = exp.swa["model"]
    else:
        model = exp.model

    model.eval()

    if split != "inference":
        loss_avg = ml_utils.RunningAverage()

        outs = []
        targs = []

        with tqdm(total=len(exp.dls[split]), desc="- " + split, ncols=100) as tqbar:
            for data_batch, target_batch in exp.dls[split]:
                with torch.no_grad():
                    data_batch = data_batch.to(exp.device)
                    target_batch = target_batch.to(exp.device)

                    if exp.preprocessor is not None:
                        data_batch = exp.preprocessor(data_batch)
                    stats = exp.params.get("stats")
                    if stats is not None:
                        data_batch = (data_batch - stats["mean"]) / stats["std"]

                    output_batch = model(data_batch)
                    loss = exp.loss(output_batch.mean(-1), target_batch)

                outs.append(output_batch.detach())
                targs.append(target_batch.detach())

                loss_avg.update(loss.item())
                tqbar.set_postfix(loss=f"{loss_avg():.4g}")
                tqbar.update()

            eval_outputs, eval_targets = exp.eval_activation(outs, targs)

            summary = {}
            if calculate_metrics:
                summary.update({
                    met: exp.metrics[met](eval_targets, eval_outputs)
                    for met in exp.metrics
                })

        return summary, eval_targets, eval_outputs
    
    else:
        outs = []
        for data_batch, _ in exp.dls:
            with torch.no_grad():
                data_batch = data_batch.to(exp.device)

                stats = exp.params.get("stats")      
                if stats is not None:
                    data_batch = (data_batch - stats["mean"]) / stats["std"]
        
                output_batch = model(data_batch)
            outs.append(output_batch.detach())
        return _, _, outs
        