"""Calculate mean and std of data"""

import math
from typing import Dict

import torch
from tqdm import tqdm

import emotion_clf.ml.utils
from emotion_clf.ml.experiment import Experiment


def calculate_stats(exp: Experiment) -> Dict[str, float]:
    """Calculate mean and std of data"""

    avg = ml.utils.RunningAverage()
    sq_avg = ml.utils.RunningAverage()

    proc = exp.preprocessor
    if proc is None:

        def proc(arg):
            return arg

    with tqdm(total=len(exp.dls["train"]), desc="calculating stats",
              ncols=100) as tqbar:
        for data, _ in exp.dls["train"]:
            with torch.no_grad():
                data = data.to(exp.device)
                output = proc(data)

            avg.update(output.mean().item())
            sq_avg.update((output**2).mean().item())

            tqbar.set_postfix(mean=f"{avg():05.3f}")
            tqbar.update()

    mean = avg()
    return {"mean": mean, "std": math.sqrt(sq_avg() - mean**2)}
