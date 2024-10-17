"""Average predictions"""

from typing import Callable, Iterable, Optional

import numpy as np
from torch import Tensor
import pdb

def average_predictions(output_paths: str,
                        targets: Tensor = None,
                        weight_metric: Optional[Callable] = None):
    """Average predictions and save result. Optionally weight with metric."""

    acc = np.zeros(1)
    weight = 1
    total = 0
    for path in output_paths:  # 6 models
        probas = np.load(path)

        if weight_metric is not None:
            pdb.set_trace()
            weight = weight_metric(targets, probas).mean()  # 56*1 mean = 1

        acc = acc + probas * weight  # 4231*56
        total += weight
    return acc / total  # 4231*56


def verify_targets(target_paths: Iterable[str]):
    """Return None if all targets are not the same, targets otherwise."""

    targets = None
    for path in target_paths:
        if targets is None:
            targets = np.load(path)
        elif not np.array_equal(np.load(path), targets):
            targets = None
            break
    return targets
