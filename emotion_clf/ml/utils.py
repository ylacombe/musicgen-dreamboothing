"""Utils"""

import logging
import random
from typing import Optional

import numpy as np
import torch


class RunningAverage():
    """Keeps running average"""
    def __init__(self) -> None:
        self.steps = 0
        self.total = 0.

    def update(self, val: float) -> None:
        """Update running average with new value"""

        self.total += val
        self.steps += 1

    def __call__(self) -> float:
        return self.total / float(self.steps)


def define_manual_seed(seed: Optional[int]) -> None:
    """Set random seed"""

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        #logging.info("Manual seed set to %s", seed)
