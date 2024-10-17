"""Logging: utilities to log messages and metrics"""
import csv
import datetime
import logging
import os
import time
from typing import Dict, Optional

import numpy as np


def set_logger(log_path: str) -> None:
    """Initialize logging to file and console"""

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    logger.handlers = []

    # Logging to file
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s:%(levelname)s: %(message)s"))
    logger.addHandler(file_handler)

    # Logging to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)


def mean(array: np.ndarray, weights: Optional[np.ndarray]) -> float:
    """Calculate weighted average"""

    if weights is None or array.ndim == 0:
        res = array.mean()
    else:
        res = (array[0:len(weights)] * weights).sum() / weights.sum()
    return float(res)


def save_metrics(directory: str,
                 split: str,
                 metrics: Dict[str, np.ndarray],
                 epoch: Optional[int],
                 weights: np.ndarray = None) -> None:
    """Append metrics to file metrics.csv"""

    path = os.path.join(directory, "metrics.csv")
    with open(path, "a", encoding="utf-8") as file:
        writer = csv.writer(file)
        for met in metrics:
            row = [split, met, epoch, mean(metrics[met], weights)]
            if metrics[met].size > 1:
                row = row + list(metrics[met])
            writer.writerow(row)


def log_metrics(start_time: float,
                metrics: Dict[str, np.ndarray],
                weights: Optional[np.ndarray] = None) -> None:
    """Log metrics"""

    logging.info("  time: %s; %s",
                 datetime.timedelta(seconds=int(time.time() - start_time)),
                 format_metrics(metrics, weights))


def format_metrics(metrics: Dict[str, np.ndarray],
                   weights: Optional[np.ndarray] = None) -> str:
    """Format dict of metrics for logging"""

    return " ; ".join(f"{k}: {mean(v, weights):.4g}"
                      for k, v in metrics.items())
