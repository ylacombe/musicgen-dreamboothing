"""Load objects from files"""
import importlib
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.swa_utils import AveragedModel

import emotion_clf.ml.losses as ml_losses
import emotion_clf.ml.metrics as ml_metrics
import emotion_clf.ml.parameters as ml_parameters
from emotion_clf.ml.experiment import Experiment
import pdb


def load_experiment(experiment_dir: str,
                    data_dir: str = "data",
                    num_workers: int = 0,
                    restart_training: bool = False,
                    restore_name: Optional[str] = "last",
                    num_epochs: Optional[int] = None,
                    learning_rate: Optional[float] = None,
                    weight_decay: Optional[float] = None,
                    dropout: Optional[float] = None,
                    batch_size: Optional[int] = None,
                    manual_seed: Optional[int] = None,
                    model: Optional[str] = None,
                    loss: Optional[str] = None,
                    **kwargs) -> Experiment:
    """Construct experiment from parameters

    Load parameters from file parameters.json in given experiment_dir.
    Parameters contained in parent directories are loaded recursively.
    Parameters are overridden by optional arguments.
    Final parameters are written in experiment_dir/parameters.json.
    Additional arguments are given to the Experiment class constructor."""

    parameters = ml_parameters.load_parameters_recursively(experiment_dir)
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir, exist_ok=True)
    #os.makedirs(experiment_dir, exist_ok=True)
    
    if restart_training:
        restore_name = None
        # Clean metrics from experiment dir
        metrics_path = os.path.join(experiment_dir, "metrics.csv")
        if os.path.exists(metrics_path):
            os.remove(metrics_path)

    restore_file = None
    if restore_name is not None:
        restore_file = os.path.join(experiment_dir, restore_name + ".pth.tar")
        if not os.path.exists(restore_file):
            print(f"File {restore_file} doesn't exists - "
                  "cannot restore checkpoint")
            restore_file = None

    if num_epochs is not None:
        parameters.set(num_epochs, "num_epochs")
        scheduler_name = parameters.get("scheduler", "name")
        if scheduler_name == "CosineAnnealingLR":
            parameters.set(num_epochs, "scheduler", "params", "T_max")

    if learning_rate is not None:
        parameters.set(learning_rate, "optimizer", "params", "lr")

    if model is not None:
        parameters.set(model, "model", "name")

    if weight_decay is not None:
        parameters.set(weight_decay, "optimizer", "params", "weight_decay")

    if dropout is not None:
        parameters.set(dropout, "model", "params", "dropout")

    if manual_seed is not None:
        parameters.set(manual_seed, "manual_seed")

    if batch_size is not None:
        for split in ["train", "valid"]:
            parameters.set(batch_size, "dataloaders", split, "batch_size")

    if loss is not None:
        parameters.set({"name": loss, "params": {}}, "loss")

    return Experiment(parameters,
                      experiment_dir,
                      data_dir,
                      num_workers=num_workers,
                      restore_file=restore_file,
                      restart_training=restart_training,
                      **kwargs)


def load_model(model_params: Dict[str, Any], device: str) -> Module:
    """Load model from models.name module"""
    model_module = importlib.import_module(f"emotion_clf.models.{model_params['name']}")
    get_model = getattr(model_module, "get_model")
    model = get_model(model_params["params"]).to(device)
    return model


def load_optimizer(optimizer_params: Dict[str, Any],
                   parameters: Iterable[Any]) -> Optimizer:
    """Load optimizer from torch.optim module"""
    optimizer_class = getattr(torch.optim, optimizer_params["name"])
    optimizer = optimizer_class(parameters, **optimizer_params["params"])
    return optimizer


def load_scheduler(scheduler_params: Dict[str, Any],
                   optimizer: Optimizer) -> _LRScheduler:
    """Load optimizer from torch.optim.lr_scheduler module"""
    scheduler = None
    if scheduler_params is not None:
        scheduler_class = getattr(torch.optim.lr_scheduler,
                                  scheduler_params["name"])
        scheduler = scheduler_class(optimizer, **scheduler_params["params"])
    return scheduler


def load_dataloaders(root, splits, dataset_params, dataloaders_params, num_workers):
    """Load dict of dataloaders

    Uses function get_dataloader in module loaders.dataset_name"""
    loader_module = importlib.import_module(f"emotion_clf.loaders.{dataset_params['name']}")  # loaders.jamendo
    data_dir = os.path.join(root, dataset_params["name"])  # data/jamendo
    get_dataloader = getattr(loader_module, "get_dataloader")
    params_all = dataset_params["params"].get("all") # directory: melspecs2
    if params_all is None:
        params_all = {}
    dls = {}
    for split in splits:
        params = params_all.copy()
        params.update(dataset_params["params"][split])  # get trackfile of a split
        dls[split] = get_dataloader(data_dir, params, dataloaders_params[split], num_workers)
    return dls


def load_loss(loss_params: Dict[str, Any], device: str) -> _Loss:
    """Load loss from module ml.losses"""
    loss_class = getattr(ml_losses, loss_params["name"])
    loss_params = loss_params["params"]
    return loss_class(**loss_params).to(device)


def load_metrics(metrics_names: List[str]):
    """Load metrics from module ml.metrics"""
    metrics = {}
    if metrics_names is not None:
        for met in metrics_names:
            metrics[met] = getattr(ml_metrics, met)
    return metrics


def load_eval_activation(name: str):
    """Load evaluation activation from module ml.metrics"""
    activation = None
    if name is not None:
        activation = getattr(ml_metrics, name)
    return activation


def load_preprocessor(preproc_params: Dict[str, Any], device: str) -> Module:
    """Load preprocessor from module preprocessors.name"""
    preproc = None
    if preproc_params is not None:
        preproc_module = importlib.import_module(
            f"emotion_clf.preprocessors.{preproc_params['name']}")
        get_preprocessor = getattr(preproc_module, "get_preprocessor")
        preproc = get_preprocessor(preproc_params["params"]).to(device)
    return preproc


def load_checkpoint(
        filepath: str,
        device: Optional[str] = None,
        model: Optional[Module] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        swa_model: Optional[AveragedModel] = None,
        swa_scheduler: Optional[_LRScheduler] = None) -> Tuple[int, float]:
    """Load checkpoint from filepath

    Returns epoch number and best validation metrics of saved training"""

    if not os.path.exists(filepath):
        raise ValueError(f"File doesn't exist {filepath}")
    checkpoint = torch.load(filepath, map_location=device)
    random_state = checkpoint.get("random_state")
    if random_state is not None:
        torch.set_rng_state(random_state.to("cpu"))

    model_state = checkpoint.get('model_state')
    if model_state is not None and model is not None:
        model.load_state_dict(model_state)

    optimizer_state = checkpoint.get('optimizer_state')
    if optimizer_state is not None and optimizer is not None:
        optimizer.load_state_dict(optimizer_state)

    scheduler_state = checkpoint.get('scheduler_state')
    if scheduler_state is not None and scheduler is not None:
        scheduler.load_state_dict(scheduler_state)

    swa_model_state = checkpoint.get('swa_model_state')
    if swa_model_state is not None and swa_model is not None:
        swa_model.load_state_dict(swa_model_state)

    swa_scheduler_state = checkpoint.get('swa_scheduler_state')
    if swa_scheduler_state is not None and swa_scheduler is not None:
        swa_scheduler.load_state_dict(swa_scheduler_state)

    return (checkpoint.get('epoch'), checkpoint.get('best_validation_metric'))
