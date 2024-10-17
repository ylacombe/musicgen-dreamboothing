"""Define the Experiment class"""

import logging
import os
import time
from typing import List, Optional

import numpy as np
import torch
import torchaudio
import librosa

import emotion_clf.ml as ml
import pdb
from emotion_clf.ml.inference_utils import *

class Experiment():
    """An Experiment contains everything to train and evaluate a model
    according to parameters."""
    def __init__(self,
                 parameters: ml.parameters.Parameters,
                 directory: str,
                 data_dir: str,
                 num_workers: int = 0,
                 restore_file: Optional[str] = None,
                 restart_training: bool = False,
                 splits: Optional[List[str]] = None,
                 calculate_stats: Optional[bool] = None) -> None:

        log_path = os.path.join(directory, "experiment.log")
        ml.logging.set_logger(log_path)

        ml.utils.define_manual_seed(parameters.get("manual_seed"))

        self.directory = directory
        self.params = parameters
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if splits is None:
            splits = self.params.get("dataloaders").keys()

        if isinstance(data_dir, torch.Tensor):
            # used when inferencing, avoid loading presaved pickle etc, to mimic the output of load_dataloaders
            wave = inference_process(data_dir)
            self.dls = [(wave, None)]
        elif not os.path.isfile(data_dir):  # usually directory, only is file when inference
            # self.dls = (data_batch, target_batch) --> data_batch=(1, 128, length)
            self.dls = ml.loading.load_dataloaders(data_dir, splits,
                                                self.params.get("dataset"),
                                                self.params.get("dataloaders"),
                                                num_workers)
        else:
            wave = inference_process(data_dir)
            self.dls = [(wave, None)]

        self.loss = ml.loading.load_loss(self.params.get("loss"), device=self.device)  # weighted BCE w/ Logit Loss
        self.metrics = ml.loading.load_metrics(self.params.get("metrics"))  # PR-AUC, ROC-AUC
        self.eval_activation = ml.loading.load_eval_activation(self.params.get("evaluation_activation"))  # averaged sigmoid
        self.preprocessor = ml.loading.load_preprocessor(self.params.get("preprocessor"), self.device)
        self.model = ml.loading.load_model(self.params.get("model"), self.device)
        self.optimizer = ml.loading.load_optimizer(self.params.get("optimizer"), self.model.parameters())
        self.scheduler = ml.loading.load_scheduler(self.params.get("scheduler"), self.optimizer)
        self.scaler = torch.cuda.amp.GradScaler()
        self.num_epochs = self.params.get("num_epochs")
        self.gradient_clipping = self.params.get("gradient_clipping")

        self.init_swa()
        self.init_mixup()

        if calculate_stats is None and self.params.get("stats") is None:
            calculate_stats = True

        if calculate_stats:
            self.calculate_stats(save=True)

        #logging.debug("Parameters: %s", self.params.show())
        self.save_parameters()

        self.epoch = 0
        self.best_valid_metric = 0.0

        if restore_file is not None:
            self.restore_checkpoint(restore_file, restart_training=restart_training)

    def init_swa(self) -> None:
        """Initialize self.swa (stochastic weight averaging)"""
        swa_params = self.params.get("swa")
        self.swa = {}
        if swa_params is not None:
            swa_start = self.num_epochs * (1 - swa_params["length"])
            swa_scheduler = None
            swa_scheduler_start = None
            if swa_params.get("scheduler") is not None:
                swa_scheduler_start = self.num_epochs * (
                    1 - swa_params["scheduler"]["length"])
                swa_anneal_epochs = int(
                    self.num_epochs * swa_params["scheduler"]["anneal_length"])
                swa_scheduler = torch.optim.swa_utils.SWALR(
                    self.optimizer,
                    swa_lr=swa_params["scheduler"]["lr"],
                    anneal_epochs=swa_anneal_epochs,
                    anneal_strategy="linear")

            self.swa = {
                "start": swa_start,
                "model": torch.optim.swa_utils.AveragedModel(self.model),
                "scheduler": swa_scheduler,
                "scheduler_start": swa_scheduler_start
            }

    def init_mixup(self) -> None:
        """Initialize self.mixup"""
        self.mixup = self.params.get("mixup")
        if self.mixup == 0:
            self.mixup = None

    def save_parameters(self) -> None:
        """Save current parameters in directory"""
        json_path = os.path.join(self.directory, "parameters.json")
        self.params.save(json_path)
        #logging.info("Parameters saved to %s", json_path)

    def train(self,
              train_split: str,
              valid_split: Optional[str] = None) -> None:
        """Train model"""
        #logging.info("Starting training for %d epoch(s)",
        #             max(0, self.num_epochs - self.epoch))

        start_time = time.time()
        valid_metrics = None

        while self.epoch < self.num_epochs:
            self.epoch += 1

            learning_rate = self.optimizer.param_groups[0]["lr"]
            #logging.info("Epoch %d/%d: lr=%.1e", self.epoch, self.num_epochs,
            #             learning_rate)

            #logging.debug("- training")
            train_metrics = ml.training.train_one_epoch(self, train_split)
            #ml.logging.save_metrics(self.directory, train_split, train_metrics,
            #                        self.epoch)
            #ml.logging.log_metrics(start_time, train_metrics)

            if valid_split is not None:
                logging.debug("- validation")
                valid_metrics, _, _ = ml.evaluation.evaluate(self, valid_split)

                #ml.logging.save_metrics(self.directory, valid_split,
                #                        valid_metrics, self.epoch)
                #ml.logging.log_metrics(start_time, valid_metrics)

                first_metric = ml.metrics.extract_first_metric(valid_metrics)
                is_best = first_metric >= self.best_valid_metric
                if is_best:
                    #logging.info("Found new best %s",
                    #             self.params.get("metrics")[0])
                    self.best_valid_metric = first_metric
            else:
                valid_metrics = {}
                is_best = False

            if self.swa.get("model") is not None:
                if self.epoch > self.swa["start"]:
                    self.swa["model"].update_parameters(self.model)

            if (self.swa.get("scheduler") is not None) and (
                    self.epoch > self.swa["scheduler_start"]):
                self.swa["scheduler"].step()
            elif self.scheduler is not None:
                self.scheduler.step()

            self.save_checkpoint("last.pth.tar", valid_metrics)
            if is_best:
                self.save_checkpoint("best.pth.tar", valid_metrics)
                #logging.debug("Best model saved to best.pth.tar")

        if self.swa.get("model") is not None and valid_metrics is not None:
            #logging.info("Updating batchnorm running stats for swa")
            ml.training.update_swa_bn(self, train_split)
            self.save_checkpoint("last.pth.tar", valid_metrics)

    def evaluate(self,
                 split: str,
                 restore_file_name: Optional[str] = None,
                 use_swa: bool = False,
                 save_predictions: bool = False):
        """Evaluate on given split and optionally save predictions"""

        desc = split
        if restore_file_name is not None:
            restore_file = os.path.join(self.directory,
                                        restore_file_name + ".pth.tar")
            self.restore_checkpoint(restore_file)
            desc += "-" + restore_file_name

        if use_swa:
            desc += "-swa"

        #logging.info("Evaluating %s", desc)

        if split != "inference":
            start_time = time.time()
            eval_metrics, targets, outputs = ml.evaluation.evaluate(
                self, split, use_swa=use_swa)

            #ml.logging.save_metrics(self.directory, desc, eval_metrics, epoch=None)
            #ml.logging.log_metrics(start_time, eval_metrics)

            if save_predictions:
                self.save_predictions(desc, targets, outputs)

        else:
            start_time = time.time()
            eval_metrics, targets, outputs = ml.evaluation.evaluate(self, split, use_swa=use_swa)

            if save_predictions:
                outputs = outputs[0].flatten().detach().cpu()
            return outputs

    def save_predictions(self, desc: str, targets: np.ndarray,
                         outputs: np.ndarray):
        """Save predictions in experiment directory"""

        predictions_dir = os.path.join(self.directory, "predictions")
        os.makedirs(predictions_dir, exist_ok=True)
        targets_path = os.path.join(predictions_dir, desc + "-targets.npy")
        outputs_path = os.path.join(predictions_dir, desc + "-outputs.npy")
        np.save(targets_path, targets)
        #logging.info("Targets saved to %s", targets_path)
        np.save(outputs_path, outputs)
        #logging.info("Predictions saved to %s", outputs_path)

    def save_checkpoint(self, filename: str, valid_metrics=None):
        """Save checkpoint to file. Optionally include validation metrics"""

        scheduler_state = None
        swa_model_state = None
        swa_scheduler_state = None
        if self.scheduler is not None:
            scheduler_state = self.scheduler.state_dict()
        if self.swa.get("model") is not None:
            swa_model_state = self.swa["model"].state_dict()
        if self.swa.get("scheduler") is not None:
            swa_scheduler_state = self.swa["scheduler"].state_dict()

        checkpoint = {
            "epoch": self.epoch,
            "validation_metrics": valid_metrics,
            "best_validation_metric": self.best_valid_metric,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": scheduler_state,
            "swa_model_state": swa_model_state,
            "swa_scheduler_state": swa_scheduler_state,
            "random_state": torch.get_rng_state()
        }

        filepath = os.path.join(self.directory, filename)
        torch.save(checkpoint, filepath)

    def restore_checkpoint(self,
                           restore_file: str,
                           restart_training: bool = False) -> None:
        """Restore checkpoint from file"""
        epoch, best_valid_metric = ml.loading.load_checkpoint(
            restore_file,
            device=self.device,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            swa_model=self.swa.get("model"),
            swa_scheduler=self.swa.get("scheduler"))
        #logging.info("Checkpoint restored from %s", restore_file)
        if not restart_training:
            self.best_valid_metric = best_valid_metric
            self.epoch = epoch
            #logging.info("Resuming at epoch %d", epoch)

    def calculate_stats(self, save: bool = True):
        """Calculate mean and std of input, optionally save them in params"""
        stats = ml.statistics.calculate_stats(self)
        #logging.info("Data statistics: %s", stats)
        if save:
            self.params.set(stats, "stats")
            self.save_parameters()
