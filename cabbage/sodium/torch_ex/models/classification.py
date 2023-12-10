#!/usr/bin/env python3
import json
import logging
import math
import os
import time
from logging import Logger
from typing import Tuple, Iterator, Callable, Any, Dict, List

import torch
from torch import Tensor, nn, optim
from torch.nn import Parameter
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, StepLR
from torch.utils.data import DataLoader

from sodium.device import DeviceSupport
from sodium.state import func_state
from sodium.torch_ex.checkpoint import torch_model_save, torch_model_load
from sodium.torch_ex.loss.classification import LossManager
from sodium.torch_ex.models.schemas import LitAutoModelSchema, LitAutoHelperSchema, LitAutoState
from sodium.torch_ex.optim import OptimCH
from sodium.torch_ex.score.classification import get_score_f1_score_precision_recall, ScoreManager, Score
from sodium.torch_ex.types import PredictionTypes, BatchType, BatchIndex, ImageTypes, TargetTypes, X_True, Y_Prediction
from sodium.types import PathType, DatasetStepLiterals
from sodium.utils import make_progress_bar, remove_fd
from sodium.x.runtime.types import void
from sodium.x.runtime.wrapper import BaseClass


class LitAutoModel(BaseClass, LitAutoModelSchema):
    logger: Logger

    model: nn.Module
    device: torch.device = DeviceSupport.pref()
    host: torch.device = torch.device("cpu")

    learning_rate: float
    momentum: float
    weight_decay: float
    betas: Tuple[float, float]
    epsilon: float
    # iou_threshold: float

    # StepLR.
    step_size: int
    gamma: float

    optim_ch: OptimCH

    # criterion: nn.Module

    scores: ScoreManager
    losses: LossManager

    epoch: int  # checkpoint: load
    epochs: int

    dataloaders: Dict[DatasetStepLiterals, DataLoader | None]

    max_train_samples: float | int
    max_valid_samples: float | int
    max_test_samples: float | int

    def __init__(self,
                 model: nn.Module,
                 dataloaders: Dict[DatasetStepLiterals, DataLoader] | None = None,
                 epochs: int = 1):
        """
            Binding Model Detection.
        :param model:
        """

        if dataloaders is None:
            dataloaders = dict(train=None, valid=None, test=None)

        # normalize dataloaders literals!

        training = dataloaders.get('training')
        if training is not None:
            dataloaders['train'] = training

        validation = dataloaders.get('validation')
        if validation is not None:
            dataloaders['valid'] = validation

        testing = dataloaders.get('testing')
        if testing is not None:
            dataloaders['test'] = testing

        # -- end --

        # Make Logger
        self.logger = logging.getLogger(self.name)

        # Binding Model with Compatible Device Pref.
        self.model = model.to(self.device)

        # Fine-Tuning, Hyper-Parameters.
        self.learning_rate = 1e-3
        self.momentum = 0.9
        self.weight_decay = 0.004
        self.betas = (0.9, 0.999)  # AdamW
        self.epsilon = 1e-07  # AdamW

        # self.ams_grad = False
        # self.iou_threshold = 0.8

        # StepLR.
        self.step_size = 30
        self.gamma = 0.1

        self.optim_ch = OptimCH.SGD

        # criterion
        # self.criterion = nn.CrossEntropyLoss()
        # self.criterion = self.criterion.to(self.device)  # set auto compatible device.

        # Binding DataLoaders.
        self.dataloaders = dataloaders

        # Binding Score Manager.
        self.scores = ScoreManager(logger=self.logger)

        # Binding Loss Manager.
        self.losses = LossManager(logger=self.logger)

        # Set Epoch.
        self.epochs = epochs
        self.epoch = -1

        # Limit Samplers.
        self.max_train_samples = math.inf
        self.max_valid_samples = math.inf
        self.max_test_samples = math.inf

        self.x_true = []
        self.y_prediction = []

    def checkpoint_load(self, path: PathType, arcname: str = "model.pt") -> Tuple[int, Tensor]:
        """
            Load Checkpoint.
        :param path:
        :param arcname:
        :return:
        """

        # self.optimizer.refresh()

        # Torch Model Load.
        optimizer = self.optimizer()
        epoch, loss = torch_model_load(self.model, optimizer, path, arcname)
        self.losses.start_epoch = epoch
        self.scores.start_epoch = epoch
        self.epoch = epoch

        self.criterion.refresh()  # need re-form criterion
        self.parameters.refresh()  # depends on scheduler
        # self.optimizer.refresh()  # was update from torch_model_load
        self.scheduler.refresh()  # need re-form scheduler

        # Return.
        return epoch, loss

    def try_checkpoint_load(self, path: PathType, arcname: str = "model.pt") -> None:
        """
            Try Load Checkpoint.
        :param path:
        :param arcname:
        :return:
        """

        try:
            self.checkpoint_load(path, arcname)

        except Exception as e:
            self.logger.warning(e)

        return None

    def checkpoint_save(self, path: PathType, arcname: str = "model.pt") -> None:
        """
            Save Checkpoint.
        :param path:
        :param arcname:
        :return:
        """

        if 0 <= self.losses.size():
            loss = torch.tensor(self.losses.last())  # current: loss
            torch_model_save(self.epoch, self.model, self.optimizer(), loss, path, arcname)
            return None

        raise Exception("Model does not perform training actions")

    def images_fix(self, images: ImageTypes, device: torch.device | None = None) -> ImageTypes:
        """
            Images Fixed, By Device Pref.
        :param images:
        :param device:
        :return:
        """

        if device is None:
            device = self.device

        images = list(images)
        for i, image in enumerate(images):
            images[i] = image.to(device)

        return tuple(images)

    def targets_fix(self, targets: TargetTypes, device: torch.device | None = None) -> TargetTypes:
        """
            Targets Fixed, By Device Pref.
        :param targets:
        :param device:
        :return:
        """

        if device is None:
            device = self.device

        targets = list(targets)  # cast to array list.

        for i, target in enumerate(targets):
            if isinstance(target, Tensor):
                targets[i] = target.to(device)

            elif isinstance(target, Dict):
                targets[i] = {k: v.to(device) for k, v in target.items()}

            elif isinstance(target, List | Tuple):
                targets[i] = tuple(v.to(device) for v in target)

            else:
                raise Exception("target is not tensor or dictionary either")

        return tuple(targets)

    def predictions_fix(self, predictions: PredictionTypes, device: torch.device | None = None) -> PredictionTypes:
        """
            Predictions Fixed, By Device Pref.
        :param predictions:
        :param device:
        :return:
        """

        # Same As Targets Fixed, By Device Pref.
        return self.targets_fix(predictions, device)

    def batch_fix(self, batch: BatchType, device: torch.device | None = None) -> BatchType:
        """
            Batch Fixed, By Device Pref.
        :param batch:
        :param device:
        :return:
        """

        if device is None:
            device = self.device

        images, targets = batch
        return self.images_fix(images, device), self.targets_fix(targets, device)

    def training(self, batch: BatchType, batch_idx: BatchIndex) -> Tensor:  # handle for "training"
        """
            Training Step, Return Loss As Float Number.
        :param batch:
        :param batch_idx:
        :return:
        """
        void(batch_idx)

        # batch already to cuda mode

        if not self.model.training:
            self.model.train()

        images, targets = batch
        criterion = self.criterion()
        optimizer = self.optimizer()

        with torch.enable_grad():
            optimizer.zero_grad()

            images = torch.stack(images)
            images = images.to(self.device)

            targets = torch.stack(targets)
            targets = targets.to(self.device)

            outputs = self.model(images)  # training just images as inputs.
            loss = criterion(outputs, targets)  # evaluation with criterion.

            loss.backward()
            optimizer.step()

            return loss

    def evaluation(self, batch: BatchType, batch_idx: BatchIndex) -> PredictionTypes:
        """
            Evaluation Purposes for Validation, And Testing.
        :param batch:
        :param batch_idx:
        :return:
        """

        void(batch_idx)

        # batch already to cuda mode

        if self.model.training:
            self.model.eval()

        with torch.no_grad():
            images = torch.stack(batch)
            images = images.to(self.device)

            outputs = self.model(images)
            outputs = torch.softmax(outputs, dim=1)  # max soft. /(0_0)/
            # indices: torch.Tensor = torch.argmax(outputs, dim=1)  # max arg. /(0_0)/
            confidence, indices = torch.max(outputs, dim=1)
            return tuple(zip(indices.to(self.host), confidence.to(self.host)))

    def validation(self, batch: BatchType, batch_idx: BatchIndex) -> Tuple[Score, Tuple[X_True, Y_Prediction]]:
        """
            Validation Step, Return Score.
        :param batch:
        :param batch_idx:
        :return:
        """

        # Targets.
        targets = batch[1]

        # Evaluation Model.
        predictions = self.evaluation(batch[0], batch_idx)

        # Fix Targets, And Predictions To Compatible CPU.
        targets = self.targets_fix(targets, self.host)
        predictions = self.predictions_fix(predictions, self.host)

        # Calculation Scores.
        score = get_score_f1_score_precision_recall(predictions, targets)

        # X_True, Y_Prediction. (make it possible)
        labels = [int(target) for target in targets], [int(idx) for idx, score in predictions]

        # Return.
        return score, labels

    def testing(self, batch: BatchType, batch_idx: BatchIndex) -> Any:
        """
            Testing Step, Like Validation Step, Return Score.
        :param batch:
        :param batch_idx:
        :return:
        """

        return self.validation(batch=batch, batch_idx=batch_idx)

    @func_state
    def criterion(self) -> nn.Module:
        """
            Return Criterion.
        :return:
        """

        # criterion
        cross = nn.CrossEntropyLoss()
        cross = cross.to(self.device)  # set auto compatible device.
        return cross

    @func_state
    def parameters(self) -> Iterator[Parameter]:
        """
            Modification Module Parameter Before Used.
        :return:
        """
        params = [param for param in self.model.parameters() if param.requires_grad]
        return iter(params)

    @func_state
    def optimizer(self) -> Optimizer:
        """
            Binding Optimizer with SGD.
        :return:
        """

        params = self.parameters()

        match self.optim_ch:
            case OptimCH.SGD:
                return optim.SGD(params=params, lr=self.learning_rate,
                                 momentum=self.momentum, weight_decay=self.weight_decay)

            case OptimCH.AdamW:
                return optim.AdamW(params=params, lr=self.learning_rate,
                                   weight_decay=self.weight_decay, betas=self.betas,
                                   eps=self.epsilon)

    @func_state
    def scheduler(self) -> LRScheduler:
        """
            Binding LRSchedular with StepLR.
        :return:
        """

        optimizer = self.optimizer()
        return StepLR(optimizer=optimizer, step_size=self.step_size,
                      gamma=self.gamma, last_epoch=self.epoch)

    def training_task(self) -> Tuple[float, float]:
        """
            Training Task.
        :return:
        """

        losses = LossManager()
        train_data = self.dataloaders.get("train")
        if train_data is None:
            raise Exception("dataset step on 'train' is not found")

        i = 0
        total = min(self.max_train_samples, len(train_data))
        for batch_idx, batch in enumerate(make_progress_bar(train_data, total=total)):
            if self.max_train_samples <= i:
                break

            batch = self.batch_fix(batch, self.device)
            loss = self.training(batch, batch_idx)
            losses.append(loss)

            i += 1

        # scores collection.
        self.losses.append(losses.avg_score())

        loss = losses.last()
        avg_loss = self.losses.avg_score()
        return loss, avg_loss

    def validation_task(self) -> Tuple[Score, Score]:
        """
            Validation Task.
        :return:
        """

        scores = ScoreManager()
        valid_data = self.dataloaders.get("valid")
        if valid_data is None:
            raise Exception("dataset step on 'valid' is not found")

        i = 0
        total = min(self.max_valid_samples, len(valid_data))
        for batch_idx, batch in enumerate(make_progress_bar(valid_data, total=total)):
            if self.max_valid_samples <= i:
                break

            batch = self.batch_fix(batch, self.device)
            score, labels = self.validation(batch, batch_idx)
            x_true, y_prediction = labels

            for x in x_true:
                self.x_true.append(x)

            for y in y_prediction:
                self.y_prediction.append(y)

            scores.append(score)

            i += 1

        # scores collection.
        score = scores.avg_score()
        self.scores.append(score)
        avg_score = self.scores.avg_score()

        return score, avg_score

    def testing_task(self) -> Tuple[Any, Any]:
        """
            Validation Task.
        :return:
        """

        scores = ScoreManager()
        test_data = self.dataloaders.get("test")
        if test_data is None:
            raise Exception("dataset step on 'test' is not found")

        i = 0
        total = min(self.max_test_samples, len(test_data))
        for batch_idx, batch in enumerate(make_progress_bar(test_data, total=total)):
            if self.max_test_samples <= i:
                break

            batch = self.batch_fix(batch, self.device)
            score, labels = self.validation(batch, batch_idx)
            x_true, y_prediction = labels

            for x in x_true:
                self.x_true.append(x)

            for y in y_prediction:
                self.y_prediction.append(y)

            scores.append(score)

            i += 1

        # # scores collection.
        # self.scores.append(scores.avg_score())
        #
        # # make it same shape as training - validation, training - testing.
        # self.losses.append(self.losses.last())

        score = Score.avg([scores.avg_score(), self.scores.last()])
        self.scores[-1] = score  # unsafe.

        # score = scores.last()
        avg_score = self.scores.avg_score()
        return score, avg_score

    def action(self, cb: Callable[[LitAutoState], Any] | None = None):
        """
            Action Training, Validation, And Testing.
        :return:
        """

        scheduler = self.scheduler()

        epoch = self.epoch + 1
        for _ in range(self.epochs):
            step = epoch + 1

            # Training Task.
            self.logger.info(f"Training Task #{step}")

            loss, avg_loss = self.training_task()
            self.logger.info(f"Loss Score {loss:.2f} Avg {avg_loss:.2f}")

            # Validation Task.
            self.logger.info(f"Validation Task #{step}")

            score, avg_score = self.validation_task()
            self.logger.info(f"Accuracy Score {score.accuracy:.2f}")
            self.logger.info(f"F1 Score {score.f1_score:.2f}")
            self.logger.info(f"Precision Score {score.precision:.2f}")
            self.logger.info(f"Recall Score {score.recall:.2f}")

            test = self.dataloaders.get("test")
            if test is not None:  # testing it's optional.

                # Testing Task.
                self.logger.info(f"Testing Task #{step}")

                score, avg_score = self.testing_task()
                self.logger.info(f"Accuracy Score {score.accuracy:.2f}")
                self.logger.info(f"F1 Score {score.f1_score:.2f}")
                self.logger.info(f"Precision Score {score.precision:.2f}")
                self.logger.info(f"Recall Score {score.recall:.2f}")

            # Scheduler Step.
            scheduler.step()

            # update epoch.
            self.epoch = epoch
            epoch += 1

            if callable(cb):
                current_state = LitAutoState(epoch=epoch,
                                             loss=loss, score=score,
                                             losses=self.losses, scores=self.scores)
                cb(current_state)


class LitAutoHelper(BaseClass, LitAutoHelperSchema):
    model: LitAutoModel
    patient: int
    workdir: str

    started: float
    finished: float

    hist: Dict[str, Any]

    logger: logging.Logger

    def __init__(self,
                 model: LitAutoModel,
                 patient: float | int = math.inf,
                 workdir: str = "torch/models/classification",
                 logger: logging.Logger | None = None):
        """
            Lit Auto Helper, Make Recovery Step On Number Patient.
        :param model:
        :param patient:
        :param workdir:
        """

        # Logging.
        if logger is None:

            # Auto Binding.
            if model.logger is None:
                model.logger = logging.getLogger(self.name)

            logger = model.logger

        self.logger = logger

        self.model = model
        self.patient = int(min(max(patient, 0), model.epochs))  # auto. (0 <= patient <= epochs)
        self.workdir = workdir
        self.hist = dict(losses=[], scores=[], recovery=[], duration=0.0)

    def action(self):
        """
            Take Action Start, Main, End.
        :return:
        """
        self._start()
        self.model.action(self._cb)
        self._done()

    def _start(self):
        """
            Initial Before Take Action.
        :return:
        """
        self.started = time.perf_counter()
        os.makedirs(self.workdir, exist_ok=True)
        remove_fd(self.workdir)

    def _cb(self, state: LitAutoState) -> None:
        """
            In Action, Mul Patient in the case.
        :param state:
        :return:
        """

        epochs = state.epoch - self.model.scores.start_epoch
        low_loss = state.loss
        high_score = state.score
        patient = self.patient

        # save checkpoints
        save_main_dir = os.path.join(self.workdir, str(epochs))
        os.makedirs(save_main_dir)
        self.model.checkpoint_save(save_main_dir)

        if epochs < patient:
            return None

        if epochs % patient:
            return None

        start = max(epochs - patient - 1, 0)  # previous, previous start index.
        end = max(epochs - 1, 0)  # previous, previous end index.
        for index in range(start, end):
            j = end - index - 1

            loss = state.losses[j]
            score = state.scores[j]

            check1 = loss <= low_loss and high_score < score
            check2 = loss < low_loss and high_score <= score

            if check1 or check2:
                save_recovery_dir = os.path.join(self.workdir, str(j + 1 + self.model.scores.start_epoch))
                self.model.try_checkpoint_load(save_recovery_dir)
                self.logger.info(f"Recovery Checkpoint #{j + 1}")
                self.logger.info(f"Loss Score {loss:.2f}")
                self.logger.info(f"Accuracy Score {score.accuracy:.2f}")
                self.logger.info(f"F1 Score {score.f1_score:.2f}")
                self.logger.info(f"Precision Score {score.precision:.2f}")
                self.logger.info(f"Recall Score {score.recall:.2f}")
                self.hist["recovery"].append(j + 1)

                # update
                low_loss = loss
                high_score = score

        for j in range(start, epochs):
            unused_checkpoint_dir = os.path.join(self.workdir, str(j + 1))
            remove_fd(unused_checkpoint_dir)

        return None

    def _done(self) -> None:
        """
            Last Action, Maybe greater than mul patient.
        :return:
        """

        model = self.model
        epochs = model.epoch - self.model.scores.start_epoch
        low_loss: float = model.losses.last()
        high_score: Score = model.scores.last()
        patient = self.patient

        k = epochs % patient  # residual.

        start = max(epochs - k - 1, 0)  # previous start index.
        end = max(epochs - 1, 0)  # previous, previous end index.
        for index in range(start, end):
            j = end - index - 1

            loss = model.losses[j]
            score = model.scores[j]

            check1 = loss <= low_loss and high_score < score
            check2 = loss < low_loss and high_score <= score

            if check1 or check2:

                save_recovery_dir = os.path.join(self.workdir, str(j + 1 + self.model.scores.start_epoch))
                model.try_checkpoint_load(save_recovery_dir)
                self.logger.info(f"Recovery Final Checkpoint #{j + 1}")
                self.logger.info(f"Loss Score {loss:.2f}")
                self.logger.info(f"Accuracy Score {score.accuracy:.2f}")
                self.logger.info(f"F1 Score {score.f1_score:.2f}")
                self.logger.info(f"Precision Score {score.precision:.2f}")
                self.logger.info(f"Recall Score {score.recall:.2f}")
                self.hist["recovery"].append(j + 1)

                # update
                low_loss = loss
                high_score = score

        for j in range(start, epochs):
            unused_checkpoint_dir = os.path.join(self.workdir, str(j + 1))
            remove_fd(unused_checkpoint_dir)

        # time duration.
        self.finished = time.perf_counter()
        elapsed = self.finished - self.started
        self.hist["duration"] = elapsed

        # save hist file.
        self.hist["losses"] = model.losses.to_list()
        self.hist["scores"] = model.scores.to_list()

        hist_file = os.path.join(self.workdir, "hist.json")
        with open(hist_file, "w") as fstream:

            if fstream.writable():
                fstream.write(json.dumps(self.hist))  # save hist file.

                # reset, clean.
                self.hist = dict(losses=[], scores=[], recovery=[], duration=0.0)  # clean, reset.

            else:
                raise Exception("failed save hist file")
        return None
