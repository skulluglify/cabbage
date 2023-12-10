#!/usr/bin/env python3
from logging import Logger
from typing import Tuple, Iterator, Callable, Any, Protocol, Dict

import attr
import torch
from torch import Tensor, nn
from torch.nn import Parameter
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from sodium.torch_ex.types import PredictionTypes, BatchType, BatchIndex, ImageTypes, TargetTypes
from sodium.types import PathType, DatasetStepLiterals
from sodium.x.runtime.wrapper import BaseClass


@attr.define
class LitAutoState(BaseClass):
    epoch: int

    loss: float  # current state.
    losses: Any

    score: Any  # current state.
    scores: Any


class LitAutoModelSchema(Protocol):
    logger: Logger

    model: nn.Module
    device: torch.device
    host: torch.device

    learning_rate: float
    momentum: float
    weight_decay: float

    step_size: int
    gamma: float

    # criterion: Any

    scores: Any
    losses: Any

    epoch: int  # checkpoint: load
    epochs: int

    dataloaders: Dict[DatasetStepLiterals, DataLoader]

    max_train_samples: float | int
    max_valid_samples: float | int
    max_test_samples: float | int

    def checkpoint_load(self, path: PathType, arcname: str = "model.pt") -> Tuple[int, Tensor]:
        """
            Load Checkpoint.
        :param path:
        :param arcname:
        :return:
        """
        pass

    def try_checkpoint_load(self, path: PathType, arcname: str = "model.pt") -> None:
        """
            Try Load Checkpoint.
        :param path:
        :param arcname:
        :return:
        """
        pass

    def checkpoint_save(self, path: PathType, arcname: str = "model.pt") -> None:
        """
            Save Checkpoint.
        :param path:
        :param arcname:
        :return:
        """
        pass

    def images_fix(self, images: ImageTypes, device: torch.device | None = None) -> ImageTypes:
        """
            Images Fixed, By Device Pref.
        :param images:
        :param device:
        :return:
        """
        pass

    def targets_fix(self, targets: TargetTypes, device: torch.device | None = None) -> TargetTypes:
        """
            Targets Fixed, By Device Pref.
        :param targets:
        :param device:
        :return:
        """
        pass

    def predictions_fix(self, predictions: PredictionTypes, device: torch.device | None = None) -> PredictionTypes:
        """
            Predictions Fixed, By Device Pref.
        :param predictions:
        :param device:
        :return:
        """
        pass

    def batch_fix(self, batch: BatchType, device: torch.device | None = None) -> BatchType:
        """
            Batch Fixed, By Device Pref.
        :param batch:
        :param device:
        :return:
        """
        pass

    def training(self, batch: BatchType, batch_idx: BatchIndex) -> Tensor:  # handle for "training"
        """
            Training Step, Return Loss As Float Number.
        :param batch:
        :param batch_idx:
        :return:
        """
        pass

    def evaluation(self, batch: BatchType, batch_idx: BatchIndex) -> PredictionTypes:
        """
            Evaluation Purposes for Validation, And Testing.
        :param batch:
        :param batch_idx:
        :return:
        """
        pass

    def validation(self, batch: BatchType, batch_idx: BatchIndex) -> Any:
        """
            Validation Step, Return Score.
        :param batch:
        :param batch_idx:
        :return:
        """
        pass

    def testing(self, batch: BatchType, batch_idx: BatchIndex) -> Any:
        """
            Testing Step, Like Validation Step, Return Score.
        :param batch:
        :param batch_idx:
        :return:
        """
        pass

    def parameters(self) -> Iterator[Parameter]:
        """
            Modification Module Parameter Before Used.
        :return:
        """
        pass

    def optimizer(self) -> Optimizer:
        """
            Binding Optimizer with SGD.
        :return:
        """
        pass

    def scheduler(self) -> LRScheduler:
        """
            Binding LRSchedular with StepLR.
        :return:
        """
        pass

    def training_task(self) -> Tuple[float, float]:
        """
            Training Task.
        :return:
        """
        pass

    def validation_task(self) -> Tuple[Any, Any]:
        """
            Validation Task.
        :return:
        """
        pass

    def testing_task(self) -> Tuple[Any, Any]:
        """
            Validation Task.
        :return:
        """
        pass

    def action(self, cb: Callable[[LitAutoState], Any] | None = None):
        """
            Action Training, Validation, And Testing.
        :return:
        """
        pass


class LitAutoHelperSchema(Protocol):

    def action(self):
        """
            Take Action Start, Main, End.
        :return:
        """
        pass
