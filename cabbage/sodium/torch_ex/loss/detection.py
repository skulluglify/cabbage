#!/usr/bin/env python3
import json
import logging
import math
from typing import List, Tuple, Iterator, Dict

import torch
from torch import Tensor

from sodium.x.runtime.types import void
from sodium.x.runtime.wrapper import BaseClass


def get_loss_from_dict(data: Dict[str, Tensor]) -> Tensor:
    """
        Get Loss From Dictionary, Keep Requires Gradient if Available.
    :param data:
    :return:
    """

    value: Tensor
    values: List[Tensor]
    values = [value for value in data.values()]
    loss = torch.sum(torch.stack(values), dtype=torch.float32)
    return loss


class LossManager(BaseClass):

    _losses: List[float]
    start_epoch: int

    logger: logging.Logger

    def __init__(self, logger: logging.Logger | None = None):
        """
            Loss Manager.
        """
        self._losses = []
        self.start_epoch = 0

        if logger is None:
            logger = logging.getLogger(self.name)
            
        self.logger = logger

    def __getitem__(self, idx: int) -> float:
        """
            Link Func GetItem to Var _Scores.
        :param idx:
        :return:
        """
        if idx < 0:
            return self._losses[idx]

        return self._losses[max(idx - self.start_epoch, 0)]

    def __contains__(self, loss: float) -> bool:
        """
            Check Contains Not Permission Enabled.
        :param loss:
        :return:
        """
        void(loss)

        # Default Value.
        return False

    def __len__(self) -> int:
        """
            Link Func Length to Func Size.
        :return:
        """
        return self.size()

    def last(self) -> float:
        """
            Last Value.
        :return:
        """

        if 0 < len(self._losses):
            return self._losses[-1]

        # Default.
        return 0.0

    def to_list(self) -> List[float | int]:
        """
            Make JSON Syntax.
        :return:
        """
        # data = [{"loss": origin} for origin in self._losses]
        data = [origin for origin in self._losses]
        return data

    def to_json(self) -> str:
        """
            Make JSON Syntax.
        :return:
        """

        return json.dumps(self.to_list())

    def append(self, loss: float | Tensor) -> None:
        """
            Append Loss Value in Losses Values.
        :param loss:
        :return:
        """

        # Loss is Tensor.
        if isinstance(loss, Tensor):

            # Make it Compatible On CPU
            loss = loss.cpu()

            # Check Is Array.
            shape = loss.shape
            if 0 < len(shape):

                # Value On First Index.
                loss = float(loss[0])
            else:

                # Literally Number Float.
                loss = float(loss)

        # Skipping NaN or Infinite
        if math.isnan(loss) or math.isinf(loss):
            err = Exception("Skipping Loss Value, is NaN or Inf")
            self.logger.warning(err)
            return None

        self._losses.append(loss)
        return None

    def history(self) -> Tuple[float, ...]:
        """
            Watching Historical Loss Values.
        :return:
        """
        data: Iterator[float] = iter(self._losses)
        return tuple(data)

    def avg_score(self) -> float:
        """
            Average Score Loss Values.
        :return:
        """

        if 0 < len(self._losses):
            # Calculate All Values.
            return math.fsum(self._losses) / self.size()

        return math.nan

    def size(self) -> int:
        """
            Size of Losses.
        :return:
        """

        return len(self._losses)
