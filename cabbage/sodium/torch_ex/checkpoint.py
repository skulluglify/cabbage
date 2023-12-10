#!/usr/bin/env python3
from typing import Dict, Any, Tuple

import torch
from torch import nn, Tensor
from torch.optim import Optimizer

from sodium.torch_ex.externals import check_file_is_writable, check_file_is_readable
from sodium.types import PathType


def torch_model_save(epoch: int,
                     model: nn.Module,
                     optimizer: Optimizer,
                     loss: Tensor,
                     path: PathType,
                     arcname: str = "model.pt"):
    """
        Torch Model Save.

    Examples:
        >>> torch_model_save(epoch, model, optimizer, loss, path)
        >>> ...

    :param epoch:
    :param model:
    :param optimizer:
    :param loss:
    :param path:
    :param arcname:
    :return:
    """

    path = check_file_is_writable(path, arcname)

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss.float(),
    }, path)


def torch_model_load(model: nn.Module,
                     optimizer: Optimizer,
                     path: PathType,
                     arcname: str = "model.pt") -> Tuple[int, Tensor]:
    """
        Torch Model Save.

    Examples:
        >>> m_epoch, m_loss = torch_model_load(model, optimizer, path)
        >>> ...

    :param model:
    :param optimizer:
    :param path:
    :param arcname:
    :return:
    """

    path = check_file_is_readable(path, arcname)

    data: Dict[str, Any]
    data = torch.load(path)

    epoch: int
    epoch = data.get("epoch")

    loss: Tensor
    loss = data.get("loss")

    model.load_state_dict(data.get("model_state_dict"))
    optimizer.load_state_dict(data.get("optimizer_state_dict"))

    return epoch, loss
