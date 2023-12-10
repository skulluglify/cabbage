#!/usr/bin/env python3
import math
import random

import torch
from torch import Tensor


# Conversion

def cvt_abs_range_tensor(tensor: Tensor, value: int) -> Tensor:
    """
        Conversion, Convert Array Values to Absolute Ranges if is Possible.
    :param tensor:
    :param value:
    :return:
    """

    tensor = tensor.to(torch.float32)

    v_min, v_max = torch.min(tensor), torch.max(tensor)
    v_half = torch.tensor(value / 2)

    # ex. 0. ~ 1.
    if 0 <= v_min and v_max <= 1:
        tensor = tensor * value

    # ex. -1. ~ 0.
    elif -1 <= v_min and v_max <= 0:
        tensor = (tensor + 1) * value

    # ex. -0.5 ~ 0.5
    elif -0.5 <= v_min and v_max <= 0.5:
        tensor = (tensor + 0.5) * value

    # ex. 0 ~ value
    elif 0 <= v_min and v_max <= value:
        pass

    # ex. -value ~ 0
    elif -value <= v_min and v_max <= 0:
        tensor = tensor + value

    # ex. -v_half ~ v_half
    elif -v_half <= v_min and v_max <= v_half:
        tensor = tensor + v_half

    # Not supported.
    else:
        raise Exception("it's impossible to convert array values with absolute ranges")

    # Keep floating.
    return tensor


def cvt_rel_range_tensor(tensor: Tensor, value: int) -> Tensor:
    """
        Conversion, Convert Array Values to Relative Ranges if is Possible.
    :param tensor:
    :param value:
    :return:
    """

    tensor = tensor.to(torch.float32)

    v_min, v_max = torch.min(tensor), torch.max(tensor)
    v_half = torch.tensor(value / 2)

    # ex. 0. ~ 1.
    if 0 <= v_min and v_max <= 1:
        pass

    # ex. -1. ~ 0.
    elif -1 <= v_min and v_max <= 0:
        tensor = tensor + 1

    # ex. -0.5 ~ 0.5
    elif -0.5 <= v_min and v_max <= 0.5:
        tensor = tensor + 0.5

    # ex. 0 ~ value
    elif 0 <= v_min and v_max <= value:
        tensor = tensor / value

    # ex. -value ~ 0
    elif -value <= v_min and v_max <= 0:
        tensor = (tensor + value) / value

    # ex. -v_half ~ v_half
    elif -v_half <= v_min and v_max <= v_half:
        tensor = (tensor + v_half) / value

    # Not supported.
    else:
        raise Exception("it's impossible to convert array values with absolute ranges")

    # Keep floating.
    return tensor


def cvt_f32_to_u8_tensor(tensor: Tensor) -> Tensor:
    """
        Try Conversion Float32 Array to Uint8 Array.
    :param tensor:
    :return:
    """

    tensor = cvt_abs_range_tensor(tensor, 255)
    return tensor.to(torch.uint8)


def cvt_u8_to_f32_tensor(tensor: Tensor) -> Tensor:
    """
        Try Conversion Uint8 Array to Float32 Array.
    :param tensor:
    :return:
    """

    return cvt_rel_range_tensor(tensor, 255)


def overlap(box1: Tensor, box2: Tensor) -> Tensor:
    """
        IOU TAPE FLEX. /(0_0)/
    :param box1:
    :param box2:
    :return:
    """

    if len(box1.shape) <= 0:
        raise Exception("Size of box shape is not greater than 0")

    N = (*box1.shape,)[0:1][0]  # Cast, Limit, Index.

    if box1.shape != (N,):
        raise Exception("Box shape is not equal as 1d")

    if box1.shape != box2.shape:
        raise Exception("Box shape is not equal as another box")

    x1 = torch.max(torch.stack([box1[0], box2[0]]))
    y1 = torch.max(torch.stack([box1[1], box2[1]]))
    x2 = torch.min(torch.stack([box1[2], box2[2]]))
    y2 = torch.min(torch.stack([box1[3], box2[3]]))

    zero = torch.tensor(0.0)
    intersection = torch.max(torch.stack([zero, x2 - x1])) * torch.max(torch.stack([zero, y2 - y1]))

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection

    iou = intersection / union if union != 0.0 else math.inf

    if not isinstance(iou, Tensor):
        return torch.tensor(iou)  # Cast to Tensor.

    return iou


def shuffle(tensor: Tensor) -> Tensor:
    """
        Shuffle Tensor Data Array.
    :param tensor:
    :return:
    """

    if len(tensor.shape) <= 0:
        raise Exception("Size of box shape is not greater than 0")

    if len(tensor) <= 0:
        return torch.clone(tensor)  # empty tensor list.

    data = [item for item in tensor]
    random.shuffle(data)

    return torch.stack(data)
