#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
import logging
import multiprocessing
import os
import sys

import torch


def init(debug: bool = True) -> bool:
    """
        Initialize Any Services.
    :param debug:
    :return:
    """

    name = "SODIUM_INITIAL_VALUE"
    initial_value = os.environ.get(name)
    if initial_value is None or initial_value not in ("1",):
        os.environ.setdefault(name, "1")

        if debug:  # debug mode.
            # os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
            logging.basicConfig(level=logging.INFO)

        # Multiprocessing Freezing Support.
        if sys.platform.startswith("win"):
            multiprocessing.freeze_support()

        # Torch Float32 Manipulation High Precision.
        torch.set_float32_matmul_precision("high")

        return True
    return False
