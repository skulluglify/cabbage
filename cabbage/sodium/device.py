#!/usr/bin/env python3

import torch
# import torch.backends.mkldnn
import torch.backends.mps

from .state import func_state


class DeviceSupport:
    """
    cpu, cuda, ipu,
    xpu, mkldnn, opengl,
    opencl, ideep, hip,
    ve, fpga, ort,
    xla, lazy, vulkan,
    mps, meta, hpu, mtia
    """
    name = "cuda:0"

    @classmethod
    def pref(cls) -> torch.device:
        """
            Preferences Available Compatibility Device.
        :return:
        """
        # print("pref", cls.name)  # cuda0, cuda:0, cuda/0, cuda$0
        # index = torch.cuda.current_device()
        # torch.cuda.set_device(index)

        Wrapper = cls
        bound = Wrapper()
        return bound.compatible

    @property
    def compatible(self):
        """
            Property 'compatible' from 'device_bound'.
        :return:
        """
        return self.device_bound()

    @func_state(keep=True)
    def device_bound(self) -> torch.device:
        """
        device = torch.device("cuda" if torch.cuda.is_available() else \
        "mkldnn" if torch.backends.mkldnn.is_available() else \
        "vulkan" if torch.is_vulkan_available() else \
        "mps" if torch.backends.mps.is_available() else "cpu")

        :return: torch.device
        """

        device = torch.device("cpu")
        if torch.cuda.is_available():

            try:
                torch.cuda.init()

            except Exception as error:
                raise Exception("Cuda failed to initialize: %s" % error)

            if not torch.cuda.is_initialized():
                raise Exception("Cuda is not initialized")

            device = torch.device("cuda")

        if torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                raise Exception("Current PyTorch install was not built with MPS support")

            device = torch.device("mps")

        print(f"Using {device} device")
        return device

    def refresh(self):
        """
            Mapping Func 'refresh' from 'FuncStateWrapper'.
        :return:
        """

        self.device_bound.refresh()

    def reset(self):
        """
            Mapping Func 'reset' from 'FuncStateWrapper'.
        :return:
        """

        self.device_bound.reset()
