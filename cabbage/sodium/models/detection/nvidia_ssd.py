#!/usr/bin/env python3
# ref: https://pytorch.org/hub/nvidia_deeplearningexamples_ssd/

import torch

NvSSDModel = torch.hub.load('NVIDIA/DeepLearningExamples:master', 'nvidia_ssd')
NvSSDUtils = torch.hub.load('NVIDIA/DeepLearningExamples:master', 'nvidia_ssd_processing_utils')
