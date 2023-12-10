#!/usr/bin/env python3
import math
from typing import List, Tuple, TypeVar, ParamSpec, Iterator, Any

import torch
from PIL import Image
from torch import Tensor, nn
from torchvision.datasets import VisionDataset

from sodium.utils import cvt_image_to_tensor


def mutation_fn(image: Tensor | Image.Image,
                transforms: List[nn.Module | None],
                scales: List[int]) -> Tensor:
    """
        Mutation Function.
    :param image:
    :param transforms:
    :param scales:
    :return:
    """
    if isinstance(image, Image.Image):
        image = cvt_image_to_tensor(image)

    image = image.cpu().clone()  # make it computable on CPU.

    if len(transforms) != len(scales):
        raise Exception("must be same as size transforms and scales")

    k = 0
    shape = (sum(scales), *image.shape)
    images = torch.zeros(shape, dtype=image.dtype)
    for i, transform in enumerate(transforms):
        for j in range(scales[i]):
            images[k] = image if transform is None else transform(image)
            k += 1

    return images


class MutationRandomImageBinding(VisionDataset):
    dataset: VisionDataset
    transforms: List[nn.Module | None]
    scales: List[int]  # multiple transforms.

    # caching
    image_mut_cache: Tensor | None
    image_mut_cache_index: int
    image_mut_cache_label: Any

    def __init__(self, dataset: VisionDataset,
                 transforms: List[nn.Module | None],
                 scales: List[int] | None = None):
        super().__init__(root=dataset.root)
        self.dataset = dataset
        self.transforms = transforms
        if scales is None:
            scales = [1] * len(self.transforms)
        self.scales = scales

        # initial caching.
        self.image_mut_cache = None
        self.image_mut_cache_index = -1
        self.image_mut_cache_label = 0

    def __iter__(self) -> Iterator[Tuple[Tensor, Any]]:
        """
            Use this for commonly usable.
        :return:
        """

        for data in self.dataset:
            image, label = data
            for image_mut in mutation_fn(image, self.transforms, self.scales):
                yield image_mut, label  # virtual indexes.

    def __getitem__(self, v_index: int) -> Tuple[Tensor, Any]:
        """
            GetItem Data (Tensor, Label) on Dataset.
        :param v_index:
        :return:
        """
        n = len(self.dataset)
        s = sum(self.scales)
        k = n * s  # length after transformation.

        index = math.floor(v_index * n / k)  # index before transformation.

        y = index + 1  # length before transformation. (dynamic)
        p = y * s  # prefix length before transformation. (dynamic)
        x = p - v_index - 1  # index on current transformation. (reverse)
        x = s - x - 1  # fix index start at 0. (reverse - reverse)

        image_mut: Tensor
        label: Any

        if self.image_mut_cache_index != index:
            data = self.dataset[index]
            image, label = data
            image_mut = mutation_fn(image, self.transforms, self.scales)
            self.image_mut_cache = image_mut
            self.image_mut_cache_index = index
            self.image_mut_cache_label = label

        else:
            image_mut = self.image_mut_cache
            label = self.image_mut_cache_label

        return image_mut[x], label  # virtual indexes.

    def __len__(self) -> int:
        return len(self.dataset) * sum(self.scales)  # after transformation.


_P = ParamSpec("_P")
MutationRandomImageSelf = TypeVar("MutationRandomImageSelf", bound="MutationRandomImage")


class MutationRandomImage:
    transforms: List[nn.Module | None]
    scales: List[int]  # multiple transforms.

    def __init__(self, transforms: List[nn.Module | None], scales: List[int] | None = None):
        self.transforms = transforms
        if scales is None:
            scales = [1] * len(self.transforms)
        self.scales = scales

    # slots = ("transforms", "scales")  # or use `__annotations__` instead.
    def __call__(self, dataset: VisionDataset) -> MutationRandomImageBinding:
        return MutationRandomImageBinding(dataset, transforms=self.transforms, scales=self.scales)
