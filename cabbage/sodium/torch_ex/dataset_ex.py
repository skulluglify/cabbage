#!/usr/bin/env python3
import os
import time
from typing import List, Any, Tuple, Callable

from torch import nn
from torch.utils.data import DataLoader

from sodium.projekt.yolo8x.cache import ProjektDatasetCache
from sodium.projekt.yolo8x.dataset import ProjektDataset
from sodium.types import DatasetStepLiterals

from sodium.transforms import Compose
from ..projekt.yolo8x.distribute import ProjektDatasetDistribute
from ..x.runtime.pool.flow.context import FlowContext
from ..x.runtime.pool.flow.process.common import Flow
from ..x.runtime.pool.flow.types import FlowSelect


class DatasetEx(ProjektDataset):
    _usage = Flow.usage(0.6) * 64
    flow_ctx = FlowContext(*_usage, tune=FlowSelect.MAX_CORE_SYNC)


class DatasetCacheEx(ProjektDatasetCache):
    pickle_save = True


class DatasetDistributeEx(ProjektDatasetDistribute):
    chunksize = 1024


def dataset_init_ex(projekt_dir: str,
                    projekt_cache_dir: str = "",
                    transforms: Compose | nn.Sequential | None = None,
                    preload: bool = True,
                    step: DatasetStepLiterals = "train",
                    skip_over_bbox: bool = True,
                    optimize: bool = True,
                    debug: bool = False,
                    shuffle: bool = True) -> ProjektDataset | ProjektDatasetCache | ProjektDatasetDistribute:
    """
        Dataset Initial, Make it Short of Use ProjektDataset with ProjektDatasetCache.
    :param projekt_dir:
    :param projekt_cache_dir:
    :param transforms:
    :param preload:
    :param step:
    :param skip_over_bbox:
    :param optimize:
    :param debug:
    :param shuffle:
    :return:
    """

    # String Leading And Trailing Whitespace Removed.
    projekt_dir = projekt_dir.strip()
    projekt_cache_dir = projekt_cache_dir.strip()

    # Auto Direct Link Cache Directory.
    if projekt_cache_dir == "":
        projekt_cache_dir = os.path.join(projekt_dir, "caches")

    # Dataset Initial.
    dataset = DatasetEx(projekt_dir, step=step, skip_over_bbox=True, transforms=transforms)
    dataset.skip_over_bbox = skip_over_bbox

    # Dataset Optimize.
    if optimize:
        start = time.perf_counter()

        # Dataset Cache Initial.
        cache = DatasetCacheEx(projekt_cache_dir, dataset)

        # Dataset Distribute Initial.
        distribute = DatasetDistributeEx(cache, shuffle=shuffle)

        if preload:  # preload effect.

            # Dataset Cache Preload Initial With Debug Mode.
            if debug:
                distribute.preload(pickle_save=False)  # debug: mode

            # Test Mode.
            else:
                distribute.preload()  # test: mode

        # Time Counter.
        end = time.perf_counter()
        elapsed = end - start
        print(f"Preload::{distribute.name} time {elapsed:.0f} second(s)")

        # Enable Distribution Features.
        return distribute

    # Return. (Distribution Enable Features)
    return dataset


def collate_fn(batch: List[Any]) -> Tuple[Any, ...]:
    """
        Collate Function for DataLoader.
    :param batch:
    :return:
    """
    return tuple(zip(*batch))


def make_dataloader_ex(projekt_dir: str,
                       projekt_cache_dir: str = "",
                       transforms: Compose | nn.Sequential | None = None,
                       preload: bool = True,
                       step: DatasetStepLiterals = "train",
                       skip_over_bbox: bool = True,
                       optimize: bool = False,
                       batch_size: int = 1,
                       num_workers: int = 0,
                       shuffle: bool = False,
                       callback: Callable = (lambda x: x)):
    """
        Dataloader Initial, Make it Short of Use func 'dataset_init'.
    :param callback:
    :param projekt_dir:
    :param projekt_cache_dir:
    :param transforms:
    :param preload:
    :param step:
    :param skip_over_bbox:
    :param optimize:
    :param batch_size:
    :param num_workers:
    :param shuffle:
    :return:
    """
    # Catch Dataset.
    dataset = dataset_init_ex(projekt_dir=projekt_dir,
                              projekt_cache_dir=projekt_cache_dir,
                              transforms=transforms,
                              preload=preload,
                              step=step,
                              skip_over_bbox=skip_over_bbox,
                              optimize=optimize,
                              shuffle=shuffle)

    if callable(callback):
        dataset = callback(dataset)

    # Initial DataLoader.
    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      collate_fn=collate_fn,
                      num_workers=num_workers)
