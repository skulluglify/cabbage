#!/usr/bin/env python3
import json
import logging
import os
import pathlib as p
import time
from typing import Iterable, Sized, Iterator, Tuple, Dict, TypeVar, Type, List, Any

import attrs
from torch import Tensor
from torch.utils.data import Dataset

from sodium.projekt.yolo8x.cache import ProjektDatasetCache
from sodium.projekt.yolo8x.dataset import ProjektDatasetParameterType, ProjektDataset, ProjektLoggerFormatter, \
    ProjektDatasetStepConfig, ProjektDatasetParameterTypes, ProjektDatasetResultSingleImage, \
    ProjektDatasetResultImageTypes
from sodium.state import func_state
from sodium.types import DatasetStepLiterals, BoundingBoxLiterals
from sodium.x.runtime.utils.refs import ref_obj_get_val
from sodium.x.runtime.wrapper import BaseClass, BaseConfig, Wrapper

ProjektDatasetDistributePartitionSelf = TypeVar("ProjektDatasetDistributePartitionSelf",
                                                bound="ProjektDatasetDistributePartition")

ProjektDatasetDistributePartitionType = Type[ProjektDatasetDistributePartitionSelf]


@attrs.define
class ProjektDatasetDistributePartition(BaseConfig):

    # Unique.
    path: str

    # ProjektDatasetDistributeConfig
    parts: List[str] | Tuple[str, ...]
    volumes: List[int] | Tuple[int, ...]
    chunksize: int
    size: int

    # ProjektDatasetConfig
    classes: List[str] | Tuple[str, ...]
    step: ProjektDatasetStepConfig

    @property
    def nc(self) -> int:
        """
            Auto Direct NC to Size of classes.
        :return:
        """

        return len(self.classes)

    def merge(self: ProjektDatasetDistributePartitionSelf,
              config: ProjektDatasetDistributePartitionSelf) -> ProjektDatasetDistributePartitionSelf:

        origin = self.copy()
        origin.path = config.path or origin.path
        origin.parts = config.parts or origin.parts
        origin.volumes = config.volumes or origin.volumes
        origin.chunksize = config.chunksize or origin.chunksize
        origin.size = config.size or origin.size
        origin.classes = config.classes or origin.classes
        origin.step = config.step or origin.step
        return origin

    @classmethod
    def from_dict(cls: ProjektDatasetDistributePartitionSelf,
                  data: Dict[str, Any], **kwargs) -> ProjektDatasetDistributePartitionSelf:

        path = data.get("path")
        if type(path) is not str:
            raise Exception("Value 'path' is not integer")

        # FIXME: classes maybe is sequences but item type maybe is not string
        classes = data.get("classes")
        if type(classes) is not list:
            raise Exception("Value 'classes' is not sequences")

        # FIXME: step maybe is dictionary
        step = data.get("step")
        if type(step) is not dict:
            raise Exception("Value 'step' on metadata file configuration is not dictionary")

        step_cfg = ProjektDatasetStepConfig.from_dict(step)

        # FIXME: volumes maybe is sequences but item type maybe is not integer
        volumes = data.get("volumes")
        if type(volumes) not in (list, tuple):
            raise Exception("Value 'volumes' is not sequences")

        # FIXME: parts maybe is sequences but item type maybe is not string
        parts = data.get("parts")
        if type(parts) not in (list, tuple):
            raise Exception("Value 'parts' is not sequences")

        chunksize = data.get("chunksize")
        if type(chunksize) is not int:
            raise Exception("Value 'chunksize' is not integer")

        size = data.get("size")
        if type(size) is not int:
            raise Exception("Value 'size' is not integer")

        # Verify Volumes And Size.
        if sum(volumes) != size:
            raise Exception(f"Volumes on metadata is not valid")

        return Wrapper(cls)(path=path, classes=classes, step=step_cfg,
                            parts=parts, volumes=volumes,
                            chunksize=chunksize, size=size)

    def copy(self: ProjektDatasetDistributePartitionSelf) -> ProjektDatasetDistributePartitionSelf:
        """
            Func 'Copy'.
        :return:
        """
        return Wrapper(self)(path=self.path, classes=self.classes, step=self.step,
                             parts=self.parts, volumes=self.volumes,
                             chunksize=self.chunksize, size=self.size)

    def to_dict(self: ProjektDatasetDistributePartitionSelf) -> Dict[str, Any]:
        return {
            "path": self.path,
            "classes": self.classes,
            "step": self.step.to_dict(),
            "parts": self.parts,
            "volumes": self.volumes,
            "chunksize": self.chunksize,
            "size": self.size,
        }


ProjektDatasetDistributeConfigSelf = TypeVar("ProjektDatasetDistributeConfigSelf",
                                             bound="ProjektDatasetDistributeConfig")
ProjektDatasetDistributeConfigType = Type[ProjektDatasetDistributeConfigSelf]


@attrs.define
class ProjektDatasetDistributeConfig(BaseConfig):

    train: ProjektDatasetDistributePartition | None
    valid: ProjektDatasetDistributePartition | None
    test: ProjektDatasetDistributePartition | None

    def merge(self, config: ProjektDatasetDistributeConfigSelf) -> ProjektDatasetDistributeConfigSelf:
        """
            Merging Two Configuration.
        :param config:
        :return:
        """

        origin = self.copy()
        origin.train = config.train or origin.train
        origin.valid = config.valid or origin.valid
        origin.test = config.test or origin.test
        return origin

    @classmethod
    def from_dict(cls: ProjektDatasetDistributeConfigSelf,
                  data: Dict[str, Any], path: str | None = None, **kwargs) -> ProjektDatasetDistributeConfigSelf:

        train = None
        valid = None
        test = None

        # TODO: maybe steps is list but item is not dictionary
        steps = data.get("steps")
        if type(steps) is not list:
            raise Exception("Value 'steps' is not dictionary")

        for partition in steps:
            if type(partition) is not dict:
                raise Exception("Value 'partition' is not dictionary")

            # normalize naming partition category.
            partition = ProjektDatasetDistributePartition.from_dict(partition)
            category = partition.step.category.lower()

            if path is not None:
                if p.Path(os.path.abspath(partition.path)) != p.Path(os.path.abspath(path)):
                    continue

            if category in ("train", "training"):
                partition.step.category = "train"
                train = partition
                continue

            if category in ("valid", "validation"):
                partition.step.category = "valid"
                valid = partition
                continue

            if category in ("test", "testing"):
                partition.step.category = "test"
                test = partition
                continue

        return Wrapper(cls)(train=train, valid=valid, test=test)

    def copy(self: ProjektDatasetDistributeConfigSelf) -> ProjektDatasetDistributeConfigSelf:
        """
            Func 'Copy'.
        :return:
        """
        return Wrapper(self)(train=self.train, valid=self.valid, test=self.test)

    def to_dict(self: ProjektDatasetDistributeConfigSelf) -> Dict[str, Any]:
        """
            Convert To Dictionary.
        :return:
        """

        parts = (self.train, self.valid, self.test)
        return {
            "steps": [
                part.to_dict() for part in parts if part is not None
            ],
        }


ProjektDatasetDistributeSelf = TypeVar("ProjektDatasetDistributeSelf", bound="ProjektDatasetDistribute")
ProjektDatasetDistributeType = Type[ProjektDatasetDistributeSelf]


class ProjektDatasetDistribute(BaseClass, Dataset, Iterable, Sized):
    """
        Projekt Dataset Distribute, Handling for Local, And Global Caches.
    """

    logger: logging.Logger

    dataset: ProjektDatasetCache
    dataset_projekt_distribute_config: ProjektDatasetDistributeConfig | None
    chunksize: int

    ignore_errors: bool
    shuffle: bool

    _dataset_projekt_dir: str  # for `preload` required.
    _dataset_projekt_cache_dir: str  # for `__getitem__` required.
    _was_preload: bool  # for `preload` required.

    def __init__(self: ProjektDatasetDistributeSelf,
                 dataset: ProjektDataset | ProjektDatasetCache,
                 pickle_save: bool = True,
                 make_cache_dir: bool = True,
                 ignore_errors: bool = True,
                 chunksize: int = 1024,
                 shuffle: bool = True,
                 logger: logging.Logger | None = None):

        # Enable Caching.
        if isinstance(dataset, ProjektDataset):
            dataset_projekt_cache_dir = os.path.join(dataset.dataset_projekt_dir, "caches")
            dataset = ProjektDatasetCache(dataset_projekt_cache_dir=dataset_projekt_cache_dir,
                                          dataset=dataset, pickle_save=pickle_save,
                                          make_cache_dir=make_cache_dir,
                                          ignore_errors=ignore_errors)

        # Logging.
        if logger is None:
            logger = logging.getLogger(self.name)

        self.logger = logger

        # Initial Logger Formatter.
        logfmt = ProjektLoggerFormatter()
        logfmt.hook(self.logger)

        # Initial.
        self.dataset = dataset
        self.ignore_errors = ignore_errors
        self.dataset_projekt_distribute_config = None  # trigger for preload
        self.chunksize = chunksize
        self.shuffle = shuffle

        self._dataset_projekt_dir = dataset.dataset.dataset_projekt_dir
        self._dataset_projekt_cache_dir = dataset.dataset_projekt_cache_dir
        self._was_preload = False

    @property
    def root(self) -> str:
        return self.dataset.dataset.dataset_projekt_dir

    @property
    def images(self: ProjektDatasetDistributeSelf) -> ProjektDatasetResultImageTypes:
        """
            Mapping Parameters.
        :return:
        """

        err = Exception("It's deprecated property in dataset distribution")
        if self.ignore_errors:
            self.logger.warning(err)

        else:
            raise err

        return self.dataset.images

    @images.setter
    def images(self: ProjektDatasetDistributeSelf,
               images: ProjektDatasetResultImageTypes):
        """
            Mapping Parameters.
        :param images:
        :return:
        """

        err = Exception("It's deprecated property in dataset distribution")
        if self.ignore_errors:
            self.logger.warning(err)

        else:
            raise err

        self.dataset.images = images

    @property
    def params(self: ProjektDatasetDistributeSelf) -> ProjektDatasetParameterTypes:
        """
            Mapping Parameters.
        :return:
        """

        return self.dataset.params

    @params.setter
    def params(self: ProjektDatasetDistributeSelf,
               params: ProjektDatasetParameterTypes):
        """
            Mapping Parameters.
        :param params:
        :return:
        """

        self.dataset.params = params

    def open_projekt_config(self: ProjektDatasetDistributeSelf,
                            path: str = "metadata.json",
                            ignore_errors: bool | None = None) -> ProjektDatasetDistributeConfig:
        """
            Open Projekt Dataset Distribute Config. (Metadata File Configuration)
        :param path:
        :param ignore_errors:
        :return:
        """

        if ignore_errors is None:
            ignore_errors = self.ignore_errors

        dataset_projekt_distribute_config: ProjektDatasetDistributeConfig
        default = ProjektDatasetDistributeConfig(train=None, valid=None, test=None)

        # Initial.
        dataset_projekt_distribute_file_config = os.path.join(self._dataset_projekt_cache_dir, path)
        dataset_projekt_distribute_file_config_path = p.Path(dataset_projekt_distribute_file_config)

        verify = 0
        if dataset_projekt_distribute_file_config_path.exists():
            if dataset_projekt_distribute_file_config_path.is_file():
                verify = 1

        if not verify:
            err = Exception(f"FileConfig on '{dataset_projekt_distribute_file_config}' is not found")
            if ignore_errors:
                self.logger.warning(err)
                return default

            else:
                raise err

        verify = 0
        with open(dataset_projekt_distribute_file_config, "rb") as fstream:
            if fstream.readable():

                path = self._dataset_projekt_dir
                dataset_projekt_distribute_config = ProjektDatasetDistributeConfig.from_json(fstream.read(), path=path)
                verify = 1

            if not verify:
                err = Exception(f"File '{dataset_projekt_distribute_file_config}' is not give read permission")
                if ignore_errors:
                    self.logger.warning(err)
                    return default

                else:
                    raise err

        step = self.dataset.dataset.step
        partition = getattr(dataset_projekt_distribute_config, step)

        if partition is None:
            if ignore_errors:
                err = Exception("Recovery data config, no setup required")
                self.logger.warning(err)
                return dataset_projekt_distribute_config  # maybe merging.

            else:
                raise Exception("Value 'step' is not valid")

        # Verify Partition Volumes Check.
        volumes = partition.volumes
        size = partition.size

        if sum(volumes) != size:
            err = Exception(f"Volumes on metadata is not valid")
            if ignore_errors:
                self.logger.warning(err)
                return default

            else:
                raise err

        # Setup.
        self.chunksize = partition.chunksize
        self.dataset_projekt_distribute_config = dataset_projekt_distribute_config
        return dataset_projekt_distribute_config

    def check_initial_preload(self: ProjektDatasetDistributeSelf,
                              step: DatasetStepLiterals | None = None,
                              label_ext: str | None = None,
                              fmt: BoundingBoxLiterals | None = None) -> bool:
        """
            Check Initial Preload.
        :param step:
        :param label_ext:
        :param fmt:
        :return:
        """

        if self._was_preload:
            return True

        return self.dataset.check_initial_preload(step=step, label_ext=label_ext, fmt=fmt)

    def preload(self: ProjektDatasetDistributeSelf,
                step: DatasetStepLiterals | None = None,
                label_ext: str | None = None,
                fmt: BoundingBoxLiterals | None = None,
                params: Iterable[ProjektDatasetParameterType] | None = None,
                pickle_save: bool | None = None,
                ignore_errors: bool | None = None,
                shuffle: bool | None = None) -> bool:
        """
            Projekt Dataset Distribution. (Preload)
        :param step:
        :param label_ext:
        :param fmt:
        :param params:
        :param pickle_save:
        :param ignore_errors:
        :param shuffle:
        :return:
        """

        config = self.open_projekt_config(ignore_errors=True)
        dataset = self.dataset.dataset

        if step is None:
            step = dataset.step

        # Skipping Preload, If Config Was Created.
        # if config is not None:
        #     partition = getattr(config, step)
        #
        #     if partition is not None:
        #         self.chunksize = partition.chunksize  # update chunksize
        #         return False

        if fmt is None:
            fmt = dataset.fmt

        if label_ext is None:
            label_ext = dataset.label_ext

        if pickle_save is None:
            pickle_save = self.dataset.pickle_save

        if params is None:
            params = self.dataset.parameters(step=step, label_ext=label_ext,
                                             fmt=fmt, ignore_errors=ignore_errors,
                                             progress=True)

        if ignore_errors is None:
            ignore_errors = self.ignore_errors

        if shuffle is None:
            shuffle = self.shuffle

        # params must be slicing of chunk.

        n = 0  # size of partitions
        volumes: List[int] = []
        params_size = len(params)
        current_dataset_size = 0
        dataset_projekt_cache_dir = self._dataset_projekt_cache_dir
        for params_idx in range(0, params_size, self.chunksize):
            self.logger.info(f"Preload::{self.name} chunksize={self.chunksize} step={n}")

            # Initial.
            params_chunk = params[params_idx:params_idx + self.chunksize]
            dataset_projekt_cache_chunk_dir = os.path.join(dataset_projekt_cache_dir, str(n))

            # Preload.
            dataset = ProjektDatasetCache(dataset_projekt_cache_chunk_dir,
                                          dataset=self.dataset.dataset,
                                          make_cache_dir=True,
                                          logger=self.logger)

            # Preload.
            dataset.preload(step=step, label_ext=label_ext,
                            fmt=fmt, params=params_chunk,
                            pickle_save=pickle_save,
                            ignore_errors=ignore_errors,
                            shuffle=shuffle)

            # Update.
            dataset_size = dataset.size()
            current_dataset_size += dataset_size
            volumes.append(dataset_size)
            n += 1

            # Free Memory.
            dataset.clear(ignore_errors=ignore_errors)

        # Return Back.
        # self._dataset_projekt_cache_dir = dataset_projekt_cache_dir
        parts = [str(i) for i in range(n)]

        # TODO: create metadata.json,
        #  check existing file metadata
        #  and load indexes distribution cache
        #  to iteration of dataset

        train = None
        valid = None
        test = None

        dataset_projekt_config = self.dataset.dataset.dataset_projekt_config
        step = dataset.dataset.step

        if step in ("train", "training"):
            train = ProjektDatasetDistributePartition(
                path=dataset.dataset.dataset_projekt_dir,
                classes=dataset_projekt_config.classes,
                step=dataset_projekt_config.train,
                parts=parts, volumes=volumes,
                chunksize=self.chunksize,
                size=current_dataset_size
            )

        elif step in ("valid", "validation"):
            valid = ProjektDatasetDistributePartition(
                path=dataset.dataset.dataset_projekt_dir,
                classes=dataset_projekt_config.classes,
                step=dataset_projekt_config.valid,
                parts=parts, volumes=volumes,
                chunksize=self.chunksize,
                size=current_dataset_size
            )

        elif step in ("test", "testing"):
            test = ProjektDatasetDistributePartition(
                path=dataset.dataset.dataset_projekt_dir,
                classes=dataset_projekt_config.classes,
                step=dataset_projekt_config.test,
                parts=parts, volumes=volumes,
                chunksize=self.chunksize,
                size=current_dataset_size
            )

        dataset_projekt_distribute_config = ProjektDatasetDistributeConfig(train=train, valid=valid, test=test)

        # store values.
        self.dataset_projekt_distribute_config = dataset_projekt_distribute_config

        # create metadata file configuration
        # FIXME: metadata must be includes sha1 verify for pickle files
        metadata_file_config_path = "metadata.json"
        dataset_projekt_distribute_file_config = os.path.join(dataset_projekt_cache_dir, metadata_file_config_path)

        # Open Metadata file configuration for Any Steps.
        config_origin: Any | Dict[str, Any] | List[Any] | Tuple[Any, ...]
        dataset_projekt_distribute_file_config_path = p.Path(dataset_projekt_distribute_file_config)

        verify = 0
        if dataset_projekt_distribute_file_config_path.exists():
            if dataset_projekt_distribute_file_config_path.is_file():
                verify = 1

        if verify:
            with open(dataset_projekt_distribute_file_config, "r") as fstream:

                verify = 0
                if fstream.readable():
                    config_origin = json.loads(fstream.read())
                    verify = 1

                if not verify:
                    raise Exception("Couldn't write metadata file configuration")

        else:
            config_origin = {}

        verify = 0
        if type(config_origin) is dict:

            steps = config_origin.get("steps")
            if type(steps) is list:
                verify = 1

        if not verify:
            self.logger.warning("Config Origin Exists is not valid")
            config_origin = {
                "steps": []
            }

        any_steps = []
        steps = config_origin.get("steps", [])
        for part in steps:  # steps, parts

            path = part.get("path")
            if type(path) is not str:
                raise Exception("Value 'path' is not string")

            if p.Path(os.path.abspath(path)) != p.Path(os.path.abspath(self._dataset_projekt_dir)):
                any_steps.append(part)

        # Merging Current Step.
        config = config.merge(dataset_projekt_distribute_config)

        # Merging for Any Steps.
        parts = (config.train, config.valid, config.test)
        config_origin["steps"] = any_steps + [part.to_dict() for part in parts if part is not None]

        # Writable Metadata file configuration.
        with open(dataset_projekt_distribute_file_config, "w") as fstream:

            verify = 0
            if fstream.writable():
                fstream.write(json.dumps(config_origin))
                verify = 1

            if not verify:
                raise Exception("Couldn't write metadata file configuration")

        self._was_preload = True
        return True

    def __getitem__(self: ProjektDatasetDistributeSelf, idx: int) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
            Get Item Per Index. (Bad Idea for Dataset Distribution)

        Using Iteration Instead Of Get Item Per Index. (Optimizing)
        :param idx:
        :return:
        """

        if not self._was_preload:  # if not preload yet.
            return self.dataset[idx]

        config = self.open_projekt_config(ignore_errors=self.ignore_errors)

        # get current partition.
        step = self.dataset.dataset.step
        partition = getattr(config, step)
        # dataset = self.dataset.dataset

        if partition is None:
            raise Exception("Value 'step' is not valid")

        parts_size = 0
        pickle_save = self.dataset.pickle_save
        for volume_idx, volume in enumerate(partition.volumes):
            part = partition.parts[volume_idx]
            parts_size += volume

            if parts_size <= idx:
                continue

            # Initial.
            dataset = self.dataset
            dataset_projekt_cache_chunk_dir = os.path.join(self._dataset_projekt_cache_dir, part)

            # Binding Current Dataset.
            if dataset.dataset_projekt_cache_dir != dataset_projekt_cache_chunk_dir:
                dataset = ProjektDatasetCache(dataset_projekt_cache_chunk_dir,
                                              dataset=self.dataset.dataset,
                                              pickle_save=pickle_save,
                                              ignore_errors=self.ignore_errors,
                                              make_cache_dir=False,
                                              logger=self.logger)

                # Reset.
                dataset.reset()

                # Preload.
                dataset.preload(pickle_save=self.dataset.pickle_save,
                                shuffle=self.shuffle)

                # Debug Mode.
                # time.sleep(1.0)

            # Return Back for Caching.
            self.dataset = dataset

            prev_parts_size = parts_size - volume
            idx = idx - prev_parts_size

            # Preload It's Bad Idea for Getting Single Image.
            result = dataset[idx]
            if result is None:
                raise Exception("Unable to reach data image")

            return result
        raise Exception("Index out of bound array")

    def open_projekt_images_iter(self: ProjektDatasetDistributeSelf,
                                 step: DatasetStepLiterals | None = None,
                                 label_ext: str | None = None,
                                 fmt: BoundingBoxLiterals | None = None,
                                 ignore_errors: bool | None = None,
                                 shuffle: bool | None = None) -> Iterator[ProjektDatasetResultSingleImage]:
        """
            Mapping Open Projekt Images Iteration.
        :param step:
        :param label_ext:
        :param fmt:
        :param ignore_errors:
        :param shuffle:
        :return:
        """

        config = self.open_projekt_config(ignore_errors=self.ignore_errors)
        dataset = self.dataset.dataset

        if step is None:
            step = dataset.step

        if fmt is None:
            fmt = dataset.fmt

        if label_ext is None:
            label_ext = dataset.label_ext

        if ignore_errors is None:
            ignore_errors = self.ignore_errors

        if shuffle is None:
            shuffle = self.shuffle

        partition = getattr(config, step)

        if partition is None:
            raise Exception("Value 'step' is not valid")

        pickle_save = self.dataset.pickle_save
        for volume_idx, volume in enumerate(partition.volumes):
            part = partition.parts[volume_idx]

            # Initial.
            dataset_projekt_cache_chunk_dir = os.path.join(self._dataset_projekt_cache_dir, part)

            # Preload.
            dataset = ProjektDatasetCache(dataset_projekt_cache_chunk_dir,
                                          dataset=self.dataset.dataset,
                                          ignore_errors=ignore_errors,
                                          make_cache_dir=True,
                                          logger=self.logger)

            # Reset.
            dataset.reset()

            # Preload.
            dataset.preload(step=step, label_ext=label_ext, fmt=fmt, params=None,
                            pickle_save=pickle_save, ignore_errors=ignore_errors,
                            shuffle=shuffle)

            for data in iter(dataset.dataset.images):
                yield data

            # Free Memory.
            dataset.clear(ignore_errors=ignore_errors)
        return

    def sync(self) -> Iterator[Tuple[Tensor, Dict[str, Tensor]]]:
        """
            Mapping Synchronously Iteration.
        :return:
        """

        if not self._was_preload:
            return iter(self.dataset)

        # for result in map(self.dataset.dataset.check_result_bbox_over_image_size,
        #                   self.open_projekt_images_iter()):
        #
        #     if result is None:
        #         continue
        #
        #     yield result.img, result.target

        for result in self.open_projekt_images_iter():
            yield result.img, result.target

    def __iter__(self: ProjektDatasetDistributeSelf) -> Iterator[Tuple[Tensor, Dict[str, Tensor]]]:
        return self.sync()

    def __len__(self: ProjektDatasetDistributeSelf) -> int:
        return self.size()

    @func_state
    def size(self: ProjektDatasetDistributeSelf) -> int:
        """
            Get Size Function.
        :return:
        """

        config = self.open_projekt_config(ignore_errors=self.ignore_errors)

        step = self.dataset.dataset.step
        partition = getattr(config, step)

        if partition is None:
            raise Exception("Value 'step' is not valid")

        return partition.size

    def reset(self: ProjektDatasetDistributeSelf,
              step: DatasetStepLiterals | None = None,
              label_ext: str | None = None,
              fmt: BoundingBoxLiterals | None = None,
              ignore_errors: bool | None = None) -> bool:
        """
            Mapping Reset Function.
        :param step:
        :param label_ext:
        :param fmt:
        :param ignore_errors:
        :return:
        """

        if ignore_errors is None:
            ignore_errors = self.ignore_errors

        self.dataset_projekt_distribute_config = None
        self.ignore_errors = ignore_errors
        self._was_preload = False

        return self.dataset.reset(step=step, label_ext=label_ext,
                                  fmt=fmt, ignore_errors=ignore_errors)

    def refresh(self: ProjektDatasetDistributeSelf, ignore_errors: bool | None = None) -> bool:
        """
            Mapping Refresh Function.
        :param ignore_errors:
        :return:
        """

        if ignore_errors is None:
            ignore_errors = self.ignore_errors

        verify = 0
        func_state_refresh = ref_obj_get_val(self.size, "refresh")
        if func_state_refresh is not None:
            if callable(func_state_refresh):
                func_state_refresh()
                verify = 1

        if not verify:
            if not ignore_errors:
                raise Exception("Function 'size' is not implemented 'func_state' decorator")

            return False

        # TODO: Maybe Projekt Dataset Distribution Need Refreshing.
        return self.dataset.refresh(ignore_errors=ignore_errors)

    def clear(self: ProjektDatasetDistributeSelf, ignore_errors: bool | None = None) -> bool:
        """
            Mapping Clear Function.
        :param ignore_errors:
        :return:
        """

        if ignore_errors is None:
            ignore_errors = self.ignore_errors

        self.dataset_projekt_distribute_config = None
        self.ignore_errors = ignore_errors
        self._was_preload = False

        # TODO: Maybe Projekt Dataset Distribution Need Clear Cache.
        return self.dataset.clear(ignore_errors=ignore_errors)


def main():
    logging.basicConfig(level=logging.INFO)

    dataset = ProjektDataset("datasets/projekt",
                             step="train",
                             # transforms=Compose([
                             #     Resize((224, 244), antialias=False),
                             # ]),
                             skip_over_bbox=True)

    distribute = ProjektDatasetDistribute(dataset)
    distribute.preload()

    start = time.perf_counter()
    # for i in range(distribute.size()):
    #     print(distribute[i])
    #     print(i)

    # for i, data in enumerate(distribute):  # 113 seconds
    #     print(data)
    #     print(i)

    # dataset + 1core = ~526 seconds
    # dataset + flow = ~268 seconds
    # distribute + caching = ~113 seconds

    for _ in distribute:
        pass

    end = time.perf_counter()
    elapsed = end - start
    print(f"time elapsed {elapsed:.2f} second(s)")


if str(__name__).upper() in ("__MAIN__",):
    main()
