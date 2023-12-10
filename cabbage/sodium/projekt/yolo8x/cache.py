#!/usr/bin/env python3
import logging
import os
import pathlib as p
import pickle
import random

from io import BufferedReader, BufferedWriter
from typing import Iterable, Sized, IO, Iterator, Tuple, Dict, Literal, TypeVar, Type

from torch import Tensor
from torch.utils.data import Dataset

from .dataset import (ProjektDataset,
                      ProjektLoggerFormatter,
                      ProjektDatasetParameterType,
                      ProjektDatasetParameterTypes,
                      ProjektDatasetResultImageTypes)

from ...types import DatasetStepLiterals, BoundingBoxLiterals
from ...utils import fs_make_fname_safe, fs_make_relpath
from ...x.runtime.registry import cwd
from ...x.runtime.utils.refs import ref_obj_get_val

__all__ = [
    "ProjektDatasetCache",
]

from ...x.runtime.wrapper import BaseClass

ProjektDatasetCacheSelf = TypeVar("ProjektDatasetCacheSelf", bound="ProjektDatasetCache")
ProjektDatasetCacheType = Type[ProjektDatasetCacheSelf]


class ProjektDatasetCache(BaseClass, Dataset, Iterable, Sized):
    """
        Enable cache features on ProjektDataset.
    """

    dataset_projekt_cache_dir: str
    dataset: ProjektDataset

    pickle_save: bool
    ignore_errors: bool
    shuffle: bool

    logger: logging.Logger

    def __init__(self: ProjektDatasetCacheSelf,
                 dataset_projekt_cache_dir: str,
                 dataset: ProjektDataset,
                 pickle_save: bool = True,
                 make_cache_dir: bool = True,
                 ignore_errors: bool = True,
                 shuffle: bool = True,
                 logger: logging.Logger | None = None):

        # Logging.
        if logger is None:

            # Auto Binding.
            if dataset.logger is None:
                dataset.logger = logging.getLogger(self.name)

            logger = dataset.logger

        self.logger = logger

        # Initial Logger Formatter.
        logfmt = ProjektLoggerFormatter()
        logfmt.hook(self.logger)

        # Check and Verify directory 'dataset_projekt_cache_dir'.
        verify = 0
        if len(dataset_projekt_cache_dir) != 0:
            dataset_cache_dir_path = p.Path(dataset_projekt_cache_dir)

            if dataset_cache_dir_path.exists():
                if dataset_cache_dir_path.is_dir():
                    verify = 1

            if not verify:
                if not make_cache_dir:
                    raise Exception(f"Directory '{dataset_projekt_cache_dir}' is not found")

                else:
                    self.logger.info(f"Make::Directory('{dataset_projekt_cache_dir}')")
                    os.makedirs(dataset_projekt_cache_dir, mode=755, exist_ok=True)

        else:
            raise Exception("Value 'dataset_projekt_cache_dir' is empty string")

        self.dataset_projekt_cache_dir = dataset_projekt_cache_dir
        self.dataset = dataset

        # current state of value 'pickle_save'.
        self.pickle_save = pickle_save

        # current state of value 'self.dataset.ignore_errors'.
        self.dataset.ignore_errors = ignore_errors
        self.ignore_errors = ignore_errors
        self.shuffle = shuffle

    @property
    def root(self) -> str:
        return self.dataset.dataset_projekt_dir

    @property
    def images(self: ProjektDatasetCacheSelf) -> ProjektDatasetResultImageTypes:
        """
            Mapping Parameters. (Distribution Requirements)
        :return:
        """

        return self.dataset.images

    @images.setter
    def images(self: ProjektDatasetCacheSelf,
               images: ProjektDatasetResultImageTypes):
        """
            Mapping Parameters. (Distribution Requirements)
        :param images:
        :return:
        """

        self.dataset.images = images

    @property
    def params(self: ProjektDatasetCacheSelf) -> ProjektDatasetParameterTypes:
        """
            Mapping Parameters. (Distribution Requirements)
        :return:
        """

        return self.dataset.params

    @params.setter
    def params(self: ProjektDatasetCacheSelf,
               params: ProjektDatasetParameterTypes):
        """
            Mapping Parameters. (Distribution Requirements)
        :param params:
        :return:
        """

        self.dataset.params = params

    def parameters(self: ProjektDatasetCacheSelf,
                   step: DatasetStepLiterals | None = None,
                   label_ext: str | None = None,
                   fmt: BoundingBoxLiterals | None = None,
                   ignore_errors: bool | None = None,
                   progress: bool = False) -> ProjektDatasetParameterTypes:
        """
            Mapping Parameters. (Distribution Requirements)
        :param step:
        :param label_ext:
        :param fmt:
        :param ignore_errors:
        :param progress:
        :return:
        """

        return self.dataset.parameters(step=step,
                                       label_ext=label_ext,
                                       fmt=fmt,
                                       ignore_errors=ignore_errors,
                                       progress=progress)

    @property
    def fpath(self: ProjektDatasetCacheSelf) -> p.Path:
        """
            File Path for Pickle File Save.
        :return:
        """

        dataset_projekt_step_config = ref_obj_get_val(self.dataset.dataset_projekt_config, self.dataset.step)

        if dataset_projekt_step_config is None:
            raise Exception(f"Directory '{self.dataset.step}' is not valid")

        dataset_projekt_step_dir = ref_obj_get_val(dataset_projekt_step_config, "images")
        dataset_projekt_step_dir_path = p.Path(dataset_projekt_step_dir)

        if dataset_projekt_step_dir is None:
            raise Exception(f"Directory '{dataset_projekt_step_config}' is not valid")

        dataset_projekt_cache_step = "." + ".".join(dataset_projekt_step_dir_path.parts)

        # Make it relative
        dataset_projekt_dir = fs_make_relpath(cwd, self.dataset.dataset_projekt_dir)
        dataset_projekt_dir_path = p.Path(dataset_projekt_dir)

        dataset_projekt_cache_name = ".".join(dataset_projekt_dir_path.parts)
        dataset_projekt_cache_ext = ".pkl"

        # powerful 'os.path.abspath'.
        # check same as base directory.
        dataset_projekt_dir = os.path.abspath(self.dataset.dataset_projekt_dir)
        dataset_projekt_cache_dir = os.path.abspath(self.dataset_projekt_cache_dir)

        # FIXME: maybe path different style with current operating system used.
        #  Using `.startswith` not more reliable for any purposes.
        same_as_base_dir = dataset_projekt_cache_dir.startswith(dataset_projekt_dir)

        dataset_projekt_cache_file: str
        # handling for file name if cache dir is same as base directory.
        if not same_as_base_dir:
            # different base directory, must be specific for file name generator.
            dataset_projekt_cache_file = (dataset_projekt_cache_name +
                                          dataset_projekt_cache_step +
                                          dataset_projekt_cache_ext)

        else:
            # minimal file name for same as base directory.
            dataset_projekt_cache_file = (self.dataset.step + dataset_projekt_cache_ext)

        # Safe file name.
        dataset_projekt_cache_file = fs_make_fname_safe(dataset_projekt_cache_file)
        dataset_projekt_cache_file = dataset_projekt_cache_file.lower()

        dataset_projekt_cache_path = p.Path(os.path.join(self.dataset_projekt_cache_dir, dataset_projekt_cache_file))

        return dataset_projekt_cache_path

    def open_file_projekt_cache(self: ProjektDatasetCacheSelf, mode: Literal["rb", "wb"] | str = "rb") -> IO[bytes]:
        """
            OpenFile Projekt Cache, File stream.
        :return:
        """

        verify: int
        dataset_projekt_cache_path = self.fpath

        if mode in ("rb",):

            # Check and verify dataset_projekt_cache_path.
            verify = 0
            if dataset_projekt_cache_path.exists():
                if dataset_projekt_cache_path.is_file():
                    verify = 1

            if not verify:
                raise Exception(f"File '{dataset_projekt_cache_path}' is not found")

        if mode in ("rb", "wb"):

            fstream = open(dataset_projekt_cache_path, mode)

            if isinstance(fstream, BufferedReader) or isinstance(fstream, BufferedWriter):
                return fstream

            fstream.close()
            raise Exception(f"File stream is not valid")

        else:
            raise Exception(f"File stream mode '{mode}' is not supported")

    def check_initial_preload(self: ProjektDatasetCacheSelf,
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

        return self.dataset.check_initial_preload(step=step, label_ext=label_ext, fmt=fmt)

    def preload(self: ProjektDatasetCacheSelf,
                step: DatasetStepLiterals | None = None,
                label_ext: str | None = None,
                fmt: BoundingBoxLiterals | None = None,
                params: Iterable[ProjektDatasetParameterType] | None = None,
                pickle_save: bool | None = None,
                ignore_errors: bool | None = None,
                shuffle: bool | None = None) -> bool:
        """
            Cache After Preload, Auto Mapping Direct To Dataset Preload.
        :param step:
        :param label_ext:
        :param fmt:
        :param params:
        :param pickle_save:
        :param ignore_errors:
        :param shuffle:
        :return:
        """

        if pickle_save is None:
            pickle_save = self.pickle_save

        if ignore_errors is None:
            ignore_errors = self.ignore_errors

        if shuffle is None:
            shuffle = self.shuffle

        dataset_projekt_cache_path = self.fpath

        verify = 0
        if dataset_projekt_cache_path.exists():
            if dataset_projekt_cache_path.is_file():
                verify = 1

        if not verify:
            self.logger.info(f"Preload::{self.dataset.name}")
            self.dataset.preload(step=step, label_ext=label_ext, fmt=fmt,
                                 params=params, ignore_errors=ignore_errors)

            if pickle_save:
                self.logger.info(f"Pickle::Save('{self.fpath}')")
                with self.open_file_projekt_cache(mode="wb") as fstream:

                    if fstream.writable():
                        images = tuple(self.dataset.images)  # make it immutable sequence.
                        pickle.dump(images, fstream)

                    else:
                        raise Exception(f"File '{dataset_projekt_cache_path}' not give writable permission")

        else:
            if not self.check_initial_preload(step=step, fmt=fmt, label_ext=label_ext):
                self.logger.info(f"Pickle::Load('{self.fpath}')")

                # Reset On Dataset.
                self.dataset.reset(step=step, label_ext=label_ext, fmt=fmt, ignore_errors=ignore_errors)

                verify = 0
                with self.open_file_projekt_cache(mode="rb") as fstream:

                    if fstream.readable():
                        images = list(pickle.load(fstream))  # make it mutable sequence.

                        if shuffle:  # shuffle images, realtime.
                            random.shuffle(images)

                        self.dataset.images = images
                        verify = 1

                    if not verify:
                        err = Exception(f"File '{dataset_projekt_cache_path}' not give readable permission")
                        if ignore_errors:
                            self.logger.warning(err)
                            return False

                        else:
                            raise err

            else:
                self.logger.info(f"Current::Load({self.dataset.name})")

            return False
        return True

    def __getitem__(self: ProjektDatasetCacheSelf, idx: int) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
            Mapping get item.
        :param idx:
        :return:
        """

        return self.dataset[idx]

    def __iter__(self: ProjektDatasetCacheSelf) -> Iterator[Tuple[Tensor, Dict[str, Tensor]]]:
        """
            Passing Simple Iteration.
        :return:
        """

        return self.sync()

    def sync(self: ProjektDatasetCacheSelf) -> Iterator[Tuple[Tensor, Dict[str, Tensor]]]:
        """
            Mapping iteration.
        :return:
        """

        img: Tensor
        target: Dict[str, Tensor]
        for img, target in iter(self.dataset):
            yield img, target

    def __len__(self: ProjektDatasetCacheSelf) -> int:
        """
            Mapping size of images.
        :return:
        """

        return self.size()

    def size(self: ProjektDatasetCacheSelf) -> int:
        """
            Mapping size of images.
        :return:
        """

        return self.dataset.size()

    def reset(self: ProjektDatasetCacheSelf,
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

        return self.dataset.reset(step=step, label_ext=label_ext,
                                  fmt=fmt, ignore_errors=ignore_errors)

    def refresh(self: ProjektDatasetCacheSelf, ignore_errors: bool | None = None) -> bool:
        """
            Mapping Refresh Function.
        :param ignore_errors:
        :return:
        """

        return self.dataset.refresh(ignore_errors=ignore_errors)

    def clear(self: ProjektDatasetCacheSelf, ignore_errors: bool | None = None) -> bool:
        """
            Mapping Clear Function.
        :param ignore_errors:
        :return:
        """

        return self.dataset.clear(ignore_errors=ignore_errors)
