#!/usr/bin/env python3
import logging
import mimetypes
import os.path
import pathlib as p
import random
from typing import Dict, Iterator, IO, List, Tuple, TypeVar, Type, Literal, Mapping, Any, Iterable, Sized

import attrs
import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
from numpy.typing import ArrayLike
from torch import Tensor
from torch.utils.data import Dataset

from ...state import func_state
from ...torch_ex.utils import cvt_rel_range_tensor
from ...transforms import Compose
from ...types import BoundingBoxLiterals, BoundingBox, DatasetStepLiterals, ImageInfo, LoggerFormatter
from ...utils import (cvt_tensor_to_array,
                      cvt_image_to_tensor,
                      cvt_codes_to_datapoints_xyxy,
                      cvt_abs_range_array,
                      cvt_tensor_to_x_array,
                      cvt_rel_range_array,
                      make_progress_bar)
from ...x.runtime.pool.flow.context import FlowContext
from ...x.runtime.pool.flow.process.common import Flow
from ...x.runtime.pool.flow.types import FlowSelect
from ...x.runtime.wrapper import BaseClass, BaseConfig, Wrapper

__all__ = [
    "ProjektDataset",
    "ProjektDatasetAssets",
    "ProjektDatasetConfig",
    "ProjektDatasetConfigType",
    "ProjektDatasetStepConfig",
    "ProjektDatasetStepConfigType",
    "ProjektDatasetImageAnnotationTypes",
    "ProjektDatasetImageSingleAnnotation",
    "ProjektDatasetImageSingleAnnotationType",
    "ProjektDatasetParameterType",
    "ProjektDatasetParameterTypes",
    "ProjektDatasetResultImageTypes",
    "ProjektDatasetResultImageIterator",
    "ProjektDatasetResultSingleImage",
    "ProjektDatasetResultSingleImageType",
    "ProjektLoggerFormatter",
    "ProjektLoggerFormatterType",
]

from ...x.runtime.utils.refs import ref_obj_get_val

ProjektDatasetStepConfigSelf = TypeVar("ProjektDatasetStepConfigSelf", bound="ProjektDatasetStepConfig")
ProjektDatasetStepConfigType = Type[ProjektDatasetStepConfigSelf]


@attrs.define
class ProjektDatasetStepConfig(BaseConfig):
    category: DatasetStepLiterals
    images: str
    labels: str

    def merge(self: ProjektDatasetStepConfigSelf,
              config: ProjektDatasetStepConfigSelf) -> ProjektDatasetStepConfigSelf:
        """
            Merging Two Configuration.
        :param config:
        :return:
        """

        origin = self.copy()
        origin.category = config.category or origin.category
        origin.images = config.images or origin.images
        origin.labels = config.labels or origin.labels
        return origin

    @classmethod
    def from_dict(cls: ProjektDatasetStepConfigType, data: Dict[str, str], **kwargs) -> ProjektDatasetStepConfigSelf:
        if type(data) is not dict:
            raise Exception("Value 'data' is not dictionary")

        category = data.get("category")
        if type(category) is not str:
            raise Exception("Value 'category' is not string")

        if category not in ("train", "training", "valid", "validation", "test", "testing"):
            raise Exception("Value 'category' is not valid")

        images = data.get("images")
        if type(images) is not str:
            raise Exception("Value 'images' is not string")

        labels = data.get("labels")
        if type(labels) is not str:
            raise Exception("Value 'labels' is not string")

        return Wrapper(cls)(category=category, images=images,
                            labels=labels)

    def copy(self: ProjektDatasetStepConfigSelf) -> ProjektDatasetStepConfigSelf:
        return Wrapper(self)(category=self.category, images=self.images,
                             labels=self.labels)

    @staticmethod
    def _step(step: str) -> str:
        """
            Make Naming Step
        Suitable for Any Dataset.
        :param step:
        :return:
        """
        if step in ("train", "training"):
            return "train"

        if step in ("valid", "validation"):
            return "valid"

        if step in ("test", "testing"):
            return "test"

        return "train"

    def to_dict(self: ProjektDatasetStepConfigSelf) -> Dict[str, str]:
        return {
            "category": self._step(self.category),
            "images": self.images,
            "labels": self.labels,
        }


ProjektDatasetConfigSelf = TypeVar("ProjektDatasetConfigSelf", bound="ProjektDatasetConfig")
ProjektDatasetConfigType = Type[ProjektDatasetConfigSelf]


@attrs.define
class ProjektDatasetConfig(BaseConfig):
    """
        References YOLOv8 dataset.
    """

    classes: List[str] | Tuple[str, ...]
    train: ProjektDatasetStepConfig = ProjektDatasetStepConfig(category="training", images="train/images",
                                                               labels="train/labels")

    valid: ProjektDatasetStepConfig = ProjektDatasetStepConfig(category="validation", images="valid/images",
                                                               labels="valid/labels")

    test: ProjektDatasetStepConfig = ProjektDatasetStepConfig(category="testing", images="test/images",
                                                              labels="test/labels")

    @property
    def nc(self) -> int:
        """
            Auto Direct NC to Size of classes.
        :return:
        """

        return len(self.classes)

    def merge(self: ProjektDatasetConfigSelf, config: ProjektDatasetConfigSelf) -> ProjektDatasetConfigSelf:
        """
            Merging Two Configuration.
        :param config:
        :return:
        """

        origin = self.copy()
        origin.classes = config.classes or origin.classes
        origin.train = config.train or origin.train
        origin.valid = config.valid or origin.valid
        origin.test = config.test or origin.test
        return origin

    @classmethod
    def from_dict(cls: ProjektDatasetConfigSelf, data: Dict[str, Any], **kwargs) -> ProjektDatasetConfigSelf:
        # FIXME: classes maybe is sequences but item type maybe is not string
        classes = data.get("classes")
        if type(classes) not in (list, tuple):
            raise Exception("Value 'classes' is not sequences")

        train = data.get("train")
        training = data.get("training", train)
        if training is None:
            raise Exception("Key 'train' or 'training' not found")

        train_cfg = ProjektDatasetStepConfig.from_dict(training)

        valid = data.get("valid")
        validation = data.get("validation", valid)
        if validation is None:
            raise Exception("Key 'valid' or 'validation' not found")

        valid_cfg = ProjektDatasetStepConfig.from_dict(validation)

        test = data.get("test")
        testing = data.get("testing", test)
        if testing is None:
            raise Exception("Key 'test' or 'testing' not found")

        test_cfg = ProjektDatasetStepConfig.from_dict(testing)

        return Wrapper(cls)(classes=classes, train=train_cfg,
                            valid=valid_cfg, test=test_cfg)

    def copy(self: ProjektDatasetConfigSelf) -> ProjektDatasetConfigSelf:
        return Wrapper(self)(classes=self.classes, train=self.train,
                             valid=self.valid, test=self.test)

    def to_dict(self: ProjektDatasetConfigSelf) -> Dict[str, Any]:
        return {
            "classes": self.classes,
            "training": self.train.to_dict(),
            "validation": self.valid.to_dict(),
            "testing": self.test.to_dict(),
        }


ProjektDatasetImageSingleAnnotationSelf = TypeVar("ProjektDatasetImageSingleAnnotationSelf",
                                                  bound="ProjektAnnotationSingleImage")

ProjektDatasetImageSingleAnnotationType = Type[ProjektDatasetImageSingleAnnotationSelf]


@attrs.define
class ProjektDatasetImageSingleAnnotation:
    # class id
    id: int

    # positions top-left
    xtl: float
    ytl: float

    # positions bottom-right
    xbr: float
    ybr: float

    @classmethod
    def from_tensor(cls: ProjektDatasetImageSingleAnnotationType,
                    tensor: Tensor) -> ProjektDatasetImageSingleAnnotationSelf:
        data: np.ndarray = cvt_tensor_to_x_array(tensor)

        if data.shape != (5,):
            raise Exception("Data tensor is not valid")

        codes: List[float | int] = data.tolist()

        return Wrapper(cls)(id=int(codes[0]),
                            xtl=float(codes[1]),
                            ytl=float(codes[2]),
                            xbr=float(codes[3]),
                            ybr=float(codes[4]))

    @staticmethod
    def cvt_to_tensor(annotation: ProjektDatasetImageSingleAnnotationSelf) -> Tensor:
        """
            Convert Annotation to Tensor.
        :param annotation:
        :return:
        """

        data = np.asarray([
            annotation.id,
            annotation.xtl,
            annotation.ytl,
            annotation.xbr,
            annotation.ybr
        ], dtype=np.float32)

        return torch.from_numpy(data)

    def to_tensor(self: ProjektDatasetImageSingleAnnotationSelf) -> Tensor:
        return self.cvt_to_tensor(self)


ProjektDatasetImageAnnotationTypes = (List[ProjektDatasetImageSingleAnnotation] |
                                      Tuple[ProjektDatasetImageSingleAnnotation, ...])

ProjektDatasetResultSingleImageConfigSelf = TypeVar("ProjektDatasetResultSingleImageConfigSelf",
                                                    bound="ProjektDatasetResultSingleImageConfig")

ProjektDatasetResultSingleImageConfigType = Type[ProjektDatasetResultSingleImageConfigSelf]


class ProjektDatasetResultSingleImageConfig:
    label_background_id: int = 0
    bbox_is_absolute: bool = True

    @classmethod
    def set_label_background_id(cls: ProjektDatasetResultSingleImageConfigType, label_background_id: int = 0):
        cls.label_background_id = label_background_id

    @classmethod
    def set_bbox_absolute(cls: ProjektDatasetResultSingleImageConfigType, bbox_is_absolute: bool = True):
        cls.bbox_is_absolute = bbox_is_absolute


ProjektDatasetResultSingleImageSelf = TypeVar("ProjektDatasetResultSingleImageSelf",
                                              bound="ProjektDatasetResultSingleImage")
ProjektDatasetResultSingleImageType = Type[ProjektDatasetResultSingleImageSelf]


@attrs.define
class ProjektDatasetResultSingleImage:
    """
        Result SingleImage, Data and Information about Dataset.
    """

    # Background Purposes, property of '__background__'.

    img: Tensor
    annotations: ProjektDatasetImageAnnotationTypes
    boxes: Tensor
    ids: Tensor

    config = ProjektDatasetResultSingleImageConfig()

    @classmethod
    def set_label_background_id(cls: ProjektDatasetResultSingleImageType, label_background_id: int = 0):
        cls.config.set_label_background_id(label_background_id)

    @classmethod
    def set_bbox_absolute(cls: ProjektDatasetResultSingleImageType, bbox_is_absolute: bool = True):
        cls.config.set_bbox_absolute(bbox_is_absolute)

    @classmethod
    def skip_over_bbox(cls, data: List[float], info: ImageInfo):
        """
            Skip Over Bounding Box.
        :param data:
        :param info:
        :return:
        """

        # Unpack Data Points.
        x1, y1, x2, y2 = data

        # Auto Skipping Bounding Box Over Relative Values.
        if cls.config.bbox_is_absolute:

            # normalize.
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            min_x = 0.0 <= x1 < x2
            min_y = 0.0 <= y1 < y2
            max_x = x1 < x2 <= info.width
            max_y = y1 < y2 <= info.height

            vw = min_x and max_x
            vh = min_y and max_y
            verified = vw and vh

        else:

            # norm 1.00 value.
            x1 = float(int(x1 * 100) / 100)
            y1 = float(int(y1 * 100) / 100)
            x2 = float(int(x2 * 100) / 100)
            y2 = float(int(y2 * 100) / 100)

            min_x = 0.0 <= x1 < x2
            min_y = 0.0 <= y1 < y2
            max_x = x1 < x2 <= 1.0
            max_y = y1 < y2 <= 1.0

            vw = min_x and max_x
            vh = min_y and max_y
            verified = vw and vh

        # Not Verified Auto Skip.
        if not verified:
            return None

        return BoundingBox.create([x1, y1, x2, y2], fmt="xyxy")

    @classmethod
    def open_single_boxes(cls: ProjektDatasetResultSingleImageType,
                          img: Tensor,
                          annotation: ProjektDatasetImageSingleAnnotation) -> BoundingBox:
        """
            OpenSingleBoxes, Conversion Single Annotation into Boxes with Absolute Range.
        ex. [0., 0., 1., 1.] x (640, 640) -> [0., 0., 640., 640.].
        :param img:
        :param annotation:
        :return:
        """

        info = ImageInfo.detect(img)

        x1: float | int
        y1: float | int
        x2: float | int
        y2: float | int

        if cls.config.bbox_is_absolute:

            x1, x2 = cvt_abs_range_array([
                annotation.xtl,
                annotation.xbr,
            ], info.width)

            y1, y2 = cvt_abs_range_array([
                annotation.ytl,
                annotation.ybr,
            ], info.height)

        else:

            x1, x2 = cvt_rel_range_array([
                annotation.xtl,
                annotation.xbr,
            ], info.width)

            y1, y2 = cvt_rel_range_array([
                annotation.ytl,
                annotation.ybr,
            ], info.height)

        return BoundingBox.create([x1, y1, x2, y2], fmt="xyxy")

    @classmethod
    def open_boxes_iter(cls: ProjektDatasetResultSingleImageType,
                        img: Tensor,
                        annotations: ProjektDatasetImageAnnotationTypes) -> Iterator[BoundingBox]:
        """
            OpenBoxes, Conversion Annotations into Boxes with Absolute Range.
        ex. [0., 0., 1., 1.] x (640, 640) -> [0., 0., 640., 640.].
        :param img:
        :param annotations:
        :return:
        """

        info = ImageInfo.detect(img)

        for annotation in annotations:

            x1: float | int
            y1: float | int
            x2: float | int
            y2: float | int

            if cls.config.bbox_is_absolute:

                x1, x2 = cvt_abs_range_array([
                    annotation.xtl,
                    annotation.xbr,
                ], info.width)

                y1, y2 = cvt_abs_range_array([
                    annotation.ytl,
                    annotation.ybr,
                ], info.height)

            else:

                x1, x2 = cvt_rel_range_array([
                    annotation.xtl,
                    annotation.xbr,
                ], info.width)

                y1, y2 = cvt_rel_range_array([
                    annotation.ytl,
                    annotation.ybr,
                ], info.height)

            verified: bool

            yield BoundingBox.create([x1, y1, x2, y2], fmt="xyxy")

    @classmethod
    def open_boxes(cls: ProjektDatasetResultSingleImageType,
                   img: Tensor,
                   annotations: ProjektDatasetImageAnnotationTypes) -> Tuple[BoundingBox, ...]:
        """
            OpenBoxes, Conversion Annotations into Boxes with Absolute Range.
        ex. [0., 0., 1., 1.] x (640, 640) -> [0., 0., 640., 640.].
        :param img:
        :param annotations:
        :return:
        """

        return tuple(cls.open_boxes_iter(img, annotations))

    @classmethod
    def open_single_boxes_from_array(cls: ProjektDatasetResultSingleImageType,
                                     img: Tensor,
                                     data: ArrayLike,
                                     fmt: BoundingBoxLiterals) -> BoundingBox:
        """
            OpenSingleBoxes, Conversion Single Array into Boxes with Absolute Range.
        ex. [0., 0., 1., 1.] x (640, 640) -> [0., 0., 640., 640.].
        :param img:
        :param data:
        :param fmt:
        :return:
        """

        if isinstance(data, Tensor):
            data = cvt_tensor_to_array(data)

        if isinstance(data, np.ndarray):
            data = data.tolist()

        if len(data) != 4:
            raise Exception("Bounding box is not valid")

        info = ImageInfo.detect(img)

        xtl, ytl, xbr, ybr = cvt_codes_to_datapoints_xyxy(data, fmt=fmt)

        x1: float | int
        y1: float | int
        x2: float | int
        y2: float | int

        if cls.config.bbox_is_absolute:

            x1, x2 = cvt_abs_range_array([xtl, xbr], info.width)
            y1, y2 = cvt_abs_range_array([ytl, ybr], info.height)

        else:

            x1, x2 = cvt_rel_range_array([xtl, xbr], info.width)
            y1, y2 = cvt_rel_range_array([ytl, ybr], info.height)

        return BoundingBox.create([x1, y1, x2, y2], fmt="xyxy")

    @classmethod
    def open_boxes_from_array_iter(cls: ProjektDatasetResultSingleImageType,
                                   img: Tensor,
                                   data: ArrayLike,
                                   fmt: BoundingBoxLiterals) -> Iterator[BoundingBox]:
        """
            OpenBoxes, Conversion Nested Arrays into Boxes with Absolute Range.
        ex. [0., 0., 1., 1.] x (640, 640) -> [0., 0., 640., 640.].
        :param img:
        :param data:
        :param fmt:
        :return:
        """

        if isinstance(data, Tensor):
            data = cvt_tensor_to_array(data)

        if isinstance(data, np.ndarray):
            data = data.tolist()

        data = np.asarray(data)

        if data.shape not in ((1, 4),):
            raise Exception("Bounding box is not valid")

        info = ImageInfo.detect(img)

        for box in data:
            xtl, ytl, xbr, ybr = cvt_codes_to_datapoints_xyxy(box, fmt=fmt)

            x1: float | int
            y1: float | int
            x2: float | int
            y2: float | int

            if cls.config.bbox_is_absolute:

                x1, x2 = cvt_abs_range_array([xtl, xbr], info.width)
                y1, y2 = cvt_abs_range_array([ytl, ybr], info.height)

            else:

                x1, x2 = cvt_rel_range_array([xtl, xbr], info.width)
                y1, y2 = cvt_rel_range_array([ytl, ybr], info.height)

            yield BoundingBox.create([x1, y1, x2, y2], fmt="xyxy")

    @classmethod
    def open_boxes_from_array(cls: ProjektDatasetResultSingleImageType,
                              img: Tensor,
                              data: ArrayLike,
                              fmt: BoundingBoxLiterals) -> Tuple[BoundingBox, ...]:
        """
            OpenBoxes, Conversion Nested Arrays into Boxes with Absolute Range.
        ex. [0., 0., 1., 1.] x (640, 640) -> [0., 0., 640., 640.].
        :param img:
        :param data:
        :param fmt:
        :return:
        """

        return tuple(cls.open_boxes_from_array_iter(img, data, fmt))

    @classmethod
    def create(cls: ProjektDatasetResultSingleImageType,
               img: Tensor,
               annotations: ProjektDatasetImageAnnotationTypes,
               fmt: BoundingBoxLiterals = "xyxy") -> ProjektDatasetResultSingleImageSelf:
        n = len(annotations)
        boxes = np.zeros(shape=(n, 4), dtype=np.float32)
        ids = np.zeros(shape=(n,), dtype=np.int64)

        i: int
        annotation: ProjektDatasetImageSingleAnnotation
        for i, annotation in enumerate(annotations):
            tensor = annotation.to_tensor()
            data: np.ndarray = tensor.numpy()

            boxes[i] = cls.open_single_boxes_from_array(img, data[1:5], fmt=fmt).numpy()
            ids[i] = data[0].astype(dtype=np.int64)

        return Wrapper(cls)(img=img,
                            annotations=annotations,
                            boxes=torch.from_numpy(boxes),
                            ids=torch.from_numpy(ids))

    @property
    def labels(self: ProjektDatasetResultSingleImageSelf) -> Tensor:
        """
            Get Labels from Annotations.
        :return:
        """

        k = len(self.annotations)
        data = np.zeros((k, 5), dtype=np.float32)

        i: int
        annotation: ProjektDatasetImageSingleAnnotation
        for i, annotation in enumerate(self.annotations):
            data[i] = annotation.to_tensor()

        return torch.from_numpy(data).type(dtype=torch.float32)

    @property
    def target(self) -> Dict[str, Tensor]:

        return {
            "boxes": self.boxes.type(dtype=torch.float32),

            # after label background id
            "labels": self.ids.type(dtype=torch.int64) + self.config.label_background_id + 1,
        }


ProjektDatasetResultImageIterator = Iterator[ProjektDatasetResultSingleImage]
ProjektDatasetResultImageTypes = List[ProjektDatasetResultSingleImage] | Tuple[ProjektDatasetResultSingleImage, ...]
ProjektLoggerFormatterSelf = TypeVar("ProjektLoggerFormatterSelf", bound="ProjektLoggerFormatter")
ProjektLoggerFormatterType = Type[ProjektLoggerFormatterSelf]


# Custom Logging Formatter
class ProjektLoggerFormatter(LoggerFormatter):

    def __init__(self: ProjektLoggerFormatterSelf,
                 fmt: str | None = None,
                 datefmt: str | None = None,
                 style: Literal["%", "{", "$"] = "%",
                 validate: bool = True,
                 defaults: Mapping[str, Any] | None = None):
        """
            Customize Logger Formatter.
        :param fmt:
        :param datefmt:
        :param style:
        :param validate:
        :param defaults:
        """

        super().__init__(fmt, datefmt=datefmt,
                         style=style, validate=validate,
                         defaults=defaults)


# Define Assets

@attrs.define
class ProjektDatasetAssets:
    """
        ProjektDatasetAssets, Store Values.
    """

    dataset_projekt_images: List[str]
    dataset_projekt_labels: List[str]
    dataset_projekt_images_dir: str
    dataset_projekt_labels_dir: str
    dataset_projekt_format: BoundingBoxLiterals
    dataset_projekt_annotation_ext: str
    dataset_projekt_size: int


# Pool Executor Supported
ProjektDatasetParameterType = Tuple[str, str, str, str, str, BoundingBoxLiterals, bool]
ProjektDatasetParameterTypes = List[ProjektDatasetParameterType] | Tuple[ProjektDatasetParameterType, ...]
ProjektDatasetSelf = TypeVar("ProjektDatasetSelf", bound="ProjektDataset")
ProjektDatasetType = Type[ProjektDatasetSelf]


# Main Dataset

class ProjektDataset(BaseClass, Dataset, Iterable, Sized):
    """
        Support format YOLOv8 dataset.
    """
    logger: logging.Logger

    steps: List[DatasetStepLiterals] = ["train", "valid", "test"]
    dataset_projekt_config: ProjektDatasetConfig
    dataset_projekt_dir: str

    transforms: Compose | nn.Sequential | None
    images: ProjektDatasetResultImageTypes
    params: ProjektDatasetParameterTypes

    step: DatasetStepLiterals
    label_ext: str
    fmt: BoundingBoxLiterals

    ignore_errors: bool
    skip_over_bbox: bool
    shuffle: bool

    # ProjektDatasetResultSingleImage.
    label_background_id: int
    bbox_is_absolute: bool

    # Flex: <concurrent.futures.ThreadPoolExecutor>
    # Error: <concurrent.futures.ProcessPoolExecutor>
    # wrapper: Callable[[str], ProjektDatasetResultSingleImage | None] | None
    flow_ctx: FlowContext

    def __init__(self: ProjektDatasetSelf,
                 dataset_projekt_dir: str,
                 transforms: Compose | nn.Sequential | None = None,
                 step: DatasetStepLiterals = "train",
                 label_ext: str = "text/plain",
                 fmt: BoundingBoxLiterals = "cxcywh",
                 ignore_errors: bool = True,
                 flow_ctx: FlowContext | None = None,
                 skip_over_bbox: bool = False,
                 label_background_id: int = 0,
                 bbox_is_absolute: bool = True,
                 open_config: bool = True,
                 shuffle: bool = True,
                 logger: logging.Logger | None = None):

        verify = 0
        if len(dataset_projekt_dir) != 0:
            dataset_projekt_dir_path = p.Path(dataset_projekt_dir)

            if dataset_projekt_dir_path.exists():
                if dataset_projekt_dir_path.is_dir():
                    verify = 1

            if not verify:
                raise Exception(f"Directory '{dataset_projekt_dir}' is not found")

        else:
            raise Exception("Variable 'dataset_projekt_dir' is empty string")

        if logger is None:
            logger = logging.getLogger(self.name)

        # Logging.
        self.logger = logger
        # self.logger.setLevel(logging.INFO)

        # Initial Logger Formatter.
        logfmt = ProjektLoggerFormatter()
        logfmt.hook(self.logger)

        self.dataset_projekt_dir = dataset_projekt_dir
        self.transforms = transforms

        self.images = []
        self.params = ()

        check_label_ext, label_ext = self.check_label_ext(label_ext, ignore_errors=False)
        if not check_label_ext:
            pass

        self.step = step
        self.label_ext = label_ext
        self.fmt = fmt

        self.ignore_errors = ignore_errors
        self.skip_over_bbox = skip_over_bbox

        # ProjektDatasetResultSingleImage.
        self.label_background_id = label_background_id
        ProjektDatasetResultSingleImage.set_label_background_id(self.label_background_id)

        # ProjektDatasetResultSingleImage.
        self.bbox_is_absolute = bbox_is_absolute
        ProjektDatasetResultSingleImage.set_bbox_absolute(self.bbox_is_absolute)

        if flow_ctx is None:
            jobs, processes, chunksize = Flow.usage(0.6) * 16
            # benchmark = FlowBenchmark(jobs, processes, chunksize, tune=FlowSelect.AUTO_BENCHMARK_V2)
            # benchmark.logger = self.logger
            # flow_ctx = benchmark.ctx()
            flow_ctx = FlowContext(jobs, processes, chunksize)

        self.flow_ctx = flow_ctx

        # Open Dataset Config.
        if open_config:
            self.open_projekt_config(ignore_errors=ignore_errors)

        self.shuffle = shuffle

    @property
    def root(self) -> str:
        return os.path.join(self.dataset_projekt_dir, self.step)

    def thread(self: ProjektDatasetSelf, jobs: int | None = None):
        """
            Use ThreadPoolExecutor instead of Single Run.
        :param jobs:
        :return:
        """

        self.flow_ctx.tune = FlowSelect.CORE_THREAD_SYNC
        self.flow_ctx.jobs = jobs

    def process(self: ProjektDatasetSelf, jobs: int | None = None):
        """
            Use ProcessPoolExecutor instead of Single Run.
        :param jobs:
        :return:
        """

        self.flow_ctx.tune = FlowSelect.CORE_MP_SYNC
        self.flow_ctx.jobs = jobs

    @staticmethod
    def check_label_ext(label_ext: str, ignore_errors: bool = True) -> Tuple[bool, str]:
        """
            Check Label Extension.
        :param label_ext:
        :param ignore_errors:
        :return:
        """

        verify = 0
        label_ext = label_ext.strip()

        if label_ext != "":
            verify = 1

            if not label_ext.startswith("."):
                label_ext = mimetypes.guess_extension(label_ext)

                if label_ext is None:
                    verify = 0

        # Todo: If Label_Ext is None, Make it Skipping Open Labels
        if not verify:
            if not ignore_errors:
                raise Exception("Variable 'label_ext' is None")

            return False, label_ext

        return True, label_ext

    def reset(self: ProjektDatasetSelf,
              step: DatasetStepLiterals | None = None,
              label_ext: str | None = None,
              fmt: BoundingBoxLiterals | None = None,
              ignore_errors: bool | None = None) -> bool:
        """
            Soft Reset, Clear Caches, And Refresh All Methods with FuncState.
        :param step:
        :param label_ext:
        :param fmt:
        :param ignore_errors:
        :return:
        """

        if step is None:
            step = self.step

        if label_ext is None:
            label_ext = self.label_ext

        if fmt is None:
            fmt = self.fmt

        if ignore_errors is None:
            ignore_errors = self.ignore_errors

        check_label_ext, label_ext = self.check_label_ext(label_ext, ignore_errors)
        if not check_label_ext:
            return False

        self.step = step
        self.label_ext = label_ext
        self.fmt = fmt

        return self.clear(ignore_errors)

    def refresh(self: ProjektDatasetSelf, ignore_errors: bool | None = None) -> bool:
        """
            Refresh All Methods with FuncState Decorator.
        :param ignore_errors:
        :return:
        """

        # Refresh Bounding Box.
        ProjektDatasetResultSingleImage.set_bbox_absolute(self.bbox_is_absolute)

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

        verify = 0
        func_state_refresh = ref_obj_get_val(self.assets, "refresh")
        if func_state_refresh is not None:
            if callable(func_state_refresh):
                func_state_refresh()
                verify = 1

        if not verify:
            if not ignore_errors:
                raise Exception("Function 'assets' is not implemented 'func_state' decorator")

            return False
        return True

    def clear(self: ProjektDatasetSelf, ignore_errors: bool | None = None) -> bool:
        """
            Clear All Caches, And Refresh All Methods with FuncState.
        :param ignore_errors:
        :return:
        """

        if not self.refresh(ignore_errors):
            return False

        # Clear Caches
        self.images = []
        self.params = ()

        return True

    def preload(self: ProjektDatasetSelf,
                step: DatasetStepLiterals | None = None,
                label_ext: str | None = None,
                fmt: BoundingBoxLiterals | None = None,
                params: Iterable[ProjektDatasetParameterType] | None = None,
                ignore_errors: bool | None = None,
                shuffle: bool | None = None) -> bool:
        """
            Preload Images First.
        :param step:
        :param label_ext:
        :param fmt:
        :param params:
        :param ignore_errors:
        :param shuffle:
        :return:
        """

        if step is None:
            step = self.step

        if label_ext is None:
            label_ext = self.label_ext

        if fmt is None:
            fmt = self.fmt

        if ignore_errors is None:
            ignore_errors = self.ignore_errors

        if shuffle is None:
            shuffle = self.shuffle

        if not self.reset(step, label_ext, fmt, ignore_errors):
            return False

        if params is None:
            params = self.parameters(step, label_ext=label_ext,
                                     fmt=fmt, ignore_errors=ignore_errors,
                                     progress=True)

        total = len(params)  # total same as parameters length
        if total <= 0:
            total = self.size(step, label_ext, fmt)

        # Preload.
        self.logger.info(f"Load::{self.name} bound_params='{self.open_projekt_images_params.__name__}'")
        for result in map(self.check_result_bbox_over_image_size,
                          make_progress_bar(
                              self.open_projekt_images_iter(step=step, label_ext=label_ext,
                                                            fmt=fmt, params=params,
                                                            ignore_errors=ignore_errors),
                              total=total)
                          ):

            # Skip None Image.
            if result is None:
                continue

            # Collate.
            self.images.append(result)

        # if not self.refresh(ignore_errors):
        #     return False

        # Shuffle Images.
        if shuffle:
            random.shuffle(self.images)

        # Return.
        return True

    def check_result_bbox_over_image_size(self: ProjektDatasetSelf,
                                          result: ProjektDatasetResultSingleImage) -> (ProjektDatasetResultSingleImage
                                                                                       | None):
        """
            Skipping Result if BoundingBox Over Image Size.
        :param result:
        :return:
        """

        # No Skipping Over Bounding Box.
        if not self.skip_over_bbox:
            return result

        ids: List[int] = []
        annotations: ProjektDatasetImageAnnotationTypes = []
        for annotation in result.annotations:

            xtl = annotation.xtl
            ytl = annotation.ytl
            xbr = annotation.xbr
            ybr = annotation.ybr

            # Set Minimum 0.01 Allowed.
            xtl = float(int(xtl * 100) / 100)
            ytl = float(int(ytl * 100) / 100)
            xbr = float(int(xbr * 100) / 100)
            ybr = float(int(ybr * 100) / 100)

            # Safe for used.
            min_x = 0.0 <= xtl < xbr
            min_y = 0.0 <= ytl < ybr
            max_x = xtl < xbr <= 1.0
            max_y = ytl < ybr <= 1.0

            # Check all.
            vw = min_x and max_x
            vh = min_y and max_y
            verified = vw and vh

            # Skipping Boundary Box Outside Image Size.
            if not verified:
                self.logger.warning(f"Skipping::Result(Bounding Box Over Relative Values) "
                                    f"xtl={annotation.xtl :0.2f} ytl={annotation.ytl :0.2f} "
                                    f"xbr={annotation.xbr :0.2f} ybr={annotation.ybr :0.2f}")
                continue

            annotations.append(annotation)
            ids.append(annotation.id)

        # Skipping if not found ids or annotations.

        n_ids = len(ids)
        n_annotations = len(annotations)
        n_size = n_ids or n_annotations

        if n_size == 0:
            return None

        # Open Boxes.
        boxes: Tuple[List[float], ...] = tuple(bbox.data for bbox in result.open_boxes_iter(result.img, annotations))

        # Merging Data Result.
        result.annotations = tuple(annotations)
        result.boxes = torch.from_numpy(np.asarray(boxes, dtype=np.float32))
        result.ids = torch.from_numpy(np.asarray(ids, dtype=np.float32))

        return result

    def __iter__(self) -> Iterator[Tuple[Tensor, Dict[str, Tensor]]]:
        """
            Passing Simple Iteration.
        :return:
        """
        return self.sync()

    def sync(self) -> Iterator[Tuple[Tensor, Dict[str, Tensor]]]:
        """
            Mapping Synchronously Iteration.
        :return:
        """

        for result in map(self.check_result_bbox_over_image_size,
                          self.images or self.open_projekt_images_iter()):

            if result is None:
                continue

            yield result.img, result.target

    def __getitem__(self: ProjektDatasetSelf, idx: int) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
            Auto Skipping Boundary Box Outside Image Size.
        :param idx:
        :return:
        """

        n = len(self.images)
        result: ProjektDatasetResultSingleImage | None
        result = None

        if 1 <= n:
            if n <= idx:
                raise Exception("Index out of bound array")

            result = self.images[idx]

        else:

            # Fix: make it simple cache for params,
            #  and used single param for finding single image.
            found = False
            for result in self.open_projekt_images_iter(index=idx):
                found = True
                break

            if not found:
                raise Exception("Index out of bound array")

        if result is None:
            raise Exception("Unable to reach data image")

        # Disable Checking Bounding Box, Size of Dataset has not been already preload.
        # result = self.check_result_bbox_over_image_size(result)

        # if result is None:
        #     raise Exception("bounding box is over than relative values")

        return result.img, result.target

    def __len__(self) -> int:
        """
            Get Size Of Images.
        :return:
        """

        # Return.
        return self.size(self.step)

    @func_state
    def assets(self: ProjektDatasetSelf,
               step: DatasetStepLiterals | None = None,
               label_ext: str | None = None,
               fmt: BoundingBoxLiterals | None = None) -> ProjektDatasetAssets:
        """
            OpenProjekt Images and Annotations Like YOLOv8 dataset.
        :param step:
        :param label_ext:
        :param fmt:
        :return:
        """

        if step is None:
            step = self.step

        if label_ext is None:
            label_ext = self.label_ext

        if fmt is None:
            fmt = self.fmt

        check_label_ext, label_ext = self.check_label_ext(label_ext, ignore_errors=False)
        if not check_label_ext:
            pass

        dataset_projekt_images: List[str]
        dataset_config_step: ProjektDatasetStepConfig
        default_step_images_dir: str
        default_step_labels_dir: str

        if step == "train":
            dataset_config_step = self.dataset_projekt_config.train

        elif step == "valid":
            dataset_config_step = self.dataset_projekt_config.valid

        elif step == "test":
            dataset_config_step = self.dataset_projekt_config.test

        else:
            dataset_config_step = self.dataset_projekt_config.train

        dataset_projekt_images_dir = os.path.join(self.dataset_projekt_dir,
                                                  dataset_config_step.images)

        dataset_projekt_labels_dir = os.path.join(self.dataset_projekt_dir,
                                                  dataset_config_step.labels)

        dataset_projekt_images_dir_path = p.Path(dataset_projekt_images_dir)
        dataset_projekt_labels_dir_path = p.Path(dataset_projekt_labels_dir)

        verify = 0

        if dataset_projekt_images_dir_path.exists():
            if dataset_projekt_images_dir_path.is_dir():
                verify = 1

        if not verify:
            raise Exception(f"Dataset projekt images step on '{step}' is not found")

        verify = 0

        if dataset_projekt_labels_dir_path.exists():
            if dataset_projekt_labels_dir_path.is_dir():
                verify = 1

        if not verify:
            raise Exception(f"Dataset projekt labels step on '{step}' is not found")

        # Todo: Checker On Depth with Format Extension .jpg .png .txt .ini
        dataset_projekt_images = os.listdir(dataset_projekt_images_dir_path)
        dataset_projekt_labels = os.listdir(dataset_projekt_labels_dir_path)

        dataset_projekt_images_size = len(dataset_projekt_images)
        dataset_projekt_labels_size = len(dataset_projekt_labels)

        if dataset_projekt_images_size != dataset_projekt_labels_size:
            raise Exception("Dataset projekt doesn't match size of images and labels")

        # verify = 0

        dataset_projekt_found: bool
        for dataset_projekt_image in dataset_projekt_images:
            dataset_projekt_image_name, dataset_projekt_image_ext = os.path.splitext(dataset_projekt_image)

            dataset_projekt_found = False
            for dataset_projekt_label in dataset_projekt_labels:
                dataset_projekt_label_name, dataset_projekt_label_ext = os.path.splitext(dataset_projekt_label)

                if dataset_projekt_image_name == dataset_projekt_label_name:
                    if dataset_projekt_label_ext != label_ext:
                        raise Exception("Label extension is not match")

                    dataset_projekt_found = True
                    break

            if not dataset_projekt_found:
                raise Exception(f"Label for image '{dataset_projekt_image}' is not found")

        size = dataset_projekt_images_size or dataset_projekt_labels_size

        return ProjektDatasetAssets(dataset_projekt_images=dataset_projekt_images,
                                    dataset_projekt_labels=dataset_projekt_labels,
                                    dataset_projekt_images_dir=dataset_projekt_images_dir,
                                    dataset_projekt_labels_dir=dataset_projekt_labels_dir,
                                    dataset_projekt_format=fmt,
                                    dataset_projekt_annotation_ext=label_ext,
                                    dataset_projekt_size=size)

    @func_state
    def size(self: ProjektDatasetSelf,
             step: DatasetStepLiterals | None = None,
             label_ext: str | None = None,
             fmt: BoundingBoxLiterals | None = None) -> int:
        """
            Size of Images in Dataset on Realtime.
        :param step:
        :param label_ext:
        :param fmt:
        :return:
        """

        if step is None:
            step = self.step

        if label_ext is None:
            label_ext = self.label_ext

        if fmt is None:
            fmt = self.fmt

        check_label_ext, label_ext = self.check_label_ext(label_ext, ignore_errors=False)
        if not check_label_ext:
            pass

        dataset_projekt_images_size = len(self.images)
        if dataset_projekt_images_size != 0:
            return dataset_projekt_images_size

        dataset_projekt_assets = self.assets(step, label_ext, fmt)
        dataset_projekt_size = dataset_projekt_assets.dataset_projekt_size

        return dataset_projekt_size

    def open_projekt_config(self: ProjektDatasetSelf,
                            path: str = "data.yaml",
                            ignore_errors: bool | None = None) -> ProjektDatasetConfig:
        """
            OpenProjekt Config, file: data.yaml.
            Support Custom Config file: config.yaml (Coming Soon).
        :param path:
        :param ignore_errors:
        :return:
        """

        if ignore_errors is None:
            ignore_errors = self.ignore_errors

        default = ProjektDatasetConfig(classes=[])
        dataset_projekt_dir_path = p.Path(self.dataset_projekt_dir)
        verify = 0

        # default value
        config_projekt_path = p.Path()

        if dataset_projekt_dir_path.exists():
            if dataset_projekt_dir_path.is_dir():
                config_projekt_path = dataset_projekt_dir_path.joinpath(path)
                verify = 1

        if not verify:
            err = Exception("Projekt directory is not found")
            if ignore_errors:
                self.logger.warning(err)
                return default

            else:
                raise err

        verify = 0
        with open(config_projekt_path, "rb") as fstream:
            if fstream.readable():
                config_data: Dict[str, any] = yaml.safe_load(fstream)

                # nc = int(config_data.get("nc", "0"))
                classes: List[str] = config_data.get("names", [])

                self.dataset_projekt_config = ProjektDatasetConfig(classes=classes)
                verify = 1

        if not verify:
            err = Exception("Couldn't read data configuration")
            if ignore_errors:
                self.logger.warning(err)
                return default

            else:
                raise err

        # verify = 0
        return self.dataset_projekt_config.copy()

    @classmethod
    def open_annotations_from_single_label_txt(cls: ProjektDatasetType,
                                               fstream: IO[bytes] | IO[str],
                                               fmt: BoundingBoxLiterals = "cxcywh",
                                               ) -> ProjektDatasetImageAnnotationTypes:

        if not fstream.readable():
            raise Exception("Couldn't read fstream")

        lines = fstream.readlines()
        annotations: ProjektDatasetImageAnnotationTypes = []
        line: str | bytes

        for line in lines:

            # auto convert into string
            if type(line) is bytes:
                line = line.decode("utf")

            if type(line) is not str:
                raise Exception("Couldn't decode line codes")

            # striping the line
            line = line.strip()

            # line is empty
            if line == "":
                continue

            # split the codes
            codes = line.split(" ")
            n_codes = len(codes)

            # codes is empty
            if n_codes < 1:
                raise Exception(f"Label codes from '{fstream.name}' is not valid")

            # first code is idx
            idx = int(codes[0])

            # skipping first code
            codes = codes[1:]
            n_codes = n_codes - 1

            # codes is greater than 4
            if n_codes < 4:
                raise Exception(f"Label codes from '{fstream.name}' is not valid")

            # cut off, only 4 codes left
            codes = codes[:4]
            n_codes = 4

            # codes absolutely modulo by four
            k_codes = n_codes % 4

            # codes contains residual values
            if k_codes != 0:
                raise Exception(f"Label codes from '{fstream.name}' is not valid")

            # if the code is chains with another code, 'point point point ...'
            for i in range(0, n_codes, 4):
                xtl, ytl, xbr, ybr = cvt_codes_to_datapoints_xyxy(codes, fmt=fmt)
                annotation = ProjektDatasetImageSingleAnnotation(id=idx, xtl=xtl, ytl=ytl, xbr=xbr, ybr=ybr)
                annotations.append(annotation)

        return tuple(annotations)

    @classmethod
    def open_annotations_from_single_label_xml(cls: ProjektDatasetType,
                                               fstream: IO[bytes] | IO[str],
                                               fmt: BoundingBoxLiterals = "cxcywh",
                                               ) -> ProjektDatasetImageAnnotationTypes:
        """
            Open Annotations on Single Label ext .xml. (Coming Soon).
        .. code-block:: xml

            <images>
                <image src="../butterfly.png"/>
                    <annotations>
                        <annotation id="1" xtl="0.0" ytl="0.0" xbr="0.0" ybr="0.0">
                            <datapoints xtl="0.0" ytl="0.0" xbr="0.0" ybr="0.0">
                                <datapoint vw="640"/>
                                <datapoint vh="640"/>
                            </datapoints>
                        </annotation>
                    </annotations>
                </image>
            </images>
        ..
        :param fstream:
        :param fmt:
        :return:
        """

        pass

    @classmethod
    def wrapper(cls: ProjektDatasetType,
                dataset_projekt_image: str,
                dataset_projekt_label: str,
                dataset_projekt_images_dir: str,
                dataset_projekt_labels_dir: str,
                label_ext: str,
                fmt: BoundingBoxLiterals,
                ignore_errors: bool,
                ) -> ProjektDatasetResultSingleImage | None:
        """
            OpenProjekt Images (Wrapper Only).
        .. code-block:: python

            dataset_projekt_image = data[0]
            dataset_projekt_label = data[1]
            dataset_projekt_images_dir = data[2]
            dataset_projekt_labels_dir = data[3]
            label_ext = data[4]
            fmt = data[5]
            ignore_errors = data[6]
        ..
        :param dataset_projekt_image:
        :param dataset_projekt_label:
        :param dataset_projekt_images_dir:
        :param dataset_projekt_labels_dir:
        :param label_ext:
        :param fmt:
        :param ignore_errors:
        :return:
        """

        # ref_obj_type_safe(data, "0", str)
        # ref_obj_type_safe(data, "1", str)
        # ref_obj_type_safe(data, "2", str)
        # ref_obj_type_safe(data, "3", str)
        # ref_obj_type_safe(data, "4", str)
        # ref_obj_type_safe(data, "5", BoundingBoxLiterals)
        # ref_obj_type_safe(data, "6", bool)

        verify = 0

        dataset_projekt_image = dataset_projekt_image.strip()

        if dataset_projekt_image != "":
            verify = 1

        if not verify:
            if not ignore_errors:
                raise Exception("Variable 'dataset_projekt_image' is empty string")

            return None

        verify = 0

        dataset_projekt_label = dataset_projekt_label.strip()

        if dataset_projekt_label != "":
            verify = 1

        if not verify:
            if not ignore_errors:
                raise Exception("Label is empty string")

            return None

        verify = 0

        dataset_projekt_images_dir = dataset_projekt_images_dir.strip()

        if dataset_projekt_images_dir != "":
            verify = 1

        if not verify:
            if not ignore_errors:
                raise Exception("Variable 'dataset_projekt_images_dir' is empty string")

            return None

        verify = 0

        dataset_projekt_labels_dir = dataset_projekt_labels_dir.strip()

        if dataset_projekt_labels_dir != "":
            verify = 1

        if not verify:
            if not ignore_errors:
                raise Exception("Labels directory is empty string")

            return None

        check_label_ext, label_ext = cls.check_label_ext(label_ext, ignore_errors)
        if not check_label_ext:
            return None

        dataset_projekt_images_dir_path = p.Path(dataset_projekt_images_dir)
        dataset_projekt_labels_dir_path = p.Path(dataset_projekt_labels_dir)

        dataset_projekt_image_path = dataset_projekt_images_dir_path.joinpath(dataset_projekt_image)
        # dataset_projekt_image_name, dataset_projekt_image_ext = os.path.splitext(dataset_projekt_image)

        # dataset_projekt_label_path = dataset_projekt_labels_dir_path.joinpath(dataset_projekt_label)
        dataset_projekt_label_name, dataset_projekt_label_ext = os.path.splitext(dataset_projekt_label)

        if dataset_projekt_label_ext != label_ext:
            if not ignore_errors:
                raise Exception("Label extension is not match")

            return None

        dataset_projekt_label = dataset_projekt_label_name + dataset_projekt_label_ext
        dataset_projekt_label_path = dataset_projekt_labels_dir_path.joinpath(dataset_projekt_label)

        verify = 0

        if dataset_projekt_label_path.exists():
            if dataset_projekt_label_path.is_file():
                verify = 1

        if not verify:
            if not ignore_errors:
                raise Exception(f"Label image for '{dataset_projekt_label}' is not found")

            return None

        verify = 0

        with Image.open(dataset_projekt_image_path) as dataset_projekt_image_fstream:
            with open(dataset_projekt_label_path, "rb") as dataset_projekt_label_fstream:

                if dataset_projekt_image_fstream.readonly:
                    if dataset_projekt_label_fstream.readable():
                        verify = 1

                if not verify:
                    raise Exception("Couldn't read data label or image is not set readonly")

                # verify = 0

                dataset_projekt_image_tensor = cvt_image_to_tensor(dataset_projekt_image_fstream)

                annotations: ProjektDatasetImageAnnotationTypes
                if dataset_projekt_label_ext == ".txt":
                    annotations = cls.open_annotations_from_single_label_txt(
                        dataset_projekt_label_fstream, fmt=fmt
                    )

                else:
                    if not ignore_errors:
                        raise Exception(f"Format {label_ext} for 'label_ext' is not supported")

                    return None

                # skipping result if empty annotations
                if len(annotations) != 0:
                    return ProjektDatasetResultSingleImage.create(dataset_projekt_image_tensor, annotations)

        return None

    def open_projekt_images_params(self: ProjektDatasetSelf,
                                   step: DatasetStepLiterals | None = None,
                                   label_ext: str | None = None,
                                   fmt: BoundingBoxLiterals | None = None,
                                   ignore_errors: bool | None = None) -> Iterator[ProjektDatasetParameterType]:
        """
            OpenProjekt Params for OpenProjekt Iteration.
        :param step:
        :param label_ext:
        :param fmt:
        :param ignore_errors:
        :return:
        """

        if step is None:
            step = self.step

        if label_ext is None:
            label_ext = self.label_ext

        if fmt is None:
            fmt = self.fmt

        if ignore_errors is None:
            ignore_errors = self.ignore_errors

        check_label_ext, label_ext = self.check_label_ext(label_ext, ignore_errors)
        if not check_label_ext:
            return None

        dataset_projekt_assets = self.assets(step, label_ext, fmt)
        dataset_projekt_images = dataset_projekt_assets.dataset_projekt_images
        dataset_projekt_labels = dataset_projekt_assets.dataset_projekt_labels
        dataset_projekt_images_dir = dataset_projekt_assets.dataset_projekt_images_dir
        dataset_projekt_labels_dir = dataset_projekt_assets.dataset_projekt_labels_dir

        n_dataset_projekt_images = len(dataset_projekt_images)
        n_dataset_projekt_labels = len(dataset_projekt_labels)

        if n_dataset_projekt_images == 0:
            if not ignore_errors:
                raise Exception("Images is empty list")

            return None

        if n_dataset_projekt_labels == 0:
            if not ignore_errors:
                raise Exception("Labels is empty list")

            return None

        dataset_projekt_image: str
        dataset_projekt_label: str

        # dataset_projekt_found: bool
        for dataset_projekt_image in dataset_projekt_images:
            dataset_projekt_image_name, dataset_projekt_image_ext = os.path.splitext(dataset_projekt_image)

            # dataset_projekt_found = False
            # for dataset_projekt_label in dataset_projekt_labels:
            #     dataset_projekt_label_name, dataset_projekt_label_ext = os.path.splitext(dataset_projekt_label)
            #
            #     if dataset_projekt_image_name == dataset_projekt_label_name:
            #         if dataset_projekt_label_ext != label_ext:
            #             raise Exception("label extension is not match")
            #
            #         yield (
            #             dataset_projekt_image,
            #             dataset_projekt_label,
            #             dataset_projekt_images_dir,
            #             dataset_projekt_labels_dir,
            #             label_ext,
            #             fmt,
            #             ignore_errors,
            #         )
            #
            #         dataset_projekt_found = True
            #         break
            #
            # if not dataset_projekt_found:
            #     raise Exception(f"label for image '{dataset_projekt_image}' is not found")

            # Make it faster (Check and Verify have been carried out on .assets)
            yield (
                dataset_projekt_image,
                dataset_projekt_image_name + label_ext,
                dataset_projekt_images_dir,
                dataset_projekt_labels_dir,
                label_ext,
                fmt,
                ignore_errors,
            )

        return

    def parameters(self: ProjektDatasetSelf,
                   step: DatasetStepLiterals | None = None,
                   label_ext: str | None = None,
                   fmt: BoundingBoxLiterals | None = None,
                   ignore_errors: bool | None = None,
                   progress: bool = False) -> ProjektDatasetParameterTypes:
        """
            Preload Parameters.
        :param step:
        :param label_ext:
        :param fmt:
        :param ignore_errors:
        :param progress:
        :return:
        """

        if not self.refresh(ignore_errors):
            return tuple()  # default: value

        if step is None:
            step = self.step

        if label_ext is None:
            label_ext = self.label_ext

        if fmt is None:
            fmt = self.fmt

        params: ProjektDatasetParameterTypes
        params = []

        verify = 0
        if step == self.step and \
                label_ext == self.label_ext and \
                fmt == self.fmt:

            # Assign.
            params = tuple(self.params)

            # Check Available Parameters.
            if 0 < len(params):
                verify = 1

        if not verify:

            # Initial.
            n = self.size(step, label_ext, fmt)
            parameters = self.open_projekt_images_params(step=step, label_ext=label_ext,
                                                         fmt=fmt, ignore_errors=ignore_errors)

            # Scanning.
            if progress:
                self.logger.info(f"Scan::{self.name} step='{self.step}' label_ext='{self.label_ext}' size={n}")
                params = tuple(make_progress_bar(parameters, total=n))

            else:
                params = tuple(parameters)

        # Binding And Return.
        self.params = params
        return params

    def open_projekt_images_iter(self: ProjektDatasetSelf,
                                 step: DatasetStepLiterals | None = None,
                                 label_ext: str | None = None,
                                 fmt: BoundingBoxLiterals | None = None,
                                 params: Iterable[ProjektDatasetParameterType] | None = None,
                                 ignore_errors: bool | None = None,
                                 index: int | None = None) -> Iterator[ProjektDatasetResultSingleImage]:
        """
            OpenProjekt Images and Annotations Like YOLOv8 dataset.
            With Thread Pools, but no source images has been preload.
            Update size of images must be using Preload.
        :param step:
        :param label_ext:
        :param fmt:
        :param params:
        :param ignore_errors:
        :param index:
        :return:
        """

        # if not self.refresh(ignore_errors):
        #     return

        if params is None:
            params = self.parameters(step, label_ext=label_ext,
                                     fmt=fmt, ignore_errors=ignore_errors)

        if index is None:

            # Flow
            with self.flow_ctx as ctx:

                result: ProjektDatasetResultSingleImage | None
                for result in ctx.run(self.wrapper, params, unpack=True).sync():

                    if result is None:
                        continue

                    # make it relative array.
                    result.img = cvt_rel_range_tensor(result.img, 255)

                    # transforms if needed.
                    if self.transforms is not None:
                        result.img = self.transforms(result.img)

                    yield result

        else:

            # Single Result.
            if index < len(params):
                result = self.wrapper(*params[index])

                if result is None:
                    return

                # make it relative array.
                result.img = cvt_rel_range_tensor(result.img, 255)

                # transforms if needed.
                if self.transforms is not None:
                    result.img = self.transforms(result.img)

                yield result

            else:
                raise Exception("Index out of bound array")

        return

    def open_projekt_images(self: ProjektDatasetSelf,
                            step: DatasetStepLiterals | None = None,
                            label_ext: str | None = None,
                            fmt: BoundingBoxLiterals | None = None,
                            params: Iterable[ProjektDatasetParameterType] | None = None,
                            ignore_errors: bool | None = None,
                            cache: bool = False) -> ProjektDatasetResultImageTypes:
        """
            Preload all Images.
        :param step:
        :param label_ext:
        :param fmt:
        :param params:
        :param cache:
        :param ignore_errors:
        :return:
        """

        if cache:
            if params is None:
                if self.check_initial_preload(step=step, label_ext=label_ext, fmt=fmt):
                    return self.images

            if not self.reset(step, label_ext, fmt, ignore_errors):
                pass

            self.images = tuple(self.open_projekt_images_iter(step=step, label_ext=label_ext,
                                                              fmt=fmt, params=params,
                                                              ignore_errors=ignore_errors))
            return self.images

        return tuple(self.open_projekt_images_iter(step=step, label_ext=label_ext,
                                                   fmt=fmt, params=params,
                                                   ignore_errors=ignore_errors))

    def check_initial_preload(self: ProjektDatasetSelf,
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

        if ((step is None or self.step == step) and
                (label_ext is None or self.label_ext == label_ext) and
                (fmt is None or self.fmt == fmt)):

            if len(self.images) != 0:
                return True

        return False
