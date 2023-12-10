#!/usr/bin/env python3
import enum
import json
import logging
import os
from typing import Any, Dict, List, Literal, Tuple, Type, TypeVar, Mapping, TextIO

import attrs
import numpy as np
from PIL import Image
from numpy.typing import ArrayLike
from torch import Tensor

from .transforms import to_pil_image
from .x.runtime.utils.refs import ref_obj_type_safe


# Colors On Command Shell
class Colors(enum.Enum):
    """
        Shell Command Color Supported.
        Default Background is Black.
        Foreground is Relative.
    """

    BLACK = "\x1b[0;30;40m"
    DARK_GREY = "\x1b[1;30;40m"
    RED = "\x1b[0;31;40m"
    RED_BOLD = "\x1b[1;31;40m"
    GREEN = "\x1b[0;32;40m"
    GREEN_BOLD = "\x1b[1;32;40m"
    YELLOW = "\x1b[0;33;40m"
    YELLOW_BOLD = "\x1b[1;33;40m"
    BLUE = "\x1b[0;34;40m"
    BLUE_BOLD = "\x1b[1;34;40m"
    PURPLE = "\x1b[0;35;40m"
    PURPLE_BOLD = "\x1b[1;35;40m"
    CYAN = "\x1b[0;36;40m"
    CYAN_BOLD = "\x1b[1;36;40m"
    WHITE = "\x1b[0;37;40m"
    WHITE_BOLD = "\x1b[1;37;40m"
    NORMAL = "\x1b[0;39;40m"
    NORMAL_BOLD = "\x1b[1;39;40m"
    RESET = "\x1b[0m"


ColorLiterals = Literal[Colors.BLACK,
                        Colors.RED,
                        Colors.GREEN,
                        Colors.YELLOW,
                        Colors.BLUE,
                        Colors.PURPLE,
                        Colors.CYAN,
                        Colors.WHITE]


# Custom Logger Formats.
class LoggerFormat(enum.Enum):
    """
        Logger Format Enumerate Std.
    """

    ERROR = f"{Colors.RED_BOLD.value}%(levelname)s{Colors.RESET.value} - " \
            f"{Colors.NORMAL_BOLD.value}%(asctime)s{Colors.RESET.value} - " \
            f"{Colors.RED_BOLD.value}%(message)s{Colors.RESET.value}"

    INFO = f"{Colors.GREEN_BOLD.value}%(levelname)s{Colors.RESET.value} - " \
           f"{Colors.NORMAL_BOLD.value}%(asctime)s{Colors.RESET.value} - " \
           f"{Colors.GREEN_BOLD.value}%(message)s{Colors.RESET.value}"

    WARN = f"{Colors.YELLOW_BOLD.value}%(levelname)s{Colors.RESET.value} - " \
           f"{Colors.NORMAL_BOLD.value}%(asctime)s{Colors.RESET.value} - " \
           f"{Colors.YELLOW_BOLD.value}%(message)s{Colors.RESET.value}"

    DEBUG = WARN
    WARNING = WARN
    CRITICAL = ERROR


LoggerFormatterSelf = TypeVar("LoggerFormatterSelf", bound="LoggerFormatter")
LoggerFormatterType = Type[LoggerFormatterSelf]


# Custom Logger Formatter.
class LoggerFormatter(logging.Formatter):
    """
        Logger Formatter Std.
    """

    channels: Dict[int, LoggerFormat] = {
        logging.DEBUG: LoggerFormat.DEBUG,
        logging.INFO: LoggerFormat.INFO,
        logging.WARNING: LoggerFormat.WARNING,
        logging.ERROR: LoggerFormat.ERROR,
        logging.CRITICAL: LoggerFormat.CRITICAL,
    }

    def __init__(self: LoggerFormatterSelf,
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

    def make_handler(self: LoggerFormatterSelf, level: int = logging.INFO) -> logging.StreamHandler[TextIO]:
        """
            Create Channel for Logger.
        :param level:
        :return:
        """
        ch = logging.StreamHandler()
        ch.setFormatter(self)
        ch.setLevel(level)
        return ch

    def hook(self: LoggerFormatterSelf, logger: logging.Logger, level=logging.INFO):
        """
            Logger Hook.
        :param logger:
        :param level:
        :return:
        """

        logger.addHandler(self.make_handler(level))

    def format(self: LoggerFormatterSelf, record: logging.LogRecord) -> str:
        fmt: LoggerFormat = self.channels.get(record.levelno)
        formatter = logging.Formatter(fmt.value)
        return formatter.format(record)


DatasetStepLiterals = Literal["train", "training", "valid", "validation", "test", "testing"] | str
BoundingBoxDataPoints = List[float] | Tuple[float, float, float, float] | Tensor | np.ndarray
BoundingBoxLiterals = Literal["cxcywh", "xywh", "xyxy"] | str


class BoundingBoxFormat(enum.Enum):
    """
        Like <torchvision.datapoints.BoundingBoxFormat>
        Without Warning Message.
    """

    CXCYWH: BoundingBoxLiterals = "cxcywh"
    XYWH: BoundingBoxLiterals = "xywh"
    XYXY: BoundingBoxLiterals = "xyxy"


BoundingBoxSelf = TypeVar("BoundingBoxSelf", bound="Boxes")
BoundingBoxType = Type[BoundingBoxSelf]


@attrs.define
class BoundingBox:
    data: List[float] | Tuple[float, ...]
    format: BoundingBoxLiterals

    @staticmethod
    def to_numpy(tensor: Tensor) -> np.ndarray:
        """
            Convert to Numpy Array.
        :param tensor:
        :return:
        """

        if tensor.requires_grad:
            return tensor.cpu().numpy()

        return tensor.detach().cpu().numpy()

    @classmethod
    def create(cls: BoundingBoxType, data: ArrayLike, fmt: BoundingBoxLiterals) -> BoundingBoxSelf:

        Wrapper: Type[BoundingBox] = cls

        if isinstance(data, Tensor):
            data = cls.to_numpy(data)

        if isinstance(data, np.ndarray):
            data = data.tolist()

        if len(data) != 4:
            raise Exception("bounding box is not valid")

        ref_obj_type_safe(data, "0", float | int)
        ref_obj_type_safe(data, "1", float | int)
        ref_obj_type_safe(data, "2", float | int)
        ref_obj_type_safe(data, "3", float | int)

        return Wrapper(data=data, format=fmt)

    @classmethod
    def create_from_dict(cls: BoundingBoxType,
                         data: Dict[str, float | int],
                         fmt: BoundingBoxLiterals) -> BoundingBoxSelf:

        temp: List[float] = [0., 0., 0., 0.]

        if fmt == "cxcywh":

            ref_obj_type_safe(data, "cx", float | int)
            ref_obj_type_safe(data, "cy", float | int)
            ref_obj_type_safe(data, "w", float | int)
            ref_obj_type_safe(data, "h", float | int)

            temp[0] = data["cx"]
            temp[1] = data["cy"]
            temp[2] = data["w"]
            temp[3] = data["h"]

        elif fmt == "xywh":

            ref_obj_type_safe(data, "x", float | int)
            ref_obj_type_safe(data, "y", float | int)
            ref_obj_type_safe(data, "w", float | int)
            ref_obj_type_safe(data, "h", float | int)

            temp[0] = data["x"]
            temp[1] = data["y"]
            temp[2] = data["w"]
            temp[3] = data["h"]

        else:

            ref_obj_type_safe(data, "x1", float | int)
            ref_obj_type_safe(data, "y1", float | int)
            ref_obj_type_safe(data, "x2", float | int)
            ref_obj_type_safe(data, "y2", float | int)

            temp[0] = data["x1"]
            temp[1] = data["y1"]
            temp[2] = data["x2"]
            temp[3] = data["y2"]

        return cls.create(temp, fmt=fmt)

    def numpy(self: BoundingBoxSelf) -> np.ndarray:
        """
            Supported Conversion to Numpy Array.
        :return:
        """
        return np.asarray(self.data, dtype=np.float32)

    @classmethod
    def open_fmt(cls: BoundingBoxType, data: Dict[str, float | int] | BoundingBoxSelf) -> BoundingBoxLiterals:

        Wrapper: Type[BoundingBox] = cls

        if isinstance(data, Wrapper):
            return data.format

        if isinstance(data, Dict):

            x = ref_obj_type_safe(data, "x", float | int, ignore_errors=True)
            y = ref_obj_type_safe(data, "y", float | int, ignore_errors=True)
            x1 = ref_obj_type_safe(data, "x1", float | int, ignore_errors=True)
            y1 = ref_obj_type_safe(data, "y1", float | int, ignore_errors=True)
            x2 = ref_obj_type_safe(data, "x2", float | int, ignore_errors=True)
            y2 = ref_obj_type_safe(data, "y2", float | int, ignore_errors=True)
            cx = ref_obj_type_safe(data, "cx", float | int, ignore_errors=True)
            cy = ref_obj_type_safe(data, "cy", float | int, ignore_errors=True)
            vw = ref_obj_type_safe(data, "w", float | int, ignore_errors=True)
            vh = ref_obj_type_safe(data, "h", float | int, ignore_errors=True)

            if cx and cy and vw and vh:
                return "cxcywh"

            elif x and y and vw and vh:
                return "xywh"

            elif x1 and y1 and x2 and y2:
                return "xyxy"

            else:
                raise Exception("bounding box is not valid")

    @staticmethod
    def cvt_to_dict(bbox: BoundingBoxSelf) -> Dict[str, float | int]:

        if len(bbox.data) != 4:
            raise Exception("bounding box is not valid")

        if bbox.format == "cxcywh":
            return {
                "cx": bbox.data[0],
                "cy": bbox.data[1],
                "w": bbox.data[2],
                "h": bbox.data[3],
            }

        elif bbox.format == "xywh":
            return {
                "x": bbox.data[0],
                "y": bbox.data[1],
                "w": bbox.data[2],
                "h": bbox.data[3],
            }

        else:
            return {
                "x1": bbox.data[0],
                "y1": bbox.data[1],
                "x2": bbox.data[2],
                "y2": bbox.data[3],
            }

    def to_dict(self: BoundingBoxSelf) -> Dict[str, float | int]:
        return self.cvt_to_dict(self)

    @classmethod
    def cvt_to_json(cls: BoundingBoxType, bbox: BoundingBoxSelf) -> str:
        return json.dumps(cls.cvt_to_dict(bbox))

    def to_json(self: BoundingBoxSelf) -> str:
        return self.cvt_to_json(self)


ImageInfoSelf = TypeVar("ImageInfoSelf", bound="ImageInfo")
ImageInfoType = Type[ImageInfoSelf]


@attrs.define
class ImageInfo:
    """
        Image Info for Pillow Image Supported.
    """

    name: str
    format: str
    mode: str
    width: int
    height: int

    @classmethod
    def detect(cls: ImageInfoType, img: Tensor) -> ImageInfoSelf:
        """
            Detection Image Information from Tensor Image.
        :param img:
        :return:
        """
        Wrapper = cls

        image: Image.Image
        with to_pil_image(img.cpu(), mode="RGB") as image:
            return Wrapper(name="<pytorch.Tensor>",
                           format=image.format,
                           mode=image.mode,
                           width=image.width,
                           height=image.height)


PredictionSingleDataSelf = TypeVar("PredictionSingleDataSelf", bound="PredictionSingleData")
PredictionSingleDataType = Type[PredictionSingleDataSelf]


@attrs.define
class PredictionSingleData:
    """
        Prediction Single Data, Contains Class Identity, Class Name, Bounding Box, Confidence.
    """

    id: int
    name: str
    bbox: BoundingBox
    confidence: float

    @classmethod
    def create(cls: PredictionSingleDataType,
               __id: int,
               name: str,
               bbox: BoundingBox,
               confidence: float) -> PredictionSingleDataSelf:
        Wrapper: Type[PredictionSingleData] = cls

        return Wrapper(id=__id, name=name, bbox=bbox, confidence=confidence)

    @classmethod
    def create_from_dict(cls: PredictionSingleDataType,
                         data: Dict[str, Any],
                         fmt: BoundingBoxLiterals) -> PredictionSingleDataSelf:
        ref_obj_type_safe(data, "id", int)
        ref_obj_type_safe(data, "name", str)
        ref_obj_type_safe(data, "box", dict)
        ref_obj_type_safe(data, "confidence", float)

        if not isinstance(data["confidence"], float):
            raise Exception("key 'confidence' is not floating number")

        return cls.create(data["class"],
                          name=data["name"],
                          bbox=BoundingBox.create_from_dict(data["box"], fmt=fmt),
                          confidence=data["confidence"])

    @staticmethod
    def cvt_to_dict(prediction: PredictionSingleDataSelf) -> Dict[str, Any]:
        return {
            "class": prediction.id,
            "name": prediction.name,
            "box": BoundingBox.cvt_to_dict(prediction.bbox),
            "confidence": prediction.confidence,
        }

    def to_dict(self: PredictionSingleDataSelf) -> Dict[str, Any]:
        return self.cvt_to_dict(self)

    @classmethod
    def cvt_to_json(cls: PredictionSingleDataType, prediction: PredictionSingleDataSelf) -> str:
        return json.dumps(cls.cvt_to_dict(prediction))

    def to_json(self: PredictionSingleDataSelf) -> str:
        return self.cvt_to_json(self)


PredictionTypes = List[PredictionSingleData] | Tuple[PredictionSingleData, ...]
PredictionMapType = Dict[str, Dict[str, float] | Any]
PredictionMapTypes = List[PredictionMapType] | Tuple[PredictionMapType, ...]

BoundingBoxColorMapSelf = TypeVar("BoundingBoxColorMapSelf", bound="BoundingBoxColorMap")
BoundingBoxColorMapType = Type[BoundingBoxColorMapSelf]


@attrs.define
class BoundingBoxColorMap:
    """
        Bounding Box Color Map Single Data.
    """

    name: str
    color: str


BoundingBoxColorMapTypes = List[BoundingBoxColorMap] | Tuple[BoundingBoxColorMap, ...]

# Universal
PathType = str | os.PathLike[str]
