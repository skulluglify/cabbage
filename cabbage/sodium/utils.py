#!/usr/bin/env python3
import os
import pathlib as p
import posixpath
import shutil
import string
import sys
import time
from typing import Any, Dict, List, Tuple, Iterable, TypeVar, Iterator

import numpy as np
import torch
import tqdm
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike
from torch import Tensor
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_bounding_boxes

from .types import (BoundingBox,
                    PredictionSingleData,
                    PredictionMapTypes,
                    PredictionTypes,
                    BoundingBoxDataPoints,
                    BoundingBoxLiterals,
                    BoundingBoxColorMapTypes, ColorLiterals, PathType, Colors, ImageInfo)
from .x.runtime.utils.refs import ref_obj_get_val


# Shortcut Conversion

def cvt_tensor_to_x_array(tensor: Tensor) -> np.ndarray:
    """
    Convert Tensor Into Array in CPU mode
    :param tensor:
    :return:
    """
    # Auto Cloning if Tensor Requires Gradient.
    if tensor.requires_grad:
        tensor = torch.clone(tensor)

    # Make it Computable on CPU.
    tensor = tensor.cpu()

    # Auto Cloning And Computable On CPU.
    # tensor = torch.clone(tensor).cpu()

    return tensor.detach().numpy()


def cvt_tensor_to_array(tensor: Tensor) -> List[Any]:
    """
    Convert Tensor Into Array in CPU mode
    :param tensor:
    :return:
    """
    return cvt_tensor_to_x_array(tensor).tolist()


# Shortcut Image.Image Conversion

def cvt_image_as_array_to_image(inp: Tensor | np.ndarray) -> Image.Image:
    img = to_pil_image(inp, mode="RGB")
    return img


def cvt_tensor_to_image(tensor: Tensor) -> Image.Image:
    img = to_pil_image(tensor.cpu(), mode="RGB")
    return img


def cvt_image_to_image_as_array(img: Image.Image) -> np.ndarray:
    inp = np.asarray(img, dtype=np.uint8)
    inp = inp.transpose((2, 0, 1))
    inp = inp.copy()
    return inp


def cvt_tensor_to_image_as_array(tensor: Tensor) -> np.ndarray:
    img = cvt_tensor_to_image(tensor)
    inp = cvt_image_to_image_as_array(img)
    return inp


def cvt_image_as_array_to_tensor(inp: np.ndarray) -> Tensor:
    tensor = torch.from_numpy(inp)
    return tensor.to(torch.uint8)


def cvt_image_to_tensor(img: Image.Image) -> Tensor:
    inp = cvt_image_to_image_as_array(img)
    return cvt_image_as_array_to_tensor(inp)


# Bounding Boxes for RCNN

def cvt_x_array_to_bbox(x: ArrayLike, fmt: BoundingBoxLiterals = "xyxy") -> Dict[str, float]:
    if isinstance(x, Tensor):
        x = cvt_tensor_to_array(x)

    if len(x) != 4:
        raise Exception("x is not accepted")

    if fmt == "xyxy":
        return dict([
            ("x1", x[0]),
            ("y1", x[1]),
            ("x2", x[2]),
            ("y2", x[3]),
        ])

    elif fmt == "cxcywh":
        return dict([
            ("cx", x[0]),
            ("cy", x[1]),
            ("w", x[2]),
            ("h", x[3]),
        ])

    else:
        return dict([
            ("x", x[0]),
            ("y", x[1]),
            ("w", x[2]),
            ("h", x[3]),
        ])


def cvt_f32_to_u8_x_array(data: ArrayLike) -> np.ndarray:
    """
        Try Conversion Float32 Array to Uint8 Array.
    :param data:
    :return:
    """

    inp = cvt_abs_range_x_array(data, 255)
    return inp.astype(np.uint8)


def cvt_u8_to_f32_x_array(data: ArrayLike) -> np.ndarray:
    """
        Try Conversion Uint8 Array to Float32 Array.
    :param data:
    :return:
    """

    return cvt_rel_range_x_array(data, 255)


# Conversion Utilities

def cvt_abs_range_x_array(data: ArrayLike, value: int) -> np.ndarray:
    """
        Conversion, Convert Array Values to Absolute Ranges if is Possible.
    :param data:
    :param value:
    :return:
    """

    if isinstance(data, Tensor):
        data = cvt_tensor_to_x_array(data)

    if isinstance(data, List) or isinstance(data, Tuple):
        data = np.asarray(data, dtype=np.float32)

    if not isinstance(data, np.ndarray):
        raise Exception("couldn't convert data to numpy array")

    data = data.astype(np.float32)

    v_min, v_max = np.min(data), np.max(data)
    v_half = value / 2

    # ex. 0. ~ 1.
    if 0 <= v_min and v_max <= 1:
        data = data * value

    # ex. -1. ~ 0.
    elif -1 <= v_min and v_max <= 0:
        data = (data + 1) * value

    # ex. -0.5 ~ 0.5
    elif -0.5 <= v_min and v_max <= 0.5:
        data = (data + 0.5) * value

    # ex. 0 ~ value
    elif 0 <= v_min and v_max <= value:
        pass

    # ex. -value ~ 0
    elif -value <= v_min and v_max <= 0:
        data = data + value

    # ex. -v_half ~ v_half
    elif -v_half <= v_min and v_max <= v_half:
        data = data + v_half

    # Not supported.
    else:
        raise Exception("it's impossible to convert array values with absolute ranges")

    # Keep floating.
    return data


def cvt_abs_range_array(data: ArrayLike, value: int) -> List[int]:
    """
        Conversion, Convert Array Values to Absolute Ranges if is Possible.
    :param data:
    :param value:
    :return:
    """

    return cvt_abs_range_x_array(data, value).tolist()


def cvt_rel_range_x_array(data: ArrayLike, value: int) -> np.ndarray:
    """
        Conversion, Convert Array Values to Relative Ranges if is Possible.
    :param data:
    :param value:
    :return:
    """

    if isinstance(data, Tensor):
        data = cvt_tensor_to_x_array(data)

    if isinstance(data, List) or isinstance(data, Tuple):
        data = np.asarray(data, dtype=np.float32)

    if not isinstance(data, np.ndarray):
        raise Exception("couldn't convert data to numpy array")

    data = data.astype(np.float32)

    v_min, v_max = np.min(data), np.max(data)
    v_half = value / 2

    # ex. 0. ~ 1.
    if 0 <= v_min and v_max <= 1:
        pass

    # ex. -1. ~ 0.
    elif -1 <= v_min and v_max <= 0:
        data = data + 1

    # ex. -0.5 ~ 0.5
    elif -0.5 <= v_min and v_max <= 0.5:
        data = data + 0.5

    # ex. 0 ~ value
    elif 0 <= v_min and v_max <= value:
        data = data / value

    # ex. -value ~ 0
    elif -value <= v_min and v_max <= 0:
        data = (data + value) / value

    # ex. -v_half ~ v_half
    elif -v_half <= v_min and v_max <= v_half:
        data = (data + v_half) / value

    # Not supported.
    else:
        raise Exception("it's impossible to convert array values with absolute ranges")

    # Keep floating.
    return data


def cvt_rel_range_array(data: ArrayLike, value: int) -> List[float]:
    """
        Conversion, Convert Array Values to Relative Ranges if is Possible.
    :param data:
    :param value:
    :return:
    """

    return cvt_rel_range_x_array(data, value).tolist()


# Conversion Coordinates

def cvt_datapoints_xyxy_to_codes(
        datapoints: List[float | int] | Tuple[float | int, ...],
        fmt: BoundingBoxLiterals = "cxcywh"
) -> Tuple[float, float, float, float]:
    """
        Convert datapoints with format 'xyxy' to codes.
    :param datapoints:
    :param fmt:
    :return:
    """

    xtl: float
    ytl: float
    xbr: float
    ybr: float

    if len(datapoints) != 4:
        raise Exception("codes is not valid")

    xtl, ytl, xbr, ybr = datapoints

    if fmt == "cxcywh":
        vw, vh = abs(xbr - xtl), abs(ybr - ytl)

        cx = xtl + (vw / 2)
        cy = ytl + (vh / 2)

        return cx, cy, vw, vh

    if fmt == "xywh":

        vw, vh = abs(xbr - xtl), abs(ybr - ytl)

        return xtl, ytl, vw, vh

    else:

        return xtl, ytl, xbr, ybr


def cvt_codes_to_datapoints_xyxy(
        codes: List[float | int | str] | Tuple[float | int | str, ...],
        start: int = 0,
        fmt: BoundingBoxLiterals = "cxcywh"
) -> Tuple[float, float, float, float]:
    """
        Convert codes to datapoints with format 'xyxy'.
    :param codes:
    :param start:
    :param fmt:
    :return:
    """

    xtl: float
    ytl: float
    xbr: float
    ybr: float

    if len(codes) != 4:
        raise Exception("codes is not valid")

    if fmt == "cxcywh":

        cx, cy = float(codes[start]), float(codes[start + 1])
        vw, vh = float(codes[start + 2]), float(codes[start + 3])

        xtl = cx - (vw / 2)
        ytl = cy - (vh / 2)

        xbr = xtl + vw
        ybr = ytl + vh

    elif fmt == "xywh":

        xtl, ytl = float(codes[start]), float(codes[start + 1])
        vw, vh = float(codes[start + 2]), float(codes[start + 3])

        xbr = xtl + vw
        ybr = ytl + vh

    else:

        # coordinates 'xyxy' 'top-left bottom-right'
        xtl, ytl = float(codes[start]), float(codes[start + 1])
        xbr, ybr = float(codes[start + 2]), float(codes[start + 3])

    return xtl, ytl, xbr, ybr


def cvt_datapoints_fmt(
        datapoints: BoundingBoxDataPoints,
        in_fmt: BoundingBoxLiterals,
        out_fmt: BoundingBoxLiterals,
) -> Tuple[float, float, float, float]:
    """
        Convert datapoints to datapoints with other format.
    :param datapoints:
    :param in_fmt:
    :param out_fmt:
    :return:
    """

    codes: List[float] = [0, 0, 0, 0]

    if isinstance(datapoints, Tensor):
        datapoints = cvt_tensor_to_array(datapoints)

    if isinstance(datapoints, np.ndarray):
        datapoints = datapoints.tolist()

    if len(datapoints) != 4:
        raise Exception("coordinates is not valid")

    codes[0] = float(datapoints[0])
    codes[1] = float(datapoints[1])
    codes[2] = float(datapoints[2])
    codes[3] = float(datapoints[3])

    return cvt_datapoints_xyxy_to_codes(
        cvt_codes_to_datapoints_xyxy(codes, fmt=in_fmt),
        fmt=out_fmt
    )


# Normalize Data

def data_prediction_normalize(categories: List[str] | Tuple[str, ...],
                              labels: ArrayLike,
                              scores: ArrayLike,
                              boxes: ArrayLike,
                              fmt: BoundingBoxLiterals = "xyxy") -> PredictionTypes:
    if isinstance(labels, Tensor):
        labels = cvt_tensor_to_array(labels)

    if isinstance(scores, Tensor):
        scores = cvt_tensor_to_array(scores)

    if isinstance(boxes, Tensor):
        boxes = cvt_tensor_to_array(boxes)

    bounders = tuple(zip(labels, map(lambda x: categories[x], labels), boxes, scores))
    return tuple(map(lambda x: PredictionSingleData.create(x[0],
                                                           name=x[1],
                                                           bbox=BoundingBox.create(x[2], fmt=fmt),
                                                           confidence=x[3]),
                     bounders))


def data_prediction_normalize_dict(categories: List[str] | Tuple[str, ...],
                                   labels: ArrayLike,
                                   scores: ArrayLike,
                                   boxes: ArrayLike,
                                   fmt: BoundingBoxLiterals = "xyxy") -> PredictionMapTypes:
    if isinstance(labels, Tensor):
        labels = cvt_tensor_to_array(labels)

    if isinstance(labels, np.ndarray):
        labels = labels.tolist()

    if isinstance(scores, Tensor):
        scores = cvt_tensor_to_array(scores)

    if isinstance(scores, np.ndarray):
        scores = scores.tolist()

    if isinstance(boxes, Tensor):
        boxes = cvt_tensor_to_array(boxes)

    if isinstance(boxes, np.ndarray):
        boxes = boxes.tolist()

    bounders = tuple(zip(labels, map(lambda x: categories[x], labels), boxes, scores))
    return tuple(map(lambda x: dict([
        ("class", x[0]),
        ("name", x[1]),
        ("box", cvt_x_array_to_bbox(x[2], fmt)),
        ("confidence", x[3])
    ]),
                     bounders))


# Drawing Utilities

def draw_boxing_boxes_colored(img: Tensor,
                              predictions: PredictionMapTypes | PredictionTypes,
                              colormaps: BoundingBoxColorMapTypes,
                              color_default: str = "red",
                              font: str | None = None,
                              font_size: int | None = None,
                              width: int = 4) -> Tensor:
    for prediction in predictions:

        name: str = ref_obj_get_val(prediction, "name")
        box: Dict[str, float | int] | BoundingBox = ref_obj_get_val(prediction, "box")
        confidence: float | int = ref_obj_get_val(prediction, "confidence")

        if name is None:
            raise Exception("data prediction has no name value")

        if box is None:
            raise Exception("data prediction has no box value")

        if confidence is None:
            raise Exception("data prediction has no confidence value")

        color = color_default

        for colormap in colormaps:

            if name == colormap.name:
                color = colormap.color
                break

        if isinstance(box, Dict):
            box = BoundingBox.create_from_dict(box, BoundingBox.open_fmt(box))

        if not isinstance(box, BoundingBox):
            raise Exception("bounding box is not valid")

        # Normalize datapoints
        datapoints = cvt_datapoints_fmt(box.data, in_fmt=box.format, out_fmt="xyxy")
        boxes = torch.from_numpy(np.asarray([datapoints], dtype=np.float32))

        img = torch.from_numpy(cvt_f32_to_u8_x_array(img))

        img = draw_bounding_boxes(img,
                                  boxes=boxes,
                                  colors=color,
                                  font=font,
                                  font_size=font_size,
                                  labels=[name],
                                  width=width)

    return img


# Shortcut Plot Image.Image Show

def imshow(img: Any, axis: bool = False, path: PathType | None = None, axes: Axes | None = None):
    inp: Tensor

    if isinstance(img, Image.Image):
        img = img
        inp = cvt_image_to_tensor(img)
    elif isinstance(img, np.ndarray):
        img = cvt_image_as_array_to_image(img)
        inp = cvt_image_to_tensor(img)
    elif isinstance(img, Tensor):
        inp = img
        img = cvt_tensor_to_image(img)
    else:
        raise Exception("img is not valid")

    dpi = 92
    info = ImageInfo.detect(inp)
    figsize = (info.width / dpi, info.height / dpi)

    ax: Axes
    fig: Figure
    fig, ax = plt.subplots(figsize=figsize)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    if axes is not None:  # another axes.
        axes.axis("on" if axis else "off")
        axes.imshow(img)  # inject.

        if path is not None:
            raise Exception("Axes is not valid")
        return

    ax.axis("on" if axis else "off")
    ax.imshow(img)

    if path is not None:

        fig.savefig(path)
        fig.clear()
        plt.close()
        return

    plt.show()
    return


# Utilities FS


def fs_make_fname_safe(name: str) -> str:
    """
        Make it safe for file name equipment.
    :param name:
    :return:
    """

    if len(name) != 0:

        temp = ""
        for x in name:

            if x in string.ascii_uppercase:
                temp += x
                continue

            if x in string.ascii_lowercase:
                temp += x
                continue

            if x in (".", "-", "_"):
                temp += x
                continue

            if x in (" ",):
                temp += "_"
                continue

        return temp

    else:
        raise Exception("value name is empty string")


def fs_make_relpath(base: PathType, path: PathType, case_sensitive: bool = True) -> str:
    """
        Make it Relative Path from Base, Path.
    .. code-block:: text

        # base: /foo/bar/main/local
        # path: /foo/bar/root/local
        # relpath: root/local
    ..

    :param base:
    :param path:
    :param case_sensitive:
    :return:
    """

    base = p.Path(base)
    path = p.Path(path)

    base_parts = base.parts
    path_parts = path.parts

    n_base_parts = len(base_parts)

    relpath_parts: List[str] = []

    stop_search = False
    for i, path_part in enumerate(path_parts):
        path_part_real = path_part

        if not case_sensitive:
            path_part = path_part.lower()

        if not stop_search:
            if i < n_base_parts:
                base_part = base_parts[i]

                if not case_sensitive:
                    base_part = base_part.lower()

                if i == 0:

                    # Windows Drive Disk.
                    base_drive = base.drive.upper()
                    path_drive = path.drive.upper()

                    # Base, Path using WindowsPath.
                    if base_drive and path_drive:
                        if base_drive == path_drive:
                            continue

                        else:
                            raise Exception("different disk drive bays")

                    # Base, or Path using WindowsPath.
                    if base_drive or path_drive:
                        if ("/" in (base_part, path_part) or
                                "\\" in (base_part, path_part)):
                            continue

                    # Base, and Path not using WindowsPath.
                    if base_part in ("/", "\\") and path_part in ("/", "\\"):
                        continue

                if base_part == path_part:
                    continue

            stop_search = True

        relpath_parts.append(path_part_real)

    n_relpath_parts = len(relpath_parts)
    if n_relpath_parts == 0:
        return "."

    return posixpath.join(*relpath_parts)


def name_of_colors(color: ColorLiterals) -> str:
    """
        Get Name of Colors.
    :param color:
    :return:
    """

    return str(color.name).lower()


_T = TypeVar("_T")


def make_progress_bar(iterable: Iterable[_T], total: int, color: ColorLiterals = Colors.GREEN) -> Iterator[_T]:
    """
        Make Progress Bar.
    :param iterable:
    :param total:
    :param color:
    :return:
    """

    animate_flow = 0.001
    start = time.perf_counter()
    for result in tqdm.tqdm(iterable, colour=name_of_colors(color), file=sys.stdout, total=total):

        end = time.perf_counter()
        elapsed = end - start

        # Make it animate flow.
        if elapsed < animate_flow:
            time.sleep(animate_flow - elapsed)

        # Refresh.
        start = end

        # Return.
        yield result


def remove_fd(path: PathType) -> bool:
    path = p.Path(path)
    if path.exists():
        if path.is_dir():
            for path, dirs, files in os.walk(path):
                for directory in dirs:
                    directory = os.path.join(path, directory)
                    shutil.rmtree(directory)

                for file in files:
                    file = os.path.join(path, file)
                    os.unlink(file)
                break
            return True

        elif path.is_file() or path.is_symlink():
            os.unlink(path)
            return True

        else:
            return False
    return False


def auto_resize_keep_resolution(image: Image.Image, max_width: int = 1080, max_height: int = 720) -> Image.Image:
    """
        calculation width and height of image,\n
    keep resolution with minimal possible can be through.
    :param image:
    :param max_width:
    :param max_height:
    :return:
    """

    width, height = image.width, image.height

    max_width = 1080
    max_height = 720

    s = min(width, max_width) / width
    width, height = width * s, height * s

    s = min(height, max_height) / height
    width, height = width * s, height * s

    width = int(width)
    height = int(height)

    return image.resize(size=(width, height))
