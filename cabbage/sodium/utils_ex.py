#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
import io
import os
import pathlib as p
from typing import List, Any, Dict, Tuple

import attrs
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from torch import Tensor

from sodium.types import Colors, PathType, PredictionMapTypes, PredictionTypes, BoundingBoxColorMapTypes, BoundingBox, \
    ImageInfo
from sodium.utils import imshow, cvt_datapoints_xyxy_to_codes, cvt_image_to_tensor, cvt_datapoints_fmt, \
    cvt_f32_to_u8_x_array
from sodium.x.runtime.utils.refs import ref_obj_get_val


@attrs.define
class BoxStyle:
    name: str
    color: Colors | str
    datapoints: List[float | int] | Tuple[float | int, ...]
    edgecolor: Colors | str = Colors.BLACK


@attrs.define
class DrawBoxStyleConfig:
    _cwd: str = os.path.dirname(os.path.abspath(__file__))
    font: PathType = p.Path(_cwd, "extra/fonts/Poppins/Poppins-Italic.ttf")


def draw_box_style(img: Any, box_styles: List[BoxStyle], config: DrawBoxStyleConfig = DrawBoxStyleConfig()) -> Tensor:
    if isinstance(img, Image.Image):
        img = cvt_image_to_tensor(img)
    elif isinstance(img, np.ndarray):
        img = torch.from_numpy(img)
    elif isinstance(img, Tensor):
        img = img
    else:
        raise Exception("img is not valid")

    dpi = 92
    info = ImageInfo.detect(img)
    figsize = (info.width / dpi, info.height / dpi)

    ax: Axes
    fig: Figure
    fig, ax = plt.subplots(figsize=figsize)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    ax.axis("off")
    imshow(img, axes=ax)

    for box_style in box_styles:
        color: str
        edgecolor: str

        if isinstance(box_style.color, Colors):
            color = box_style.color.name
        elif isinstance(box_style.color, str):
            color = box_style.color
        else:
            raise Exception("box_style.color is not valid")

        if isinstance(box_style.edgecolor, Colors):
            edgecolor = box_style.edgecolor.name
        elif isinstance(box_style.edgecolor, str):
            edgecolor = box_style.edgecolor
        else:
            raise Exception("box_style.edgecolor is not valid")

        x, y, w, h = cvt_datapoints_xyxy_to_codes(box_style.datapoints, fmt="xywh")

        ax.text(x=x, y=y, s=box_style.name,
                bbox=dict(facecolor=color, alpha=0.5, edgecolor=edgecolor),
                fontdict=dict(size=18, color="white", font=config.font))

        rect = Rectangle(xy=(x, y),
                         width=w,
                         height=h,
                         angle=0.0,
                         rotation_point="xy",
                         fill=False,
                         edgecolor=color,
                         linewidth=2)

        ax.add_patch(rect)

    buffer = io.BytesIO()
    fig.savefig(buffer, format="jpeg", dpi=dpi)
    fig.clear()

    # plt.close(fig=fig)  # windows hasn't closed all.
    buffer.seek(0)

    img = Image.open(buffer)
    inp = cvt_image_to_tensor(img)

    plt.close()  # close all windows.

    return inp


def draw_boxing_boxes_colored_ex(img: Tensor,
                                 predictions: PredictionMapTypes | PredictionTypes,
                                 colormaps: BoundingBoxColorMapTypes,
                                 color_default: Colors | str = Colors.RED,
                                 draw_box_style_config: DrawBoxStyleConfig = DrawBoxStyleConfig()) -> Tensor:
    box_styles: List[BoxStyle] = []
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

        # stringify.
        if isinstance(color_default, Colors):
            color_default = color_default.name

        color = color_default

        for colormap in colormaps:

            if name == colormap.name:
                color = colormap.color
                break

        if isinstance(box, Dict):
            box = BoundingBox.create_from_dict(box, BoundingBox.open_fmt(box))

        if not isinstance(box, BoundingBox):
            raise Exception("bounding box is not valid")

        # Normalize datapoints, box_color.
        datapoints = cvt_datapoints_fmt(box.data, in_fmt=box.format, out_fmt="xyxy")

        box_styles.append(BoxStyle(name=name, datapoints=datapoints, color=color))

    img = torch.from_numpy(cvt_f32_to_u8_x_array(img))
    img = draw_box_style(img, box_styles=box_styles, config=draw_box_style_config)
    return img
