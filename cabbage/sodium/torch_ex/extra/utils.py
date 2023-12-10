#!/usr/bin/env python3
import logging
import os
from threading import Semaphore
from typing import List, Tuple, Dict, Iterator

import torch
from torch import Tensor

from sodium.motext.text import text_safe_name_kv
from sodium.torch_ex.types import PredictionType
from sodium.transforms import crop
from sodium.types import PathType, PredictionMapTypes, PredictionTypes, BoundingBox
from sodium.utils import cvt_tensor_to_image, cvt_datapoints_fmt
from sodium.x.runtime.utils.refs import ref_obj_get_val
from sodium.x.runtime.wrapper import BaseClass


class BatchTool(BaseClass):
    logger: logging.Logger
    labels_iter: Dict[str, int]
    semaphore: Semaphore

    def __init__(self):
        self.logger = logging.getLogger(self.name)
        self.labels_iter = {}
        self.semaphore = Semaphore(1)

    def next_iter(self, name: str) -> int:
        self.semaphore.acquire(blocking=True)
        name = text_safe_name_kv(name)
        num_iter = self.labels_iter.setdefault(name, 0)
        self.labels_iter[name] += 1
        self.semaphore.release()
        return num_iter

    def save(self, path: PathType,
             batch: Tuple[Tensor, List[str | float | int]],
             classes: List[str] | None = None, width: int = 7):
        os.makedirs(path, exist_ok=True)  # base dir.

        # loop.
        for image, label in zip(*batch):
            name: str

            if not isinstance(label, str):
                if isinstance(label, float | int):
                    if classes is None:
                        raise Exception("classes is not valid")

                    name = classes[int(label)]

                else:
                    raise Exception("label is not string or number either")
            else:
                name = label

            workdir = os.path.join(path, name)
            os.makedirs(workdir, exist_ok=True)  # sub folders, labels.
            self.logger.info("MAKE " + workdir)

            num_iter = self.next_iter(name)
            num_iter_str = str(num_iter).rjust(width, "0")  # like .zfill
            file_path = os.path.join(workdir, f"{num_iter_str}.jpg")
            self.logger.info("SAVE " + file_path)

            pil_image = cvt_tensor_to_image(image)
            pil_image.save(file_path, format="jpeg")
            self.logger.info("DONE " + file_path)


def crop_by_predictions(image: Tensor,
                        predictions: PredictionMapTypes | PredictionTypes
                        ) -> Iterator[Tuple[Tensor, str, float | int]]:
    if not (len(image.shape) == 3 and image.shape[0] >= 3):
        raise Exception("image shape is not valid")

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

        if isinstance(box, Dict):
            box = BoundingBox.create_from_dict(box, BoundingBox.open_fmt(box))

        if not isinstance(box, BoundingBox):
            raise Exception("bounding box is not valid")

        # Normalize datapoints, box_color.
        datapoints = cvt_datapoints_fmt(box.data, in_fmt=box.format, out_fmt="xywh")
        xtl, ytl, width, height = datapoints

        yield crop(torch.clone(image), top=int(ytl), left=int(xtl),
                   width=int(width), height=int(height)), name, confidence


def nms(prediction: PredictionType, iou_threshold: float = 0.8) -> PredictionType:

    if isinstance(prediction, dict):
        labels: Tensor = prediction["labels"]
        scores: Tensor = prediction["scores"]
        boxes: Tensor = prediction["boxes"]

        t_labels = []
        t_scores = []
        t_boxes = []

        i: int
        score: Tensor
        for i, score in enumerate(scores):
            if iou_threshold <= float(score):
                label = labels[i]
                box = boxes[i]

                box: Tensor

                t_labels.append(label.tolist())
                t_scores.append(score.tolist())
                t_boxes.append(box.tolist())

        labels = torch.tensor(t_labels, dtype=labels.dtype)
        scores = torch.tensor(t_scores, dtype=scores.dtype)
        boxes = torch.tensor(t_boxes, dtype=boxes.dtype)

        return dict(labels=labels, scores=scores, boxes=boxes)
