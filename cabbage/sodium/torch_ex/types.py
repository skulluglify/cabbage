#!/usr/bin/env python3
from typing import Literal, List, Tuple, Dict, TypeVar, Type

import attrs
from torch import Tensor

from sodium.x.runtime.wrapper import Wrapper, BaseConfig

TarReadonlyModeLiterals = Literal["r", "r:gz", "r:bz2", "r:xz"] | str
TarWriteModeLiterals = Literal["w", "w:gz", "w:bz2", "w:xz"] | str
TarModeLiterals = TarReadonlyModeLiterals | TarWriteModeLiterals
ProvideLiterals = Literal["CUDAExecutionProvider", "CPUExecutionProvider"]

# deprecated implementation used.
ImageTypes = Tensor | List[Tensor] | Tuple[Tensor, ...]

# detection
TargetType = Tensor | Dict[str, Tensor]  # labels or something else.
TargetTypes = Tensor | List[TargetType] | Tuple[TargetType, ...]
PredictionType = TargetType
PredictionTypes = TargetTypes

# batches
BatchType = Tuple[ImageTypes, TargetTypes] | Tuple[ImageTypes] | List[ImageTypes]
BatchIndex = int

# classification
X_True = List[int]
Y_Prediction = List[int]

# new implementation for next base config.
detection_truth_box_self = TypeVar("detection_truth_box_self", bound="detection_truth_box")
detection_truth_box_type = Type[detection_truth_box_self]


@attrs.define
class detection_truth_box(BaseConfig):

    labels: Tensor
    boxes: Tensor

    def merge(self, box: detection_truth_box_self) -> detection_truth_box_self:

        origin = self.copy()
        origin.labels = box.labels or origin.labels
        origin.boxes = box.boxes or origin.boxes
        return origin

    def copy(self) -> detection_truth_box_self:

        return Wrapper(self)(labels=self.labels, boxes=self.boxes)

    @classmethod
    def from_dict(cls, data: Dict[str, Tensor], *args, **kwargs) -> detection_truth_box_self:

        labels = data.get("labels")
        if labels is None:
            raise Exception("Data labels is not defined")

        boxes = data.get("boxes")
        if boxes is None:
            raise Exception("Data boxes is not defined")

        return Wrapper(cls)(labels=labels, boxes=boxes)

    def to_dict(self) -> Dict[str, Tensor]:
        return dict(labels=self.labels, boxes=self.boxes)


detection_prediction_box_self = TypeVar("detection_prediction_box_self", bound="detection_prediction_box")
detection_prediction_box_type = Type[detection_prediction_box_self]


@attrs.define
class detection_prediction_box(BaseConfig):

    labels: Tensor
    boxes: Tensor
    scores: Tensor

    def merge(self, box: detection_prediction_box_self) -> detection_prediction_box_self:

        origin = self.copy()
        origin.labels = box.labels or origin.labels
        origin.boxes = box.boxes or origin.boxes
        origin.scores = box.scores or origin.scores
        return origin

    def copy(self) -> detection_prediction_box_self:

        return Wrapper(self)(labels=self.labels, boxes=self.boxes,
                             scores=self.scores)

    @classmethod
    def from_dict(cls, data: Dict[str, Tensor], *args, **kwargs) -> detection_prediction_box_self:

        labels = data.get("labels")
        if labels is None:
            raise Exception("Data labels is not defined")

        boxes = data.get("boxes")
        if boxes is None:
            raise Exception("Data boxes is not defined")

        scores = data.get("scores")
        if scores is None:
            raise Exception("Data scores is not defined")

        return Wrapper(cls)(labels=labels, boxes=boxes,
                            scores=scores)

    def to_dict(self) -> Dict[str, Tensor]:
        return dict(labels=self.labels, boxes=self.boxes,
                    scores=self.scores)


# new implementation for next base config.
classification_labels_self = TypeVar("classification_labels_self", bound="classification_labels")
classification_labels_type = Type[classification_labels_self]


@attrs.define
class classification_labels(BaseConfig):

    x_true: Tensor
    y_prediction: Tensor

    def merge(self, box: classification_labels_self) -> classification_labels_self:

        origin = self.copy()
        origin.x_true = box.x_true or origin.x_true
        origin.y_prediction = box.y_prediction or origin.y_prediction
        return origin

    def copy(self) -> classification_labels_self:

        return Wrapper(self)(x_true=self.x_true, y_prediction=self.y_prediction)

    @classmethod
    def from_dict(cls, data: Dict[str, Tensor], *args, **kwargs) -> classification_labels_self:

        x_true = data.get("x_true")
        if x_true is None:
            raise Exception("Data x_true is not defined")

        y_prediction = data.get("y_prediction")
        if y_prediction is None:
            raise Exception("Data y_prediction is not defined")

        return Wrapper(cls)(x_true=x_true, y_prediction=y_prediction)

    def to_dict(self) -> Dict[str, Tensor]:
        return dict(x_true=self.x_true, y_prediction=self.y_prediction)
