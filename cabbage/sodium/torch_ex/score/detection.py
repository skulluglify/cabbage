#!/usr/bin/env python3
import json
import logging
import math
from typing import Dict, Type, TypeVar, List, Tuple, Iterator

import attrs
import torch
from torchmetrics.detection import MeanAveragePrecision, IntersectionOverUnion, CompleteIntersectionOverUnion

from sodium.torch_ex.types import PredictionTypes, TargetTypes
from sodium.types import BoundingBoxLiterals
from sodium.x.runtime.types import void
from sodium.x.runtime.utils.refs import ref_obj_get_val
from sodium.x.runtime.wrapper import BaseClass, BaseConfig

ScoreSelf = TypeVar("ScoreSelf", bound="Score")
ScoreAnother = ScoreSelf
ScoreType = Type[ScoreSelf]


@attrs.define(order=True)
class Score(BaseConfig):
    mean_ap: float
    iou: float
    ciou: float

    @classmethod
    def zeros(cls: ScoreType, num: int = 1) -> Tuple[ScoreSelf, ...]:
        Wrapper = cls
        return tuple(Wrapper(mean_ap=0.0, iou=0.0, ciou=0.0) for _ in range(num))

    @classmethod
    def ones(cls: ScoreType, num: int = 1) -> Tuple[ScoreSelf, ...]:
        Wrapper = cls
        return tuple(Wrapper(mean_ap=1.0, iou=1.0, ciou=1.0) for _ in range(num))

    def isvalid(self):
        """
            Check is valid value or NaN/Inf.
        :return:
        """

        if (math.isnan(self.mean_ap) or math.isinf(self.mean_ap)) and \
                (math.isnan(self.iou) or math.isinf(self.iou)) and \
                (math.isnan(self.ciou) or math.isinf(self.ciou)):

            return False
        return True

    def merge(self: ScoreSelf, config: ScoreSelf) -> ScoreSelf:
        """
            Merging Two Configuration.
        :param config:
        :return:
        """

        origin = self.copy()
        origin.mean_ap = config.mean_ap or origin.mean_ap
        origin.iou = config.iou or origin.iou
        origin.ciou = config.ciou or origin.ciou
        return origin

    @classmethod
    def from_dict(cls: ScoreType, data: Dict[str, float], **kwargs):
        """
            Create from Dictionary.
        :param data:
        :return:
        """
        Wrapper = cls

        score_map = ref_obj_get_val(data, "map", 0.0)
        score_iou = ref_obj_get_val(data, "iou", 0.0)
        score_ciou = ref_obj_get_val(data, "ciou", 0.0)

        return Wrapper(mean_ap=score_map,
                       iou=score_iou,
                       ciou=score_ciou)

    def to_dict(self: ScoreSelf) -> Dict[str, float | int]:
        """
            Return as Dictionary.
        :return:
        """

        return {
            "accuracy": self.iou,  # default using.
            "map": self.mean_ap,
            "iou": self.iou,
            "ciou": self.ciou,
        }

    def copy(self) -> ScoreSelf:
        """
            Copy Existing Object.
        :return:
        """
        Wrapper: ScoreType = self.__class__
        return Wrapper(mAP=self.mean_ap, IoU=self.iou, cIoU=self.ciou)

    @classmethod
    def avg(cls, scores: List[ScoreSelf] | Tuple[ScoreSelf, ...]) -> ScoreSelf:
        Wrapper = cls

        n = len(scores)
        avg_score = Wrapper(mean_ap=0.0, iou=0.0, ciou=0.0)

        for score in scores:
            avg_score.mean_ap += score.mean_ap
            avg_score.iou += score.iou
            avg_score.ciou += score.ciou

        avg_score.mean_ap /= n
        avg_score.iou /= n
        avg_score.ciou /= n

        return avg_score


def get_score_mean_ap_iou_ciou(predictions: PredictionTypes,
                               targets: TargetTypes,
                               box_format: BoundingBoxLiterals = "xyxy") -> Score:
    """
        Get Score Mean Average Precision, Intersection Over Union, And Complete Intersection Over Union.
    :param predictions:
    :param targets:
    :param box_format:
    :return:
    """

    # Mean Average Precision (mAP).
    mean_ap = MeanAveragePrecision(box_format=box_format, iou_type="bbox")
    mean_ap.update(predictions, targets)
    data_mean_ap = mean_ap.compute()

    # Intersection Over Union (IoU).
    iou = IntersectionOverUnion(box_format=box_format)
    iou.update(predictions, targets)
    data_iou = iou.compute()

    # Complete Intersection Over Union (cIoU).
    ciou = CompleteIntersectionOverUnion(box_format=box_format)
    ciou.update(predictions, targets)
    data_ciou = ciou.compute()

    val_mean_ap = ref_obj_get_val(data_mean_ap, "map", torch.tensor(0.0)).cpu()
    val_iou = ref_obj_get_val(data_iou, "iou", torch.tensor(0.0)).cpu()
    val_ciou = ref_obj_get_val(data_ciou, "ciou", torch.tensor(0.0)).cpu()

    val_mean_ap = val_mean_ap if not math.isnan(val_mean_ap) else torch.tensor(0.0)
    val_iou = val_iou if not math.isnan(val_iou) else torch.tensor(0.0)
    val_ciou = val_ciou if not math.isnan(val_ciou) else torch.tensor(0.0)

    return Score.from_dict({
        "map": float(val_mean_ap),
        "iou": float(val_iou),
        "ciou": float(val_ciou),
    })


def get_predictions_iou_threshold(predictions: PredictionTypes, iou_threshold: float | int) -> PredictionTypes:
    """
        Like NMS, but it is my own creation!.
    :param predictions:
    :param iou_threshold:
    :return:
    """

    predictions = list(predictions)
    for i, prediction in enumerate(predictions):
        boxes = prediction["boxes"]
        scores = prediction["scores"]
        labels = prediction["labels"]

        # like nms, but more properly.
        keep = []
        for indices, score in enumerate(scores):
            if iou_threshold <= score:
                keep.append(indices)

        # transforms.
        keep = torch.tensor(keep, dtype=torch.int)

        m_boxes = torch.zeros(size=(len(keep), 4), dtype=boxes.dtype)
        m_scores = torch.zeros(size=(len(keep),), dtype=scores.dtype)
        m_labels = torch.zeros(size=(len(keep),), dtype=labels.dtype)

        for j, indices in enumerate(keep):
            box = boxes[indices]
            score = scores[indices]
            label = labels[indices]

            m_boxes[j] = box
            m_scores[j] = score
            m_labels[j] = label

        prediction["boxes"] = m_boxes
        prediction["scores"] = m_scores
        prediction["labels"] = m_labels

        predictions[i] = prediction

    # filter empty.
    # predictions = [prediction for prediction in predictions if len(prediction["boxes"]) > 0]
    predictions = tuple(predictions)
    return predictions


class ScoreManager(BaseClass):
    _scores: List[Score]
    start_epoch: int

    logger = logging.Logger

    def __init__(self, logger: logging.Logger | None = None):
        """
            Score Manager.
        """
        self._scores = []
        self.start_epoch = 0

        if logger is None:
            logger = logging.getLogger(self.name)

        self.logger = logger

    def __getitem__(self, idx: int) -> Score:
        """
            Link Func GetItem to Var _Scores.
        :param idx:
        :return:
        """
        if idx < 0:
            return self._scores[idx]

        return self._scores[max(idx - self.start_epoch, 0)]

    def __setitem__(self, idx: int, score: Score) -> None:
        """
            Link Func GetItem to Var _Scores.
        :param idx:
        :return:
        """
        if idx < 0:
            self._scores[idx] = score
            return None

        self._scores[max(idx - self.start_epoch, 0)] = score
        return None

    def __contains__(self, score: Score) -> bool:
        """
            Check Contains Not Permission Enabled.
        :param score:
        :return:
        """
        void(score)

        # Default Value.
        return False

    def __len__(self) -> int:
        """
            Link Func Length to Func Size.
        :return:
        """
        return self.size()

    def last(self) -> Score:
        """
            Last Value.
        :return:
        """

        if 0 < len(self._scores):
            return self._scores[-1]

        # Default.
        return Score(mean_ap=0.0, iou=0.0, ciou=0.0)

    def update(self, predictions: PredictionTypes, targets: TargetTypes):
        """
            Update And Append Score From Predictions, And Targets.
        :param predictions:
        :param targets:
        :return:
        """

        score = get_score_mean_ap_iou_ciou(predictions, targets)
        self.append(score)

    def to_list(self) -> List[Dict[str, float | int]]:
        """
            Make JSON Syntax.
        :return:
        """
        data = [origin.to_dict() for origin in self._scores]
        return data

    def to_json(self) -> str:
        """
            Make JSON Syntax.
        :return:
        """
        data = [origin.to_dict() for origin in self._scores]
        return json.dumps(data)

    def append(self, score: Score) -> None:
        """
            Append Score object.
        :param score:
        :return:
        """

        if score.isvalid():
            self._scores.append(score)
            return None

        err = Exception("Skipping Loss Value, is NaN or Inf")
        self.logger.warning(err)
        return None

    def history(self) -> Tuple[Score, ...]:
        data: Iterator[Score] = map(lambda origin: origin.copy(), self._scores)
        return tuple(data)

    def avg_score(self) -> Score:
        """
            Average Score Accuracy Values.
        :return:
        """

        if 0 < len(self._scores):
            # Calculate All Values.
            return Score.avg(self._scores)

        return Score(mean_ap=math.nan, iou=math.nan, ciou=math.nan)

    def size(self) -> int:
        return len(self._scores)


def main():
    """
        Main Testing.
    :return:
    """

    score_manager = ScoreManager()
    predictions = [
        {
            "boxes": torch.tensor([[258.0, 41.0, 606.0, 285.0]]),
            "scores": torch.tensor([0.536]),
            "labels": torch.tensor([0]),
        },
    ]

    targets = [
        {
            "boxes": torch.tensor([[214.0, 41.0, 562.0, 285.0]]),
            "labels": torch.tensor([0]),
        },
    ]

    score_manager.update(predictions, targets)
    print(score_manager.avg_score())
    print(score_manager.history())

    return
