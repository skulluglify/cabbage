#!/usr/bin/env python3
import json
import logging
import math
from typing import Dict, Type, TypeVar, List, Tuple, Iterator

import attrs
import torch
from sklearn.metrics import classification_report
from torch import Tensor

from sodium.torch_ex.types import PredictionTypes, TargetTypes
from sodium.x.runtime.types import void
from sodium.x.runtime.utils.refs import ref_obj_get_val
from sodium.x.runtime.wrapper import BaseClass, BaseConfig

ScoreSelf = TypeVar("ScoreSelf", bound="Score")
ScoreAnother = ScoreSelf
ScoreType = Type[ScoreSelf]


@attrs.define(order=True)
class Score(BaseConfig):
    accuracy: float
    f1_score: float
    precision: float
    recall: float

    @classmethod
    def zeros(cls: ScoreType, num: int = 1) -> Tuple[ScoreSelf, ...]:
        Wrapper = cls
        return tuple(Wrapper(accuracy=0.0, f1_score=0.0, precision=0.0, recall=0.0) for _ in range(num))

    @classmethod
    def ones(cls: ScoreType, num: int = 1) -> Tuple[ScoreSelf, ...]:
        Wrapper = cls
        return tuple(Wrapper(accuracy=1.0, f1_score=1.0, precision=1.0, recall=1.0) for _ in range(num))

    def isvalid(self):
        """
            Check is valid value or NaN/Inf.
        :return:
        """

        if (math.isnan(self.accuracy) or math.isinf(self.accuracy)) and \
                (math.isnan(self.f1_score) or math.isinf(self.f1_score)) and \
                (math.isnan(self.precision) or math.isinf(self.precision)) and \
                (math.isnan(self.recall) or math.isinf(self.recall)):
            return False
        return True

    def merge(self: ScoreSelf, config: ScoreSelf) -> ScoreSelf:
        """
            Merging Two Configuration.
        :param config:
        :return:
        """

        origin = self.copy()
        origin.accuracy = config.accuracy or origin.accuracy
        origin.f1_score = config.f1_score or origin.f1_score
        origin.precision = config.precision or origin.precision
        origin.recall = config.recall or origin.recall
        return origin

    @classmethod
    def from_dict(cls: ScoreType, data: Dict[str, float], **kwargs):
        """
            Create from Dictionary.
        :param data:
        :return:
        """
        Wrapper = cls

        score_accuracy = ref_obj_get_val(data, "accuracy", 0.0)
        score_f1_score = ref_obj_get_val(data, "f1-score", 0.0)
        score_precision = ref_obj_get_val(data, "precision", 0.0)
        score_recall = ref_obj_get_val(data, "recall", 0.0)

        return Wrapper(accuracy=score_accuracy,
                       f1_score=score_f1_score,
                       precision=score_precision,
                       recall=score_recall)

    def to_dict(self: ScoreSelf) -> Dict[str, float | int]:
        """
            Return as Dictionary.
        :return:
        """

        return {
            "accuracy": self.accuracy,  # default using.
            "f1-score": self.f1_score,
            "precision": self.precision,
            "recall": self.recall,
        }

    def copy(self) -> ScoreSelf:
        """
            Copy Existing Object.
        :return:
        """
        Wrapper: ScoreType = self.__class__
        return Wrapper(accuracy=self.accuracy, f1_score=self.f1_score, precision=self.precision, recall=self.recall)

    @classmethod
    def avg(cls, scores: List[ScoreSelf] | Tuple[ScoreSelf, ...]) -> ScoreSelf:
        Wrapper = cls

        n = len(scores)
        avg_score = Wrapper(accuracy=0.0, f1_score=0.0, precision=0.0, recall=0.0)

        for score in scores:
            avg_score.accuracy += score.accuracy
            avg_score.f1_score += score.f1_score
            avg_score.precision += score.precision
            avg_score.recall += score.recall

        avg_score.accuracy /= n
        avg_score.f1_score /= n
        avg_score.precision /= n
        avg_score.recall /= n

        return avg_score


def get_score_f1_score_precision_recall(predictions: PredictionTypes, targets: TargetTypes) -> Score:
    """
        Get Score F1 Score, Precision, And Recall.
    :param predictions:
    :param targets:
    :return:
    """

    # dummy = {
    #     "accuracy": 0.8827951388888889,
    #     "macro avg": {
    #         "precision": 0.8854495341924469,
    #         "recall": 0.8818668049565374,
    #         "f1-score": 0.882371269693139,
    #         "support": 230400
    #     },
    #     "weighted avg": {
    #         "precision": 0.8849379010183035,
    #         "recall": 0.8827951388888889,
    #         "f1-score": 0.8825891996345749,
    #         "support": 230400
    #     }
    # }

    # tape block, spaghetti code

    if isinstance(predictions, Tensor):
        predictions = predictions.cpu()

    if isinstance(targets, Tensor):
        targets = targets.cpu()

    # Tensor in Listed.
    if isinstance(predictions, List | Tuple):
        temp = []
        for prediction in predictions:
            if isinstance(prediction, Tensor):
                temp.append(prediction.cpu())

            if isinstance(prediction, List | Tuple):
                temp.append(prediction[0].cpu())  # idx, score, ...
        predictions = temp

    if isinstance(targets, List | Tuple):
        temp = []
        for target in targets:
            if isinstance(target, Tensor):
                temp.append(target.cpu())

            if isinstance(target, List | Tuple):
                temp.append(target[0].cpu())  # idx, another, ...
        targets = temp

    x_true = torch.stack(predictions)
    y_pred = torch.stack(targets)

    report = classification_report(x_true, y_pred, output_dict=True, zero_division=0.0)  # cpu only.

    accuracy = report.get("accuracy", 0.0)
    weighted_avg = report.get("weighted avg", {
        "f1-score": 0.0,
        "precision": 0.0,
        "recall": 0.0,
    })

    val_f1_score = weighted_avg.get("f1-score", 0.0)
    val_precision = weighted_avg.get("precision", 0.0)
    val_recall = weighted_avg.get("recall", 0.0)

    val_f1_score = val_f1_score if not math.isnan(val_f1_score) else 0.0
    val_precision = val_precision if not math.isnan(val_precision) else 0.0
    val_recall = val_recall if not math.isnan(val_recall) else 0.0

    return Score.from_dict({
        "accuracy": accuracy,
        "f1-score": val_f1_score,
        "precision": val_precision,
        "recall": val_recall,
    })


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
        return Score(accuracy=0.0, f1_score=0.0, precision=0.0, recall=0.0)

    def update(self, predictions: PredictionTypes, targets: TargetTypes):
        """
            Update And Append Score From Predictions, And Targets.
        :param predictions:
        :param targets:
        :return:
        """

        score = get_score_f1_score_precision_recall(predictions, targets)
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

        return Score(accuracy=math.nan, f1_score=math.nan, precision=math.nan, recall=math.nan)

    def size(self) -> int:
        return len(self._scores)


def main():
    """
        Main Testing.
    :return:
    """

    score_manager = ScoreManager()
    predictions = [
        torch.tensor(1),
        torch.tensor(2),
        torch.tensor(1),
        torch.tensor(0),
    ]

    targets = [
        torch.tensor(2),
        torch.tensor(2),
        torch.tensor(1),
        torch.tensor(0),
    ]

    score_manager.update(predictions, targets)
    print(score_manager.avg_score())
    print(score_manager.history())

    return
