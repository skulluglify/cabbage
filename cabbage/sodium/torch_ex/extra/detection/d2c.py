#!/usr/bin/env python3
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor

from sodium.torch_ex.types import (detection_truth_box, detection_prediction_box, classification_labels, TargetTypes,
                                   PredictionTypes, X_True, Y_Prediction)
from sodium.torch_ex.utils import overlap, shuffle


def d2c_box(truth_box: detection_truth_box,
            prediction_box: detection_prediction_box,
            iou_threshold: float = 0.5) -> classification_labels:
    """
        Convert Detection Box To Classification Labels, Not High Precision,
    But Suitable for Generate Visual Confusion Matrix From Detection Module.
    :param truth_box:
    :param prediction_box:
    :param iou_threshold:
    :return:
    """

    # stack labels undetected
    stack_labels_undetected: List[Tensor]
    stack_labels_undetected = []

    # truth_box_labels > prediction_box_labels
    for label in truth_box.labels:
        if label not in prediction_box.labels:
            stack_labels_undetected.append(label)
        continue

    # prediction_box_labels > truth_box_labels
    for label in prediction_box.labels:
        if label not in truth_box.labels:
            stack_labels_undetected.append(label)
        continue

    stack_labels_undetected_dropout: List[Tensor]
    stack_labels_undetected_dropout = []

    x_true: List[Tensor]
    x_true = []

    y_prediction: List[Tensor]
    y_prediction = []

    # casting to int32.
    truth_box_labels = truth_box.labels.to(torch.int32)
    prediction_box_labels = prediction_box.labels.to(torch.int32)

    # for each item from truth_box.labels.
    for t_index, t_label in enumerate(truth_box_labels):
        label_detected = t_label not in stack_labels_undetected

        # if t_label in stack_labels_undetected:
        #     continue

        t_box = truth_box.boxes[t_index]

        c_iou = torch.tensor(0.0)
        c_label = torch.tensor(0)

        # TODO: make it twice.
        for quick_mode in (1, 0):  # quick search.

            found = 0
            quick_search = label_detected and quick_mode
            prediction_box_labels = shuffle(prediction_box_labels)
            for p_index, p_label in enumerate(prediction_box_labels):
                if t_label != p_label and quick_search:
                    continue

                p_score = prediction_box.scores[p_index]
                p_box = prediction_box.boxes[p_index]

                # get iou score. (quick_search mode)
                iou = p_score if quick_search else overlap(t_box, p_box)

                # selected, is greater than or equal to iou_threshold.
                if iou < iou_threshold:
                    continue

                    # maybe finding again with no detected mode

                if c_iou < iou:
                    c_label = p_label
                    c_iou = iou

                if quick_search:  # stop iteration. (quick_search mode)
                    found = 1
                    break

                found = 1

            if found:  # triggers another deep_sampling if failed found.
                break

        # found same as label score.
        if iou_threshold <= c_iou:
            x_true.append(t_label)
            y_prediction.append(c_label)  # closed to high score.
            continue

        # maybe same label have different position on image.
        # score under iou_threshold, undetected.
        # if t_label not in stack_labels_undetected:
        #     stack_labels_undetected.append(t_label)

        # dropout.
        # if t_label not in stack_labels_undetected_dropout:
        #     stack_labels_undetected_dropout.append(t_label)
        stack_labels_undetected_dropout.append(t_label)  # iteration by `truth_box`.

    # dropout labels undetected into zero class.
    # replaces stack_labels_undetected2 to '__background__' labels.
    x_true += stack_labels_undetected_dropout
    y_prediction += [torch.tensor(0)] * len(stack_labels_undetected_dropout)

    # may problem on calculate accuracy score.
    # # stat cmp by zero confirmation.
    # n = len(truth_box_labels)
    # x_true += [torch.tensor(0)] * n
    # y_prediction += [torch.tensor(0)] * n

    # n = 1  # per batch, trigger zero background.
    # x_true += [torch.tensor(0)] * n
    # y_prediction += [torch.tensor(0)] * n

    # a, b = len(x_true), len(y_prediction)
    # k = a or b if a == b else min(a, b)  # impossible way, if 'a' is not equal 'b' value.
    # n = max(len(truth_box_labels), len(prediction_box_labels))
    # x_true, y_prediction = x_true[:min(n, k)], y_prediction[:min(n, k)]

    # return as classification labels.
    return classification_labels(x_true=torch.stack(x_true),
                                 y_prediction=torch.stack(y_prediction))


def d2c_box_task(params: Tuple[detection_truth_box, detection_prediction_box]) -> classification_labels:
    """
        Parameters As Tuple, Suitable for Multiple Tasks with Concurrent Executor.
    :param params:
    :return:
    """

    return d2c_box(*params)


def d2c_workloads_maps(targets: TargetTypes,
                       predictions: PredictionTypes,
                       num_workers: int = 4) -> Tuple[X_True, Y_Prediction]:
    """
        Legacy Func Of D2C Algorithm. (Supported Multiple Tasks)
    :param targets:
    :param predictions:
    :param num_workers:
    :return:
    """

    x_true: List[int]
    x_true = []

    y_prediction: List[int]
    y_prediction = []

    params = list(zip(map(detection_truth_box.from_dict, targets),
                      map(detection_prediction_box.from_dict, predictions)))

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for cf_labels in executor.map(d2c_box_task, params):

            cf_x_true: np.ndarray
            cf_x_true = cf_labels.x_true.cpu().numpy()

            cf_y_prediction: np.ndarray
            cf_y_prediction = cf_labels.y_prediction.cpu().numpy()

            for x in cf_x_true.tolist():
                x_true.append(x)

            for y in cf_y_prediction.tolist():
                y_prediction.append(y)

    return x_true, y_prediction


def main():
    truth_box = detection_truth_box(labels=torch.tensor([1, 1, 2, 3, 4]),
                                    boxes=torch.tensor([
                                        [0.0, 0.0, 0.5, 0.5],
                                        [0.0, 0.0, 0.5, 0.5],
                                        [0.0, 0.0, 0.5, 0.5],
                                        [0.0, 0.0, 0.5, 0.5],
                                        [0.0, 0.0, 0.5, 0.5],
                                    ]))

    prediction_box = detection_prediction_box(labels=torch.tensor([5, 3]),
                                              boxes=torch.tensor([
                                                  [0.0, 0.0, 0.5, 0.5],
                                                  [0.0, 0.0, 0.5, 0.5],
                                              ]),
                                              scores=torch.tensor([0.7, 0.5]))

    labels = d2c_box(truth_box, prediction_box)
    print(labels)  # classification_labels(x_true=tensor([1, 1, 2, 3, 4]), y_prediction=tensor([5, 5, 5, 3, 3]))


if str(__name__).upper() in ("__MAIN__",):
    main()
