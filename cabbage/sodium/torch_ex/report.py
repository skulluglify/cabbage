#!/usr/bin/env python3
import math
import numpy as np
from typing import Tuple

from matplotlib import pyplot as plt
from numpy.typing import ArrayLike

from sodium.x.runtime.types import void


def find_cross_pos(train_acc: ArrayLike, valid_acc: ArrayLike,
                   train_loss: ArrayLike, valid_loss: ArrayLike) -> Tuple[float | int, float | int]:
    """
        Find Cross for Stop Early Pos.
    :param train_acc:
    :param valid_acc:
    :param train_loss:
    :param valid_loss:
    :return:
    """

    acc = np.subtract(train_acc, valid_acc)  # maybe valid can greater than training accuracy.
    loss = np.subtract(valid_loss, train_loss)  # maybe valid can less than training loss.
    scores = np.abs(np.divide(1 + acc - loss, 2))  # distance averages.

    # impossible conduct with many of datasets
    # train_loss <= valid_loss
    # valid_acc <= train_acc

    current_idx = 0
    max_value = -math.inf
    for index, value in enumerate(scores):
        if train_loss[index] <= train_acc[index] and \
                valid_loss[index] <= train_acc[index] and \
                train_loss[index] <= valid_acc[index] and \
                valid_loss[index] <= valid_acc[index] and \
                max_value < value:
            current_idx = index
            max_value = value

    void(max_value)  # unused value.
    cross_x = current_idx
    cross_y = 0.5  # middle of view.

    return cross_x, cross_y


def plot_fit_model_view(x: ArrayLike, y: ArrayLike, x_label: str, y_label: str):
    """
        Plot Fit Model View.
    :param x:
    :param y:
    :param x_label:
    :param y_label:
    :return:
    """

    plt.title("Fit Model View")
    nrows = len(x)

    plt.plot(x, label=x_label, marker="o", linestyle="solid", markersize=4, linewidth=1)
    plt.plot(y, label=y_label, marker="o", linestyle="solid", markersize=4, linewidth=1)
    plt.xlabel("epoch")
    plt.ylabel("values")
    plt.xticks(np.arange(nrows), np.arange(1, nrows + 1))
    plt.legend(loc="upper right")
    plt.grid(visible=True)


def plot_fit_model_full_view(train_acc: ArrayLike, train_loss: ArrayLike,
                             val_acc: ArrayLike, val_loss: ArrayLike,
                             checkpoints: ArrayLike):
    """
        Plot Fit Model Full View.
    :param train_acc:
    :param train_loss:
    :param val_acc:
    :param val_loss:
    :param checkpoints:
    :return:
    """

    plot_fit_model_view(train_acc, val_acc, "train acc", "valid acc")
    plot_fit_model_view(train_loss, val_loss, "train loss", "valid loss")

    cross_x, cross_y = find_cross_pos(train_acc=train_acc, valid_acc=val_acc,
                                      train_loss=train_loss, valid_loss=val_loss)

    checkpoint_x = checkpoints - 1  # to indexes.
    checkpoint_y = [0.5] * len(checkpoints)  # middle of views.

    plt.scatter(checkpoint_x, checkpoint_y, s=16, c="orange", label="checkpoint")
    plt.scatter(cross_x, cross_y, s=16, c="red", label="cross")
    plt.legend(loc="upper right")
    plt.grid(visible=True)
