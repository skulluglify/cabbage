#!/usr/bin/env python3
import math
from typing import List, Tuple, Literal

import matplotlib.pyplot as plt
import seaborn as sn
import torch
from matplotlib.axes import Axes
from numpy.typing import ArrayLike
from sklearn.metrics import confusion_matrix
from torch import Tensor
from torchvision.utils import make_grid

from sodium.types import PathType
from sodium.utils import imshow


def imshow_grid(images: Tensor | List[Tensor] | Tuple[Tensor, ...],
                compact: bool = False, normalize: bool = False,
                axis: bool = False, path: PathType | None = None):

    # Size of Images.
    n = len(images)

    # Known Size of Rows & Columns.
    x = math.sqrt(n)
    nrows, ncols = math.ceil(x), math.floor(x)  # for grid.

    # Make it Limit of Size.
    grid_size = nrows * ncols

    # Cut off
    if grid_size < n:
        images = images[:grid_size]

    # Make it Computable on CPU.
    # Enable Item Assignment. (Both, Compact | Subplots)
    images = [image.cpu() for image in images]

    plt.axis("on" if axis else "off")

    # Single Image
    if grid_size == 1:

        plt.imshow(images[0].squeeze().permute(1, 2, 0))
        if path is not None:
            if path is not None:
                plt.savefig(path)
                plt.close()
                return

        plt.show()
        return

    if compact:
        grid = make_grid(images, nrows, normalize=normalize)
        grid = grid.permute(1, 2, 0)
        plt.imshow(grid)

        if path is not None:
            if path is not None:
                plt.savefig(path)
                plt.close()
                return

        plt.show()
        return

    # Permute Images. (Only Subplots)

    i: int
    image: Tensor
    for i, image in enumerate(images):
        images[i] = image.squeeze().permute(1, 2, 0)

    # reshape.
    nrows, ncols = min(nrows, ncols), max(nrows, ncols)

    # Create Subplots.
    fig, ax = plt.subplots(nrows, ncols)

    # Bound Images.
    for row in range(nrows):
        for col in range(ncols):
            i = (row * ncols) + col
            axes = ax[row, col]

            if isinstance(axes, Axes):
                axes.axis("on" if axis else "off")
                axes.autoscale_view("tight")

                if i < n:
                    image = images[i]
                    axes.imshow(image, origin="upper")

    if path is not None:
        if path is not None:
            plt.savefig(path)
            plt.close()
            return

    plt.show()
    return


def plot_confusion_matrix_view(y_true: ArrayLike, y_prediction: ArrayLike, labels: ArrayLike | None = None,
                               sample_weight: ArrayLike | None = None,
                               normalize: Literal["true", "pred", "all"] | None = None):
    plt.figure()

    cmatrix = confusion_matrix(y_true, y_prediction, labels=labels, sample_weight=sample_weight, normalize=normalize)

    xticklabels = labels or "auto"
    sn.heatmap(cmatrix, annot=True, fmt="g", xticklabels=xticklabels, yticklabels=xticklabels, cmap="Blues")

    plt.xlabel("Actual", fontsize=13)
    plt.ylabel("Prediction", fontsize=13)
    plt.title("Confusion Matrix", fontsize=17)
