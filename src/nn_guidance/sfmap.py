"""Utility functions for initializing and updating safety map"""

import numpy as np
import torch
from torch import Tensor
from typing import Tuple
import matplotlib.pyplot as plt


def load_sfmap(
    path: str,
    x_range: Tuple,
    y_range: Tuple,
    normalize: bool = True,
    dtype: str = "float32",
):
    """Load safety map from path"""
    sfmap = np.load(path)
    sfmap[np.isnan(sfmap)] = 0.0
    if normalize:
        sfmap = (sfmap - np.min(sfmap)) / (np.max(sfmap) - np.min(sfmap))

    nr, nc = sfmap.shape
    xmin, xmax = x_range
    ymin, ymax = y_range
    x = np.linspace(xmin, xmax, nc)
    y = np.linspace(ymin, ymax, nr)
    X, Y = np.meshgrid(x, y)

    sfmap_tensor = np.zeros((nr, nc, 3))
    sfmap_tensor[:, :, 0] = X
    sfmap_tensor[:, :, 1] = Y
    sfmap_tensor[:, :, 2] = sfmap

    sfmap_tensor = sfmap_tensor.reshape(-1, 3)

    if dtype == "float32":
        sfmap_tensor = torch.from_numpy(sfmap_tensor).float()
    elif dtype == "float64":
        sfmap_tensor = torch.from_numpy(sfmap_tensor).double()
    else:
        raise ValueError("dtype must be either float32 or float64")

    return sfmap_tensor, (nr, nc)


def make_simple_sfmap(x_range, y_range, n_points, dtype="float32"):
    """Make safety map from scratch"""

    xmin, xmax = x_range
    ymin, ymax = y_range
    x = np.linspace(xmin, xmax, n_points)
    y = np.linspace(ymin, ymax, n_points)
    X, Y = np.meshgrid(x, y)

    sfmap = np.zeros((n_points, n_points, 3))
    sfmap[:, :, 0] = X
    sfmap[:, :, 1] = Y
    sfmap[:, :, 2] = X + Y 
    sfmap[:, :, 2] = (sfmap[:, :, 2] - np.min(sfmap[:, :, 2])) / (np.max(sfmap[:, :, 2]) - np.min(sfmap[:, :, 2]))
    sfmap[:, :, 2][X > 500] = 0.0
    sfmap[:, :, 2][Y > 500] = 0.0
    sfmap[:, :, 2][X < -500] = 0.0
    sfmap[:, :, 2][Y < -500] = 0.0

    sfmap = sfmap.reshape(-1, 3)
    if dtype == "float32":
        sfmap = torch.from_numpy(sfmap).float()
    elif dtype == "float64":
        sfmap = torch.from_numpy(sfmap).double()
    else:
        raise ValueError("dtype must be either float32 or float64")

    return sfmap, (n_points, n_points)


def visualize_sfmap(sfmap: Tensor, nskip: int = 1):
    sfmap_ = sfmap.clone()
    sfmap_ = sfmap_.detach().cpu().numpy()

    sf = sfmap_[::nskip]

    fig, ax = plt.subplots()

    ax.scatter(sf[:, 0], sf[:, 1], c=sf[:, 2], s=0.5, cmap="jet", alpha=0.8)

    sm = plt.cm.ScalarMappable(cmap="jet")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.grid(True)
    plt.colorbar(sm)
    plt.show()
