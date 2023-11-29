"""Utility functions for soft landing reachable set using NN model."""
import torch
from torch import Tensor
import torch.nn as nn
from matplotlib import pyplot as plt

from ..learning import transform_ic, inverse_transform_reachsetparam
from ..reachset import reach_ellipses_torch


def get_nn_reachset_param(x0: Tensor, tgo: Tensor, model: nn.Module, fov: float):
    """Compute parameters of soft landing reachable set using NN model"""

    assert x0.shape == (7,), f"Invalid shape of x0: {x0.shape}"

    # unpack state
    r, v, z = x0[:3], x0[3:6], x0[6]
    alt = r[2]
    v_horiz = torch.norm(v[:2])
    v_vert = v[2]
    v_horiz_angle = torch.atan2(v[1], v[0])

    # prepare input
    alt_, v_horiz_, v_vert_, z_, tgo_ = transform_ic(alt, v_horiz, v_vert, z, tgo)
    nn_input = torch.tensor([alt_, v_horiz_, v_vert_, z_, tgo_])
    nn_input = nn_input.unsqueeze(0)

    # check model dtype and convert input accordingly
    model_dtype = next(model.parameters()).dtype
    if model_dtype == torch.float64:
        nn_input = nn_input.double()
    elif model_dtype == torch.float32:
        nn_input = nn_input.float()
    else:
        raise TypeError(f"Model dtype {model_dtype} not supported.")

    # compute output (= reachset parameters)
    nn_output = model(nn_input)
    nn_output = nn_output.squeeze(0)
    xmin_, xmax_, ymax_, x_ymax_ = nn_output
    xmin, xmax, ymax, x_ymax = inverse_transform_reachsetparam(xmin_, xmax_, ymax_, x_ymax_, alt, fov=fov)

    return xmin, xmax, ymax, x_ymax, v_horiz_angle, r[:2]


def rot2d_torch(theta: Tensor):
    """2D rotation matrix.

    Args:
        theta (Tensor): angle of rotation (rad)

    Returns:
        Tensor: 2D rotation matrix
    """
    return torch.tensor(
        [[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]]
    )


def visualize_nn_reachset(reach_mask: Tensor, sfmap: Tensor, nskip: int = 20):
    reach_mask_ = reach_mask.clone()
    reach_mask_ = reach_mask_.detach().cpu().numpy()
    sfmap_ = sfmap.clone()
    sfmap_ = sfmap_.detach().cpu().numpy()

    rm = reach_mask_[::nskip]
    sf = sfmap_[::nskip]

    fig, ax = plt.subplots()

    ax.scatter(sf[:, 0], sf[:, 1], c=rm, s=0.5, cmap="jet", alpha=0.5)
    ax.scatter(sf[:, 0], sf[:, 1], c=sf[:, 2], s=0.5, cmap="gray", alpha=0.3)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.grid(True)
    plt.show()
