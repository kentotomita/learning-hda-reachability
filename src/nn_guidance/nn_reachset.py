"""Utility functions for soft landing reachable set using NN model."""
import torch
from torch import Tensor
import torch.nn as nn
from matplotlib import pyplot as plt

from ..learning import transform_ic, inverse_transform_reachsetparam
from ..reachset import reach_ellipses_torch


def get_nn_reachset_param(x0: Tensor, tgo: Tensor, model: nn.Module, full=False):
    """Compute parameters of soft landing reachable set using NN model.

    Args:
        x0 (Tensor): initial state
        tgo (Tensor): time to go
        model (nn.Module): NN model
        full (bool, optional): whether to return full set of parameters. Defaults to False.

    Returns (if full=False):
        xp (Tensor): x-coordinate of intersection of two ellipses
        xc1 (Tensor): x-coordinate of left ellipse center
        xc2 (Tensor): x-coordinate of right ellipse center
        a1 (Tensor): left ellipse semi-major axis
        a2 (Tensor): right ellipse semi-major axis
        b1 (Tensor): left ellipse semi-minor axis
        b2 (Tensor): right ellipse semi-minor axis
        rotation_angle (Tensor): rotation angle of reachset
        center (Tensor): center of reachset

    Returns (if full=True):
        xmin (Tensor): minimum x coordinate of reachset
        xmax (Tensor): maximum x coordinate of reachset
        alpha (Tensor): x-coordinate of intersection of two ellipses, as a rate of (xmax - xmin) from xmin
        xp (Tensor): x-coordinate of intersection of two ellipses
        yp (Tensor): y-coordinate of intersection of two ellipses
        xc1 (Tensor): x-coordinate of left ellipse center
        xc2 (Tensor): x-coordinate of right ellipse center
        a1 (Tensor): left ellipse semi-major axis
        a2 (Tensor): right ellipse semi-major axis
        b1 (Tensor): left ellipse semi-minor axis
        b2 (Tensor): right ellipse semi-minor axis
        rotation_angle (Tensor): rotation angle of reachset
        center (Tensor): center of reachset
    """
    eps = 1e-8  # small constant to prevent numerical instability

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
    feasible, xmin_, xmax_, alpha_, yp_, a1_, a2_ = nn_output
    xmin, xmax, alpha, yp, a1, a2 = inverse_transform_reachsetparam(xmin_, xmax_, alpha_, yp_, a1_, a2_, alt)

    if full:
        xc1 = xmin + a1
        xc2 = xmax - a2
        xp = alpha * (xmax - xmin) + xmin
        b1 = torch.sqrt(torch.clamp(yp**2 * (1 - (xp - xc1)**2/(a1**2 + eps)), min=eps))
        b2 = torch.sqrt(torch.clamp(yp**2 * (1 - (xp - xc2)**2/(a2**2 + eps)), min=eps))
        return xp, xc1, xc2, a1, a2, b1, b2, v_horiz_angle, r[:2]

    else:
        return feasible, xmin, xmax, alpha, yp, a1, a2, v_horiz_angle, r[:2]



def get_nn_reachset(x0: Tensor, tgo: Tensor, model: nn.Module, n:int=100):
    """Compute discrete points of soft landing reachable set border using NN model.

    Args:
        x0 (Tensor): initial state
        tgo (Tensor): time to go
        model (nn.Module): NN model
        n (int, optional): number of points to compute. Defaults to 100.
    
    Returns:
        Tensor: reachset points, shape (n, 2)   
    """

    # compute reachset parameters
    feasible, xmin, xmax, alpha, yp, a1, a2, rotation_angle, center = get_nn_reachset_param(x0, tgo, model, full=False)

    # compute border of reachset
    reach_pts = torch.zeros((2*n - 2, 2))
    xs = torch.linspace(xmin.item(), xmax.item(), n)
    ys, _ = reach_ellipses_torch(X=xs, param=(xmin, xmax, alpha, yp, a1, a2))
    reach_pts[:n, 0] = xs
    reach_pts[:n, 1] = ys
    reach_pts[n:, 0] = xs[1:-1]
    reach_pts[n:, 1] = -ys[1:-1]

    # rotate and translate reachset point
    reach_pts = rot2d_torch(rotation_angle) @ reach_pts.T
    reach_pts += center.reshape(2, 1)
    reach_pts = reach_pts.T

    return feasible, reach_pts


def rot2d_torch(theta: Tensor):
    """2D rotation matrix.

    Args:
        theta (Tensor): angle of rotation (rad)

    Returns:
        Tensor: 2D rotation matrix
    """
    return torch.tensor([[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]])



def visualize_nn_reachset(reach_mask: Tensor, sfmap: Tensor, nskip: int=20):
    reach_mask_ = reach_mask.clone()
    reach_mask_ = reach_mask_.detach().cpu().numpy()
    sfmap_ = sfmap.clone()
    sfmap_ = sfmap_.detach().cpu().numpy()

    rm = reach_mask_[::nskip]
    sf = sfmap_[::nskip]

    fig, ax = plt.subplots()

    ax.scatter(sf[:, 0], sf[:, 1], c=rm, s=0.5, cmap='jet', alpha=0.5)
    ax.scatter(sf[:, 0], sf[:, 1], c=sf[:, 2], s=0.5, cmap='gray', alpha=0.3)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    ax.grid(True)
    plt.show()


