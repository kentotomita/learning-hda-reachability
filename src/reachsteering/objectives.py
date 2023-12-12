"""Objective functions and its utility funtions for reachability-aware guidance."""
import torch
import torch.nn as nn
import numpy as np
from numba import jit

from ..landers import Lander
from ..learning import transform_ic, inverse_transform_reachsetparam


def ic2mean_safety_npy(lander: Lander, x0: np.ndarray, tgo: float, model: nn.Module, sfmap: np.ndarray, border_sharpness=0.1,
                       return_safest_point=False):
    """Compute mean safety of the initial condition (x0, tgo) based on the soft landing reachable set.

    Args:
        lander (Lander): lander model
        x0 (np.ndarray): initial condition; [rx, ry, rz, vx, vy, vz, m]
        tgo (float): time-to-go
        model (nn.Module): neural network model
        sfmap (np.ndarray): safety map
        border_sharpness (float, optional): sharpness of the soft boundary. Defaults to 0.1.
    """
    assert lander.mdry <= x0[6] <= lander.mwet, f"Invalid mass: {x0[6]}"

    # compute reachset parameters
    xmin, xmax, ymax, x_ymax, rotation_angle, center = get_nn_reachset_param(x0, tgo, model, lander.fov)

    fov_radius = x0[2] * np.tan(lander.fov / 2)
    buffer = 0.25 * fov_radius
    buffered_fov_radius = fov_radius + buffer
    xrange = (-buffered_fov_radius + center[0], buffered_fov_radius + center[0])
    yrange = (center[1] - buffered_fov_radius, center[1] + buffered_fov_radius) 

    sfmap_cropped, crop_mask = crop_sfmap(sfmap, xrange, yrange)

    a1 = x_ymax - xmin
    a2 = xmax - x_ymax
    b = ymax

    mean_safety, soft_mask_fov = _calc_mean_safety_npy(a1, a2, b, x_ymax, rotation_angle, center, sfmap_cropped, border_sharpness, fov_radius)

    soft_mask = np.zeros_like(crop_mask).astype(np.float32)
    soft_mask[crop_mask] = soft_mask_fov

    if return_safest_point:
        # get xy coordinate that maximizes sfmap_crop * soft_mask_fov
        mask = soft_mask_fov > 0.5
        idx = np.argmax(sfmap_cropped[:, 2] * mask * soft_mask_fov)
        cx, cy = sfmap_cropped[idx, :2]
        return mean_safety, soft_mask, (cx, cy, sfmap_cropped[idx, 2])

    else:
        return mean_safety, soft_mask


def get_nn_reachset_param(x0: np.ndarray, tgo: float, model: nn.Module, fov: float):
    """Compute parameters of soft landing reachable set using NN model"""

    # unpack state
    r, v, m = x0[:3], x0[3:6], x0[6]
    z = np.log(m)
    alt = r[2]
    v_horiz = np.linalg.norm(v[:2])
    v_horiz_angle = np.arctan2(v[1], v[0])

    # prepare input
    alt_, v_horiz_, v_vert_, z_, tgo_ = transform_ic(alt, v_horiz, v[2], z, tgo)
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
    nn_output = nn_output.detach().numpy()
    xmin_, xmax_, ymax_, x_ymax_ = nn_output
    xmin, xmax, ymax, x_ymax = inverse_transform_reachsetparam(xmin_, xmax_, ymax_, x_ymax_, alt, fov=fov)

    return xmin, xmax, ymax, x_ymax, v_horiz_angle, r[:2]

@jit(nopython=True)
def crop_sfmap(sfmap, xrange, yrange):
    """Crop safety map to the specified range."""
    xmin, xmax = xrange
    ymin, ymax = yrange
    mask = (sfmap[:, 0] > xmin) & (sfmap[:, 0] < xmax) & (sfmap[:, 1] > ymin) & (sfmap[:, 1] < ymax)
    sfmap_cropped = sfmap[mask]
    return sfmap_cropped, mask


@jit(nopython=True)
def _calc_mean_safety_npy(a1, a2, b, x_ymax, rotation_angle, center, sfmap, alpha, fov_radius):
    """Compute mean safety of the initial condition (x0, tgo) based on the soft landing reachable set."""

    eps = 1e-8  # small constant to prevent numerical instability

    _xy = (sfmap[:, :2] - center) @ rot2d(-rotation_angle).T
    _x = _xy[:, 0]
    _y = _xy[:, 1]

    # compute soft boundary
    mask0 = 1.0 * (x_ymax > _x)   # 1 for x < x_ymax, 0 for x >= x_ymax
    left_ellipse = (1 - (_x - x_ymax)**2 / (a1**2 + eps) - _y**2 / (b**2 + eps)) * mask0  # positive inside the ellipse, negative outside, 0 if x >= x_ymax
    right_ellipse = (1 - (_x - x_ymax)**2 / (a2**2 + eps) - _y**2 / (b**2 + eps)) * (1 - mask0)  # positive inside the ellipse, negative outside, 0 if x < x_ymax
    soft_mask = sigmoid(alpha * (left_ellipse + right_ellipse))

    # filter out points outside of the field of view
    soft_mask_fov = soft_mask * ((_x)**2 - (_y)**2 <= fov_radius**2)  # center is at (0, 0) because we already shifted the safety map

    mean_safety = np.sum(sfmap[:, 2] * soft_mask_fov) / np.sum(soft_mask_fov + 1e-8)  # added epsilon for numerical stability

    return mean_safety, soft_mask_fov

@jit(nopython=True)
def rot2d(theta: np.ndarray):
    """2D rotation matrix."""
    return np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )

@jit(nopython=True)
def sigmoid(x: np.ndarray):
    return 1 / (1 + np.exp(-x))