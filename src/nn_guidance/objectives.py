"""Objective functions and its utility funtions for reachability-aware guidance."""
import torch
import torch.nn as nn
from torch import Tensor

from .nn_reachset import get_nn_reachset_param, rot2d_torch
from ..learning import FOV


def u_2mean_safety(u_: Tensor, tgo_next: Tensor, x0: Tensor, dt: float, rocket: nn.Module, sfmap: Tensor, model: nn.Module, fov: Tensor, border_sharpness: float = 1.0):
    """Return mean safety and reachable mask for the control sequence.

    Args:
        u_ (torch.Tensor): normalized control sequence, shape (N, 3)
        tgo_next (torch.Tensor): time to go at the next waypoint, shape (N, )
        x0 (torch.Tensor): initial state, shape (7, )
        dt (float): time step
        rocket (nn.Module): rocket model
        sfmap (torch.Tensor): safety map, [[x, y, safety], ...]], shape (n, 3)

    Returns:
        mean_safety (torch.Tensor): mean safety
        reachable_mask (torch.Tensor): reachable mask, shape (N, )
    """
    # propagate dynamics
    x = u_2x(u_, x0, dt, rocket)
    
    # compute mean safety on the reachable safety map
    mean_safety, reachable_mask = ic2mean_safety(x, tgo_next, model, sfmap, border_sharpness, fov)
    
    return mean_safety, reachable_mask


def u_2x(u_: Tensor, x0: Tensor, dt: float, rocket: nn.Module):
    """Propagate dynamics with the normalized control sequence.

    Args:
        u_ (torch.Tensor): normalized control sequence, shape (N, 3)
        x0 (torch.Tensor): initial state, shape (7, )
        dt (float): time step
        rocket (nn.Module): rocket model
    
    Returns:
        x (torch.Tensor): propagated states, shape (N, 7)
    """
    #x = x0
    # copy x0 tensor to avoid in-place operation
    x = x0.clone()
    for i in range(u_.shape[0]):
        u = inverse_transform_u(u_[i], torch.tensor(rocket.rho1), torch.tensor(rocket.rho2), torch.tensor(rocket.pa))
        x = dynamics(x, u, dt, torch.tensor(rocket.g), torch.tensor(rocket.alpha))
    return x


def dynamics(x: torch.Tensor, u: torch.Tensor, dt: float, g: torch.Tensor, alpha: float):
    """Compute next state under rocket dynamics. 

    Args:
        x (torch.Tensor): state, shape (7, ); [x, y, z, vx, vy, vz, log(mass)]
        u (torch.Tensor): control, shape (3, )
        dt (float): time step
        g (torch.Tensor): gravity, shape (3, )
        alpha (float): mass flow rate

    Returns:
        x_ (torch.Tensor): next state, shape (7, )
    """
    mass = torch.exp(x[6])
    dt22 = dt **2 / 2.0
    x_ = torch.zeros_like(x)
    x_[:3] = x[:3] + dt * x[3:6] + dt22 * (u/mass + g)
    x_[3:6] = x[3:6] + dt * (u/mass + g)
    x_[6] = x[6] - dt * alpha * torch.norm(u) / mass
    return x_


def inverse_transform_u(u_: Tensor, rho1: Tensor, rho2: Tensor, pa: Tensor):
    """Inverse transform: from normalized thrust vector u_ to thrust vector u.

    Args:
        u_ (torch.Tensor): normalized control parameter, [0, 1], shape (3, )
            - u_[0] is the normalized norm of u
            - u_[1] is the normalized x element
            - u_[2] is the normalized y element
        rho1 (Tensor): minimum thrust
        rho2 (Tensor): maximum thrust
        pa (Tensor): maximum gimbal angle
    
    Returns:
        torch.Tensor: control, shape (3, )
    """
    assert u_.shape == (3,), "Invalid shape of u_: {}".format(u_.shape)

    eps = 1e-8
    u = torch.zeros(3)
    
    u_norm = u_[0] * (rho2 - rho1) + rho1
    
    R = u_norm * torch.sin(pa)  # radius of the the circle that intersects |u|==u_norm sphere and angle(u, uz)==pa
    u_xy_ = (u_[:2] * 2 - 1) * R  # u_xy_ lies on the [-R, R] square 
    u_xy_norm = torch.norm(u_xy_)  
    u_xy_norm_clamped = torch.clamp(u_xy_norm, torch.tensor(0.), R)  # u_xy_norm_clamped is bounded by [0, R]
    u[:2] = u_xy_ * u_xy_norm_clamped / (u_xy_norm+eps)  # u_xy lies inside the circle with radius R
    u[2] = torch.sqrt(u_norm**2 - u_xy_norm_clamped**2)  # u_z is determined by u_norm and u_xy

    return u


def transform_u(u: Tensor, rho1: Tensor, rho2: Tensor, pa: Tensor):
    """Transform thrust vector u to normalized thrust vector u_.
    
    Args:
        u (torch.Tensor): control, shape (3, )
        rho1 (Tensor): minimum thrust
        rho2 (Tensor): maximum thrust
        pa (Tensor): maximum gimbal angle
    
    """
    u_ = torch.zeros(3)
    u_norm = torch.norm(u)
    u_xy_norm = torch.norm(u[:2])
    pa_num = torch.atan2(u_xy_norm, u[2])
    assert u_norm >= rho1 and u_norm <= rho2, "u_norm {} is out of range: [{}, {}]".format(u_norm, rho1, rho2)
    assert pa_num <= pa, "pa_num is out of range"

    u_[0] = (u_norm - rho1) / (rho2 - rho1)
    u_[:2] = (u[:2] / u_xy_norm + 1) / 2

    return u_


def ic2mean_safety(x0: torch.Tensor, tgo: torch.Tensor, model: nn.Module, sfmap: torch.Tensor, border_sharpness=0.1, fov=FOV):
    """Compute mean safety of the initial condition (x0, tgo) based on the soft landing reachable set.

    Args:
        x0 (torch.Tensor): initial state
        tgo (torch.Tensor): time to go
        model (nn.Module): NN model
        sfmap (torch.Tensor): safety map, [[x, y, safety], ...]], shape (n, 3)
        border_sharpness (float, optional): sharpness of reachable set border. Defaults to 0.1.
        fov (float, optional): field of view. Defaults to FOV.

    Returns:
        mean_safety (torch.Tensor): mean safety, shape (1, )
        sfmap_reachable_mask (torch.Tensor): reachable safety map mask, shape (n, )
    """

    # compute reachable safety map
    sfmap_reachable_mask = reachable_sfmap_soft(x0, tgo, model, sfmap, alpha=border_sharpness, fov=fov)

    # compute mean safety
    mean_safety = torch.sum(sfmap[:, 2] * sfmap_reachable_mask) / torch.sum(sfmap_reachable_mask + 1e-8)  # added epsilon for numerical stability

    return mean_safety, sfmap_reachable_mask



def reachable_sfmap_soft(x0: torch.Tensor, tgo: torch.Tensor, model: nn.Module, sfmap: torch.Tensor, alpha=1.0, fov=FOV):
    """Compute safety map indices of reachable set with a soft boundary; differentiable.

    Args:
        x0 (torch.Tensor): initial state
        tgo (torch.Tensor): time to go
        model (nn.Module): NN model
        sfmap (torch.Tensor): safety map, [[x, y, safety], ...]], shape (n, 3)
        alpha (float, optional): soft boundary parameter. Defaults to 1.0.

    Returns:
        soft_mask (torch.Tensor): soft boundary mask, shape (n, )
    """
    eps = 1e-8  # small constant to prevent numerical instability

    # compute reachset parameters
    xmin, xmax, ymax, x_ymax, rotation_angle, center = get_nn_reachset_param(x0, tgo, model, fov)
    a1 = x_ymax - xmin
    a2 = xmax - x_ymax
    b = ymax
    
    # shift and rotate safety map to canonical coordinate of reachset
    _xy = rot2d_torch(-rotation_angle) @ (sfmap[:, :2] - center).T
    _xy = _xy.T
    _x = _xy[:, 0]
    _y = _xy[:, 1]

    # compute soft boundary
    mask0 = torch.sigmoid(alpha * (x_ymax - _x))  # 1 for x < x_ymax, 0 for x >= x_ymax
    mask1 = torch.sigmoid(alpha * (1 - (_x - x_ymax)**2 / (a1**2 + eps) - _y**2 / (b**2 + eps)))
    mask2 = torch.sigmoid(alpha * (1 - (_x - x_ymax)**2 / (a2**2 + eps) - _y**2 / (b**2 + eps)))
    soft_mask = torch.max(torch.min(mask1, mask0), torch.min(mask2, 1 - mask0))

    # filter out points outside of the field of view
    fov_radius = x0[2] * torch.tan(fov / 2)
    fov_mask = torch.sigmoid(alpha * (fov_radius**2 - (_x)**2 - (_y)**2))  # center is at (0, 0) because we already shifted the safety map
    soft_mask_fov = torch.min(soft_mask, fov_mask)

    return soft_mask_fov
