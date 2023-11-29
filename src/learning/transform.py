import numpy as np
#import torch

# Parameters for NN input
ALT_MAX = 2000.0
ALT_MIN = 0.0
TGO_MAX = 200.0
TGO_MIN = 0.0
VX_MAX = 100.0
VX_MIN = 0.0
VZ_MAX = 100.0
VZ_MIN = -100.0
MASS_MAX = 1905.0
MASS_MIN = 1505.0
#Z_MAX = torch.log(torch.tensor(MASS_MAX))
#Z_MIN = torch.log(torch.tensor(MASS_MIN))
Z_MAX = np.log(MASS_MAX)
Z_MIN = np.log(MASS_MIN)

# Parameters for NN output
#FOV = torch.tensor(0.2617993877991494)  # 15 degrees in radian
FOV = 0.2617993877991494


def transform_ic(alt, vx, vz, z, tgo):
    """
    Transform initial conditions to normalized coordinates.
    """

    # normalize
    alt_ = (alt - ALT_MIN) / (ALT_MAX - ALT_MIN)
    vx_ = (vx - VX_MIN) / (VX_MAX - VX_MIN)
    vz_ = (vz - VZ_MIN) / (VZ_MAX - VZ_MIN)
    z_ = (z - Z_MIN) / (Z_MAX - Z_MIN)
    tgo_ = (tgo - TGO_MIN) / (TGO_MAX - TGO_MIN)

    return alt_, vx_, vz_, z_, tgo_


def inverse_transform_ic(alt_, vx_, vz_, z_, tgo_):
    """
    Inverse transform normalized coordinates to initial conditions.
    """

    # inverse normalize
    alt = alt_ * (ALT_MAX - ALT_MIN) + ALT_MIN
    vx = vx_ * (VX_MAX - VX_MIN) + VX_MIN
    vz = vz_ * (VZ_MAX - VZ_MIN) + VZ_MIN
    z = z_ * (Z_MAX - Z_MIN) + Z_MIN
    tgo = tgo_ * (TGO_MAX - TGO_MIN) + TGO_MIN

    return alt, vx, vz, z, tgo


def transform_reachsetparam(xmin, xmax, ymax, x_ymax, alt, fov=FOV):
    """transform values to [0, 1]"""
    assert np.all(ymax >= 0), "ymax must be positive"
    # normalize
    fov_radius = (alt * np.tan(fov / 2)) * 1.05  # 5% safety factor
    xmin_ = (xmin + fov_radius) / fov_radius / 2
    xmax_ = (xmax + fov_radius) / fov_radius / 2
    ymax_ = ymax / fov_radius
    x_ymax_ = (x_ymax + fov_radius) / fov_radius / 2

    return xmin_, xmax_, ymax_, x_ymax_


def inverse_transform_reachsetparam(xmin_, xmax_, ymax_, x_ymax_, alt, fov=FOV):
    # inverse normalize
    fov_radius = (alt * np.tan(fov / 2)) * 1.05  # 5% safety factor
    xmin = xmin_ * fov_radius * 2 - fov_radius
    xmax = xmax_ * fov_radius * 2 - fov_radius
    ymax = ymax_ * fov_radius
    x_ymax = x_ymax_ * fov_radius * 2 - fov_radius

    return xmin, xmax, ymax, x_ymax
