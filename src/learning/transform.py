import numpy as np
import torch

# Parameters for NN input
ALT_MAX = 1500.0
ALT_MIN = 100.0
TGO_MAX = 80.0
TGO_MIN = 0.0
VX_MAX = 50.0
VX_MIN = 0.0
VZ_MAX = 20.0
VZ_MIN = -90.0
MASS_MAX = 1825.0
MASS_MIN = 1505.0
Z_MAX = torch.log(torch.tensor(MASS_MAX))
Z_MIN = torch.log(torch.tensor(MASS_MIN))

# Parameters for NN output
FOV = torch.tensor(15 * 3.1415926535 / 180)
FOV_R_MAX = ALT_MAX * torch.tan(FOV / 2)
SMA_MAX = ALT_MAX * 10


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


def transform_reachsetparam(xmin, xmax, alpha, yp, a1, a2, alt, fov=FOV):
    """transform values to [0, 1]"""
    # normalize
    fov_radius = (alt * np.tan(fov / 2)) * 1.1  # 1.1 is a safety factor
    length_max = fov_radius * 5
    xmin_ = (xmin + fov_radius) / fov_radius / 2
    xmax_ = (xmax - xmin) / fov_radius / 2
    alpha_ = alpha
    yp_ = yp / fov_radius
    a1_ = (a1 - alpha * (xmax - xmin) / 2) / length_max
    a2_ = (a2 - (1 - alpha) * (xmax - xmin) / 2) / length_max

    return xmin_, xmax_, alpha_, yp_, a1_, a2_


def inverse_transform_reachsetparam(xmin_, xmax_, alpha_, yp_, a1_, a2_, alt, fov=FOV):
    # inverse normalize
    fov_radius = (alt * np.tan(fov / 2)) * 1.1  # 1.1 is a safety factor
    length_max = fov_radius * 5
    xmin = xmin_ * fov_radius * 2 - fov_radius
    xmax = xmax_ * fov_radius * 2 + xmin
    alpha = alpha_
    yp = yp_ * fov_radius
    a1 = a1_ * length_max + alpha * (xmax - xmin) / 2
    a2 = a2_ * length_max + (1 - alpha) * (xmax - xmin) / 2

    return xmin, xmax, alpha, yp, a1, a2
