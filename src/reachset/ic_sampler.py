"""Utility functions for sampling initial conditions for reachability analysis."""

import numpy as np
from numba import njit


@njit
def convert_to_x0(sample: np.ndarray):
    """Convert random sample from unit cube to x0

    Args:
        sample (np.ndarray): random sample from unit cube, shape (n, 5)

    Returns:
        x0_arr (np.ndarray): initial state, shape (n, 7)
        tgo (np.ndarray): time to go, shape (n, )
    """
    r_alt, r_tgo, r_vx, r_vz, r_mass = sample.T

    alt_min, alt_max = 100.0, 1500.0
    alt = alt_min + (alt_max - alt_min) * r_alt

    tgo_min, tgo_max = tgo_bound(alt)
    tgo = tgo_min + (tgo_max - tgo_min) * r_tgo
    tgo = tgo // 1  # round to integer

    vx_min, vx_max = vx_bound(alt)
    vx = vx_min + (vx_max - vx_min) * r_vx

    vz_min, vz_max = vz_bound(alt)
    vz = vz_min + (vz_max - vz_min) * r_vz

    mass_min, mass_max = mass_bound(alt)
    mass = mass_min + (mass_max - mass_min) * r_mass
    z = np.log(mass)

    zeros = np.zeros_like(alt)

    x0_arr = np.vstack((zeros, zeros, alt, vx, zeros, vz, z)).T

    return x0_arr, tgo


@njit
def vz_bound(alt):
    n = len(alt)
    ymax = 10 - 10 / 1500 * alt
    ymin = -20 + (-50 - (-20)) / 250 * alt
    mask1 = alt > 250
    ymin[mask1] = -50 + (-65 - (-50)) / (500 - 250) * (alt[mask1] - 250)
    mask2 = alt > 500
    ymin[mask2] = -65 + (-35 - (-65)) / (1500 - 500) * (alt[mask2] - 500)
    return ymin, ymax


@njit
def vx_bound(alt):
    n = len(alt)
    ymin = np.zeros(n)
    ymax = 15 + (90 - 15) / 450 * alt
    mask = alt > 450
    ymax[mask] = 90
    mask = alt > 900
    ymax[mask] = 90 + (60 - 90) / (1500 - 900) * (alt[mask] - 900)
    return ymin, ymax


@njit
def mass_bound(alt):
    ymax = 1750 + (1825 - 1750) / 1500 * alt
    ymin = 1600 + (1750 - 1600) * alt / 1300
    mask = alt <= 300
    ymax[mask] = 1650 + (1750 - 1650) / 300 * alt[mask]
    return ymin, ymax


@njit
def tgo_bound(alt):
    ymax = 30 + (60 - 30) / 1500 * alt
    ymin = 50 / 1500 * alt
    return ymin, ymax
