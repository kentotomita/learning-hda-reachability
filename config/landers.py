"""Lander configurations for each planet."""
import numpy as np
import sys
sys.path.append("../")

from src.lcvx import Rocket


def get_lander(planet: str = "mars"):
    """Return Rocket object for the given planet.
    Args:
        planet (str): planet name
    Returns:
        Rocket: Rocket object
    """
    if planet == "mars":
        return _mars_lander()
    else:
        raise ValueError(f"Undefined planet: {planet}")


def _mars_lander():
    return Rocket(
        g_=3.7114,
        mdry=1505.0,
        mwet=1905.0,
        Isp=225.0,
        rho1=4972.0,
        rho2=13260.0,
        gsa=25 * np.pi / 180,
        fov=15 * np.pi / 180,
        pa=30 * np.pi / 180,
        vmax=100.0,  # m/s
    )
