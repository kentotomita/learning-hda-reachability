"""Lander configurations for each planet."""
import numpy as np
import yaml
import sys
sys.path.append("../")

from src.lcvx import Rocket


def get_lander(planet: str = "Mars"):
    """Return Rocket object for the given planet.
    Args:
        planet (str): planet name
    Returns:
        Rocket: Rocket object
    """
    with open("config/landers.yaml", "r") as f:
        lander_data = yaml.safe_load(f)

    if planet in lander_data:
        data = lander_data[planet]
        return Rocket(
            g_=data['g'],
            mdry=data['mdry'],
            mwet=data['mwet'],
            Isp=data['isp'],
            rho1=data['rho1'],
            rho2=data['rho2'],
            gsa=data['gsa'],
            fov=data['fov'],
            pa=data['pa'],
            vmax=data['vmax']
        )
    else:
        raise ValueError(f"Undefined planet: {planet}")
