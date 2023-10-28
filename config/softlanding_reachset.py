# Parameters for soft landing reachable set
import numpy as np
import sys

sys.path.append("../")
import src.lcvx as lc

FOV = 15 * np.pi / 180


def slr_config():
    """Return soft landing reachable set config"""

    rocket = lc.Rocket(
        g=np.array([0, 0, -3.7114]),  # Gravitational acceleration (m/s^2)
        mdry=1505.0,  # Dry mass (kg)
        mwet=1825.0,  # 1905. # Wet mass (kg)
        Isp=225.0,
        rho1=4972.0,  # Minimum thrust (N)
        rho2=13260.0,  # Maximum thrust (N)
        gsa=np.pi / 2 - FOV / 2,  # Glide slope angle (rad); from horizontal
        pa=30 * np.pi / 180,  # Pointing angle (rad); from vertical
        fov=FOV,
    )
    N = 100
    return rocket, N
