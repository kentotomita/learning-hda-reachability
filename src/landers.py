"""Lander object stores parameters related to dynamics and constraints"""
import numpy as np
import yaml
from dataclasses import dataclass
from scipy.constants import g as ge  # Standard gravity (m/s^2)


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
        return Lander(
            g_=data['g'],
            mdry=data['mdry'],
            mwet=data['mwet'],
            Isp=data['isp'],
            rho1=data['rho1'],
            rho2=data['rho2'],
            R_MAX=data['R_MAX'],
            LU=data['LU'],
            gsa=data['gsa'],
            fov=data['fov'],
            pa=data['pa'],
            vmax=data['vmax'],
        )
    else:
        raise ValueError(f"Undefined planet: {planet}")


@dataclass
class Lander:
    """Parameters for the rocket dynamics and operational constraints.
    """
    # -------------------------
    # Default values
    # -------------------------

    # Lander parameters
    g_: float  # Gravitational acceleration (m/s^2), positive in the downward direction
    mdry: float  # Dry mass (kg)
    mwet: float  # Wet mass (kg)
    Isp: float  # Specific impulse (s)
    rho1: float  # Minimum thrust (N)
    rho2: float  # Maximum thrust (N)

    # Bounds and scaling factors
    R_MAX: float  # Maximum x, y, z bound (m)
    LU: float  # Length unit (m) for scaling

    # -------------------------
    # Non default parameters
    # -------------------------

    # Lander operational parameters
    gsa: float = None  # Glide slope angle; measured from horizon (rad)
    pa: float = None  # Pointing angle (rad)
    fov: float = None  # Field of view (rad)
    vmax: float = None  # Maximum velocity (m/s)

    # -------------------------
    # Derived parameters (will be computed in __post_init__)
    # -------------------------

    # Bounds and scaling factors
    TU: float = None  # Time unit (s) for scaling
    MU: float = None  # Mass unit (kg) for scaling

    # Other derived parameters (will be computed in __post_init__)
    g: np.ndarray = None  # Gravitational acceleration vector (m/s^2); flat planet approximation
    alpha: float = None  # Mass flow rate per thrust (kg/s/N)

    def __post_init__(self):
        """Post processing; assertion and computation of derived parameters"""
        assert self.g_ >= 0, "g must be non-negative"
        assert self.mdry > 0, "mdry must be positive"
        assert self.mwet > self.mdry, "mwet must be greater than mdry"

        self.TU = np.sqrt(self.LU / abs(self.g_))
        self.MU = self.mwet

        self.g = np.array([0.0, 0.0, -self.g_])
        self.alpha = 1 / (self.Isp * ge)
