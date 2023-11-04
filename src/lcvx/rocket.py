"""Rocket object stores parameters related to dynamics and constraints"""
import numpy as np
from dataclasses import dataclass
from scipy.constants import g as ge  # Standard gravity (m/s^2)
from .linear_dynamics import continuous_sys


@dataclass
class Rocket:
    """Parameters for the rocket dynamics and operational constraints.
    The relaxed linear dynamics are employed in the LCVx formulation:
        x_dot = A_c x + B_c u + p_c
    where x is the state vector, [r, v, log(m)],
    u is the control vector, [T/m, Gamma/m],
    and p_c is the constant term vector, [0, g, 0].
    """

    # Rocket parameters
    g_: float  # Gravitational acceleration (m/s^2), positive in the downward direction
    mdry: float  # Dry mass (kg)
    mwet: float  # Wet mass (kg)
    Isp: float  # Specific impulse (s)
    rho1: float  # Minimum thrust (N)
    rho2: float  # Maximum thrust (N)
    gsa: float = None  # Glide slope angle; measured from horizon (rad)
    pa: float = None  # Pointing angle (rad)
    fov: float = None  # Field of view (rad)
    vmax: float = None  # Maximum velocity (m/s)

    # Derived parameters (will be computed in __post_init__)
    g: np.ndarray = None  # Gravitational acceleration vector (m/s^2)
    alpha: float = None  # Mass flow rate per thrust (kg/s/N)
    Ac: np.ndarray = None  # Continuous-time dynamics matrix
    Bc: np.ndarray = None
    pc: np.ndarray = None
    n: int = None  # Number of states
    m: int = None  # Number of controls

    def __post_init__(self):
        """Post processing; assertion and computation of derived parameters"""
        assert self.g_ >= 0, "g must be non-negative"
        assert self.mdry > 0, "mdry must be positive"
        assert self.mwet > self.mdry, "mwet must be greater than mdry"

        self.g = np.array([0.0, 0.0, -self.g_])
        self.alpha = 1 / (self.Isp * ge)
        self.Ac, self.Bc, self.pc = continuous_sys(self.g_, self.alpha)
        self.n, self.m = self.Bc.shape
