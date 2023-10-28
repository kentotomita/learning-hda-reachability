import numpy as np
import numpy.polynomial.polynomial as poly
from numpy.linalg import norm
from numba import njit


@njit
def zemzev(
    t: float,
    state: np.ndarray,
    tf: float,
    xf: np.ndarray,
    g: np.ndarray,
    Tmax: float,
    Tmin: float,
) -> np.ndarray:
    """
    Zemzev guidance law

    Args:
        t: current time
        state: state vector, [x, y, z, vx, vy, vz, m]
        tf: final time
        xf: target vector, [x, y, z, vx, vy, vz]
        g: gravitational acceleration vector
        Tmax: maximum thrust
        Tmin: minimum thrust

    Returns:
        acceleration vector

    Reference:
    [1] Y. Guo, M. Hawkins, and B. Wie, “OPTIMAL FEEDBACK GUIDANCE ALGORITHMS FOR PLANETARY LANDING AND ASTEROID INTERCEPT”.
    """
    # Unpack state vector
    r = state[0:3]
    v = state[3:6]
    m = state[6]

    # Unpack target vector
    rf = xf[0:3]
    vf = xf[3:6]

    # Compute time to go
    tgo = tf - t

    # Acceleration command; Eq. (35) of [1]
    u = 6 * (rf - (r + tgo * v)) / tgo**2 - 2 * (vf - v) / tgo - g
    u = u * m  # Convert to thrust
    if norm(u) > Tmax:
        u = u / norm(u) * Tmax
    elif norm(u) < Tmin:
        u = u / norm(u) * Tmin

    return u


def tgo_zemzev(state: np.ndarray, target: np.ndarray, g: np.ndarray):
    """Find time-to-go for Zemzev guidance law
    Reference:
        Eq. (30) of "OPTIMAL FEEDBACK GUIDANCE ALGORITHMS FOR PLANETARY LANDING AND ASTEROID INTERCEPT", AAS 2011

    Args:
        state: state vector, [x, y, z, vx, vy, vz, m]
        target: target vector, [x, y, z, vx, vy, vz]
        g: gravitational acceleration vector
    """
    # Unpack state vector
    r = state[0:3]
    v = state[3:6]

    # Unpack target vector
    rf = target[0:3]
    vf = target[3:6]

    poly_coeff = np.array(
        [
            norm(g) ** 2,
            0,
            -2 * (norm(v) ** 2 + norm(vf) ** 2 + np.dot(v, vf)),
            12 * (np.dot(rf - r, v + vf)),
            -18 * (norm(rf - r) ** 2),
        ]
    )
    roots = np.roots(poly_coeff)
    roots = roots[np.isreal(roots)]
    roots = roots[roots > 0]

    return np.min(roots.real)
