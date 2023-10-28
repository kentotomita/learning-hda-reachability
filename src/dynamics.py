import numpy as np
from numba import njit


@njit
def pd_3dof_eom(
    x: np.ndarray, u: np.ndarray, g: np.ndarray, alpha: float
) -> np.ndarray:
    """
    3DOF Equations of Motion of Powered Descent Guidance

    Args:
        x: State vector (m, m, m, m/s, m/s, m/s, kg)
        u: Control vector (N, N, N)
        g: Gravitational acceleration vector (m/s^2)
        alpha: Fuel consumption rate (kg/s/N)

    Returns:
        State derivative vector
    """
    # Unpack state vector
    x, y, z, vx, vy, vz, m = x

    # Compute state derivative vector
    dxdt = vx
    dydt = vy
    dzdt = vz
    dvxdt = u[0] / m + g[0]
    dvydt = u[1] / m + g[1]
    dvzdt = u[2] / m + g[2]
    dmdt = -alpha * np.linalg.norm(u)

    return np.array([dxdt, dydt, dzdt, dvxdt, dvydt, dvzdt, dmdt])
