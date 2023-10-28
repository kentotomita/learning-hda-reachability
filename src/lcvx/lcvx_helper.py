"""Helper functions for lossless convexification."""
import numpy as np


def isolated_active_set_gs(r: np.ndarray, gsa: float):
    """Check if glide slope constraint is active only at the isolated points.

    Args:
        r (np.ndarray): The position vector, shape (3, N).
        gsa (float): The glide slope angle in radians.

    Returns:
        flag (bool): True if glide slope constraint is active only at the isolated points.
        i (int): The index of the first active point in case not isolated.
    """
    assert r.shape[0] == 3

    N = r.shape[1]
    rf = r[:, -1]
    active = lambda r, rf, gsa: np.sqrt(
        (r[0] - rf[0]) ** 2 + (r[1] - rf[1]) ** 2
    ) * np.tan(gsa) >= abs(r[2] - rf[2])
    flag = True
    for i in range(N - 1):
        if active(r[:, i], rf, gsa) and active(r[:, i + 1], rf, gsa):
            flag = False
            return flag, i
    return flag, None


def active_slack_var(u: np.ndarray, sigma: np.ndarray, tol=1e-6):
    """Check if slack variables are always active. If so, the relaxed problem is equivalent to the original problem.

    Args:
        u (np.ndarray): The control vector, shape (3, N).
        sigma (np.ndarray): The slack variable, shape (N, ).

    Returns:
        flag (bool): True if slack variables are active.
        i (int): The index of the first active point in case not isolated.
    """
    assert u.shape[0] == 3
    assert u.shape[1] == sigma.shape[0]

    N = u.shape[1]
    u_norm = np.linalg.norm(u, axis=0)
    flag = True
    for i in range(N - 1):
        if abs(u_norm[i] - sigma[i]) > tol:
            flag = False
            return flag, i
    return flag, None
