"""Implement discrete-time system discretization methods. 

References:
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cont2discrete.html
"""
from typing import Tuple
import numpy as np
from scipy.linalg import expm
import cvxpy as cp


def zoh(
    A: np.ndarray, B: np.ndarray, dt: float, p: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Zero-order hold discretization of a continuous-time system.

    dx/dt = Ax + Bu + p

    Args:
        A: Continuous-time state matrix.
        B: Continuous-time input matrix.
        p: Continuous-time constant matrix.
        dt: Discretization time step.


    References:
        [1] https://en.wikipedia.org/wiki/Discretization#Discretization_of_linear_state_space_models
    """

    n = A.shape[0]  # number of states
    m = B.shape[1]  # number of inputs

    if p is None:
        p = np.zeros((n, 1))
    else:
        p = p.reshape((n, 1))

    # build exponential matrix
    em = np.block([[A, B, p], [np.zeros((m + 1, n + m + 1))]])
    em = expm(em * dt)

    # extract discrete-time matrices
    Ad = em[:n, :n]
    Bd = em[:n, n : n + m]
    pd = em[:n, n + m]

    if p is None:
        return Ad, Bd
    else:
        return Ad, Bd, pd


def zoh_cp(
    A: np.ndarray, B: np.ndarray, dt: cp.Expression, p: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Zero-order hold discretization when dt is a cvxpy variable."""

    n = A.shape[0]  # number of states
    m = B.shape[1]  # number of inputs

    if p is None:
        p = np.zeros((n, 1))
    else:
        p = p.reshape((n, 1))

    # Include p in B
    Bc = np.block([B, p])

    # Discretize
    Ad = np.eye(7) + A * dt + 0.5 * A @ A * dt**2
    Bd = (np.eye(7) + 0.5 * A * dt + 1 / 6 * A @ A * dt**2) @ Bc * dt

    # Extract p
    pd = Bd[:, -1]

    # Remove p from Bd
    Bd = Bd[:, :-1]

    if p is None:
        return Ad, Bd
    else:
        return Ad, Bd, pd
