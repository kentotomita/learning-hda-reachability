"""Relaxed linear dynamics for LCVx. The equality constraint between 
the thrust vector and thrust norm is relaxed, resulting in decoupled linear dynamics."""

import numpy as np

def continuous_sys(g: float, alpha: float):
    """Returns the continuous-time dynamics of the rocket.

    Args:
        g: Gravitational acceleration vector (m/s^2)
        alpha: Mass flow rate per thrust (kg/s/N)
    """
    assert g >= 0, 'g must be non-negative'

    A = np.array([
        [0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
    ])
    B = np.array([
        [0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., -alpha]
        ])
    p = np.array([[0., 0., 0., 0., 0., -g, 0.]]).T

    return A, B, p


def discrete_sys(dt: float, g: float, alpha: float):
    """Returns the discrete-time dynamics of the rocket.

    Args:
        dt: Time step (s)
        g: Gravitational acceleration vector (m/s^2)
        alpha: Mass flow rate per thrust (kg/s/N)

    
    Exponential Matrix = ([
        [1, 0, 0, dt, 0,  0,  0,    dt**2/2, 0,        0,          0,          0], 
        [0, 1, 0, 0,  dt, 0,  0,    0,       dt**2/2,  0,          0,          0], 
        [0, 0, 1, 0,  0,  dt, 0,    0,       0,        dt**2/2,    0,          -dt**2*g/2], 
        [0, 0, 0, 1,  0,  0,  0,    dt,      0,        0,          0,          0], 
        [0, 0, 0, 0,  1,  0,  0,    0,       dt,       0,          0,          0], 
        [0, 0, 0, 0,  0,  1,  0,    0,       0,        dt,         0,          -dt*g], 
        [0, 0, 0, 0,  0,  0,  1,    0,       0,        0,          -alpha*dt,  0], 
        [0, 0, 0, 0,  0,  0,  0, 1,       0,        0, 0, 0], 
        [0, 0, 0, 0,  0,  0,  0, 0,       1,        0, 0, 0], 
        [0, 0, 0, 0,  0,  0,  0, 0,       0,        1, 0, 0], 
        [0, 0, 0, 0,  0,  0,  0, 0,       0,        0, 1, 0], 
        [0, 0, 0, 0,  0,  0,  0, 0,       0,        0, 0, 1]
        ])
    """

    Ad = np.array([
        [1., 0., 0., dt, 0., 0., 0.],
        [0., 1., 0., 0., dt, 0., 0.],
        [0., 0., 1., 0., 0., dt, 0.],
        [0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 0., 1.]
    ])
    Bd = np.array([
        [dt**2/2,   0.,         0.,         0.],
        [0.,        dt**2/2,    0.,         0.],
        [0.,        0.,         dt**2/2,    0.],
        [dt,        0.,         0.,         0.],
        [0.,        dt,         0.,         0.],
        [0.,        0.,         dt,         0.],
        [0.,        0.,         0.,         -alpha*dt]
    ])
    pd = np.array([[0., 0., -dt**2*g/2, 0., 0., -dt*g, 0.]]).T

    return Ad, Bd, pd