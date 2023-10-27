import numpy as np
from numba import njit

import sys
sys.path.append('..')

from src.integrator import rk4

if __name__=="__main__":
    @njit
    def f(t, y):
        dydt = np.zeros(y.shape)
        dydt[0] = t * np.sqrt(y[1])  # Sample ODE: dy0/dt = t * sqrt(y1)
        dydt[1] = y[0] * y[1]        # Sample ODE: dy1/dt = y0 * y1
        return dydt

    @njit
    def event(t, y):
        return y[1] - 5  # Event detection condition: y1(t) = 5

    y0 = np.array([1.0, 1.0])  # Initial conditions: y0(0) = 1, y1(0) = 1
    t0 = 0
    tf = 10
    dt = 0.1

    t, y = rk4(f, y0, t0, tf, dt, event)

    # Print results
    for i in range(len(t)):
        print(f"t: {t[i]}, y0: {y[i, 0]}, y1: {y[i, 1]}")