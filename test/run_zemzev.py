import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from scipy.integrate import solve_ivp

import sys

sys.path.append("../")

from src.dynamics import pd_3dof_eom
from src.integrator import rk4
from src.zemzev import zemzev, tgo_zemzev


def plot_3d_trajectory(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x, y, z, "b-", linewidth=2)
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    plt.show()


if __name__ == "__main__":
    print("Running Zemzev Guidance Law")
    # Simulation parameters
    g = np.array([0, 0, -3.7114])  # Gravitational acceleration (m/s^2)
    alpha = 4.53 * 10**-4  # Fuel consumption rate (kg/N)
    Tmax = 13260  # Maximum thrust (N)
    Tmin = 4972  # Minimum thrust (N)
    x0 = np.array(
        [2000.0, 0.0, 1500.0, -100.0, 30.0, -25.0, 1905.0]
    )  # Initial state vector (m, m, m, m/s, m/s, m/s, kg)
    target = np.array([0, 0, 0, 0, 0, 0])  # Final state vector (m, m, m, m/s, m/s, m/s)
    t0 = 0  # Initial time (s)
    dt = 0.001  # Time step (s)
    tmax = 100  # Maximum time (s)

    # Compute Time to Go
    tf = tgo_zemzev(x0, target, g)  # Final time (s)
    print(f"Final time: {tf} sec")

    # Closed loop system dynamics
    @njit
    def fcl(t, x):
        g = np.array([0, 0, -3.7114])
        u = zemzev(t, x, tf, target, g, Tmax, Tmin)
        return pd_3dof_eom(x, u, g, alpha)

    # Define event; detect touchdown
    def event(t, x):
        return x[2]

    # Run simulation
    print(f"Initial state: {x0}")
    t_eval = np.arange(t0, tmax, dt)
    t, x = rk4(fcl, x0, t0, tmax, dt, event)

    # Print results
    plot_3d_trajectory(x[:, 0], x[:, 1], x[:, 2])
