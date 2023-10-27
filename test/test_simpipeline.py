import numpy as np
import matplotlib.pyplot as plt
from numba import njit

import sys
sys.path.append('../')

from src.dynamics import pd_3dof_eom
from src.integrator import sim_feedback
from src.zemzev import zemzev, tgo_zemzev
from src.visualization import *


if __name__ == "__main__":
    print("Running Zemzev Guidance Law")

    # Simulation parameters
    t0 = 0  # Initial time (s)
    tmax = 100  # Maximum time (s)
    dt = 0.001  # Time step (s)
    dt_g = 1    # Guidance update time step (s)
    g = np.array([0, 0, -3.7114])  # Gravitational acceleration (m/s^2)
    alpha = 4.53 * 10**-4  # Fuel consumption rate (kg/N)
    Tmax = 13260 # Maximum thrust (N)
    Tmin = 4972 # Minimum thrust (N)
    mdry = 1505 # Dry mass (kg)
    mwet = 1905 # Wet mass (kg)
    x0 = np.array([1500, 0, 2000, -25, 30, -100, mwet])  # Initial state vector (m, m, m, m/s, m/s, m/s, kg)
    xf = np.array([0, 0, 0, 0, 0, 0])  # Final state vector (m, m, m, m/s, m/s, m/s)

    # Compute Time to Go
    tf = tgo_zemzev(x0, xf, g)  # Final time (s)
    print(f"Final time: {tf} sec")

    # Construct controller object
    controller = lambda t, x: zemzev(t, x, tf, xf, g, Tmax, Tmin)

    # Define event; detect touchdown
    def event(t, x):
        return x[2] - 1e-3  # subtract 1 mm to avoid numerical issues

    # Run simulation
    print(f"Initial state: {x0}")
    t, X, U = sim_feedback(
        f=lambda t, x, u: pd_3dof_eom(x, u, g, alpha),
        u=controller, 
        x0=x0, 
        t_span=(t0, tmax), 
        dt=dt,
        dt_g=dt_g,
        event=event)

    # Plot results
    plot_3sides(t, X, U, uskip=int(dt_g / dt))

    plot_mass(t, X, mdry, mwet)

    plot_thrust_mag(t, U, Tmax, Tmin)



