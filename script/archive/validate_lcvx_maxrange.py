import numpy as np
import cvxpy as cp
import sys
import json
import os

sys.path.append("../")

import src.lcvx as lc
from src.visualization import *


if __name__ == "__main__":
    t0 = 0.0  # Initial time (s)
    g = np.array([0, 0, -3.7114])  # Gravitational acceleration (m/s^2)
    Tmax = 13260.0  # Maximum thrust (N)
    Tmin = 4972.0  # Minimum thrust (N)
    mdry = 1505.0  # Dry mass (kg)
    mwet = 1905.0  # Wet mass (kg)
    Isp = 225.0  # Specific impulse (s)
    gsa = 4 * np.pi / 180  # Gimbal angle (rad)
    pa = 30 * np.pi / 180  # Pitch angle (rad)
    vmax = 800 * 1e3 / 3600  # Maximum velocity (m/s)

    rocket = lc.Rocket(
        g=g,
        mdry=mdry,
        mwet=mwet,
        Isp=Isp,
        rho1=Tmin,
        rho2=Tmax,
        gsa=gsa,
        pa=pa,
        vmax=vmax,
    )

    N = 100

    n = 3
    m_range = np.linspace(1805, 1905, n)
    rz_range = np.linspace(1000, 2000, n)
    vz_range = np.linspace(-100, -30, n)
    vx_range = np.linspace(0, 100, n)
    tf_range = np.linspace(50, 80, n)
    c_list = [
        np.array([np.cos(theta), np.sin(theta), 0.0])
        for theta in np.linspace(0, np.pi, n)
    ]

    for tfi, tf in enumerate(tf_range):
        for ci, c in enumerate(c_list):
            lcvx_obj = lc.LCvxMaxRange(
                rocket=rocket, N=N, parameterize_x0=True, parameterize_c=False
            )
            prob = lcvx_obj.problem(tf=tf, c=c)

            k = 0
            for vz in vz_range:
                for vx in vx_range:
                    x0 = np.array([0.0, 0.0, 1500.0, vx, 0, vz, np.log(1905.0)])
                    lc.set_params(prob, {"x0": x0})

                    # solve problem
                    # prob.solve(verbose=False, requires_grad=True, eps_abs=1e-7, eps_rel=1e-7, eps_infeas=1e-8)
                    prob.solve(
                        verbose=False,
                        requires_grad=False,
                        abstol=1e-9,
                        reltol=1e-9,
                        feastol=1e-9,
                    )
                    print(
                        f"Problem parameters  tf: {tf: .1f}, c: {c}, vz: {vz: .1f}, vx: {vx: .1f} | problem status: {prob.status}, {prob.value}"
                    )

                    if prob.status == "optimal":
                        # get solution
                        sol = lc.get_vars(prob, ["X", "U"])
                        X_sol = sol["X"]
                        U_sol = sol["U"]

                        r, v, z, u, sigma = lcvx_obj.recover_variables(X_sol, U_sol)
                        m = np.exp(z)
                        U = u.T * m[:-1].reshape(-1, 1)  # (N, 3)
                        X = np.hstack([r.T, v.T, m.reshape(-1, 1)])  # (N+1, 7)
                        u_norm = np.linalg.norm(U_sol[:3, :], axis=0)
                        sigma = U_sol[3, :]
                        t = np.linspace(t0, tf, N + 1)

                        # plot results
                        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
                        ax.plot(t[:-1], u_norm, label="u_norm")
                        ax.plot(t[:-1], sigma, label="sigma")
                        ax.legend()
                        ax.set_xlabel("t")
                        plt.show()

                        plot_3sides(t[:-1], X, U, gsa=gsa)

                        k += 1
