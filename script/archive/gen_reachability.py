import numpy as np
import cvxpy as cp
import sys
import json
import os

sys.path.append("../")

import src.lcvx as lc
from src.visualization import *


def vis_results(prob, rocket):
    print(f"Problem is DCP: {prob.is_dcp()}")
    print(f"Problem is DPP: {prob.is_dpp()}")

    sol = lc.get_vars(prob, ["X", "U"])
    X_sol = sol["X"]
    U_sol = sol["U"]
    r, v, z, u, sigma = lcvx_obj.recover_variables(X_sol, U_sol)

    # visualize
    m = np.exp(z)
    X = np.hstack((r.T, v.T, m.reshape(-1, 1)))
    U = u.T * m[:-1].reshape(-1, 1)

    # Plot results
    t = np.linspace(0, tf, lcvx_obj.N + 1)

    plot_3sides(t[:-1], X, U, uskip=1, gsa=rocket.gsa)

    # lc.plot_slack_var(t, U_sol)

    # plot_vel(t, X, rocket.vmax)

    # plot_mass(t, X, rocket.mdry, rocket.mwet)

    # plot_thrust_mag(t[:-1], U, rocket.rho2, rocket.rho1)

    # plot_pointing(t[:-1], U, rocket.pa)


if __name__ == "__main__":
    t0 = 0.0  # Initial time (s)
    g = np.array([0, 0, -3.7114])  # Gravitational acceleration (m/s^2)
    Tmax = 13260.0  # Maximum thrust (N)
    Tmin = 4972.0  # Minimum thrust (N)
    mdry = 1505.0  # Dry mass (kg)
    mwet = 1905.0  # Wet mass (kg)
    Isp = 225.0  # Specific impulse (s)
    gsa = 85 * np.pi / 180  # Gimbal angle (rad)
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
    rz_list = [1000.0]
    vz_list = np.linspace(-100, 0, n)
    vx_list = np.linspace(0, 30, n)
    tf_list = np.linspace(50, 80, 3)
    c_list = [
        np.array([np.cos(theta), np.sin(theta), 0.0])
        for theta in np.linspace(0, np.pi, n)
    ]
    c_list = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])

    visualize = True

    for tfi, tf in enumerate(tf_list):
        for ci, c in enumerate(c_list):
            lcvx_obj = lc.LCvxMaxRange(
                rocket=rocket, N=N, parameterize_x0=True, parameterize_c=False
            )
            prob = lcvx_obj.problem(tf=tf, c=c)

            k = 0
            for vz in vz_list:
                for vx in vx_list:
                    x0 = np.array([0.0, 0.0, 1500.0, vx, 0, vz, np.log(1905.0)])
                    lc.set_params(prob, {"x0": x0})

                    # solve problem
                    # prob.solve(verbose=False, requires_grad=True, eps_abs=1e-7, eps_rel=1e-7, eps_infeas=1e-8)
                    prob.solve(
                        verbose=False,
                        requires_grad=False,
                        abstol=1e-9,
                        reltol=1e-9,
                        feastol=1e-8,
                    )
                    print(
                        f"Problem parameters  tf: {tf: .1f}, c: {c}, vz: {vz: .1f}, vx: {vx: .1f} | problem status: {prob.status}, {prob.value}"
                    )

                    # save results
                    fname = os.path.join("../out", f"tf_{tfi}_c_{ci}_{k}.json")
                    params = {
                        "c": c.tolist(),
                        "gsa": gsa,
                        "pa": pa,
                        "vmax": vmax,
                        "Tmax": Tmax,
                        "Tmin": Tmin,
                        "mdry": mdry,
                        "mwet": mwet,
                        "g": g.tolist(),
                    }
                    lc.save_results(prob, lcvx_obj, tf, fname, params=params)

                    if visualize and prob.status == "optimal":
                        vis_results(prob, rocket)

                    k += 1
