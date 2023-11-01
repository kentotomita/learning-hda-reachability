import numpy as np
import cvxpy as cp
import sys

sys.path.append("../")

import src.lcvx as lc
from src.visualization import *


def config():
    rk = lc.Rocket(
        g_=3.7114,
        mdry=1505.0,
        mwet=1905.0,
        Isp=225.0,
        rho1=4972.0,
        rho2=13260.0,
        gsa=4 * np.pi / 180,
        pa=30 * np.pi / 180,
        vmax=800 * 1e3 / 3600,
    )
    N = 100
    tf = 55.0
    dt = tf / N
    x0 = np.array(
        [2000.0, 0.0, 1500.0, -10.0, 3.0, -75.0, np.log(rk.mwet)]
    )  # Initial state vector (m, m, m, m/s, m/s, m/s, kg)
    return rk, x0, N, tf, dt


def vis_results(solved_problem, rocket):
    print(f"Problem is DCP: {solved_problem.is_dcp()}")
    print(f"Problem is DPP: {solved_problem.is_dpp()}")
    print(f"Problem is DGP: {solved_problem.is_dgp()}")

    sol = lc.get_vars(solved_problem, ["X", "U"])
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

    plot_vel(t, X, rocket.vmax)

    plot_mass(t, X, rocket.mdry, rocket.mwet)

    plot_thrust_mag(t[:-1], U, rocket.rho2, rocket.rho1)

    plot_pointing(t[:-1], U, rocket.pa)


if __name__ == "__main__":
    # Simulation parameters
    rocket, x0, N, tf, dt = config()
    c = np.array([0.0, 1.0, 0.0])

    #"""
    # Test parameterize x0
    lcvx_obj = lc.LCvxMaxRange(
        rocket=rocket, N=N, parameterize_x0=True, parameterize_c=False
    )
    prob = lcvx_obj.problem(tf=tf, c=c, rc=np.zeros(3))
    lc.set_params(prob, {"x0": x0})
    prob.solve(verbose=True, requires_grad=True)
    vis_results(prob, rocket)

    # Test parameterize c
    lcvx_obj = lc.LCvxMaxRange(
        rocket=rocket, N=N, parameterize_x0=False, parameterize_c=True
    )
    prob = lcvx_obj.problem(x0=x0, tf=tf, rc=np.zeros(3))
    lc.set_params(prob, {"c": c})
    prob.solve(verbose=True, requires_grad=True)
    vis_results(prob, rocket)
    #"""

    # Test parameterize x0 and c
    lcvx_obj = lc.LCvxMaxRange(
        rocket=rocket, N=N, parameterize_x0=True, parameterize_c=True
    )
    prob = lcvx_obj.problem(tf=tf, rc=np.zeros(3))
    lc.set_params(prob, {"x0": x0})
    lc.set_params(prob, {"c": c})
    prob.solve(verbose=True, requires_grad=False)
    vis_results(prob, rocket)
