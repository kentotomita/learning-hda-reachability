"""Generate and visualize soft landing reachable set."""
import numpy as np
import cvxpy as cp
import sys

sys.path.append("../")
import matplotlib.pyplot as plt

import src.lcvx as lc
from src.visualization import *


def config():
    # Simulation parameters
    t0 = 0.0  # Initial time (s)
    tmax = 100.0  # Maximum time (s)
    dt = 0.001  # Time step (s)
    dt_g = 1.0  # Guidance update time step (s)
    g = np.array([0, 0, -3.7114])  # Gravitational acceleration (m/s^2)
    Tmax = 13260.0  # Maximum thrust (N)
    Tmin = 4972.0  # Minimum thrust (N)
    mdry = 1505.0  # Dry mass (kg)
    mwet = 1825  # 1905. # Wet mass (kg)
    # x0 = np.array([2000., 500., 1500., -10., 3., -75., mwet])  # Initial state vector (m, m, m, m/s, m/s, m/s, kg)
    x0 = np.array(
        [0.0, 0.0, 1500.0, 37.5, 0.0, -53.0, mwet]
    )  # Initial state vector (m, m, m, m/s, m/s, m/s, kg)

    rocket = lc.Rocket(
        g=g,
        mdry=mdry,
        mwet=mwet,
        Isp=225.0,
        rho1=Tmin,
        rho2=Tmax,
        gsa=82.5 * np.pi / 180,
        pa=30 * np.pi / 180,
        vmax=800 * 1e3 / 3600,
    )
    N = 100
    tf = 55.0
    dt = tf / N
    x0[6] = np.log(x0[6])

    return rocket, x0, N, tf, dt


if __name__ == "__main__":
    # Simulation parameters
    rocket, x0, N, tf, dt = config()
    # fov = 20 * np.pi / 180
    fov = np.pi - rocket.gsa * 2

    # -------------------------------
    # Generate reachable set
    # -------------------------------
    def get_rf(prob, c):
        lc.set_params(prob, {"c": c})
        # prob.solve(verbose=False, abstol=1e-9, reltol=1e-9, feastol=1e-9)
        prob.solve(
            verbose=False,
            requires_grad=True,
            eps_abs=1e-7,
            eps_rel=1e-7,
            eps_infeas=1e-8,
        )
        sol = lc.get_vars(prob, ["X", "U"])
        X_sol = sol["X"]
        U_sol = sol["U"]
        r, v, z, u, sigma = lcvx.recover_variables(X_sol, U_sol)

        isolated_gsc, idx_gsc = lc.isolated_active_set_gs(r, rocket.gsa)
        active_slack, idx_slack = lc.active_slack_var(u, sigma)

        debug = False
        if isolated_gsc or active_slack:
            print(
                f"Isolated active set: {isolated_gsc} | Active slack variable: {active_slack}"
            )
            if debug:
                visualize_traj(r, v, z, u, sigma)
            return r[:3, -1]
        else:
            if not isolated_gsc:
                print(
                    f"Not isolated active set at r = {r[:3, idx_gsc]} and r = {r[:3, idx_gsc+1]}, idx = {idx_gsc}"
                )
            if not active_slack:
                print(
                    f"Not active slack variable at u = {u[:, idx_slack]} and u = {u[:, idx_slack+1]}, idx = {idx_slack}"
                )
            if debug:
                visualize_traj(r, v, z, u, sigma)
            return None

    def visualize_traj(r, v, z, u, sigma):
        N = u.shape[1]
        m = np.exp(z)
        X = np.hstack((r.T, v.T, m.reshape(-1, 1)))
        U = u.T * m[:-1].reshape(-1, 1)
        t = np.linspace(0, tf, N + 1)
        plot_3sides(t[:-1], X, U, uskip=1, gsa=rocket.gsa)
        plot_slack(t, u, sigma)

    # Feasible point
    c_list = [
        np.array([np.cos(theta), np.sin(theta), 0.0])
        for theta in np.linspace(0, np.pi, 100)
    ]
    lcvx = lc.LCvxMaxRange(rocket, N, parameterize_c=True)
    prob = lcvx.problem(x0=x0, tf=tf)
    rp = get_rf(prob, c_list[0])  # maximum downrange
    rm = get_rf(prob, c_list[-1])  # minimum downrange
    rc = (rp + rm) / 2
    points = [rp]

    # Max ranges
    prob = lcvx.problem(x0=x0, tf=tf, rc=rc)
    for c in c_list[1:-1]:
        pt = get_rf(prob, c)
        if pt is not None:
            points.append(pt)
    points.append(rm)

    points = np.array(points)

    xy_coords = points[:, :2]
    fig, ax = plt.subplots()
    poly = plt.Polygon(xy_coords, fill=None, edgecolor="r", label="Reachable Set")
    ax.add_patch(poly)
    fov_radius = x0[2] * np.tan(fov / 2)
    fov_circle = plt.Circle(
        x0[:2],
        fov_radius,
        fill=None,
        edgecolor="k",
        linestyle="--",
        label="Sensor Field of View",
    )
    ax.add_patch(fov_circle)
    ax.scatter(xy_coords[:, 0], xy_coords[:, 1], color="k", marker="x", s=1)
    ax.scatter(rc[0], rc[1], color="k", marker="x", s=1)
    ax.axis("equal")

    xmin = min(np.min(xy_coords[:, 0]), x0[0] - fov_radius) - 100
    xmax = max(np.max(xy_coords[:, 0]), x0[0] + fov_radius) + 100
    ymin = min(np.min(xy_coords[:, 1]), x0[1] - fov_radius) - 100
    ymax = max(np.max(xy_coords[:, 1]), x0[1] + fov_radius) + 100
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title(
        f"$r_0={x0[:3]}$ [m],   $v_0={x0[3:6]}$ [m/v],   $m_0={np.exp(x0[6]):.0f}$ [kg]"
    )
    plt.legend()
    plt.grid()
    plt.show()
