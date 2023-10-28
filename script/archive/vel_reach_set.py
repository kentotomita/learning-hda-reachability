"""Generate and visualize soft landing reachable set."""
import numpy as np
import cvxpy as cp
import sys

sys.path.append("../")
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull

import src.lcvx as lc
from src.visualization import *


def visualize_convex_hull(vertices):
    """
    Visualize the 3D convex hull of the given vertices.

    Args:
        vertices (numpy.ndarray): An array of 3D points representing the vertices.
    """
    hull = ConvexHull(vertices)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c="k", marker="o")

    for simplex in hull.simplices:
        polygon = vertices[simplex]
        ax.add_collection3d(
            Poly3DCollection(
                [polygon], alpha=0.25, linewidths=0.5, edgecolors="k", facecolors="gray"
            )
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Limit axes for better visualization
    ax.set_xlim([np.min(vertices[:, 0]), np.max(vertices[:, 0])])
    ax.set_ylim([np.min(vertices[:, 1]), np.max(vertices[:, 1])])
    ax.set_zlim([np.min(vertices[:, 2]), np.max(vertices[:, 2])])

    plt.show()


def visualize_convex_hull_2d(vertices):
    """
    Visualize the 2D convex hull of the given vertices.

    Args:
        vertices (numpy.ndarray): An array of 2D points representing the vertices.
    """
    hull = ConvexHull(vertices)

    plt.figure()
    for simplex in hull.simplices:
        plt.plot(vertices[simplex, 0], vertices[simplex, 1], "k-")

    plt.plot(vertices[:, 0], vertices[:, 1], "o")
    plt.xlabel("X")
    plt.ylabel("Y")

    # Limit axes for better visualization
    plt.xlim([np.min(vertices[:, 0]) - 1, np.max(vertices[:, 0]) + 1])
    plt.ylim([np.min(vertices[:, 1]) - 1, np.max(vertices[:, 1]) + 1])

    plt.show()


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
    # x0 = np.array([0., 0., 1500., 0., 0., -10., mwet])  # Initial state vector (m, m, m, m/s, m/s, m/s, kg)

    rocket = lc.Rocket(
        g=g,
        mdry=mdry,
        mwet=mwet,
        Isp=225.0,
        rho1=Tmin,
        rho2=Tmax,
        gsa=82.5 * np.pi / 180,
        pa=30 * np.pi / 180,
        fov=15 * np.pi / 180,
    )
    N = 100
    tf = 55.0
    dt = tf / N
    x0[6] = np.log(x0[6])

    return rocket, x0, N, tf, dt


def get_var(prob, c, return_all=False):
    lc.set_params(prob, {"c": c})
    try:
        prob.solve(verbose=False)
        # prob.solve(verbose=False, requires_grad=True, eps_abs=1e-7, eps_rel=1e-7, eps_infeas=1e-8)
    except cp.SolverError:
        print("Infeasible")
        return None
    if prob.status != "optimal":
        return None

    sol = lc.get_vars(prob, ["X", "U"])
    X_sol = sol["X"]
    U_sol = sol["U"]
    r, v, z, u, sigma = lcvx.recover_variables(X_sol, U_sol)

    isolated_gsc, idx_gsc = lc.isolated_active_set_gs(r, rocket.gsa)
    active_slack, idx_slack = lc.active_slack_var(u, sigma)

    debug = False
    if isolated_gsc or active_slack:
        # print(f'Isolated active set: {isolated_gsc} | Active slack variable: {active_slack}')
        if debug:
            visualize_traj(r, v, z, u, sigma)
    else:
        """
        if not isolated_gsc:
            print(f'Not isolated active set at r = {r[:3, idx_gsc]} and r = {r[:3, idx_gsc+1]}, idx = {idx_gsc}')
        if not active_slack:
            print(f'Not active slack variable at u = {u[:, idx_slack]} and u = {u[:, idx_slack+1]}, idx = {idx_slack}')
        if debug:
            visualize_traj(r, v, z, u, sigma)
        #"""
    if return_all:
        return r, v, z, u, sigma
    else:
        return v[:, lcvx.maxk]


def visualize_traj(r, v, z, u, sigma):
    N = u.shape[1]
    m = np.exp(z)
    X = np.hstack((r.T, v.T, m.reshape(-1, 1)))
    U = u.T * m[:-1].reshape(-1, 1)
    t = np.linspace(0, tf, N + 1)
    plot_3sides(t[:-1], X, U, uskip=1, gsa=rocket.gsa)
    plot_slack(t, u, sigma)


if __name__ == "__main__":
    # Simulation parameters
    rocket, x0, N, tf, dt = config()

    # -------------------------------
    # Horizontal velocity reach set
    # -------------------------------

    x0_min = np.array([0.0, 0.0, 1500.0, 0.0, 0.0, -75, np.log(rocket.mwet * 0.9)])
    x0_max = np.array([0.0, 0.0, 1500.0, 37.5, 0.0, 0.0, np.log(rocket.mwet * 1.1)])
    x0_bounds = (x0_min, x0_max)
    r0 = np.array([0.0, 0.0, 1500.0])
    z0_bounds = (np.log(rocket.mwet * 0.9), np.log(rocket.mwet * 1.1))
    v0_max = 75.0
    v0_max_angle = 30 * np.pi / 180

    def v0_cstr(v: cp.Variable, v0_max=v0_max, v0_max_angle=v0_max_angle):
        cstr = []
        cstr.append(cp.norm(v[:, 0]) <= v0_max)
        cstr.append(cp.norm(v[0, 0] + v[1, 0]) <= -v[2, 0] * np.tan(v0_max_angle))
        return cstr

    # Feasible point
    n_per_step = 25
    c_list = [
        np.array([np.cos(theta), np.sin(theta)])
        for theta in np.linspace(0, np.pi, n_per_step)
    ]

    n_max_steps = 10
    pbar = tqdm(total=n_max_steps * n_per_step)
    vertices3d = []
    vertices2d = []

    for i in range(n_max_steps):
        maxk = int(N * i / n_max_steps)

        # find feasible point
        lcvx = lc.LCvxReachVxy(rocket, N, maxk=maxk, inner=False)
        prob = lcvx.problem(x0_bounds=x0_bounds, tf=tf, cstr_add=v0_cstr)
        v1 = get_var(prob, np.array([0.0, 1.0]))
        v2 = get_var(prob, np.array([1.0, 0.0]))
        v3 = get_var(prob, np.array([-1.0, -1.0]))

        try:
            vc = (v1 + v2 + v3) / 3
        except TypeError:
            vc = np.array([0.0, 0.0])

        lcvx = lc.LCvxReachVxy(rocket, N, maxk=maxk, inner=True)
        prob = lcvx.problem(x0_bounds=x0_bounds, tf=tf, vc=vc[:2], cstr_add=v0_cstr)
        for c in c_list:
            out = get_var(prob, c, return_all=True)
            if out is not None:
                r, v, z, u, sigma = out
                vk = v[:2, lcvx.maxk]
                alt = r[2, lcvx.maxk]
                vertices3d.append([vk[0], vk[1], alt])
                vertices3d.append([vk[0], -vk[1], alt])
                vertices2d.append([np.linalg.norm(vk), alt])

            pbar.update(1)

    pbar.close()

    vertices2d = np.array(vertices2d)
    print(vertices2d.shape)

    visualize_convex_hull_2d(vertices2d)

    vertices3d = np.array(vertices3d)
    fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
    ax.scatter(vertices3d[:, 0], vertices3d[:, 1], vertices3d[:, 2], s=1)
    ax.set_xlabel("vx")
    ax.set_ylabel("vy")
    ax.set_zlabel("alt")
    ax.set_xlim(-30, 30)
    ax.set_ylim(-30, 30)
    ax.set_zlim(0, 1500)
    plt.show()
