"""Generate and visualize soft landing reachable set."""
import numpy as np
import cvxpy as cp
import sys

sys.path.append("../")
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
import concurrent.futures
from multiprocessing import Pool

import src.lcvx as lc
from src.visualization import *


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

    plt.grid()
    plt.show()


def config():
    rocket = lc.Rocket(
        g=np.array([0, 0, -3.7114]),  # Gravitational acceleration (m/s^2)
        mdry=1505.0,  # Dry mass (kg)
        mwet=1825.0,  # 1905. # Wet mass (kg)
        Isp=225.0,
        rho1=4972.0,  # Minimum thrust (N)
        rho2=13260.0,  # Maximum thrust (N)
        gsa=82.5 * np.pi / 180,
        pa=30 * np.pi / 180,
        fov=15 * np.pi / 180,
    )
    N = 100
    tf = 55.0
    dt = tf / N
    return rocket, N, tf, dt


def solve_reachable_set(params):
    # Unpack parameters
    maxk = params["maxk"]
    rocket = params["rocket"]
    N = params["N"]
    x0_bounds = params["x0_bounds"]
    tf = params["tf"]
    v0_cstr = params["v0_cstr"]
    n_per_step = params["n_per_step"]

    # find feasible point
    lcvx = lc.LCvxReachVxy(rocket, N, maxk=maxk, inner=False)
    prob = lcvx.problem(x0_bounds=x0_bounds, tf=tf, cstr_add=v0_cstr)
    v1 = get_var(lcvx, prob, np.array([0.0, 1.0]))
    v2 = get_var(lcvx, prob, np.array([0.0, -1.0]))
    try:
        vc = (v1 + v2) / 2
    except TypeError:
        vc = np.array([0.0, 0.0])

    # generate inner convex hull
    lcvx = lc.LCvxReachVxy(rocket, N, maxk=maxk, inner=True)
    prob = lcvx.problem(x0_bounds=x0_bounds, tf=tf, vc=vc[:2], cstr_add=v0_cstr)

    vertices3d = []
    vertices2d = []

    c_list = [
        np.array([np.cos(theta), np.sin(theta)])
        for theta in np.linspace(0, np.pi, n_per_step)
    ]
    for c in c_list:
        out = get_var(lcvx, prob, c, return_all=True)
        if out is not None:
            r, v, z, u, sigma = out
            vk = v[:2, lcvx.maxk]
            alt = r[2, lcvx.maxk]
            vertices3d.append([vk[0], vk[1], alt])
            vertices3d.append([vk[0], -vk[1], alt])
            vertices2d.append([np.linalg.norm(vk), alt])

    return vertices3d, vertices2d


def get_var(lcvx, prob, c, return_all=False):
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

    if return_all:
        return r, v, z, u, sigma
    else:
        return v[:, lcvx.maxk]


def v0_cstr(v: cp.Variable, v0_max=75.0, v0_max_angle=30 * np.pi / 180):
    cstr = []
    cstr.append(cp.norm(v[:, 0]) <= v0_max)
    cstr.append(cp.norm(v[0, 0] + v[1, 0]) <= -v[2, 0] * np.tan(v0_max_angle))
    return cstr


if __name__ == "__main__":
    # Simulation parameters
    rocket, N, tf, dt = config()

    # Initial conditions
    x0_min = np.array([0.0, 0.0, 1500.0, 0.0, 0.0, -75, np.log(rocket.mwet * 0.9)])
    x0_max = np.array([0.0, 0.0, 1500.0, 37.5, 0.0, 0.0, np.log(rocket.mwet * 1.1)])
    x0_bounds = (x0_min, x0_max)
    v0_max = 75.0
    v0_max_angle = 30 * np.pi / 180

    # Hyperparameters
    n_per_step = 25
    n_max_steps = 100

    # Problem parameters
    params_list = [
        {
            "maxk": int(N * i / n_max_steps),
            "rocket": rocket,
            "N": N,
            "x0_bounds": x0_bounds,
            "tf": tf,
            "v0_cstr": v0_cstr,
            "n_per_step": n_per_step,
        }
        for i in range(n_max_steps)
    ]

    # Solve reachable set problem
    # """
    with Pool() as p:
        results = p.map(solve_reachable_set, params_list)
    # """
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(solve_reachable_set, params_list))
    # """
    """
    results = []
    for params in tqdm(params_list):
        results.append(solve_reachable_set(params))
    """

    # Unpack results
    vertices3d = []
    vertices2d = []
    for result in results:
        vertices3d += result[0]
        vertices2d += result[1]

    # Plot reachable set

    vertices2d = np.array(vertices2d)
    visualize_convex_hull_2d(vertices2d)

    vertices3d = np.array(vertices3d)
    fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
    ax.scatter(vertices3d[:, 0], vertices3d[:, 1], vertices3d[:, 2], s=1)
    ax.set_xlabel("vx")
    ax.set_ylabel("vy")
    ax.set_zlabel("alt")
    # Limit axes for better visualization
    ax.set_xlim([np.min(vertices3d[:, 0]), np.max(vertices3d[:, 0])])
    ax.set_ylim([np.min(vertices3d[:, 1]), np.max(vertices3d[:, 1])])
    ax.set_zlim([np.min(vertices3d[:, 2]), np.max(vertices3d[:, 2])])
    plt.grid()
    plt.show()
