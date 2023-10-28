import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from tqdm import tqdm

import sys

sys.path.append("../")
from src import lcvx as lc
from src.nn_guidance import get_nn_reachset
from src.learning import NeuralReach
from config import slr_config

debug = False


def solve_soft_landing(rocket, N, x0, tgo):
    lcvx = lc.LCvxMinFuel(rocket, N)
    prob = lcvx.problem(x0=x0, tf=tgo)
    prob.solve(verbose=False)
    sol = lc.get_vars(prob, ["X", "U"])
    X_sol = sol["X"]
    U_sol = sol["U"]
    r, v, z, u, sigma = lcvx.recover_variables(X_sol, U_sol)
    return r, v, z, u, sigma


def main():
    # Parameters
    rocket, N = slr_config()
    x0 = np.array([0, 0, 1500, 5.0, 20.0, -30.0, np.log(rocket.mwet)])
    tgo = 60.0
    dt = tgo / N

    # Solve minimum fuel soft landing problem
    r, v, z, u, sigma = solve_soft_landing(rocket, N, x0, tgo)

    # Load NN model
    model = NeuralReach()
    model.load_state_dict(torch.load("../out/models/20230804_150021/model_20000.pth"))
    model.eval()

    # Compute NN reachset
    reach_sets = []
    for i in tqdm(range(N)):
        # reach_set = nn_reachset(r[:, i], v[:, i], z[i], tgo - i*dt, model, fov=rocket.fov)
        x0_i = torch.from_numpy(np.hstack((r[:, i], v[:, i], z[i]))).float()
        tgo_i = torch.tensor(tgo - i * dt).float()
        feasible, reach_set = get_nn_reachset(x0_i, tgo_i, model)
        reach_set = reach_set.detach().numpy()
        # sort reachset points by angle
        center = np.mean(reach_set[:, :2], axis=0)
        angles = np.arctan2(reach_set[:, 1] - center[1], reach_set[:, 0] - center[0])
        idx = np.argsort(angles)
        reach_sets.append(reach_set[idx])
    reach_sets = np.array(reach_sets)

    # Plot
    fig, ax = plt.subplots()
    for i, reach_set in enumerate(reach_sets):
        if i % 10 == 0:
            # reachability set
            poly = plt.Polygon(reach_set, fill=False, linewidth=0.5)
            ax.add_patch(poly)
            # FOV
            fov_radius = r[2, i] * np.tan(rocket.fov / 2)
            fov = plt.Circle(
                (r[0, i], r[1, i]),
                fov_radius,
                fill=False,
                linewidth=0.5,
                linestyle="--",
            )
            ax.add_patch(fov)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    fov_radius0 = x0[2] * np.tan(rocket.fov / 2)
    plt.xlim(-fov_radius0 * 1.2, fov_radius0 * 1.2)
    plt.ylim(-fov_radius0 * 1.2, fov_radius0 * 1.2)
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
