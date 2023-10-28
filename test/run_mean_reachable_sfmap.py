import numpy as np
import torch
import torch.nn as nn
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append("../")
from src import lcvx as lc
from src.nn_guidance import get_nn_reachset, ic2mean_safety
from src.learning import MLP
from config import slr_config


def solve_soft_landing(rocket, N, x0, tgo):
    lcvx = lc.LCvxMinFuel(rocket, N)
    prob = lcvx.problem(x0=x0, tf=tgo)
    prob.solve(verbose=False)
    sol = lc.get_vars(prob, ["X", "U"])
    X_sol = sol["X"]
    U_sol = sol["U"]
    r, v, z, u, sigma = lcvx.recover_variables(X_sol, U_sol)
    return r, v, z, u, sigma


def make_sfmap():
    sfmap = np.load("../saved/00000098.npy")
    sfmap[np.isnan(sfmap)] = 0.0
    sfmap = (sfmap - np.min(sfmap)) / (np.max(sfmap) - np.min(sfmap))
    nr, nc = sfmap.shape

    xmin, xmax = -500, 500
    ymin, ymax = -500, 500
    x = np.linspace(xmin, xmax, nc)
    y = np.linspace(ymin, ymax, nr)
    X, Y = np.meshgrid(x, y)

    sfmap_ = np.zeros((nr, nc, 3))
    sfmap_[:, :, 2] = sfmap
    sfmap_[:, :, 0] = X
    sfmap_[:, :, 1] = Y

    sfmap_[:, :, 2][(X > 0) * (Y > 0)] = 1.0

    sfmap_ = sfmap_.reshape(-1, 3)

    # torch
    sfmap_ = torch.from_numpy(sfmap_).float()
    return sfmap_, (nr, nc)


def main():
    # Parameters
    rocket, N = slr_config()
    x0 = np.array([0, 0, 1500, 5.0, 20.0, -30.0, np.log(rocket.mwet)])
    tgo = 60.0
    dt = tgo / N

    # Solve minimum fuel soft landing problem
    r, v, z, u, sigma = solve_soft_landing(rocket, N, x0, tgo)

    # Load NN model
    model = MLP(
        input_dim=5,  # alt, vx, vz, z, tgo
        output_dim=6,  # a1, a2, b1, b2, xmin, xmax
        hidden_layers=[32, 64, 32],
        activation_fn=nn.ReLU(),
        output_activation=nn.Sigmoid(),
    )
    model.load_state_dict(torch.load("../out/model.pth"))
    model.eval()

    dt = 1.0
    for i in range(0, N, 5):
        x0_i = torch.from_numpy(np.hstack((r[:, i], v[:, i], z[i]))).float()
        tgo_i = torch.tensor(tgo - i * dt).float()
        print("x0_i: ", x0_i)
        print("tgo_i: ", tgo_i)

        reach_set = get_nn_reachset(x0_i, tgo_i, model)
        reach_set = reach_set.detach().numpy()
        center = np.mean(reach_set[:, :2], axis=0)
        angles = np.arctan2(reach_set[:, 1] - center[1], reach_set[:, 0] - center[0])
        idx = np.argsort(angles)
        reach_set = reach_set[idx]

        sfmap, map_shape = make_sfmap()
        mean_safety, sfmap_reachable_mask = ic2mean_safety(
            x0_i, tgo_i, model, sfmap, border_sharpness=10
        )

        # """
        fig1, ax1 = plt.subplots()
        sfmap_reachable_mask = sfmap_reachable_mask.detach().numpy()
        sfmap_reachable_mask = sfmap_reachable_mask[::20]
        sf = sfmap[::20, :]
        ax1.scatter(
            sf[:, 0], sf[:, 1], c=sfmap_reachable_mask, s=0.5, cmap="jet", alpha=0.5
        )
        ax1.set_aspect("equal")
        ax1.set_xlabel("x [m]")
        ax1.set_ylabel("y [m]")
        plt.grid()
        plt.show()
        # """

        # Plot
        fig, ax = plt.subplots()

        # Safety map
        sfmap = sfmap.numpy()
        sfmap = sfmap[::20, :]
        ax.scatter(
            sfmap[:, 0], sfmap[:, 1], c=sfmap[:, 2], s=0.5, cmap="jet", alpha=0.5
        )

        # Reach set
        poly = plt.Polygon(reach_set, fill=False, linewidth=1.5)
        ax.add_patch(poly)

        # FOV
        fov_radius = r[2, i] * np.tan(rocket.fov / 2)
        fov = plt.Circle(
            (r[0, i], r[1, i]), fov_radius, fill=False, linewidth=1.5, linestyle="--"
        )
        ax.add_patch(fov)

        ax.set_title("Mean safety: {:.3f}".format(mean_safety))
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

        plt.grid()
        plt.show()


if __name__ == "__main__":
    main()
