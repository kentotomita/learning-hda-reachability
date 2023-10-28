import numpy as np
import sys
import matplotlib.pyplot as plt

sys.path.append("../")
from src.reachset import read_reachset

fov = 15 * np.pi / 180

np.set_printoptions(precision=1)

if __name__ == "__main__":
    _, _, reachset = read_reachset("../out/20230803_021318/reachset_sample.json")

    print(reachset.shape)

    num_invalid = 0
    # randamize reachset
    reachset = reachset[np.random.permutation(reachset.shape[0])]

    for i in range(min(len(reachset), 30)):
        data = reachset[i]
        x0 = data["x0"]
        tgo = data["tgo"]
        rc = data["rc"]
        c_xy = data["c"][:, :2].reshape(-1, 2)
        xy_coords = data["rf"][:, :2].reshape(-1, 2)

        print("idx: {}, x0: {}, tgo: {}".format(i, x0, tgo))

        fov_radius = x0[2] * np.tan(fov / 2)

        fig, ax = plt.subplots()
        fov_circle = plt.Circle(
            x0[:2],
            fov_radius,
            fill=None,
            edgecolor="k",
            linestyle="--",
            label="Sensor Field of View",
        )
        ax.add_patch(fov_circle)
        ax.scatter(rc[0], rc[1], label="center", marker="x", color="k")
        for i in range(len(c_xy)):
            ax.scatter(xy_coords[i, 0], xy_coords[i, 1], label=str(i))
            r = np.linalg.norm(xy_coords[i, :2] - rc[:2])
            ax.plot(
                [rc[0], rc[0] + c_xy[i, 0] * r],
                [rc[1], rc[1] + c_xy[i, 1] * r],
                label="c{}".format(i),
                linestyle="--",
            )
        ax.axis("equal")
        ax.legend()
        plt.show()

        if np.any(xy_coords[:, 1] < 0.0):
            num_invalid += 1

    print("Number of reachsets: ", len(reachset))
    print("Number of invalid reachsets: ", num_invalid)
