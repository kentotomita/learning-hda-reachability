import numpy as np
import sys
import matplotlib.pyplot as plt

sys.path.append('../')
from config import slr_config
from fit_ellipses import reachset_model
np.set_printoptions(precision=1)


if __name__=="__main__":
    dataset = np.load('../out/reachset_fitted.npy')
    reachset = np.load('../out/reachset.npy')

    rocket, _ = slr_config()


    nc = int((reachset.shape[1] - 8)/2)

    num_invalid = 0

    # randamize dataset
    dataset = dataset[np.random.permutation(dataset.shape[0])]

    for i in range(min(len(dataset), 50)):
        data = dataset[i]
        idx = int(data[0])
        x0 = data[1:8]
        tgo = data[8]
        a1, a2, b1, b2, xmin, xmax = data[9:]
        rx = reachset[idx, 8::2]
        ry = reachset[idx, 9::2]
        # remove np.nan
        valid_idx = ~np.isnan(rx) & ~np.isnan(ry)
        rx = rx[valid_idx]
        ry = ry[valid_idx]

        fov_radius = x0[2] * np.tan(rocket.fov/2)
    
        fig, ax = plt.subplots()
        # FOV circle
        fov_circle = plt.Circle(x0[:2], fov_radius, fill=None, edgecolor='k', linestyle='--', label='Sensor Field of View')
        ax.add_patch(fov_circle)
        # reachable set
        ax.scatter(rx, ry, label=str(idx))
        # conic section
        xs = np.linspace(xmin, xmax, 100)
        ys = reachset_model(xs, a1, a2, b1, b2, xmin, xmax)
        ax.plot(xs, ys, color='r', linestyle='--', label='Conic Fit')

        ax.axis('equal')
        ax.legend()
        plt.title('a1: {:.2f}, a2: {:.2f}, b1: {:.2f}, b2: {:.2f}'.format(a1, a2, b1, b2))
        plt.grid()
        plt.show()

        
    