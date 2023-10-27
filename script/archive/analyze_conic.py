import numpy as np
import sys
import matplotlib.pyplot as plt

sys.path.append('../')
from config import slr_config
from fit_conic import extract_inner_points, conic

np.set_printoptions(precision=1)

if __name__=="__main__":
    dataset = np.load('../out/reachset_ellipse.npy')
    reachset = np.load('../out/reachset.npy')

    rocket, _ = slr_config()


    nc = int((reachset.shape[1] - 8)/2)

    num_invalid = 0

    for i in range(min(len(dataset), 30)):
        data = dataset[i]
        idx = int(data[0])
        x0 = data[1:8]
        tgo = data[8]
        a, b, c, d, e = data[9:]
        xy_coords = reachset[idx, 8:].reshape(nc, 2) 

        fov_radius = x0[2] * np.tan(rocket.fov/2)
    
        fig, ax = plt.subplots()
        # FOV circle
        fov_circle = plt.Circle(x0[:2], fov_radius, fill=None, edgecolor='k', linestyle='--', label='Sensor Field of View')
        ax.add_patch(fov_circle)
        # reachable set
        ax.scatter(xy_coords[:, 0], xy_coords[:, 1], label=str(idx))
        # points used for conic fitting
        xs, ys = extract_inner_points(xy_coords[:, 0], xy_coords[:, 1], fov_radius)
        ax.scatter(xs, ys, color='g', marker='x', label='Points used for fitting')
        # conic section
        x = np.linspace(np.min(xs), np.max(xs), 100)
        y = conic(x, a, b, c, d, e)
        ax.plot(x, y, color='r', linestyle='--', label='Conic Fit')

        ax.axis('equal')
        ax.legend()
        plt.title('a: {:.2f}, b: {:.2f}, c: {:.2f}, d: {:.2f}, e: {:.2f}'.format(a, b, c, d, e))
        plt.grid()
        plt.show()

        
    