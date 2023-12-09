from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


def load_sfmap(
    path: str,
    x_range: Tuple,
    y_range: Tuple,
    normalize: bool = True,
):
    """Load safety map from path"""
    sfgrid = np.load(path)
    sfgrid[np.isnan(sfgrid)] = 0.0
    if normalize:
        sfgrid = (sfgrid - np.min(sfgrid)) / (np.max(sfgrid) - np.min(sfgrid))

    nr, nc = sfgrid.shape
    xmin, xmax = x_range
    ymin, ymax = y_range
    x = np.linspace(xmin, xmax, nc)
    y = np.linspace(ymin, ymax, nr)
    X, Y = np.meshgrid(x, y)

    sfmap = np.zeros((nr, nc, 3))
    sfmap[:, :, 0] = X
    sfmap[:, :, 1] = Y
    sfmap[:, :, 2] = sfgrid

    return sfmap, (nr, nc)


def make_simple_sfmap(x_range, y_range, n_points):
    """Make safety map from scratch"""

    xmin, xmax = x_range
    ymin, ymax = y_range
    x = np.linspace(xmin, xmax, n_points)
    y = np.linspace(ymin, ymax, n_points)
    X, Y = np.meshgrid(x, y)

    sfmap = np.zeros((n_points, n_points, 3))
    sfmap[:, :, 0] = X
    sfmap[:, :, 1] = Y
    sfmap[:, :, 2] = X + Y 
    sfmap[:, :, 2] = (sfmap[:, :, 2] - np.min(sfmap[:, :, 2])) / (np.max(sfmap[:, :, 2]) - np.min(sfmap[:, :, 2]))
    sfmap[:, :, 2][X > 500] = 0.0
    sfmap[:, :, 2][Y > 500] = 0.0
    sfmap[:, :, 2][X < -500] = 0.0
    sfmap[:, :, 2][Y < -500] = 0.0
    sfmap = sfmap.reshape(-1, 3)

    return sfmap, (n_points, n_points)


def visualize_sfmap(sfmap: np.ndarray):
    """Visualize safety map.
    
    Args:
        sfmap (np.ndarray): safety map; shape (N, M); contains safety values.
    """
    n = int(np.sqrt(sfmap.shape[0]))

    x = sfmap[:, 0].reshape(n, n)
    y = sfmap[:, 1].reshape(n, n)
    safety = sfmap[:, 2].reshape(n, n)
    
    fig, ax = plt.subplots(figsize=(8, 6))

    # Safety Map
    safety_img = ax.pcolormesh(x, y, safety, shading='auto', cmap='gray')
    plt.colorbar(safety_img, ax=ax, orientation='vertical', label='Safety Level')

    ax.set_title('Safety Map with Reachability Overlay')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    plt.tight_layout()
    plt.show()