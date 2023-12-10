from typing import Tuple
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage.filters import gaussian_filter


class DynamicSafetyMap:
    """Dynamic safety map class."""

    def __init__(self, x_range: Tuple, y_range: Tuple, npoints: int, relative_path: str = "."):
        """Initialize dynamic safety map.

        Args:
            npoints (int): number of points in x and y directions.
        """
        self.npoints = npoints

        xmin, xmax = x_range
        ymin, ymax = y_range
        x = np.linspace(xmin, xmax, npoints)
        y = np.linspace(ymin, ymax, npoints)
        self.X, self.Y = np.meshgrid(x, y)

        self.sfmap_dir = os.path.join(relative_path, "saved/safetymap")
        self.alt0_fname = "truth.npy"
        self.alt1_fname = "gsd_150.npy"
        self.alt2_fname = "gsd_200.npy"
        self.alt3_fname = "gsd_300.npy"

        self.alt0 = 0.0
        self.alt1 = 500.0
        self.alt2 = 750.0
        self.alt3 = 1500.0

        self.sfmap_alt0 = np.load(os.path.join(self.sfmap_dir, self.alt0_fname))
        self.sfmap_alt1 = np.load(os.path.join(self.sfmap_dir, self.alt1_fname))
        self.sfmap_alt2 = np.load(os.path.join(self.sfmap_dir, self.alt2_fname))
        self.sfmap_alt3 = np.load(os.path.join(self.sfmap_dir, self.alt3_fname))

    def resize_sfmap(self, sfmap: np.ndarray, nr: int, nc: int) -> np.ndarray:
        # Convert the array to an image
        original_image = Image.fromarray(np.uint8(sfmap * 255))
        # Resize the image to (1024, 1024)
        resized_image = original_image.resize((nr, nc), Image.BILINEAR)
        # Convert the resized image back to an array
        resized_array = np.asarray(resized_image) / 255.0
        return resized_array

    def get_sfmap(self, alt: float) -> np.ndarray:
        """Get safety map at the specified altitude.

        Args:
            alt (float): altitude.

        Returns:
            np.ndarray: safety map; shape (N, M, 3); each row is [x, y, safety].
        """
        assert alt >= self.alt0 and alt <= self.alt3
        if alt <= self.alt1:
            # interpolate between alt0 and alt1
            alpha = (alt - self.alt0) / (self.alt1 - self.alt0)
            sfmap = self.sfmap_alt0 * (1 - alpha) + self.sfmap_alt1 * alpha
        elif alt < self.alt2:
            # interpolate between alt1 and alt2
            alpha = (alt - self.alt1) / (self.alt2 - self.alt1)
            sfmap = self.sfmap_alt1 * (1 - alpha) + self.sfmap_alt2 * alpha
        elif alt <= self.alt3:
            # interpolate between alt2 and alt3
            alpha = (alt - self.alt2) / (self.alt3 - self.alt2)
            sfmap = self.sfmap_alt2 * (1 - alpha) + self.sfmap_alt3 * alpha

        # resize the safety map
        sfmap = self.resize_sfmap(sfmap, self.npoints, self.npoints)

        # smooth the safety map using scipy.ndimage.filters.gaussian_filter
        sfmap = gaussian_filter(sfmap, sigma=3.0)

        # reshape the safety map into (N, 3)
        sfmap_ = np.zeros((self.npoints, self.npoints, 3))
        sfmap_[:, :, 0] = self.X
        sfmap_[:, :, 1] = self.Y
        sfmap_[:, :, 2] = sfmap
        sfmap = sfmap_.reshape(-1, 3)
        return sfmap
    
    def get_grid_sfmap(self, sfmap: np.ndarray) -> np.ndarray:
        """Get safety map at the specified altitude.

        Args:
            alt (float): altitude.

        Returns:
            np.ndarray: safety map; shape (N, M); each row is [x, y, safety].
        """
        return sfmap.reshape(self.npoints, self.npoints, 3)


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