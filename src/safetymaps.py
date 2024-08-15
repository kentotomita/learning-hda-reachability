from typing import Tuple, List
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
import random


class SafetyMap:
    """Base class for safety map class."""
    def __init__(self, x_range: Tuple, y_range: Tuple, npoints: int):
        self.x_range = x_range
        self.y_range = y_range
        self.npoints = npoints

    def get_sfmap(self, alt: float) -> np.ndarray:
        """Get safety map at the specified altitude.

        Args:
            alt (float): altitude.

        Returns:
            np.ndarray: safety map; shape (N, M, 3); each row is [x, y, safety].
        """
        raise NotImplementedError
    
    def get_grid_sfmap(self, sfmap: np.ndarray) -> np.ndarray:
        """Get safety map at the specified altitude.

        Args:
            alt (float): altitude.

        Returns:
            np.ndarray: safety map; shape (N, M); each row is [x, y, safety].
        """
        return sfmap.reshape(self.npoints, self.npoints, 3)
    

class StaticSafetyMap(SafetyMap):
    """Static safety map class."""
    
    def __init__(self, x_range: Tuple, y_range: Tuple, npoints: int, sfmap: np.ndarray=None):
        """Initialize static safety map.

        Args:
            npoints (int): number of points in x and y directions.
        """
        super().__init__(x_range, y_range, npoints)

        if sfmap is None:
            sfmap, _ = make_simple_sfmap(x_range, y_range, npoints)

        self.sfmap = sfmap

    def get_sfmap(self, alt: float) -> np.ndarray:
        """Get safety map at the specified altitude.

        Args:
            alt (float): altitude.

        Returns:
            np.ndarray: safety map; shape (N, M, 3); each row is [x, y, safety].
        """
        return self.sfmap
    

class StaticNoisedSafetyMap(SafetyMap):
    """Static safety map class."""
    
    def __init__(self, x_range: Tuple, y_range: Tuple, npoints: int, sfmap: np.ndarray=None):
        """Initialize static safety map.

        Args:
            npoints (int): number of points in x and y directions.
        """
        super().__init__(x_range, y_range, npoints)

        if sfmap is None:
            sfmap, _ = make_simple_sfmap(x_range, y_range, npoints)

        self.sfmap = sfmap

    def get_sfmap(self, alt: float) -> np.ndarray:
        """Get safety map at the specified altitude.

        Args:
            alt (float): altitude.

        Returns:
            np.ndarray: safety map; shape (N, M, 3); each row is [x, y, safety].
        """
        n_noise = max(1, min(10, int(self.sfmap.shape[0] * 0.001)))
        random_indices = np.random.choice(self.sfmap.shape[0], n_noise, replace=False)
        self.sfmap[random_indices, 2] = 1.0
        
        return self.sfmap
    

class DesignedSafetyMap(SafetyMap):
    """Designed safety map class."""
    
    def __init__(self, x_range: Tuple, y_range: Tuple, npoints: int, n_hazards: List[int], hazard_sizes: List[int]):
        """Initialize designed safety map.

        Args:
            npoints (int): number of points in x and y directions.
        """
        super().__init__(x_range, y_range, npoints)

        xmin, xmax = x_range
        ymin, ymax = y_range
        x = np.linspace(xmin, xmax, npoints)
        y = np.linspace(ymin, ymax, npoints)
        self.X, self.Y = np.meshgrid(x, y)

        sfmap = np.zeros((npoints, npoints, 3))
        sfmap[:, :, 0] = self.X
        sfmap[:, :, 1] = self.Y
        
        assert len(n_hazards) == len(hazard_sizes), "Number of hazards and sizes do not match."
        n_level = len(n_hazards)
        self.alt_levels = np.linspace(0, 1500, n_level)
        self.hazard_maps = []
        for i in range(n_level):
            self.hazard_maps.append(generate_binary_disk_array((npoints, npoints), n_hazards[i], hazard_sizes[i]))

        self.safety_maps = []
        for i in range(n_level):
            #safety = 0.5 + (n_level - i) / n_level * 0.5
            safety = 1.0
            for j in range(n_level - i):
                safety *= (1 - self.hazard_maps[j])
            self.safety_maps.append(safety)


    def get_sfmap(self, alt: float) -> np.ndarray:
        """Get safety map at the specified altitude.

        Args:
            alt (float): altitude.

        Returns:
            np.ndarray: safety map; shape (N, M, 3); each row is [x, y, safety].
        """
        if alt <= self.alt_levels[0]:
            safety = self.safety_maps[0]
        elif alt >= self.alt_levels[-1]:
            safety = self.safety_maps[-1]
        else:
            for i in range(1, len(self.alt_levels)):
                if alt < self.alt_levels[i]:
                    c = (alt - self.alt_levels[i-1]) / (self.alt_levels[i] - self.alt_levels[i-1])
                    safety = self.safety_maps[i-1] * (1 - c) + self.safety_maps[i] * c
                    break

        sfmap = np.zeros((self.npoints, self.npoints, 3))
        sfmap[:, :, 0] = self.X
        sfmap[:, :, 1] = self.Y
        sfmap[:, :, 2] = safety

        return sfmap.reshape(-1, 3)
    

class DynamicSafetyMap(SafetyMap):
    """Dynamic safety map class."""

    def __init__(self, x_range: Tuple, y_range: Tuple, npoints: int, relative_path: str = "."):
        """Initialize dynamic safety map.

        Args:
            npoints (int): number of points in x and y directions.
        """
        super().__init__(x_range, y_range, npoints)

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
        eps = 1e-6
        assert alt >= self.alt0 - eps and alt <= self.alt3 + eps, f"Altitude {alt} out of range [{self.alt0}, {self.alt3}]"
        if alt <= self.alt1:
            # interpolate between alt0 and alt1
            alpha = (alt - self.alt0) / (self.alt1 - self.alt0)
            sfmap = self.sfmap_alt0 * (1 - alpha) + self.sfmap_alt1 * alpha
        elif alt < self.alt2:
            # interpolate between alt1 and alt2
            alpha = (alt - self.alt1) / (self.alt2 - self.alt1)
            sfmap = self.sfmap_alt1 * (1 - alpha) + self.sfmap_alt2 * alpha
        else:
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
    #sfmap[:, :, 2] = 1 - np.exp(-np.abs(2 * (2 * X + Y) / (xmax + ymax)))
    x_best = 0.8 * xmax + 0.2 * xmin
    y_best = 0.6 * ymax + 0.4 * ymin
    mu = np.array([x_best, y_best])
    cov = np.array([[5, 1], [1, 3]]) * max(xmax - xmin, ymax - ymin) * 5
    inv_cov = np.linalg.inv(cov)
    sfmap[:, :, 2] = np.exp(-np.sum((np.dstack((X, Y)) - mu) @ inv_cov * (np.dstack((X, Y)) - mu), axis=2))
    sfmap[:, :, 2] = (sfmap[:, :, 2] - np.min(sfmap[:, :, 2])) / (np.max(sfmap[:, :, 2]) - np.min(sfmap[:, :, 2]))
    sfmap[:, :, 2][X > 500] = 0.0
    sfmap[:, :, 2][Y > 500] = 0.0
    sfmap[:, :, 2][X < -500] = 0.0
    sfmap[:, :, 2][Y < -500] = 0.0

    x_safe, y_safe = -50.0, -20.0
    sfmap[:, :, 2][(X > x_safe - 5) * (Y > y_safe - 5) * (X < x_safe + 5) * (Y < y_safe + 5)] = 0.9

    plt.figure()
    plt.pcolormesh(X, Y, sfmap[:, :, 2], shading='auto', cmap='gray', vmin=0, vmax=1)
    plt.colorbar()
    plt.show()

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


def generate_binary_disk_array(array_size, num_disks, disk_diameter):
    """
    Generates a 2D binary array with specified number of binary disks.

    Parameters:
    - array_size: Tuple of (height, width) of the binary array.
    - num_disks: Number of binary disks to be placed in the array.
    - disk_diameter: Diameter of each disk.

    Returns:
    - A 2D binary numpy array.
    """
    height, width = array_size
    array = np.zeros((height, width), dtype=int)
    
    radius = disk_diameter // 2

    def is_valid_position(x, y):
        """Check if the disk can be placed at the position (x, y) without overlapping."""
        for i in range(max(0, x - radius), min(height, x + radius + 1)):
            for j in range(max(0, y - radius), min(width, y + radius + 1)):
                if array[i, j] == 1 and (i - x)**2 + (j - y)**2 <= radius**2:
                    return False
        return True

    def place_disk(x, y):
        """Place a disk centered at (x, y)."""
        for i in range(max(0, x - radius), min(height, x + radius + 1)):
            for j in range(max(0, y - radius), min(width, y + radius + 1)):
                if (i - x)**2 + (j - y)**2 <= radius**2:
                    array[i, j] = 1

    placed_disks = 0
    attempts = 0
    max_attempts = 10000

    while placed_disks < num_disks and attempts < max_attempts:
        x = random.randint(radius, height - radius - 1)
        y = random.randint(radius, width - radius - 1)
        
        if True: #is_valid_position(x, y):
            place_disk(x, y)
            placed_disks += 1
        attempts += 1

    if attempts >= max_attempts:
        print("Warning: Max attempts reached. Not all disks may have been placed.")

    return array


class DynamicSafetyMap2(SafetyMap):
    """Dynamic safety map class."""

    def __init__(self, x_range: Tuple, y_range: Tuple, npoints: int, relative_path: str = "."):
        """Initialize dynamic safety map.

        Args:
            npoints (int): number of points in x and y directions.
        """
        super().__init__(x_range, y_range, npoints)

        xmin, xmax = x_range
        ymin, ymax = y_range
        x = np.linspace(xmin, xmax, npoints)
        y = np.linspace(ymin, ymax, npoints)
        #self.X, self.Y = np.meshgrid(x, y)
        self.Y, self.X = np.meshgrid(y, x)

        self.sfmap_dir = os.path.join(relative_path, "saved/sfmap_shd_based")
        self.alt0_fname = "sfmap_data_truth.npz"
        #self.alt0_fname = "sfmap_data_0.npz"
        self.alt1_fname = "sfmap_data_1.npz"
        self.alt2_fname = "sfmap_data_2.npz"
        self.alt3_fname = "sfmap_data_3.npz"

        self.alt0 = 0.0
        self.alt1 = 500.0
        self.alt2 = 750.0
        self.alt3 = 1500.0

        self.sfmap_alt0 = np.load(os.path.join(self.sfmap_dir, self.alt0_fname))['site_safe']
        self.sfmap_alt1 = np.load(os.path.join(self.sfmap_dir, self.alt1_fname))['site_safe']
        self.sfmap_alt2 = np.load(os.path.join(self.sfmap_dir, self.alt2_fname))['site_safe']
        self.sfmap_alt3 = np.load(os.path.join(self.sfmap_dir, self.alt3_fname))['site_safe']

        # resize the safety map
        self.sfmap_alt0 = self.resize_sfmap(self.sfmap_alt0, self.npoints, self.npoints)
        self.sfmap_alt1 = self.resize_sfmap(self.sfmap_alt1, self.npoints, self.npoints)
        self.sfmap_alt2 = self.resize_sfmap(self.sfmap_alt2, self.npoints, self.npoints)
        self.sfmap_alt3 = self.resize_sfmap(self.sfmap_alt3, self.npoints, self.npoints)

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
        eps = 1e-6
        assert alt >= self.alt0 - eps and alt <= self.alt3 + eps, f"Altitude {alt} out of range [{self.alt0}, {self.alt3}]"
        if alt <= self.alt1:
            # interpolate between alt0 and alt1
            alpha = (alt - self.alt0) / (self.alt1 - self.alt0)
            sfmap = self.sfmap_alt0 * (1 - alpha) + self.sfmap_alt1 * alpha
        elif alt < self.alt2:
            # interpolate between alt1 and alt2
            alpha = (alt - self.alt1) / (self.alt2 - self.alt1)
            sfmap = self.sfmap_alt1 * (1 - alpha) + self.sfmap_alt2 * alpha
        else:
            # interpolate between alt2 and alt3
            alpha = (alt - self.alt2) / (self.alt3 - self.alt2)
            sfmap = self.sfmap_alt2 * (1 - alpha) + self.sfmap_alt3 * alpha

        # resize the safety map
        sfmap = self.resize_sfmap(sfmap, self.npoints, self.npoints)

        # smooth the safety map using scipy.ndimage.filters.gaussian_filter
        sfmap = gaussian_filter(sfmap, sigma=1.0)

        # reshape the safety map into (N, 3)
        sfmap_ = np.zeros((self.npoints, self.npoints, 3))
        sfmap_[:, :, 0] = self.X
        sfmap_[:, :, 1] = self.Y
        sfmap_[:, :, 2] = sfmap
        sfmap = sfmap_.reshape(-1, 3)
        return sfmap