"""Utility functions for sampling initial conditions from controllable set."""
import numpy as np
import numba
from scipy.optimize import linprog
import time
from tqdm import tqdm


@numba.njit
def inside_hull(point, simplex_eqs):
    """Check if a point is inside a convex hull defined by a set of linear equations.

    Args:
        point (np.ndarray): Point to check. (n_dim,)
        simplex_eqs (list): List of linear equations defining the convex hull. (n_points, n_dim).  Ax + b <= 0.

    """
    for eq in simplex_eqs:
        if not (np.dot(eq[:-1], point) + eq[-1] <= 0):
            return False
    return True


def random_sampling_in_hull(simplex_eqs, bounds, n_samples, seed=0):
    """Sample random points in a convex hull; a hull is defined by a set of linear equations.

    Args:
        simplex_eqs (list): List of linear equations defining the convex hull. (n_points, n_dim).  Ax + b <= 0.
        bounds (tuple): Bounds of the convex hull. (lb, ub)
        n_samples (int): Number of samples to generate.
    """
    lb, ub = bounds
    samples = np.empty((n_samples, len(lb)))
    rng = np.random.RandomState(seed=seed)
    i = 0
    while i < n_samples:
        #random_point = lb + (ub - lb) * np.random.random(size=lb.shape)
        random_point = lb + (ub - lb) * rng.random(size=lb.shape)
        if inside_hull(random_point, simplex_eqs):
            samples[i] = random_point
            i += 1

    return samples


def random_sampling_outside_hull(simplex_eqs, bounds, n_samples, seed=0):
    """Sample random points outside a convex hull; a hull is defined by a set of linear equations.

    Args:
        simplex_eqs (list): List of linear equations defining the convex hull. (n_points, n_dim).  Ax + b <= 0.
        bounds (tuple): Bounds of the convex hull. (lb, ub)
        n_samples (int): Number of samples to generate.
    """
    lb, ub = bounds
    samples = np.empty((n_samples, len(lb)))
    rng = np.random.RandomState(seed=seed)
    pbar = tqdm(total=n_samples)
    i = 0
    while i < n_samples:
        #random_point = lb + (ub - lb) * np.random.random(size=lb.shape)
        random_point = lb + (ub - lb) * rng.random(size=lb.shape)
        if not inside_hull(random_point, simplex_eqs):
            samples[i] = random_point
            i += 1
            pbar.update(1)

    return samples


def structured_sample_points_in_convex_hull(hull, n_per_dim, points, buffer=0.1):
    """Samples points within a convex hull by partitioning the hull for each dimension.
    The subsequent partitions are made within the bounds of the subspace defined by the previous partitions.

    Args:
        hull (scipy.spatial.ConvexHull): Convex hull.
        n_per_dim (int): Number of partitions per dimension.
        points (np.ndarray): Points defining the convex hull. (n_points, n_dim)

    Returns:
        np.ndarray: Sampled points within the convex hull.
    """
    simplex_eqs = hull.equations
    n_dim = points.shape[1]

    # Compute the bounds for each dimension
    def get_bounds(dim, fixed_values):
        c = np.zeros(n_dim)
        c[dim] = 1  # Objective function to maximize/minimize the current dimension

        A_ub = simplex_eqs[:, :-1]
        b_ub = -simplex_eqs[:, -1]

        # Setting equality constraints for fixed dimensions
        A_eq = np.zeros((len(fixed_values), n_dim))
        b_eq = np.zeros(len(fixed_values))
        for idx, (d, val) in enumerate(fixed_values.items()):
            A_eq[idx, d] = 1
            b_eq[idx] = val

        # Linear programming to find min and max for current dimension
        res_min = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(None, None))
        res_max = linprog(-c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(None, None))

        if res_min.success and res_max.success:
            return res_min.fun, -res_max.fun
        else:
            return None, None

    def add_buffer(bounds, buffer):
        lb, ub = bounds
        scale = (ub - lb) * buffer
        lb += scale / 2
        ub -= scale / 2
        return lb, ub

    # Sampling points within the convex hull
    samples = []
    start = time.time()
    for dim in range(n_dim):
        if dim == 0:
            min_val, max_val = np.min(points[:, dim]), np.max(points[:, dim])
            min_val, max_val = add_buffer((min_val, max_val), buffer)
            sample_points = np.linspace(min_val, max_val, n_per_dim)
            samples = [[val] for val in sample_points]
        else:
            new_samples = []
            for fixed_values in samples:
                fixed_dict = {i: fixed_values[i] for i in range(len(fixed_values))}
                min_val, max_val = get_bounds(dim, fixed_dict)
                min_val, max_val = add_buffer((min_val, max_val), buffer)
                if min_val is not None and max_val is not None:
                    for point in np.linspace(min_val, max_val, n_per_dim):
                        new_samples.append(fixed_values + [point])

                # len(samples)/n_per_dim ** n_dim is the current progress
                elapsed = time.time() - start
                n_samples = max(len(samples), len(new_samples))
                t_per_sample = elapsed / n_samples
                t_remaining = (n_per_dim ** n_dim - n_samples) * t_per_sample
                t_total = elapsed + t_remaining
                print(f"{n_samples / n_per_dim ** n_dim * 100:.2f}% completed. {t_remaining/60:.2f}m remaining ({t_total/60:.2f}m)", end="\r")

            samples = new_samples

    return np.array(samples)
