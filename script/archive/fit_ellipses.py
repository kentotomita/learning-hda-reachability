"""Fit ellipse to soft landing reachable set and save dataset"""
import numpy as np
import sys
import multiprocessing as mp
from tqdm import tqdm
import time
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import warnings
from scipy.optimize import OptimizeWarning
from numba import njit


sys.path.append("../")
from config import slr_config

debug = False
np.set_printoptions(precision=2)  # for debug
warnings.filterwarnings("ignore", category=OptimizeWarning)


def fit_ellipse(reachset: np.ndarray, fov: float, nmin=5, ic_idx0=0):
    """Fit ellipse to soft landing reachable set

    Args:
        reachset (np.ndarray): reachable set, shape: (n, nc * 2 + 1); [idx, x1, y1, x2, y2, ...]
        fov (float): field of view (rad)
        nmin (int): minimum number of points to fit ellipse

    Returns:
        out (list): list of [idx, x0, tgo, xc, a, b]
    """

    out = []
    for i in tqdm(range(len(reachset))):
        # load data
        x0 = reachset[i, :7]
        tgo = reachset[i, 7]
        rx = reachset[i, 8::2]
        ry = reachset[i, 9::2]

        # remove np.nan
        valid_idx = ~np.isnan(rx) & ~np.isnan(ry)
        rx = rx[valid_idx]
        ry = ry[valid_idx]

        # sort by x
        sort_idx = np.argsort(rx)
        rx = rx[sort_idx]
        ry = ry[sort_idx]

        # check if y < 0 exists; if so skip this IC
        if np.any(ry < 0):
            if debug:
                print(f"IC {i} has y < 0 as ry={ry}")
            continue
        # check if there are less than nmin points; if so skip this IC
        if len(rx) < nmin:
            if debug:
                print(f"IC {i} has less than {nmin} points; rx={rx}")
            continue

        fov_radius = x0[2] * np.tan(fov / 2)
        xmin, xmax = np.min(rx), np.max(rx)
        on_fov_circle = np.abs(np.sqrt(rx**2 + ry**2) - fov_radius) < 1e-2
        # If all points on FOV circle, skip fitting
        if np.argmin(on_fov_circle) == 0:
            a1, a2, b1, b2 = fov_radius, fov_radius, fov_radius, fov_radius

        # if there are some points on FOV circle, fit only left ellipse
        elif np.sum(on_fov_circle) > 1:
            # pick points off FOV circle and the first point on FOV circle from left
            first_on_fov_circle = np.argmin(rx[on_fov_circle])
            use_for_fit = np.arange(first_on_fov_circle + 1)
            # right boundary is FOV circle
            a2 = fov_radius
            b2 = fov_radius
            # initialize parameter
            argmax_y = np.argmax(ry)
            if argmax_y in on_fov_circle:
                a1_0 = (rx[argmax_y] - xmin) * 1.2
            else:
                a1_0 = rx[argmax_y] - xmin

            # b1 is a function of a1 because we know it intersects with the first point on FOV circle
            def func_b1(a1, xmin, x_first_on_fov_circle, y_first_on_fov_circle):
                xc = xmin + a1
                return np.sqrt(
                    y_first_on_fov_circle**2
                    / (1 - (x_first_on_fov_circle - xc) ** 2 / a1**2)
                )

            # fit left ellipse
            try:
                popt, _ = curve_fit(
                    f=lambda x, a1: reachset_model(
                        x,
                        a1,
                        a2,
                        func_b1(
                            a1, xmin, rx[first_on_fov_circle], ry[first_on_fov_circle]
                        ),
                        b2,
                        xmin=xmin,
                        xmax=xmax,
                    ),
                    xdata=rx[use_for_fit],
                    ydata=ry[use_for_fit],
                    p0=[a1_0],
                    bounds=([(rx[first_on_fov_circle] - xmin) / 2], [1e4]),
                )
            except (ValueError, RuntimeError):
                if debug:
                    print(f"IC {i} failed to fit with rx={rx}, ry={ry}")
                    fig, ax = plt.subplots()
                    ax.plot(rx, ry, "o")
                    plt.show()
                continue
            a1 = popt[0]
            b1 = func_b1(a1, xmin, rx[first_on_fov_circle], ry[first_on_fov_circle])
            if b1 is None or np.isnan(b1):
                print(a1, xmin, rx[first_on_fov_circle], first_on_fov_circle)
                continue

        # if there are no points on FOV circle other than x-max, fit both ellipses
        else:
            # initialize parameter
            argmax_y = np.argmax(ry)
            a1_0 = rx[argmax_y] - xmin
            a2_0 = xmax - rx[argmax_y]
            b1_0 = ry[argmax_y]
            b2_0 = ry[argmax_y]
            try:
                popt, _ = curve_fit(
                    f=lambda x, a1, a2, b1, b2: reachset_model(
                        x, a1, a2, b1, b2, xmin=xmin, xmax=xmax
                    ),
                    xdata=rx,
                    ydata=ry,
                    p0=[a1_0, a2_0, b1_0, b2_0],
                    bounds=([1e-3, 1e-3, 1e-3, 1e-3], [1e4, 1e4, 1e4, 1e4]),
                )
            except (ValueError, RuntimeError):
                if debug:
                    print(f"IC {i} failed to fit with rx={rx}, ry={ry}")
                    fig, ax = plt.subplots()
                    ax.plot(rx, ry, "o")
                    plt.show()
                continue
            a1, a2, b1, b2 = popt

        # store data
        out.append([ic_idx0 + i] + list(x0) + [tgo, a1, a2, b1, b2, xmin, xmax])

    return out


@njit
def reachset_model_old(X, a1, a2, b1, b2, xmin, xmax):
    """Reachset model using two ellipses"""
    eps = 1e-3
    yinf = (xmax - xmin) * 2
    n = len(X)
    Y = np.zeros(n)

    for i, x in enumerate(X):
        xc1 = xmin + a1
        xc2 = xmax - a2
        if -a1 <= x - xc1 <= a1:
            y1 = np.sqrt(b1**2 * (1 - (x - xc1) ** 2 / a1**2))
        if -a2 <= x - xc2 <= a2:
            y2 = np.sqrt(b2**2 * (1 - (x - xc2) ** 2 / a2**2))

        if x < xc1 and x > xc2:
            y = min(y1, y2)
        elif x < xc1 and x <= xc2:
            y = y1
        elif x >= xc1 and x > xc2:
            y = y2
        elif xc1 <= x <= xc1 + a1 and xc2 - a2 <= x <= xc2:
            y = max(y1, y2)
        else:
            y = yinf

        Y[i] = y
    return Y


def reachset_model(X, a1, a2, b1, b2, xmin, xmax):
    """Reachset model using two ellipses.
    Ellipse 1 account for left boundary of the reachable set; xc1 > xmin and runs through [xmin, 0] at left limit.
    Ellipse 2 account for right boundary of the reachable set; xc2 < xmax and runs through [xmax, 0] at right limit.
    """
    eps = 1e-9
    yinf = 1e6
    n = len(X)
    Y = np.zeros(n)

    # compute intersection
    xc1 = xmin + a1
    xc2 = xmax - a2
    # Ax^2 + Bx + C = 0
    A = b2**2 / a2**2 - b1**2 / a1**2
    B = 2 * (b1**2 / a1**2 * xc1 - b2**2 / a2**2 * xc2)
    C = (
        -(b1**2) / a1**2 * xc1**2
        + b2**2 / a2**2 * xc2**2
        + (b1**2 - b2**2)
    )
    D = B**2 - 4 * A * C
    if D < 0:  # no intersection
        for i, x in enumerate(X):
            y = 0.0
            if x < xc1 + a1:
                y = np.sqrt(b1**2 * (1 - (x - xc1) ** 2 / a1**2))
            else:
                y = np.sqrt(b2**2 * (1 - (x - xc2) ** 2 / a2**2))
            Y[i] = y
    else:
        if abs(A) < eps:
            if abs(xc1 - xc2) < eps:
                boundary = xc1
            else:
                const = -(
                    -(b1**2) / a1**2 * xc1**2 + b2**2 / a2**2 * xc2**2
                ) - (b1**2 - b2**2)
                coef = 2 * (b1**2 / a1**2 * xc1 - b2**2 / a2**2 * xc2)
                boundary = const / coef
        else:
            # Ax^2 + Bx + C = 0
            A = b2**2 / a2**2 - b1**2 / a1**2
            B = 2 * (b1**2 / a1**2 * xc1 - b2**2 / a2**2 * xc2)
            C = (
                -(b1**2) / a1**2 * xc1**2
                + b2**2 / a2**2 * xc2**2
                + (b1**2 - b2**2)
            )
            boundary = min((-B - np.sqrt(D)) / (2 * A), (-B + np.sqrt(D)) / (2 * A))

        # compute y
        for i, x in enumerate(X):
            if x < boundary:
                y = np.sqrt(b1**2 * (1 - (x - xc1) ** 2 / a1**2))
            else:
                y = np.sqrt(b2**2 * (1 - (x - xc2) ** 2 / a2**2))
            Y[i] = y
    return Y


def main(n_proc: int = 1, n_per_proc: int = None, nmax: int = None, nmin_fit: int = 3):
    """
    Args:
        n_proc (int): number of processes
        n_per_proc (int): number of samples per process
        nmax (int): maximum number of samples
    """

    # ----------------- load -----------------
    rocket, _ = slr_config()
    reachset = np.load("../out/reachset.npy")  # reachable set, shape: (n, nc * 2 + 1)

    if nmax is not None:
        reachset = reachset[:nmax]

    n_ic = reachset.shape[0]

    # ----------------- ellipse fitting -----------------
    if n_proc > 1:
        # prepare args
        args = []
        for i in range(0, n_ic, n_per_proc):
            if i + n_per_proc < n_ic:
                args.append((reachset[i : i + n_per_proc], rocket.fov, nmin_fit, i))
            else:
                args.append((reachset[i:], rocket.fov, nmin_fit, i))

        # solve
        with mp.Pool(processes=n_proc) as p:
            data = []
            for out in tqdm(p.starmap(fit_ellipse, args), total=len(args)):
                data += out

    else:
        data = fit_ellipse(reachset, rocket.fov, nmin_fit, 0)

    # sort by IC index
    data = np.array(data)
    data = data[data[:, 0].argsort()]

    # ----------------- save -----------------
    n_valid = len(data)
    n_total = n_ic
    np.save("../out/reachset_fitted.npy", data)
    print(f"Valid samples: {n_valid}/{n_total} ({n_valid/n_total*100:.2f}%)")


if __name__ == "__main__":
    n_per_proc = 1000
    n_proc = 27
    nmax = 1000

    start = time.time()
    main(n_proc, n_per_proc, nmax)
    print("--- %.3f minutes ---" % ((time.time() - start) / 60))
