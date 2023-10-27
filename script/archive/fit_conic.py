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


sys.path.append('../')
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
    for i in range(len(reachset)):
        # load data
        x0 = reachset[i, :7]
        tgo = reachset[i, 7]
        rx = reachset[i, 8::2]
        ry = reachset[i, 9::2]

        # remove np.nan
        valid_idx = ~np.isnan(rx) & ~np.isnan(ry)
        rx = rx[valid_idx]
        ry = ry[valid_idx]

        # check if y < 0 exists; if so skip this IC
        if np.any(ry < 0):
            if debug:
                print(f'IC {i} has y < 0 as ry={ry}')
            continue

        else:
            # Extract points inside the circle and the first point on the circle.
            # If all points on the circle, return all points.
            fov_radius = x0[2] * np.tan(fov/2)
            rx, ry = extract_inner_points(rx, ry, fov)

            # check if there are less than nmin points; if so skip this IC
            if len(rx) < nmin:
                if debug:
                    print(f'IC {i} has less than {nmin} points; rx={rx}')
                continue
            
            try:
                popt, _ = curve_fit(
                    f=conic, 
                    xdata=rx, 
                    ydata=ry,
                )
            except RuntimeError:
                if debug:
                    print(f'IC {i} failed to fit with rx={rx}, ry={ry}')
                    fig, ax = plt.subplots()
                    ax.plot(rx, ry, 'o')
                    plt.show()
                continue
 
            a, b, c, d, e = popt

            # store data
            out.append([ic_idx0 + i] + list(x0) + [tgo, a, b, c, d, e])

    return out

def extract_inner_points(rx, ry, fov_radius):
    """Extract points inside the circle and the first point on the circle.
    If all points on the circle, return all points.
    """
    # sort by x in ascending order
    indices = np.argsort(rx)
    rx = rx[indices]
    ry = ry[indices]

    # find the first point inside the circle 
    eps = fov_radius * 1e-2
    inner_idx =  rx**2 + ry **2 <= (fov_radius - eps)**2
    if np.any(inner_idx):
        last_inner_idx = np.where(inner_idx)[0][-1]
        rx = rx[:last_inner_idx+2]
        ry = ry[:last_inner_idx+2]
    return rx, ry


def conic(x, a, b, c, d, e):
    return a * x + b * np.sqrt(c * x ** 2 + d * x + e)


def main(n_proc:int=1, n_per_proc:int=None, nmax:int=None, nmin_fit:int=3):
    """
    Args: 
        n_proc (int): number of processes
        n_per_proc (int): number of samples per process
        nmax (int): maximum number of samples
    """

    # ----------------- load -----------------
    rocket, _ = slr_config()
    reachset = np.load('../out/reachset.npy')  # reachable set, shape: (n, nc * 2 + 1)

    if nmax is not None:
        reachset = reachset[:nmax]

    n_ic = reachset.shape[0]

    # ----------------- ellipse fitting -----------------
    if n_proc > 1:
        # prepare args
        args = []
        for i in range(0, n_ic, n_per_proc):
            if i + n_per_proc < n_ic:
                args.append((reachset[i:i+n_per_proc], rocket.fov, nmin_fit, i))
            else:
                args.append((reachset[i:], rocket.fov, nmin_fit, i))
        
        # solve
        with mp.Pool(processes=n_proc) as p:
            data  = []
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
    np.save('../out/reachset_ellipse.npy', data)
    print(f'Valid samples: {n_valid}/{n_total} ({n_valid/n_total*100:.2f}%)')


if __name__ == '__main__':
    n_per_proc = 10
    n_proc = 1
    nmax = 100

    start = time.time()
    main(n_proc, n_per_proc, nmax)
    print("--- %.3f minutes ---" % ((time.time() - start)/60))