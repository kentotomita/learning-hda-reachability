"""Generate reachable set data for training; only a few representative points are computed."""
import cvxpy as cp
import argparse
import multiprocessing as mp
import numpy as np
import os
import datetime
import pickle
import time
from tqdm import tqdm
from typing import List
import sys
sys.path.append('..')
import src.lcvx as lc


def main():
    start = time.time()

    # read command line inputs
    parser  = argparse.ArgumentParser()
    parser.add_argument('--n_proc', type=int, default=8)
    parser.add_argument('--tgo_round', type=float, default=0.1)
    parser.add_argument('--n_datamax', type=int, default=int(1e10))
    args = parser.parse_args()

    # load initial conditions
    ic_data_random = np.load('saved/controllable_set/ic_set/random_samples.npy')
    ic_data_structured = np.load('saved/controllable_set/ic_set/structured_samples.npy')
    # load picked lander
    with open('saved/controllable_set/lander.pkl', 'rb') as f:
        lander = pickle.load(f)
    N = 100

    # Truncate
    if ic_data_random.shape[0] > args.n_datamax:
        ic_data_random = ic_data_random[:args.n_datamax, :]
    if ic_data_structured.shape[0] > args.n_datamax:
        ic_data_structured = ic_data_structured[:args.n_datamax, :]

    # Solve
    data_random = solve_parallel(ic_data_random, N, lander, args.tgo_round, args.n_proc)
    data_structured = solve_parallel(ic_data_structured, N, lander, args.tgo_round, args.n_proc)

    # Save
    dtstring = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join('saved/controllable_set/reachset_train/', dtstring)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, 'data_random.npy'), data_random)
    np.save(os.path.join(out_dir, 'data_structured.npy'), data_structured)
    data_header = ['h', 'vx', 'vz', 'm', 'tgo', 'rf_xmax', 'rf_xmin', 'rf_ymax']
    with open(os.path.join(out_dir, 'data_header.txt'), 'w') as f:
        f.write('\n'.join(data_header))

    # save meta data; command line inputs and time elapsed
    with open(os.path.join(out_dir, 'meta.txt'), 'w') as f:
        f.write('n_proc: {}\n'.format(args.n_proc))
        f.write('tgo_round: {}\n'.format(args.tgo_round))
        f.write('n_datamax: {}\n'.format(args.n_datamax))
        f.write('time_elapsed: {:.2f} m\n'.format((time.time() - start) / 60))

    print('Total time: {:.2f} m'.format((time.time() - start) / 60))


def solve_parallel(ic_data: np.ndarray, N: int, lander: lc.Rocket, tgo_round: float, n_proc: int):
    """Solve the convex program for given initial conditions in parallel.

    Args:
        ic_data: initial conditions; each row is (h, vx, vz, m, tgo). Shape: (n, 5)
        N: number of time steps
        lander: Rocket object
        tgo_round: rounding value for tgo
        n_proc: number of processes

    Returns:
        data: data points; each row is (h, vx, vz, m, tgo, rf(xmax), rf(xmin), rf(ymax)). Shape: (n, 11)
    """

    print('Preparing data...')
    # Round tgo and sort
    ic_data[:, 4] = np.round(ic_data[:, 4] / tgo_round) * tgo_round
    ic_data = ic_data[np.argsort(ic_data[:, 4])]  # sort by tgo in ascending order

    # Split data into chunks with the same tgo
    tgo_unique = np.unique(ic_data[:, 4])
    ic_data_split = []
    for tgo in tgo_unique:
        ic_data_split.append(ic_data[ic_data[:, 4] == tgo, :])

    # Print info
    print('Number of initial conditions: {}'.format(ic_data.shape[0]))
    print('Number of unique tgo: {}'.format(tgo_unique.shape[0]))

    # Prepare arguments
    args = []
    for ic_chunk in ic_data_split:
        args.append((ic_chunk, lander, N))

    # Solve for x-max and x-min
    print('Solving for x-max and x-min...')
    data = []
    if n_proc == 1:
        for arg in tqdm(args):
            data.extend(solve(*arg))
    else:
        with mp.Pool(n_proc) as p:
            for out in tqdm(p.starmap(solve, args), total=len(args)):
                data.extend(out)

    # Post-processing
    data = np.array(data)
    print('Number of data points: {}'.format(data.shape[0]))

    return data


def solve(ic_list: List[np.ndarray], lander: lc.Rocket, N: int):
    """Solve for the reachable bound for given initial conditions.

    Args:
        ic_list: list of initial conditions; each initial condition is a numpy array of shape (5, )
        lander: Rocket object
        N: number of discretization

    Returns:
        list: list of data points; each data point is a list of [alt, vx, vz, m, tgo, rf(xmax), rf(xmin), rf(ymax)]. Shape (n, 11)
    """

    # --------------------
    # solve for x-max and x-min
    # --------------------
    ic_xbounds = []
    rf_xmax = []
    rf_xmin = []
    tgo_before = None
    lcvx = lc.LCvxReachabilityRxy(rocket=lander, N=N, directional_cstr=True)
    for ic in ic_list:
        # unpack initial condition
        h, vx, vz, m, tgo = ic

        # create new problem if tgo changes
        if tgo_before is None or tgo_before != tgo:
            prob = lcvx.problem(tf=tgo)

        # solve for x-max
        lc.set_params(prob, {'alt': h,
                             'z_mass': np.log(m),
                             'vx': vx,
                             'vz': vz,
                             'c': np.array([1.0, 0.0]),
                             'c_xc_arr': np.zeros((2, 2))})
        sol_xmax = _solve(lcvx, prob, lander)

        # solve for x-min
        lc.set_params(prob, {'alt': h,
                             'z_mass': np.log(m),
                             'vx': vx,
                             'vz': vz,
                             'c': np.array([-1.0, 0.0]),
                             'c_xc_arr': np.zeros((2, 2))})
        sol_xmin = _solve(lcvx, prob, lander)

        if sol_xmax is not None and sol_xmin is not None:
            ic_xbounds.append(ic)
            r_max, _, _, _, _ = sol_xmax
            r_min, _, _, _, _ = sol_xmin
            rf_xmax.append(r_max[:2, -1])
            rf_xmin.append(r_min[:2, -1])
        else:
            ic_xbounds.append(None)
            rf_xmax.append(None)
            rf_xmin.append(None)

    # --------------------
    # solve for y-max
    # --------------------
    ic_ymax = []
    rf_ymax = []
    tgo_before = None
    lcvx = lc.LCvxReachabilityRxy(rocket=lander, N=N, directional_cstr=False)
    for ic in ic_list:
        # unpack initial condition
        h, vx, vz, m, tgo = ic

        # create new problem if tgo changes
        if tgo_before is None or tgo_before != tgo:
            prob = lcvx.problem(tf=tgo)

        # solve for y-max
        lc.set_params(prob, {'alt': h,
                             'z_mass': np.log(m),
                             'vx': vx,
                             'vz': vz,
                             'c': np.array([0.0, 1.0]),
                             'c_xc_arr': np.zeros((2, 2))})
        sol_ymax = _solve(lcvx, prob, lander)

        if sol_ymax is not None:
            ic_ymax.append(ic)
            r, _, _, _, _ = sol_ymax
            rf_ymax.append(r[:2, -1])
        else:
            ic_ymax.append(None)
            rf_ymax.append(None)

    # Post-processing
    data_out = []
    for i in range(len(ic_list)):
        if ic_xbounds[i] is not None and ic_ymax[i] is not None:
            data_i = list(ic_list[i])
            data_i.extend(rf_xmax[i])
            data_i.extend(rf_xmin[i])
            data_i.extend(rf_ymax[i])
            data_out.append(data_i)

    return data_out


def _solve(lcvx, prob, lander):
    try:
        prob.solve(solver=cp.ECOS, verbose=False)
    except cp.SolverError:
        return None
    if prob.status != 'optimal':
        return None

    # Get solution
    sol = lc.get_vars(prob, ['X', 'U'])
    X_sol = sol['X']
    U_sol = sol['U']
    r, v, z, u, sigma = lcvx.recover_variables(X_sol, U_sol)
    valid, _ = lc.isolated_active_set_gs(r, lander.gsa)

    if not valid:
        return None
    return r, v, z, u, sigma


if __name__ == '__main__':
    main()
