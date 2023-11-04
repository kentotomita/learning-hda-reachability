"""Generate controllable set; possible initial states given the terminal state set"""

import numpy as np
import cvxpy as cp
import sys
import os
import multiprocessing as mp
import itertools
from tqdm import tqdm
import datetime
sys.path.append('../')
import src.lcvx as lc
from config.landers import get_lander


def main():
    # Simulation parameters
    lander = get_lander(planet='mars')
    lander.gsa = (np.pi - lander.fov) / 2
    xf_bounds = (
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.log(lander.mdry)]), # Final state vector (m, m, m, m/s, m/s, m/s, kg)
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.log(1750.0)]),  # Final state vector (m, m, m, m/s, m/s, m/s, kg)
    )
    N = 100
    alt_set = np.linspace(100, 1500, 101)
    tgo_max = alt_set / 10.0  # max time to go
    tgo_min = alt_set / 50.0  # min time to go
    d_tgo = 1.0
    theta_list = np.linspace(0, np.pi, 51)
    n_proc = 16

    # Prepare output directory
    dtstring = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join('../out/controllable_set/', dtstring)
    os.makedirs(out_dir, exist_ok=True)

    data_header = ['rx', 'ry', 'rz', 'vx', 'vy', 'vz', 'm', 'tgo']
    data = []

    # Prepare arguments
    args = []
    for i, alt in enumerate(alt_set):
        for tgo in np.arange(tgo_min[i], tgo_max[i] + d_tgo, d_tgo):
            args.append((lander, N, xf_bounds, alt, tgo, theta_list))

    # Solve
    if n_proc == 1:
        for arg in tqdm(args):
            data.extend(solve(*arg))
    else:
        with mp.Pool(n_proc) as p:
            for out in tqdm(p.starmap(solve, args), total=len(args)):
                data.extend(out)

    # Save data
    # save header and data
    np.savetxt(os.path.join(out_dir, 'data_header.txt'), data_header, fmt='%s')
    np.save(os.path.join(out_dir, 'data.txt'), np.array(data))


def solve(rocket: lc.Rocket, N: int, xf_bounds: tuple, alt: float, tgo: float, theta_list: np.ndarray):
    # Set up problem
    lcvx = lc.LCvxControllability(rocket=rocket, N=N, alt0=alt)
    directional_cstr=[False, False, False, True, True, True, False]
    xc = np.zeros(7)
    prob = lcvx.problem(xf_bounds=xf_bounds, tf=tgo, xc=xc, directional_cstr=directional_cstr)

    data = []
    for theta, m_direction in itertools.product(theta_list, [-1, 1]):
        # Define directional vector
        c = np.array([0.0, 0.0, 0.0, np.sin(theta), 0.0, -np.cos(theta), m_direction])
        # Handle parameter; set the directional constraint
        lc.set_params(prob, {'c': c})

        # Solve
        try:
            prob.solve(solver=cp.ECOS, verbose=False)
        except cp.SolverError:
            continue
        if prob.status != 'optimal':
            continue
            
        # Get solution
        sol = lc.get_vars(prob, ['X', 'U'])
        X_sol = sol['X']
        U_sol = sol['U']
        r, v, z, u, sigma = lcvx.recover_variables(X_sol, U_sol)

        data.append([r[0, 0], r[1, 0], r[2, 0], v[0, 0], v[1, 0], v[2, 0], np.exp(z[0]), tgo])

    return data


if __name__ == '__main__':
    # measure execution time
    start = datetime.datetime.now()
    main()
    print(f'Execution time (min): {(datetime.datetime.now() - start).total_seconds() / 60.0}')