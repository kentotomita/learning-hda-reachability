"""Generate controllable set; possible initial states given the terminal state set. (Step 1)"""
import yaml
import argparse
import pickle
import numpy as np
import cvxpy as cp
import os
import multiprocessing as mp
from itertools import product
from tqdm import tqdm
import datetime
import time
import src.lcvx as lc
from src import Lander, get_lander


def main(n_proc: int=1):
    start = time.time()

    # Parameters
    planet = 'Mars'
    lander = get_lander(planet=planet)
    lander.gsa = (np.pi - lander.fov) / 2  # set glide-slope angle to be FOV/2
    
    with open("config/ctrlset.yaml", "r") as f:
        ctrlset_data = yaml.safe_load(f)
    config = ctrlset_data[planet]

    # Prepare output directory
    dtstring = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join('out/controllable_set/', dtstring)
    os.makedirs(out_dir, exist_ok=True)

    # Save config
    with open(os.path.join(out_dir, 'config.yaml'), 'w') as f:
        yaml.dump(ctrlset_data, f)
    # pickle lander object
    with open(os.path.join(out_dir, 'lander.pkl'), 'wb') as f:
        pickle.dump(lander, f)

    # Prepare arguments
    xf_bounds = (config['xf_min'], config['xf_max'])
    tgo_set = np.linspace(config['tgo_min'], config['tgo_max'], config['n_tgo'])
    theta_list = np.linspace(0.0, np.pi, config['n_theta'])  # angle of velocity vector; 0.0 = vertiargs = []
    args = []
    for tgo in tgo_set:
        alt_max_ = min(config['alt_max'], config['tgo2alt_max'] * tgo)
        alt_min_ = max(config['alt_min'], config['tgo2alt_min'] * tgo)
        alt_set = np.arange(alt_min_, alt_max_ + config['d_alt'], config['d_alt'])
        args.append((lander, config['N'], xf_bounds, tgo, alt_set, theta_list, config['d_mass']))

    data = []
    # Solve
    if n_proc == 1:
        for arg in tqdm(args):
            data.extend(solve(*arg))
    else:
        with mp.Pool(n_proc) as p:
            for out in tqdm(p.starmap(solve, args), total=len(args)):
                data.extend(out)

    # Post-processing
    data = np.array(data)
    data = data[(data[:, 3] >= 0)]  # Remove negative vx

    # Save data
    # save header and data
    data_header = ['rx', 'ry', 'rz', 'vx', 'vy', 'vz', 'm0', 'tgo', 'mf']
    np.savetxt(os.path.join(out_dir, 'data_header.txt'), data_header, fmt='%s')
    np.save(os.path.join(out_dir, 'data.npy'), np.array(data))
    # save meta data
    with open(os.path.join(out_dir, 'meta.txt'), 'w') as f:
        f.write('data shape: {}\n'.format(data.shape))
        f.write('n_proc: {}\n'.format(n_proc))
        f.write('time: {} min'.format((time.time() - start)/60))


def solve(lander: Lander, N: int, xf_bounds: tuple, tgo: float, alt_set: list, theta_list: list, d_mass: float):
    """Solve for controllable set for given parameters.
    
    Args:
        lander (Lander): lander object
        N (int): number of discretization
        xf_bounds (tuple): bounds for terminal state
        tgo (float): time to go
        alt_set (list): list of altitude
        theta_list (list): list of theta
        d_mass (float): mass discretization

    Returns:
        list: list of data points; each data point is a list of [rx, ry, rz, vx, vy, vz, m0, tgo, mf]
    """

    # solve for mass bounds and feasible state space
    state_bounds = np.zeros((len(alt_set), 2, 3))  # (alt, min/max, (mass, vx, vz))
    lcvx = lc.LCvxControllability(lander=lander, N=N)
    prob = lcvx.problem(
        xf_bounds=xf_bounds,
        tf=tgo,
        x0_paramed=[False, False, True, False, True, False, False],  # altitude varied, vy fixed to 0
        c_paramed=[False, False, False, False, False, False, True],  # mass for maximization
        directional_cstr=False   # no directional constrains
        )
    for (i, alt), (j, m_direction) in product(enumerate(alt_set), enumerate([-1.0, 1.0])):
        lc.set_params(prob, {'c': np.array([m_direction])})
        lc.set_params(prob, {'x0': np.array([alt, 0.0])})
        _, data_points = _solve([], lcvx, prob, tgo)
        if data_points is not None:
            state_bounds[i, j, :] = data_points[6], data_points[3], data_points[5]  # mass, vx, vz

    # solve for feasible velocity space
    data = []
    lcvx = lc.LcVxControllabilityVxz(lander=lander, N=N)
    prob = lcvx.problem(xf_bounds=xf_bounds, tf=tgo)
    for i, alt in enumerate(alt_set):
        if state_bounds[i, 0, 0] == 0.0 or state_bounds[i, 1, 0] == 0.0:
            continue
        m_max = state_bounds[i, 1, 0]
        m_min = state_bounds[i, 0, 0]
        mass_list = np.arange(m_min, m_max + d_mass, d_mass)
        for mass in mass_list:
            lc.set_params(prob, {'alt': alt})
            lc.set_params(prob, {'z_mass': np.log(mass)})
            alpha = (mass - m_min) / (m_max - m_min)
            vx_center = alpha * state_bounds[i, 1, 1] + (1 - alpha) * state_bounds[i, 0, 1]
            vz_center = alpha * state_bounds[i, 1, 2] + (1 - alpha) * state_bounds[i, 0, 2]
            if vx_center != 0.0:
                theta_max = 2 * np.pi
                n_theta = len(theta_list) * 2
                print("vx_center != 0.0")
            else:
                theta_max = np.pi
                n_theta = len(theta_list)
            for theta in np.linspace(0.0, theta_max, n_theta):
                c = np.array([np.sin(theta), -np.cos(theta)])
                lc.set_params(prob, {'c': c})
                lc.set_params(prob, {'c_xc_arr':
                                     np.array([
                                         [c[0] * vx_center, c[0] * vz_center],
                                         [c[1] * vx_center, c[1] * vz_center]])})
                data, _ = _solve(data, lcvx, prob, tgo)
    return data


def _solve(data, lcvx, prob, tgo):
    """Solve the problem and return data and data point."""
    try:
        prob.solve(solver=cp.ECOS, verbose=False)
    except cp.SolverError:
        return data, None
    if prob.status != 'optimal':
        return data, None

    # Get solution
    sol = lc.get_vars(prob, ['X', 'U'])
    X_sol = sol['X']
    U_sol = sol['U']
    r, v, z, u, sigma = lcvx.recover_variables(X_sol, U_sol)
    data_point = [r[0, 0], r[1, 0], r[2, 0], v[0, 0], v[1, 0], v[2, 0], np.exp(z[0]), tgo, np.exp(z[-1])]
    data.append(data_point)
    return data, data_point


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--n_proc', type=int, default=8, help='number of processors')
    args = parser.parse_args()

    # measure execution time
    start = datetime.datetime.now()
    main(n_proc=args.n_proc)
    print(f'Execution time (min): {(datetime.datetime.now() - start).total_seconds() / 60.0}')
