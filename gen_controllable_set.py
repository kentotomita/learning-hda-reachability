"""Generate controllable set; possible initial states given the terminal state set"""

import numpy as np
import cvxpy as cp
import sys
import os
import multiprocessing as mp
from itertools import product
from tqdm import tqdm
import datetime
import src.lcvx as lc
from config.landers import get_lander


def main():
    # Simulation parameters
    lander = get_lander(planet='mars')
    lander.gsa = (np.pi - lander.fov) / 2  # set glide-slope angle to be FOV/2
    xf_bounds = (
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.log(lander.mdry)]), # Final state vector (m, m, m, m/s, m/s, m/s, kg)
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.log(1750.0)]),  # Final state vector (m, m, m, m/s, m/s, m/s, kg)
    )
    N = 100
    alt_max = 1500.0
    alt_min = 10.0
    tgo2alt_max = 100.0  # max velocity; max altitude = tgo * tgo2alt_max
    tgo2alt_min = 1.0  # min velocity; min altitude = tgo * tgo2alt_min
    tgo_set = np.arange(1.0, 151.0, 1.0)
    d_alt = 20.0  # granularity of altitude
    d_mass = 5.0  # granularity of mass
    theta_list = np.linspace(0.0, np.pi, 51)
    n_proc = 64

    if False:  # debug config
        tgo_set = np.arange(1.0, 151.0, 20.0)
        d_alt = 200.0
        d_mass = 20.0
        theta_list = np.linspace(0.0, np.pi, 11)
        n_proc = 1

    # Prepare output directory
    dtstring = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join('out/controllable_set/', dtstring)
    os.makedirs(out_dir, exist_ok=True)
    data_header = ['rx', 'ry', 'rz', 'vx', 'vy', 'vz', 'm0', 'tgo', 'mf']
    data = []

    # Prepare arguments
    args = []
    for tgo in tgo_set:
        alt_max_ = min(alt_max, tgo2alt_max * tgo)
        alt_min_ = max(alt_min, tgo2alt_min * tgo)
        alt_set = np.arange(alt_min_, alt_max_ + d_alt, d_alt)
        args.append((lander, N, xf_bounds, tgo, alt_set, theta_list, d_mass))

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
    np.save(os.path.join(out_dir, 'data.npy'), np.array(data))


def solve(rocket: lc.Rocket, N: int, xf_bounds: tuple, tgo: float, alt_set: list, theta_list: list, d_mass: float):
    
    # solve for mass bounds and feasible state space
    state_bounds = np.zeros((len(alt_set), 2, 3))  # (alt, min/max, (mass, vx, vz))
    lcvx = lc.LCvxControllability(rocket=rocket, N=N)
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
    lcvx = lc.LcVxControllabilityVxz(rocket=rocket, N=N)
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
    # measure execution time
    start = datetime.datetime.now()
    main()
    print(f'Execution time (min): {(datetime.datetime.now() - start).total_seconds() / 60.0}')
