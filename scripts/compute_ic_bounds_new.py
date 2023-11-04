"""Compute the reachable set given the set of initial state for the landing problem.
From the generated reachable set, initial conditions are sampled for the reachable surface dataset generation
"""
import numpy as np
import cvxpy as cp
import sys
import multiprocessing as mp
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import time

sys.path.append('../')
import src.lcvx as lc
from config.landers import get_lander


def config():
    rocket = get_lander(planet='mars')
    N = 100
    x0_min = np.array([0., 0., 1500., 20., 0., -30, np.log(rocket.mwet)])
    x0_max = np.array([0., 0., 1500., 30., 0., -20., np.log(rocket.mwet)])
    x0_bounds = (x0_min, x0_max)
    tf_list = np.linspace(50, 100, 6)
    return rocket, N, x0_bounds, tf_list


def solve_reach(lcvx, prob, c):
    # set parameter
    lc.set_params(prob, {'c': c})

    # solve
    try:
        prob.solve(solver=cp.ECOS, verbose=False)
    except cp.SolverError:
        print('SolverError')
        return None
    if prob.status != 'optimal':
        #print('status: ', prob.status)
        return None

    # get solution
    sol = lc.get_vars(prob, ['X', 'U'])
    X_sol = sol['X']
    U_sol = sol['U']
    r, v, z, u, sigma = lcvx.recover_variables(X_sol, U_sol)
    return r, v, z, u, sigma


def remove_empty_list(l):
    return [x for x in l if x !=[]]

"""
def v0_cstr(v: cp.Variable, v0_max=75., v0_max_angle=30 * np.pi / 180):
    cstr = []
    cstr.append(cp.norm(v[:, 0]) <= v0_max)
    cstr.append(cp.norm(v[0, 0] + v[1, 0]) <= -v[2, 0] * np.tan(v0_max_angle))
    return cstr
"""


def solve_vz_bound(rocket, N, tf, x0_bounds, steps, n_proc=1):
    print(f'Solving Vertical Velocity Reachable Set with tf={tf}...')
    # prepare parameters
    params = [(i, rocket, N, tf, x0_bounds) for i in steps]

    # solve
    if not n_proc==1:
        results = []
        for param in tqdm(params):
            results.extend(solve_vz_bound_k(*param))
    else:
        with mp.Pool(n_proc) as p:
            #"""
            results = []
            for result in tqdm(p.starmap(solve_vz_bound_k, params), total=len(params)):
                results.extend(result)
            #"""
            #outs = p.starmap(solve_vz_bound_k, params)
        #results = [out for out in outs if out is not None]

    return remove_empty_list(results)


def solve_vx_bound(rocket, N, tf, x0_bounds, steps, nc, n_proc=1):
    print(f'Solving Horizontal Velocity Reachable Set with tf={tf}...')

    # prepare parameters
    params = [(i, rocket, N, tf, x0_bounds, nc) for i in steps]

    # solve
    if n_proc==1:
        results = []
        for param in tqdm(params):
            results.extend(solve_vx_bound_k(*param))
    else:
        with mp.Pool(n_proc) as p:
            #"""
            results = []
            for result in tqdm(p.starmap(solve_vx_bound_k, params), total=len(params)):
                results.extend(result)
            #"""
            #outs = p.starmap(solve_vx_bound_k, params)
        #results = [out for out in outs if out is not None]

    return remove_empty_list(results)


def solve_vz_bound_k(maxk, rocket, N, tf, x0_bounds):
    dt = tf / N

    lcvx = lc.LCvxReachability(rocket=rocket, N=N, maxk=maxk)
    prob = lcvx.problem(x0_bounds=x0_bounds, tf=tf, xc=np.zeros(7))
    
    vertices = []
    for c_ in [1., -1.]:
        c = np.zeros(7)
        c[5] = c_
        out = solve_reach(lcvx, prob, c)
        if out is not None:
            r, v, _, _, _ = out
            vz = v[2, maxk]
            alt = r[2, maxk]
            tgo = tf - maxk * dt  # time to go
            vertices.append([vz, alt, tgo])

    return vertices


def solve_vx_bound_k(maxk, rocket, N, tf, x0_bounds, nc):
    dt = tf / N

    # find a feasible horizontal velocity at step k=maxk
    lcvx = lc.LCvxReachability(rocket=rocket, N=N, maxk=maxk, directional_cstr=[False, False, False, True, True, False, False])
    prob = lcvx.problem(x0_bounds=x0_bounds, tf=tf, xc=np.zeros(7))

    thetas = np.linspace(0, np.pi, nc)
    vertices = []
    for i in range(nc):
        c = np.zeros(7)
        c[3] = np.cos(thetas[i])
        c[4] = np.sin(thetas[i])
        out = solve_reach(lcvx, prob, c)
        if out is not None:
            r, v, z, u, sigma = out
            v_horizontal = np.linalg.norm(v[:2, maxk])
            alt = r[2, maxk]
            tgo = tf - maxk * dt  # time to go
            vertices.append([v_horizontal, alt, tgo])
            vertices.append([0, alt, tgo])
    return vertices


def main(n_proc=1):
    rocket, N, x0_bounds, tf_list = config()
    steps = list(range(1, N, 5))  # start, stop, step

    vz_data = []
    vx_data = []

    # measure total time
    start_time = time.time()
    for tf in tqdm(tf_list):
        vx_data.extend(solve_vx_bound(rocket, N, tf, x0_bounds, steps, 10, n_proc))
        #vz_data.extend(solve_vz_bound(rocket, N, tf, x0_bounds, steps, n_proc))

    print("--- %s minutes ---" % ((time.time() - start_time)/60))

    #vz_data = np.array(vz_data)
    vx_data = np.array(vx_data)
    #np.save('../out/vz_data_new.npy', vz_data)
    np.save('../out/vx_data_new.npy', vx_data)
    
    



if __name__=="__main__":
    main(n_proc=16)


