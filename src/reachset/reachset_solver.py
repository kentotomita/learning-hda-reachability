"""Functions for numerically finding solf landing reachable set"""

import numpy as np
import cvxpy as cp
from tqdm import tqdm

from ..lcvx import Rocket, LCvxMaxRange, set_params, get_vars, isolated_active_set_gs

def solve_maxrange(lcvx: LCvxMaxRange, prob: cp.Problem, debug=False):
    """Solve a maximum range problem

    Returns:
        rx_min (float): minimum downrange position
        feasibility (bool): whether the problem is feasible
    """

    # solve
    try:
        prob.solve(verbose=debug, abstol=1e-9, reltol=1e-9, feastol=1e-8)
    except cp.SolverError:
        if debug:
            print('SolverError')
        return None, False
    if prob.status != 'optimal':
        if debug:
            print('status: ', prob.status)
        
        if prob.status == 'infeasible':
            return np.array([np.nan, np.nan, np.nan]), False
        else:
            return None, False
    
    sol = get_vars(prob, ['X', 'U'])
    X_sol = sol['X']
    U_sol = sol['U']
    r, _, _, _, _ = lcvx.recover_variables(X_sol, U_sol)
    rf = r[:, -1]  # final position in x

    # check if GSC is isolated; if not, solution is not exact.
    isolated_gsc, idx_gsc = isolated_active_set_gs(r, lcvx.rocket.gsa)
    if isolated_gsc:
        return rf, True
    else:
        if debug:
            print('GSC is not isolated at', idx_gsc)
        return rf, False


def get_reachpoints(rocket, N, icset, c, ic_idx_list, c_idx, debug=False):
    """Find soft landing reachable set by solving a set of maximum range problems for each direction.
    
    Args:
        rocket (lc.Rocket): rocket model containing physical parameters
        N (int): number of discretization points for convex optimization
        icset (list): list of feasible initial conditions; [x0:(7), tf:(1), xmin:(1)]
        c (np.ndarray): direction vector, shape=(3,)
        ic_idx0 (int): index of initial condition to start with
        c_idx (int): index of direction vector
    
    Returns:
        out (List[Dict]): [ic_idx (1), c_idx (1), c (3), rc (3), rf (3)]
    """

    lcvx = LCvxMaxRange(rocket, N, parameterize_x0=True, parameterize_rc=True)
    n_ic = len(icset)  # number of initial conditions
    
    out = []

    pbar = tqdm(total=n_ic)

    tf_before = None

    for i in range(n_ic):
        # extract initial condition data
        ic = icset[i]
        x0 = ic[:7]
        tf = ic[7]
        xmin = ic[8]

        # prepare parameters and problem
        alt = x0[2]
        fov_radius = alt * np.tan(rocket.fov/2)
        alpha = 0.3
        rc_x = (1-alpha) * xmin + alpha * fov_radius
        rc = np.array([rc_x, 0., 0.])
        if tf != tf_before:  # Update the problem only if tf is different from the previous one; saving compitation time
            prob = lcvx.problem(tf=tf, c=c)

        # set parameter
        set_params(prob, {'x0': x0})
        set_params(prob, {'rc': rc})

        # solve problem
        rf, feasible = solve_maxrange(lcvx, prob)

        # add to output
        if type(ic_idx_list) == int:
            ic_idx = ic_idx_list
        else:
            ic_idx = ic_idx_list[i]
        if feasible:
            if not np.linalg.norm(rf[:2]) * np.tan(rocket.gsa) <= alt + 1e-3:
                print('GSA Cstr Failed: rf={}, alt={}, gsa={}'.format(rf, alt, rocket.gsa))
                rf = np.array([np.nan, np.nan, np.nan])
            if not fov_radius + 1e-3 >= np.linalg.norm(rf[:2]):
                print('FOV Cstr Failed: rf={}, fov_radius={}, x0: {}'.format(rf, fov_radius, x0))
                rf = np.array([np.nan, np.nan, np.nan])
        else:
            rf = np.array([np.nan, np.nan, np.nan])
        out.append({'ic_idx': ic_idx, 'c_idx': c_idx, 'c': c, 'rc': rc, 'rf': rf})

        if debug:
            if feasible:
                print('OK | tf={:.1f}, c={}, rf={}, x0={}'.format(tf, c, rf[:2], x0)) 
            else:
                print('X  | tf={:.1f}, c={}, rf={}, x0={}'.format(tf, c, rf, x0)) 

        # Update local parameters 
        tf_before = tf
        pbar.update(1)
    
    return out


def get_ic_list(rocket, N, x0_arr, tgo, debug=False):
    """Solve problem of minimum downrange position with given initial state -> check feasibility.

    Args:
        rocket (lc.Rocket): rocket model
        N (int): number of discretization
        x0_arr (np.ndarray): initial state, shape (n, 7)
        tgo (np.ndarray): time to go, shape (n, )

    Returns:
        ic_list (List[Dict]): list of feasible initial conditions, [x0:(7), tgo:(1), rf:(3)]
    """
    lcvx = LCvxMaxRange(rocket, N, parameterize_x0=True)

    ic_list = []
    tf_before = None
    for i, tf in enumerate(tqdm(tgo)):
        if tf != tf_before:  # define problem only when tf changes
            prob = lcvx.problem(tf=tf, c=np.array([-1., 0., 0.]), rc=np.array([0., 0., 0.]))
        
        # solve problem
        x0 = x0_arr[i]
        set_params(prob, {'x0': x0})
        rf, feasible = solve_maxrange(lcvx, prob, debug=debug)

        tf_before = tf

        if feasible:
            ic_list.append({'x0': x0, 'tgo': tf, 'rf': rf})
        else:
            if type(rf) == np.ndarray:
                if np.sum(np.isnan(rf)) == 3:
                    ic_list.append({'x0': x0, 'tgo': tf, 'rf': rf})

        if debug:
            if feasible:
                print(f'Feasible   | rf={rf}, tgo={tf:.2f}, x0={x0}')
            else:
                print(f'Infeasible | rf={rf}, tgo={tf:.2f}, x0={x0}')

    return ic_list