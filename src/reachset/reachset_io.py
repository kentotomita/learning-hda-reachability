import os
import numpy as np
from typing import List, Dict
import time
import json

from ..lcvx import Rocket

def save_icset(rocket: Rocket, N: int, ic_list: List[Dict], outdir: str='.', fname: str='icset.json', return_data: bool=False):
    """Save initial condition set to file.

    Args:
        rocket (lc.Rocket): rocket model
        N (int): number of discretization
        ic_list (List[Dict]): list of feasible initial conditions, [x0:(7), tgo:(1), rf:(3)]
    """

    # get time stamp
    tstamp = time.strftime('%Y-%m%d-%H%M%S')

    # convert ic data to array
    icarr = np.zeros((len(ic_list), 11))
    for i, ic in enumerate(ic_list):
        icarr[i, :7] = ic['x0']
        icarr[i, 7] = ic['tgo']
        icarr[i, 8:] = ic['rf']
    # sort by tgo
    idx = np.argsort(icarr[:, 7])
    icarr = icarr[idx]

    # prepare data
    icset = {}
    icset['tstamp'] = tstamp
    icset['rocket'] = {
        'g': rocket.g.tolist(),
        'mdry': rocket.mdry,
        'mwet': rocket.mwet,
        'Isp': rocket.Isp,
        'rho1': rocket.rho1,
        'rho2': rocket.rho2,
        'gsa': rocket.gsa,
        'pa': rocket.pa,
        'fov': rocket.fov,
    }
    icset['N'] = N
    icset['fields'] = ['rx0', 'ry0', 'rz0', 'vx0', 'vy0', 'vz0', 'z0', 'tgo', 'rxf', 'ryf', 'rzf']
    icset['data'] = icarr.tolist()

    # save as json
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    with open(os.path.join(outdir, fname), 'w') as f:
        json.dump(icset, f, indent=4)
    
    if return_data:
        return icset
    

def read_icset(fpath: str='icset.json'):
    """Read initial condition set from file.

    Args:
        fpath (str): path to file

    Returns:
        rocket (lc.Rocket): rocket model
        N (int): number of discretization
        icarr (np.ndarray): initial condition array, shape (n, 11)
    """
    with open(fpath, 'r') as f:
        icset = json.load(f)

    rocket = Rocket(
        g=np.array(icset['rocket']['g']),  # Gravitational acceleration (m/s^2)
        mdry=icset['rocket']['mdry'], # Dry mass (kg)
        mwet=icset['rocket']['mwet'], # Wet mass (kg)
        Isp=icset['rocket']['Isp'],
        rho1=icset['rocket']['rho1'],   # Minimum thrust (N)
        rho2=icset['rocket']['rho2'],  # Maximum thrust (N)
        gsa=icset['rocket']['gsa'],  # Glide slope angle (rad); from horizontal
        pa=icset['rocket']['pa'],  # Pointing angle (rad); from vertical
        fov=icset['rocket']['fov'],
    )
    N = icset['N']

    icarr = np.array(icset['data'])

    return rocket, N, icarr


def save_reachset(rocket: Rocket, N: int, icset: np.ndarray, reachpoints: List[Dict], nc: int, outdir: str='.', fname: str='reachset.json', return_data: bool=False):
    """Save soft landing reachable set to file.

    Args:
        rocket (lc.Rocket): rocket model
        N (int): number of discretization
        icset (np.ndarray): initial condition array, shape (n, 11)
        reachpoints (List[Dict]): list of reachpoints, [ic_idx:(1), c_idx:(1), c:(3), rc:(3), rf:(3)]
        nc (int): number of directions

    Returns:
        reachset_data (dict): reachset data
    """

    reachset = np.array([
        {'x0': np.zeros(7) * np.nan,
        'tgo': np.nan,
        'rc': np.zeros(3) * np.nan,
        'c': np.zeros((nc, 3)) * np.nan,
        'rf': np.zeros((nc, 3)) * np.nan}
        for _ in range(icset.shape[0])])

    for reachpoint in reachpoints:
        ic_idx = reachpoint['ic_idx']
        c_idx = reachpoint['c_idx']
        x0 = icset[ic_idx, :7]
        tgo = icset[ic_idx, 7]
        reachset[ic_idx]['x0'] = x0
        reachset[ic_idx]['tgo'] = tgo
        reachset[ic_idx]['rc'] = reachpoint['rc']
        reachset[ic_idx]['c'][c_idx, :] = reachpoint['c']
        reachset[ic_idx]['rf'][c_idx, :] = reachpoint['rf']
    
    # convert arrays to list
    for i in range(len(reachset)):
        reachset[i]['x0'] = reachset[i]['x0'].tolist()
        reachset[i]['tgo'] = [reachset[i]['tgo']]
        reachset[i]['rc'] = reachset[i]['rc'].tolist()
        reachset[i]['c'] = reachset[i]['c'].tolist()
        reachset[i]['rf'] = reachset[i]['rf'].tolist()
    reachset = reachset.tolist()

    # prepare data
    reachset_data = {}
    reachset_data['tstamp'] = time.strftime('%Y-%m%d-%H%M%S')
    reachset_data['rocket'] = {
        'g': rocket.g.tolist(),
        'mdry': rocket.mdry,
        'mwet': rocket.mwet,
        'Isp': rocket.Isp,
        'rho1': rocket.rho1,
        'rho2': rocket.rho2,
        'gsa': rocket.gsa,
        'pa': rocket.pa,
        'fov': rocket.fov,
    }
    reachset_data['N'] = N
    reachset_data['fields'] = ['x0', 'tgo', 'rc', 'c', 'rf']
    reachset_data['data'] = reachset

    # save
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    with open(os.path.join(outdir, fname), 'w') as f:
        json.dump(reachset_data, f, indent=4)

    if return_data:
        return reachset_data


def read_reachset(fpath: str='reachset.json', nmax: int=None):
    """Read soft landing reachable set from file.
    
    Args:
        fpath (str): path to file
        nmax (int): maximum number of datapoints to read

    Returns:
        rocket (lc.Rocket): rocket model
        N (int): number of discretization
        reachset (np.ndarray): soft landing reachable set, shape (n, 11)
    """

    with open(fpath, 'r') as f:
        reachset_data = json.load(f)

    rocket = Rocket(
        reachset_data['rocket']['g'],  # Gravitational acceleration (m/s^2)
        reachset_data['rocket']['mdry'], # Dry mass (kg)
        reachset_data['rocket']['mwet'], # Wet mass (kg)
        reachset_data['rocket']['Isp'],
        reachset_data['rocket']['rho1'],   # Minimum thrust (N)
        reachset_data['rocket']['rho2'],  # Maximum thrust (N)
        reachset_data['rocket']['gsa'],  # Glide slope angle (rad); from horizontal
        reachset_data['rocket']['pa'],  # Pointing angle (rad); from vertical
        reachset_data['rocket']['fov'],
    )
    N = reachset_data['N']

    reachset = np.array(reachset_data['data'])

    nmax = len(reachset) if nmax is None or nmax > len(reachset) else nmax
    reachset = reachset[:nmax]

    for i in range(len(reachset)):
        reachset[i]['x0'] = np.array(reachset[i]['x0']).astype(float)
        reachset[i]['tgo'] = np.array(reachset[i]['tgo']).astype(float)
        reachset[i]['rc'] = np.array(reachset[i]['rc']).astype(float)
        reachset[i]['c'] = np.array(reachset[i]['c']).astype(float)
        reachset[i]['rf'] = np.array(reachset[i]['rf']).astype(float)

    return rocket, N, reachset



