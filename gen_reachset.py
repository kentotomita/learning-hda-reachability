"""Generate soft landing reachable set data"""
import numpy as np
import sys
import multiprocessing as mp
import warnings
import time
import argparse
import os

sys.path.append('../')
from rahao.reachset import get_reachpoints, save_reachset, read_icset, read_reachset

np.set_printoptions(precision=1)  # for debug 
warnings.filterwarnings("ignore", category=UserWarning)

def main(n_proc:int=1, n_per_proc:int=None, nmax:int=None, nc:int=30, debug:bool=False, datadir:str='out'):
    """Main function to generate soft landing reachable set
    
    Args:
        n_proc (int): number of processes
        n_per_proc (int): number of initial conditions per process
        nmax (int): maximum number of initial conditions to solve
        nc (int): number of directions to solve
    """

    # Load initial conditions (IC) from json file
    rocket, N, icset_all = read_icset(os.path.join(datadir, 'icset.json'))

    # Maximum range directions
    cs = [np.array([np.cos(t), np.sin(t), 0.]) for t in np.linspace(np.pi, 0., nc)]

    # Process infeasible initial conditions (IC) from json file
    # ----------------- process unfeasible ic -----------------
    if nmax is not None:
        icset_all = icset_all[:nmax]
    n_ic_all = icset_all.shape[0]

    reachpoints = []
    ic_list = []
    ic_idx_list = []
    for ic_idx, ic in enumerate(icset_all):
        if np.isnan(ic[8]):
            for c_idx, c in enumerate(cs):
                reachpoints.append({
                    'ic_idx': ic_idx, 
                    'c_idx': c_idx, 
                    'c': c, 
                    'rc': np.array([np.nan, np.nan, np.nan]), 
                    'rf': np.array([np.nan, np.nan, np.nan]),
                    })
        else:
            ic_list.append(ic)
            ic_idx_list.append(ic_idx)
    icset = np.array(ic_list)
    ic_idxset = np.array(ic_idx_list)

    # sort by tgo 
    tgo = icset[:,7]
    idx = np.argsort(tgo)
    icset = icset[idx]
    ic_idxset = ic_idxset[idx]

    n_ic = icset.shape[0]
    print(f'Feasible IC: {n_ic}, Infeasible IC: {n_ic_all - n_ic}')

    # ----------------- solve -----------------
    if n_proc > 1:
        # prepare args
        args = []
        for c_idx, c in enumerate(cs):
            for i in range(0, n_ic, n_per_proc):
                ic_idxs = ic_idxset[i]
                if i + n_per_proc < n_ic:
                    args.append((rocket, N, icset[i:i+n_per_proc], c, ic_idxset[i:i+n_per_proc], c_idx, debug))
                else:
                    args.append((rocket, N, icset[i:], c, ic_idxset[i:], c_idx, debug))

        # solve
        with mp.Pool(processes=n_proc) as p:
            for out in p.starmap(get_reachpoints, args):
                reachpoints.extend(out)

    else:  # single process
        for c_idx, c in enumerate(cs):
            reachpoints.extend(get_reachpoints(rocket, N, icset, c, ic_idxset, c_idx, debug))

    # ----------------- save -----------------
    save_reachset(rocket, N, icset_all, reachpoints, nc, outdir=datadir, fname='reachset.json')
    _, _, reachset = read_reachset(os.path.join(datadir, 'reachset.json'))

    n_infeasible = np.sum([np.all(np.isnan(reachset[i]['rf'])) * nc for i in range(len(reachset))])
    n_valid = np.sum([np.sum(~np.isnan(reachset[i]['rf'])) /3 for i in range(len(reachset))])
    n_tot = reachset.shape[0] * nc
    print('Number of data points: ', n_tot)
    print('Number of infeasible data points: ', n_infeasible)
    print('Number of valid data points: ', n_valid)
    print('Solve rate: {:.2f}%'.format(100 *  n_valid / (n_tot - n_infeasible)))


if __name__ == '__main__':

    # read command line inputs
    parser  = argparse.ArgumentParser()
    parser.add_argument('--n_proc', type=int, default=28)
    parser.add_argument('--n_per_proc', type=int, default=1000)
    parser.add_argument('--nmax', type=int, default=None)
    parser.add_argument('--nc', type=int, default=15)
    parser.add_argument('--datadir', type=str)

    args = parser.parse_args()
    n_proc = args.n_proc
    n_per_proc = args.n_per_proc
    nmax = args.nmax
    nc = args.nc
    datadir = args.datadir

    start = time.time()
    main(n_proc, n_per_proc, nmax, nc, datadir=datadir)
    print("--- %.3f minutes ---" % ((time.time() - start)/60))

