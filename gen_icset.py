"""Generate a set of feasible initial conditions"""
import numpy as np
import sys
import multiprocessing as mp
from tqdm import tqdm
import warnings
import time
import argparse
from datetime import datetime
import os

sys.path.append("../")
from src.reachset import convert_to_x0, save_icset, get_ic_list
from config import slr_config

debug = False
np.set_printoptions(precision=1)
warnings.filterwarnings("ignore", category=UserWarning)


def main(n_sample, n_proc, n_per_process, outdir, debug=False):
    """Main function to find feasible initial conditions.

    Args:
        n_sample (int): number of random samples
        n_proc (int): number of processes
        n_per_process (int): number of samples per process
    """
    rocket, N = slr_config()

    # generate random samples
    sample = np.random.rand(n_sample, 5)
    x0_arr, tgo = convert_to_x0(sample)

    # sort by tgo
    idx = np.argsort(tgo)
    x0_arr = x0_arr[idx]
    tgo = tgo[idx]

    # ----------------- solve problem -----------------
    icset = []

    if n_proc > 1:
        # prepare parameters
        params = []
        for i in range(0, n_sample, n_per_process):
            if i + n_per_process < n_sample:
                params.append(
                    (
                        rocket,
                        N,
                        x0_arr[i : i + n_per_process],
                        tgo[i : i + n_per_process],
                        debug,
                    )
                )
            else:
                params.append((rocket, N, x0_arr[i:], tgo[i:], debug))

        # solve
        with mp.Pool(processes=n_proc) as p:
            icset = []
            for out in tqdm(p.starmap(get_ic_list, params), total=len(params)):
                icset.extend(out)

    else:
        icset.extend(get_ic_list(rocket, N, x0_arr, tgo, debug))

    if len(icset) == 0:
        raise ValueError("No feasible initial condition found.")

    # ----------------- post processing -----------------
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d_%H%M%S")
    outdir = outdir + "/" + dt_string
    os.makedirs(outdir, exist_ok=True)
    icset = save_icset(
        rocket, N, icset, outdir=outdir, fname="icset.json", return_data=True
    )

    # calc sampling efficiency
    rfz_arr = np.array(icset["data"])[:, -1]
    n_data = len(rfz_arr)
    n_feasible = n_sample - np.isnan(rfz_arr).sum()

    print("Number of generate data: ", n_data)
    print("Number of feasible initial conditions: ", n_feasible)


if __name__ == "__main__":
    print("Generate a set of feasible initial conditions.")

    # read command line inputs
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_sample", type=int, default=int(1e7))
    parser.add_argument("--n_proc", type=int, default=28)
    parser.add_argument("--n_per_proc", type=int, default=1000)
    parser.add_argument("--outdir", type=str, default="out")

    args = parser.parse_args()

    n_sample = args.n_sample
    n_proc = args.n_proc
    n_per_proc = min(args.n_per_proc, int(n_sample / n_proc))
    outdir = args.outdir

    # measure time
    start = time.time()
    main(n_sample, n_proc, n_per_proc, outdir)
    print("--- %.3f minutes ---" % ((time.time() - start) / 60))
