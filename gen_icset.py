"""Generate initial conditions by sampling points from the controllable set."""
import argparse
import multiprocessing as mp
import numpy as np
import os
import datetime
import time
from scipy.spatial import ConvexHull
import sys
sys.path.append('../')
from src.reachset.ic_sampler import random_sampling_in_hull, structured_sample_points_in_convex_hull


def main():
    start = time.time()

    # read command line inputs
    parser  = argparse.ArgumentParser()
    parser.add_argument('--n_random', type=int, default=int(1e4))
    parser.add_argument('--n_per_dim', type=int, default=5)
    parser.add_argument('--n_proc', type=int, default=8)
    args = parser.parse_args()

    # load controllable set
    data = np.load('saved/controllable_set/data.npy')
    
    # Create convex hull
    print('Creating convex hull...')
    indices = [2, 3, 5, 6, 7]
    data_5d = data[:, indices]
    data_bounds = (np.min(data_5d, axis=0), np.max(data_5d, axis=0))
    data_normalized = (data_5d - data_bounds[0]) / (data_bounds[1] - data_bounds[0])
    hull_5d = ConvexHull(data_normalized, qhull_options='Q12')

    # Random sampling
    print('Random sampling...')
    if args.n_proc == 1:
        random_samples = random_sampling_in_hull(np.ascontiguousarray(hull_5d.equations), (np.zeros(5), np.ones(5)), args.n_random)
    else:
        params = []
        for i in range(args.n_proc):
            params.append((np.ascontiguousarray(hull_5d.equations), (np.zeros(5), np.ones(5)), args.n_random, i))
        with mp.Pool(args.n_proc) as p:
            out = p.starmap(random_sampling_in_hull, params)
            random_samples = np.vstack(out)

    # denormalize
    random_samples = random_samples * (data_bounds[1] - data_bounds[0]) + data_bounds[0]

    # structured sampling
    print('Structured sampling...')
    if args.n_proc == 1:
        structured_samples = structured_sample_points_in_convex_hull(hull_5d, args.n_per_dim, hull_5d.points)
    else:
        params = []
        buffer_list = np.linspace(0.05, 0.3, args.n_proc)
        for i in range(args.n_proc):
            params.append((hull_5d, args.n_per_dim, hull_5d.points, buffer_list[i]))
        with mp.Pool(args.n_proc) as pool:
            out = pool.starmap(structured_sample_points_in_convex_hull, params)
            structured_samples = np.vstack(out)

    # denormalize
    structured_samples = structured_samples * (data_bounds[1] - data_bounds[0]) + data_bounds[0]

    # save
    print('Saving...')
    dtstring = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join('saved/controllable_set/ic_set/', dtstring)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, 'random_samples.npy'), random_samples)
    np.save(os.path.join(out_dir, 'structured_samples.npy'), structured_samples)

    # save mata data; shapes of generated samples
    with open(os.path.join(out_dir, 'meta.txt'), 'w') as f:
        f.write('random_samples: {}\n'.format(random_samples.shape))
        f.write('structured_samples: {}\n'.format(structured_samples.shape))
        f.write('time: {}\n min'.format((time.time() - start)/60))


if __name__ == '__main__':
    # measure time
    start = time.time()
    main()
    print("--- %.3f minutes ---" % ((time.time() - start)/60))
