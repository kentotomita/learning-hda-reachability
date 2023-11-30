"""Generate dataset for feasibility learning. (Step 4)"""
import argparse
import multiprocessing as mp
import numpy as np
import time
from scipy.spatial import ConvexHull
import sys
sys.path.append('../')
from src.learning import transform_ic, inverse_transform_ic
from src.reachset.ic_sampler import random_sampling_outside_hull


def main():
    start = time.time()

    # read command line inputs
    parser = argparse.ArgumentParser()
    parser.add_argument('--reduced_ctrlable_set', action='store_true')
    args = parser.parse_args()

    # load reachset
    data_random = np.load("saved/controllable_set/reachset_train/20231115-025054/data_random.npy")
    data_structured = np.load("saved/controllable_set/reachset_train/20231115-025054/data_structured.npy")

    # Create feasibility dataset using reachset; known to be feasible
    feasibility_random = np.hstack((data_random[:, :5], np.ones((data_random.shape[0], 1))))
    feasibility_structured = np.hstack((data_structured[:, :5], np.ones((data_structured.shape[0], 1))))
    feasibility_data = np.vstack((feasibility_random, feasibility_structured))

    n_feasible = feasibility_data.shape[0]  # number of feasible points

    # Sample infeasible points from the convex hull spanned by the controllable set
    feasibility_data_infeasible = sample_outside_ctrlable_hull(n_feasible, args.reduced_ctrlable_set)

    # combine feasible and infeasible data
    feasibility_data = np.vstack((feasibility_data, feasibility_data_infeasible))

    # save
    print('Saving...')
    np.save('saved/controllable_set/reachset_train/20231115-025054//feasibility_data.npy', feasibility_data)

    # save mata data; shapes of generated samples, number of feasible and infeasible points, time
    with open('saved/controllable_set/reachset_train/20231115-025054//meta.txt', 'w') as f:
        f.write('feasibility_data: {}\n'.format(feasibility_data.shape))
        f.write('n_feasible: {}\n'.format(n_feasible))
        f.write('n_infeasible: {}\n'.format(feasibility_data_infeasible.shape[0]))
        f.write('time: {}\n min'.format((time.time() - start)/60))


def sample_outside_ctrlable_hull(n_samples, reduced_ctrlable_set=False):

    # load controllable set
    ctrlable_set = np.load('saved/controllable_set/data.npy')

    # create convex hull of controllable set; negative vx is allowed.
    print('Creating convex hull...')
    indices = [2, 3, 5, 6, 7]
    ctrlable_set = ctrlable_set[:, indices]
    ctrlable_set = ctrlable_set[ctrlable_set[:, 1] > 0.0, :]

    alt, vx, vz, mass, tgo = ctrlable_set[:, 0], ctrlable_set[:, 1], ctrlable_set[:, 2], ctrlable_set[:, 3], ctrlable_set[:, 4]
    z = np.log(mass)
    alt_, vx_, vz_, z_, tgo_ = transform_ic(alt, vx, vz, z, tgo)
    ctrlable_set_ = np.vstack((alt_, vx_, vz_, z_, tgo_)).T

    # create an array where vx (index 1) is flipped
    ctrlable_set_negative_vx_ = ctrlable_set_.copy()
    ctrlable_set_negative_vx_[:, 1] = -ctrlable_set_negative_vx_[:, 1]
    
    if reduced_ctrlable_set:
        random_binary = np.random.randint(0, 2, ctrlable_set.shape[0])
        ctrlable_set_ = ctrlable_set_[random_binary == 0, :]
        ctrlable_set_negative_vx_ = ctrlable_set_negative_vx_[random_binary == 1, :]
    ctrlable_set_ = np.vstack((ctrlable_set_, ctrlable_set_negative_vx_))

    hull_ctrlable = ConvexHull(ctrlable_set_, qhull_options='Q12 QJ')

    # sample outside the convex hull
    print('Random sampling...')
    lb = np.min(ctrlable_set_, axis=0)
    ub = np.max(ctrlable_set_, axis=0)

    samples = random_sampling_outside_hull(hull_ctrlable.equations, (lb, ub), n_samples)
    
    # denormalize
    alt_, vx_, vz_, z_, tgo_ = samples[:, 0], samples[:, 1], samples[:, 2], samples[:, 3], samples[:, 4]
    alt, vx, vz, z, tgo = inverse_transform_ic(alt_, vx_, vz_, z_, tgo_)
    samples = np.vstack((alt, vx, vz, z, tgo)).T

    # create feasibility dataset
    feasibility_data = np.hstack((samples, np.zeros((samples.shape[0], 1))))

    return feasibility_data


if __name__ == '__main__':
    # measure time
    t0 = time.time()
    main()
    print("--- %.3f minutes ---" % ((time.time() - t0)/60))

