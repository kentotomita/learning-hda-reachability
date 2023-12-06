"""Generate dataset for feasibility learning. (Step 4)"""
import argparse
import multiprocessing as mp
import numpy as np
import time
from tqdm import tqdm
from scipy.spatial import ConvexHull
import sys
sys.path.append('../')
from src.learning import transform_ic, inverse_transform_ic
from src.reachset.ic_sampler import random_sampling_outside_hull, inside_hull


def main():
    start = time.time()

    # read command line inputs
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_proc', type=int, default=8)
    parser.add_argument('--reduced_ctrlable_set', action='store_true')
    args = parser.parse_args()

    # load controllable set and create convex hull
    hull_ctrlable, ctrlable_set_ = get_ctrlable_hull(args.reduced_ctrlable_set)

    # load reachset
    data_random = np.load("saved/controllable_set/reachset_train/20231115-025054/data_random.npy")
    data_structured = np.load("saved/controllable_set/reachset_train/20231115-025054/data_structured.npy")

    feasibility_random = np.hstack((data_random[:, :5], np.zeros((data_random.shape[0], 1))))
    feasibility_structured = np.hstack((data_structured[:, :5], np.zeros((data_structured.shape[0], 1))))
    feasibility_data = np.vstack((feasibility_random, feasibility_structured))

    # Randomly flip vx; the second column
    random_binary = np.random.randint(0, 2, feasibility_data.shape[0])
    feasibility_data[random_binary == 1, 1] = -feasibility_data[random_binary == 1, 1]

    # normalize data
    feasibility_data_ = feasibility_data.copy()
    alt, vx, vz, mass, tgo = feasibility_data_[:, 0], feasibility_data_[:, 1], feasibility_data_[:, 2], feasibility_data_[:, 3], feasibility_data_[:, 4]
    alt_, vx_, vz_, z_, tgo_ = transform_ic(alt, vx, vz, np.log(mass), tgo, vx_negative_allowed=True)
    feasibility_data_ = np.vstack((alt_, vx_, vz_, z_, tgo_, np.zeros_like(tgo_))).T

    # Check feasibility
    print('Checking feasibility of reachable set data...')
    if args.n_proc == 1:
        for i in tqdm(range(feasibility_data_.shape[0])):
            if inside_hull(feasibility_data_[i, :5], hull_ctrlable.equations):
                feasibility_data[i, -1] = 1.0
    else:
        params = []
        for i in range(feasibility_data_.shape[0]):
            params.append((feasibility_data_[i], hull_ctrlable.equations))

        feasibility_data__list = []
        with mp.Pool(args.n_proc) as pool:
            for out in pool.starmap(compute_feasibility, params):
                feasibility_data__list.append(out)
        feasibility_data_ = np.vstack(feasibility_data__list)
        alt_, vx_, vz_, z_, tgo_, feasibility_ = feasibility_data_[:, 0], feasibility_data_[:, 1], feasibility_data_[:, 2], feasibility_data_[:, 3], feasibility_data_[:, 4], feasibility_data_[:, 5]
        alt, vx, vz, z, tgo = inverse_transform_ic(alt_, vx_, vz_, z_, tgo_, vx_negative_allowed=True)
        mass = np.exp(z)
        feasibility_data = np.vstack((alt, vx, vz, mass, tgo, feasibility_)).T

    n_feasible = np.sum(feasibility_data[:, -1] == 1.0)
    n_infeasible = np.sum(feasibility_data[:, -1] == 0.0)
    print('n_feasible: {}, n_infeasible: {}'.format(n_feasible, n_infeasible))
    lb = np.min(ctrlable_set_, axis=0)
    ub = np.max(ctrlable_set_, axis=0)

    # Sample infeasible points from the convex hull spanned by the controllable set
    n_more = n_feasible - n_infeasible
    assert n_more > 0
    feasibility_data_infeasible = sample_outside_ctrlable_hull(n_more, hull_ctrlable, lb, ub)

    # combine feasible and infeasible data
    feasibility_data = np.vstack((feasibility_data, feasibility_data_infeasible))

    # save
    print('Saving...')
    np.save('saved/controllable_set/reachset_train/20231115-025054/feasibility_data.npy', feasibility_data)

    # save mata data; shapes of generated samples, number of feasible and infeasible points, time
    with open('saved/controllable_set/reachset_train/20231115-025054/meta.txt', 'w') as f:
        f.write('feasibility_data: {}\n'.format(feasibility_data.shape))
        f.write('n_feasible: {}\n'.format(n_feasible))
        f.write('n_infeasible: {}\n'.format(feasibility_data_infeasible.shape[0]))
        f.write('time: {}\n min'.format((time.time() - start)/60))


def compute_feasibility(feasibility_data_point_, equations):

    feasibility = inside_hull(feasibility_data_point_[:5], equations)

    feasibility_data_point_[-1] = feasibility

    return feasibility_data_point_


def get_ctrlable_hull(reduced_ctrlable_set=False):

    # load controllable set
    ctrlable_set = np.load('saved/controllable_set/data.npy')

    np.random.seed(0)
    rand_idx = np.random.choice(ctrlable_set.shape[0], 1000, replace=False)
    ctrlable_set = ctrlable_set[rand_idx, :]
    print('ctrlable_set: {}'.format(ctrlable_set.shape))

    # create convex hull of controllable set; negative vx is allowed.
    print('Creating convex hull...')
    indices = [2, 3, 5, 6, 7]
    ctrlable_set = ctrlable_set[:, indices]
    ctrlable_set = ctrlable_set[ctrlable_set[:, 1] > 0.0, :]

    # create an array where vx (index 1) is flipped
    if reduced_ctrlable_set:
        random_binary = np.random.randint(0, 2, ctrlable_set.shape[0])
        ctrlable_set = ctrlable_set[random_binary == 0, :]
        ctrlable_set_negative_vx = ctrlable_set[random_binary == 1, :]
    else:
        ctrlable_set_negative_vx = ctrlable_set.copy()
        ctrlable_set_negative_vx[:, 1] = -ctrlable_set_negative_vx[:, 1]
    ctrlable_set = np.vstack((ctrlable_set, ctrlable_set_negative_vx))
    # shuffle controllable set
    np.random.shuffle(ctrlable_set)

    alt, vx, vz, mass, tgo = ctrlable_set[:, 0], ctrlable_set[:, 1], ctrlable_set[:, 2], ctrlable_set[:, 3], ctrlable_set[:, 4]
    z = np.log(mass)
    alt_, vx_, vz_, z_, tgo_ = transform_ic(alt, vx, vz, z, tgo, vx_negative_allowed=True)
    ctrlable_set_ = np.vstack((alt_, vx_, vz_, z_, tgo_)).T
    
    hull_ctrlable = ConvexHull(ctrlable_set_, qhull_options='Q12 QJ')

    return hull_ctrlable, ctrlable_set_


def sample_outside_ctrlable_hull(n_samples, hull_ctrlable, lb, ub):

    # sample outside the convex hull
    print('Random sampling...')
    samples = random_sampling_outside_hull(hull_ctrlable.equations, (lb, ub), n_samples)
    
    # denormalize
    alt_, vx_, vz_, z_, tgo_ = samples[:, 0], samples[:, 1], samples[:, 2], samples[:, 3], samples[:, 4]
    alt, vx, vz, z, tgo = inverse_transform_ic(alt_, vx_, vz_, z_, tgo_, vx_negative_allowed=True)
    mass = np.exp(z)
    samples = np.vstack((alt, vx, vz, mass, tgo)).T

    # create feasibility dataset
    feasibility_data = np.hstack((samples, np.zeros((samples.shape[0], 1))))

    return feasibility_data


if __name__ == '__main__':
    # measure time
    t0 = time.time()
    main()
    print("--- %.3f minutes ---" % ((time.time() - t0)/60))

