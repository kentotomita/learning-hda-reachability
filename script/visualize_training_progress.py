import numpy as np
import sys
import matplotlib.pyplot as plt
import os
import torch
import matplotlib
from tqdm import tqdm

sys.path.append('../')
from src.reachset import read_reachset
fov = 15 * np.pi / 180
import sys
sys.path.append('../')
from src import lcvx as lc
from src.nn_guidance import get_nn_reachset
from src.learning import NeuralReach

np.set_printoptions(precision=1)

matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Times New Roman'
matplotlib.rcParams['mathtext.it'] = 'Times New Roman:italic'
matplotlib.rcParams['mathtext.bf'] = 'Times New Roman:bold'
matplotlib.rcParams['font.size'] = 16


if __name__=="__main__":
    _, _, reachset = read_reachset('../out/20230803_021318/reachset_sample.json')

    rng = np.random.RandomState(0)
    reachset = rng.permutation(reachset)

    data_list = []
    data_idx_list = []
    for i in range(min(len(reachset), 100)):
        data = reachset[i]
        rf = np.array(data['rf'])
        if not np.all(np.isnan(rf)):
            data_list.append(data)
            data_idx_list.append(i)

    n_data = len(data_list)
    print('Number of data points: ', n_data)

    # list NN model
    dt_string = '20230804_150021'
    model_dir = f'../out/models/{dt_string}'
    model_files = os.listdir(model_dir)
    # only extract .pth files
    model_files = [f for f in model_files if f.endswith('.pth')]
    model_paths = [os.path.join(model_dir, f) for f in model_files]
    model = NeuralReach()
    # skip number of models
    n_skip = 1
    model_paths = model_paths[::n_skip]
    max_iter = 30000

    # print number of data
    print('Number of data: ', len(data_list))

    for i, data in enumerate(data_list):
        # print progress
        print(f'Progress: {i+1}/{n_data}')
        out_dir = f'../out/figures/{dt_string}/reachset/data_{i}'
        os.makedirs(out_dir, exist_ok=True)

        x0 = data['x0']
        tgo = data['tgo']
        rc = data['rc']
        c_xy = data['c'][:, :2].reshape(-1, 2)
        xy_coords = data['rf'][:, :2].reshape(-1, 2) 

        print('idx: {}, x0: {}, tgo: {}'.format(i, x0, tgo))

        fov_radius = x0[2] * np.tan(fov/2)

        for f in tqdm(model_paths):
            # extract iteration number from file name
            fbase = os.path.basename(f)
            fbase = fbase.split('.')[0]
            fbase = fbase.split('_')[-1]
            iter_num = int(fbase)
            fig, ax = plt.subplots()
            fov_circle = plt.Circle(x0[:2], fov_radius, fill=None, edgecolor='k', linestyle='--', label='Sensor Field of View')
            ax.add_patch(fov_circle)
            ax.scatter(xy_coords[:, 0], xy_coords[:, 1], marker='x', color='k', label='Reachable Set Data')

            model.load_state_dict(torch.load(f))
            model.eval()
            
            # convert to torch tensor
            x0 = torch.tensor(x0, dtype=torch.float32)
            tgo = torch.tensor(tgo, dtype=torch.float32)
            feasible, reach_set = get_nn_reachset(x0, tgo, model)
            reach_set = reach_set.detach().numpy()
            # sort reachset points by angle
            center = np.mean(reach_set[:, :2], axis=0)
            angles = np.arctan2(reach_set[:, 1] - center[1], reach_set[:, 0] - center[0])
            idx = np.argsort(angles)
            reach_set = reach_set[idx]

            # make facecolor transparent with alpha=0.5
            poly = plt.Polygon(reach_set, fill=True, linewidth=0.5, edgecolor='k', label='Reachable Set Prediction', alpha=0.1, facecolor='r')
            ax.add_patch(poly)

            ax.axis('equal')
            # make legend at the bottom right corner
            ax.legend(loc='lower right', bbox_to_anchor=(1, 0.05))
            ax.set_xlabel('x, m')
            ax.set_ylabel('y, m')
            ax.grid(True)
            ax.set_title(f'Reachable Set Prediction: iteration {iter_num:06d}')

            # save figure
            fig.savefig(os.path.join(out_dir, f'reachset_iter_{iter_num:06d}.png'), dpi=300)

            # close figure
            plt.close(fig)

            if iter_num >= max_iter:
                break
            

