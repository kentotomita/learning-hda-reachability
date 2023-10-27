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
    reachset_path = '../out/20230805_133102/reachset.json'    # testing data
    #reachset_path = '../out/20230803_021318/reachset_sample.json'    # training data
    _, _, reachset = read_reachset(reachset_path)
    data_dtstring = reachset_path.split('/')[-2]
    model_dir = f'../out/models/20230810_214337'
    model_path = os.path.join(model_dir, 'model_final.pth')
    model_dtstring = model_dir.split('/')[-1]
    model_fname = model_path.split('/')[-1].split('.')[0]
    out_dir = f'../out/qualitative_analysis/{data_dtstring}_{model_dtstring}_{model_fname}'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # save reachset path and model path as a text file
    with open(os.path.join(out_dir, 'reachset_path.txt'), 'w') as f:
        f.write(reachset_path)
        f.write('\n')
        f.write(model_path)
        f.write('\n')

    # load model
    model = NeuralReach()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    rng = np.random.RandomState(0)
    reachset = rng.permutation(reachset)

    data_list = []
    data_idx_list = []
    for i in range(min(len(reachset), 1000)):
        data = reachset[i]
        rf = np.array(data['rf'])
        if not np.all(np.isnan(rf)):
            data_list.append(data)
            data_idx_list.append(i)

    # print number of data
    print('Number of data: ', len(data_list))

    for i, data in enumerate(tqdm(data_list)):
        # print progress
        x0 = data['x0']
        tgo = data['tgo'][0]
        rc = data['rc']
        c_xy = data['c'][:, :2].reshape(-1, 2)
        xy_coords = data['rf'][:, :2].reshape(-1, 2) 

        print('idx: {}, x0: {}, tgo: {}'.format(i, x0, tgo))

        fov_radius = x0[2] * np.tan(fov/2)

        fig, ax = plt.subplots()
        fov_circle = plt.Circle(x0[:2], fov_radius, fill=None, edgecolor='k', linestyle='--', label='Sensor Field of View')
        ax.add_patch(fov_circle)
        ax.scatter(xy_coords[:, 0], xy_coords[:, 1], marker='x', color='k', label='Reachable Set Data')
        
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
        
        ax.set_title(f'Alt: {x0[2]:.0f} m, V: ({x0[3]:.0f}, {x0[5]:.0f}) m/s, TGO: {tgo:.0f} s')

        # save figure
        fig.savefig(os.path.join(out_dir, f'reachset_{i}.png'), bbox_inches='tight', dpi=300)

        # close figure
        plt.close(fig)


