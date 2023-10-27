import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from tqdm import tqdm
import matplotlib

import sys
sys.path.append('../')
from src import lcvx as lc
from src.nn_guidance import get_nn_reachset
from src.learning import NeuralReach
from config import slr_config


matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Times New Roman'
matplotlib.rcParams['mathtext.it'] = 'Times New Roman:italic'
matplotlib.rcParams['mathtext.bf'] = 'Times New Roman:bold'
matplotlib.rcParams['font.size'] = 16

def solve_soft_landing(rocket, N, x0, tgo):
    lcvx = lc.LCvxMinFuel(rocket, N)
    prob = lcvx.problem(x0=x0, tf=tgo)
    prob.solve(verbose=False)
    sol = lc.get_vars(prob, ['X', 'U'])
    X_sol = sol['X']
    U_sol = sol['U']
    r, v, z, u, sigma = lcvx.recover_variables(X_sol, U_sol)
    return r, v, z, u, sigma

def main():
    
    # Parameters
    rocket, N = slr_config()
    x0 = np.array([0, 0, 1500, 30., 0., -30., np.log(rocket.mwet)])
    tgo = 60.
    dt = tgo / N

    # Solve minimum fuel soft landing problem
    r, v, z, u, sigma = solve_soft_landing(rocket, N, x0, tgo)

    # Load NN model
    model = NeuralReach()
    model.load_state_dict(torch.load('../out/models/20230804_150021/model_final.pth'))
    model.eval()

    # Compute NN reachset
    reach_sets = []
    for i in tqdm(range(N)):
        x0_i = torch.from_numpy(np.hstack((r[:, i], v[:, i], z[i]))).float()
        tgo_i = torch.tensor(tgo - i*dt).float()
        feasible, reach_set = get_nn_reachset(x0_i, tgo_i, model)
        reach_set = reach_set.detach().numpy()
        # sort reachset points by angle
        center = np.mean(reach_set[:, :2], axis=0)
        angles = np.arctan2(reach_set[:, 1] - center[1], reach_set[:, 0] - center[0])
        idx = np.argsort(angles)
        reach_sets.append(reach_set[idx])
    reach_sets = np.array(reach_sets)
    
    """
    # Plot all reachsets
    fig, ax = plt.subplots()
    for i, reach_set in enumerate(reach_sets):
        if i % 10 == 0:
            # reachability set 
            poly = plt.Polygon(reach_set, fill=False, linewidth=0.5)
            ax.add_patch(poly)
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]')
    fov_radius0 = x0[2] * np.tan(rocket.fov/2)
    plt.xlim(-fov_radius0*1.2, fov_radius0*1.2)
    plt.ylim(-fov_radius0*1.2, fov_radius0*1.2)
    plt.grid()
    plt.show()

    # Plot trajectory
    fig, ax = plt.subplots()
    ax.plot(r[0], r[2], 'k', linewidth=0.5)
    ax.set_xlabel('x, m')
    ax.set_ylabel('z, m')
    # Plot glide slope angle constraint
    xmin, xmax = ax.get_xlim()
    gsa = rocket.gsa
    x_range = np.linspace(xmin, xmax, 100)
    ax.plot(x_range, np.tan(gsa) * np.abs(x_range - r[0, -1]), 'k--', linewidth=0.5, alpha=0.2)
    ax.fill_between(x_range, np.tan(gsa) * np.abs(x_range - r[0,-1]), color='gray', alpha=0.5)
        
    plt.show()
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    spacing = 0.05 * (max(r[2]) - min(r[2])) / len(reach_sets)  # This can be adjusted for optimal spacing.

    all_x_points = []
    all_y_points = []
    for i, reach_set in enumerate(reach_sets):
        if i % 10 == 0:
            # reachability set 
            poly = plt.Polygon(reach_set, fill=False, linewidth=1.0)
            ax1.add_patch(poly)
            # Annotate reachable sets with time instances on ax1
            centroid = np.mean(reach_set, axis=0)
            #ax1.annotate(f'k={i}', (centroid[0], centroid[1]), fontsize=8, ha='center')
            if i <= 50:
                x_annotate = min(reach_set[:, 0])
            else:
                x_annotate = max(reach_set[:, 0])
            y_annotate = 0
            ax1.annotate(str(i), (x_annotate, y_annotate), fontsize=9, ha='left')
            # Store all x and y points for adjusting limits later
            all_x_points.extend([point[0] for point in reach_set])
            all_y_points.extend([point[1] for point in reach_set])
            # Annotate trajectory with time instances on ax2
            if i < r[0].shape[0]:  # Ensure there's a corresponding trajectory point
                ax2.annotate(f'k={i}', (r[0][i]*1.02, r[2][i]), fontsize=9)

    ax1.set_aspect('equal')
    ax1.set_xlabel('x, m')
    ax1.set_ylabel('y, m')
    # Adjust x and y limits based on reachable set points
    ax1.set_xlim(min(all_x_points) * 1.1, max(all_x_points) * 1.1)
    ax1.set_ylim(min(all_y_points) * 1.1, max(all_y_points) * 1.1)
    ax1.grid()

    # Plot trajectory on ax2
    ax2.plot(r[0], r[2], 'k', linewidth=0.5)
    ax2.set_xlabel('x, m')
    ax2.set_ylabel('z, m')

    # Plot glide slope angle constraint
    xmin, xmax = ax2.get_xlim()
    gsa = rocket.gsa
    x_range = np.linspace(xmin, xmax, 100)
    ax2.plot(x_range, np.tan(gsa) * np.abs(x_range - r[0, -1]), 'k--', linewidth=0.5, alpha=0.2)
    ax2.fill_between(x_range, np.tan(gsa) * np.abs(x_range - r[0,-1]), color='gray', alpha=0.5)

    plt.tight_layout()

    # save as pdf and png
    plt.savefig('../out/reachset.pdf')
    plt.savefig('../out/reachset.png', dpi=300)
    




    

if __name__ == '__main__':
    main()

