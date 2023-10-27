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
from src.nn_guidance import get_nn_reachset_param
from src.learning import NeuralReach
from src.reachset import reach_ellipses_torch

np.set_printoptions(precision=5)

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
    model_path = os.path.join(model_dir, 'model_12000.pth')
    model_dtstring = model_dir.split('/')[-1]
    model_fname = model_path.split('/')[-1].split('.')[0]
    out_dir = f'../out/quantitative_nn_analysis/{data_dtstring}_{model_dtstring}_{model_fname}'
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

    feas_preds = []
    feas_true = []
    points_errors = []
    state_list = []

    for data in reachset:
        x0 = data['x0']
        tgo = data['tgo']
        rc = data['rc']
        c_xy = data['c'][:, :2].reshape(-1, 2)
        xy_coords = data['rf'][:, :2].reshape(-1, 2) 

        alt = x0[2]
        if alt >=250:

            xmin_true = np.nanmin(xy_coords[:, 0])
            xmax_true = np.nanmax(xy_coords[:, 0])
            
            # compute prediction
            x0 = torch.tensor(x0, dtype=torch.float32)
            tgo = torch.tensor(tgo, dtype=torch.float32)
            
            feas_pred, xmin, xmax, alpha, yp, a1, a2, rotation_angle, center = get_nn_reachset_param(x0, tgo, model, full=False)
            feas_preds.append(feas_pred.detach().numpy())

            if np.all(np.isnan(xy_coords)):
                feas_true.append(False)
            else:
                feas_true.append(True)

                # extract non-nan data
                xy_coords = xy_coords[~np.isnan(xy_coords[:, 0])]
            
                # convert to torch
                xy_coords = torch.tensor(xy_coords, dtype=torch.float32)
                xmax_true = torch.tensor(xmax_true, dtype=torch.float32)
                xmin_true = torch.tensor(xmin_true, dtype=torch.float32)
                xs = xmin + (xmax - xmin) /(xmax_true - xmin_true) * (xy_coords[:, 0] - xmin_true)
                ys, _ = reach_ellipses_torch(X=xs, param=(xmin, xmax, alpha, yp, a1, a2))
                # convert to numpy
                xs = xs.detach().numpy()
                ys = ys.detach().numpy()
                xy_coords = xy_coords.detach().numpy()

                # compute error as percent
                points_error = np.linalg.norm(xy_coords - np.hstack([xs.reshape(-1, 1), ys.reshape(-1, 1)]), axis=1)
                #points_error = points_error / np.linalg.norm(xy_coords, axis=1)
                points_errors.append(np.mean(points_error))

                state_list.append(data['x0'])

    # compute accuracy
    feas_preds_binary = np.array(feas_preds) >= 0.5
    feas_true = np.array(feas_true)
    points_errors = np.array(points_errors)
    accuracy = np.mean(feas_preds_binary == feas_true)
    # compute mean and sigma of points error
    mean_points_error = np.mean(points_errors)
    sigma_points_error = np.std(points_errors)
    print(f'Number of data: {len(feas_preds)}')
    print(f'Feasibility Accuracy: {accuracy}')
    print(f'Reach set error: {mean_points_error} +/- {sigma_points_error} m')

    # save accuracy and mean points error
    with open(os.path.join(out_dir, 'accuracy.txt'), 'w') as f:
        f.write(f'Number of data: {len(feas_preds)}')
        f.write('\n')
        f.write(f'Feasibility Accuracy: {accuracy}')
        f.write('\n')
        f.write(f'Reach set error: {mean_points_error} +/- {sigma_points_error} m')
        f.write('\n')
    
    n_data = len(feas_preds)
    n_feas = np.sum(feas_true)
    n_infeas = n_data - n_feas
    feas_thresholds = np.linspace(0.05, 0.95, 1000)
    false_infeas = []
    false_feas = []
    for feas_thresh in feas_thresholds:
        feas_preds_binary = feas_preds >= feas_thresh
        false_infeas.append(np.sum(np.logical_and(feas_preds_binary == False, feas_true == True))/n_infeas)
        false_feas.append(np.sum(np.logical_and(feas_preds_binary == True, feas_true == False))/n_infeas)


    # plot feasibility
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(feas_thresholds, false_infeas, label='False infeasible', c='k', linestyle='--')
    ax.plot(feas_thresholds, false_feas, label='False feasible', c='k', linestyle='-')
    # specify x-ticks
    ax.set_xticks(np.arange(0.05, 0.95 + 0.1, 0.1))
    ax.legend()
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Probability')
    ax.set_title('Feasibility Misclassification Probability')
    ax.grid(True)
    # save as png and pdf
    plt.savefig(os.path.join(out_dir, 'feasibility_misclassification.png'))
    plt.savefig(os.path.join(out_dir, 'feasibility_misclassification.pdf'))

    # plot points error distribution
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.hist(points_errors, bins=100, color='gray')
    ax.set_xlabel('Reachable Set Errors (m)')
    ax.set_ylabel('Counts')
    ax.set_title('Reachable set error distribution')
    ax.grid(True)
    # save as png and pdf
    plt.savefig(os.path.join(out_dir, 'error_distribution.png'))
    plt.savefig(os.path.join(out_dir, 'error_distribution.pdf'))

    # plot points error vs altitude
    altitudes = np.array(state_list)[:, 2]
    num_bins = 5
    bin_edges = np.linspace(250, 1500, num_bins + 1)
    bins = np.digitize(altitudes, bin_edges)
    grouped_errors = {i: [] for i in range(1, num_bins + 1)}

    for b, error in zip(bins, list(points_errors)):
        grouped_errors[b].append(error)

    # Get lists of errors for boxplot
    box_data = [grouped_errors[i] for i in range(1, num_bins + 1)]

    # x-tick labels to represent the altitude ranges
    xticks_labels = [f"{bin_edges[i]:.0f}m - {bin_edges[i+1]:.0f}m" for i in range(num_bins)]

    # Plotting
    plt.figure(figsize=(6, 6))
    plt.boxplot(box_data)
    plt.xticks(np.arange(1, num_bins + 1), xticks_labels, rotation=45, ha="right")
    plt.xlabel("Altitude (m)")
    plt.ylabel("Reachable Set Errors (m)")
    plt.grid(True)
    plt.tight_layout()
    # save as png and pdf
    plt.savefig(os.path.join(out_dir, 'error_vs_altitude.png'))
    plt.savefig(os.path.join(out_dir, 'error_vs_altitude.pdf'))

    # plot points error vs horizontal velocity
    vels = np.array(state_list)[:, 3]
    num_bins = 5
    bin_edges = np.linspace(0, 50, num_bins + 1)
    bins = np.digitize(vels, bin_edges)
    grouped_errors = {i: [] for i in range(1, num_bins + 1)}

    for b, error in zip(bins, list(points_errors)):
        grouped_errors[b].append(error)

    # Get lists of errors for boxplot
    box_data = [grouped_errors[i] for i in range(1, num_bins + 1)]

    # x-tick labels to represent the altitude ranges
    xticks_labels = [f"{bin_edges[i]:.0f}m/s - {bin_edges[i+1]:.0f}m/s" for i in range(num_bins)]

    # Plotting
    plt.figure(figsize=(6, 6))
    plt.boxplot(box_data)
    plt.xticks(np.arange(1, num_bins + 1), xticks_labels, rotation=45, ha="right")
    plt.xlabel("Horizontal Velocity (m/s)")
    plt.ylabel("Reachable Set Errors (m)")
    plt.grid(True)
    plt.tight_layout()
    # save as png and pdf
    plt.savefig(os.path.join(out_dir, 'error_vs_velocity.png'))
    plt.savefig(os.path.join(out_dir, 'error_vs_velocity.pdf'))
    






    