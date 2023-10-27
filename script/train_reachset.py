import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import numpy as np
import sys
import time
import os
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.append('../')
from src.learning import NeuralReach, ReachsetDataset, inverse_transform_reachsetparam
from src.reachset import reach_ellipses_torch, read_reachset
from src.learning.transform import FOV
FOV_npy = FOV.numpy()

debug = False

torch.autograd.set_detect_anomaly(True)

def main(device='cpu'):
    start = time.time()
    print('start preprocessing')

    # make dirs
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d_%H%M%S")
    model_dir = f'../out/models/{dt_string}'
    os.makedirs(model_dir, exist_ok=True)

    # tensorboard
    os.makedirs(f'../out/tensorboard/{dt_string}', exist_ok=True)
    writer = SummaryWriter(f'../out/tensorboard/{dt_string}')

    # parameters
    lr = 1e-3  # 1e-4
    batch_size = 2048  # 1024
    num_epochs = 100  # 100
    print_itr = 100  # 100 
    val_itr = 100  # 100
    save_itr = 100  # 100
    max_datapoints = None  # None

    # create model
    model = NeuralReach()
    model = model.to(device)

    # load data
    _, _, reachset = read_reachset('../out/20230803_021318/reachset.json', nmax=max_datapoints)
    dataset = ReachsetDataset(reachset)
    n_data = len(dataset)
    train_size = int(0.95 * n_data)
    test_size = n_data - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])

    # create dataloader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    # create loss function and optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print('end preprocessing, time: {:.3f}'.format(time.time() - start))
    print('start training')

    # train
    itr = 0
    start = time.time()
    model.train()
    running_loss_dict = {
        'total': 0.,
        'feasibility': 0.,
        'points': 0.,
        'bound': 0.,
        'jump': 0.
    }
    for epoch in range(num_epochs):
        for i, data in enumerate(train_loader):
            # transfer data to device
            for j in range(len(data)):
                data[j] = data[j].to(device)

            ic_ = data[0]
            #ic_, x_data, y_data, xmin_data, xmax_data, alt, fov_radius, feasible_data = data
            #ic_, x_data, y_data, xmin_data, xmax_data, alt, fov_radius, feasible_data = ic_.to(device), x_data.to(device), y_data.to(device), xmin_data.to(device), xmax_data.to(device), alt.to(device), fov_radius.to(device), feasible_data.to(device)

            # NN forward pass
            nn_output = model(ic_)

            # compute feasible reachset
            feasible_data = data[7]
            if torch.sum(feasible_data) > 0:  # if there is at least one feasible reachset
                x_data, y_data, xmin_data, xmax_data, x_pred, y_pred, xmin_pred, xmax_pred, jump_at_xp, params_pred = feasible_reachset(data, nn_output)

            # --------------------
            # compute loss
            # --------------------
            feasible_pred = nn_output[:, 0]
            fov_radius = data[6]
            loss, feasible_loss, points_loss, bound_loss, jump_loss = compute_loss(x_data, y_data, xmin_data, xmax_data, x_pred, y_pred, xmin_pred, xmax_pred, feasible_data, feasible_pred, fov_radius, jump_at_xp)

            # --------------------
            # update weights
            # --------------------

            optimizer.zero_grad()
            loss.backward()

            if debug:
                # Check the gradients
                for name, param in model.named_parameters():
                    print(f"Gradient of {name} wrt loss: {param.grad}")

            optimizer.step()
            if debug:
                # Again, check the gradients after optimizer step
                for name, param in model.named_parameters():
                    print(f"Gradient of {name} after optimization step: {param.grad}")

            # ------------------- #
            #  Save model weights #
            # ------------------- #
            if itr % save_itr == 0:
                torch.save(model.state_dict(), os.path.join(model_dir, 'model_{}.pth'.format(itr)))

            # ------------------- #
            #    Monitor loss     #
            # ------------------- #

            running_loss_dict = {
                'total': running_loss_dict.get('total', 0) + loss.item(),
                'feasibility': running_loss_dict.get('feasibility', 0) + feasible_loss.item(),
                'points': running_loss_dict.get('points', 0) + points_loss.item(),
                'bound': running_loss_dict.get('bound', 0) + bound_loss.item(),  # 'bound' is a reserved keyword in python, so we use 'jump' instead of 'bound
                'jump': running_loss_dict.get('jump', 0) + jump_loss.item()
            }
            
            if itr % print_itr == print_itr - 1:
                elapsed = time.time() - start
                print('[{}, {}] loss: {:.3f}, time: {:.3f}'.format(epoch+1, i+1, running_loss_dict['total']/print_itr, elapsed))
                print('  feasibility: {:.3f}'.format(running_loss_dict['feasibility']/print_itr))
                print('  points: {:.3f}'.format(running_loss_dict['points']/print_itr))
                print('  bound: {:.3f}'.format(running_loss_dict['bound']/print_itr))
                print('  jump: {:.3f}'.format(running_loss_dict['jump']/print_itr))

                # Tensorboard
                writer.add_scalar('Train/total-loss', running_loss_dict['total']/print_itr, itr)
                for key, value in running_loss_dict.items():
                    writer.add_scalar(f'Train/{key}-loss', value/print_itr, itr)

                fig = visualize_prediction(data, feasible_pred, params_pred, x_pred, y_pred, return_fig=True)
                writer.add_figure('Train/prediction', fig, itr)

                # reset running loss
                for key in running_loss_dict.keys():
                    running_loss_dict[key] = 0.
            
            itr += 1

            # ------------------- #
            #   Validate model    #
            # ------------------- #
            if itr % val_itr == val_itr - 1:
                model.eval()
                test_model(model, test_loader, writer, itr, device=device)
                model.train()

    # save final model
    torch.save(model.state_dict(), os.path.join(model_dir, 'model_final.pth'))
    writer.close()


def visualize_prediction(data, feasible_pred, params_pred, x_pred, y_pred, idx=0, return_fig=False):

    _, x_data, y_data, _, _, _, fov_r, feasible_data = data
    xmin_pred, xmax_pred, alpha_pred, yp_pred, a1_pred, a2_pred = params_pred

    # Load reachable set point data
    feasible_data = feasible_data[idx].detach().numpy()
    feasible_pred = feasible_pred[idx].detach().numpy()
    x_pred = x_pred[idx, :].detach().numpy()
    y_pred = y_pred[idx, :].detach().numpy()
    x_data = x_data[idx, :].detach().numpy()
    y_data = y_data[idx, :].detach().numpy()
    fov_r = fov_r[idx].detach().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # add title
    ax.set_title(f'Feasibility score: {feasible_pred:.3f} | Ground truth: {feasible_data}')

    # training data points of reachable set
    fov_circle = plt.Circle((0, 0), fov_r, fill=None, edgecolor='k', linestyle='--', label='Sensor Field of View')
    ax.add_patch(fov_circle)
    ax.scatter(x_data, y_data, label='center', marker='x', color='k')

    # predicted points of reachable set
    ax.scatter(x_pred, y_pred, label='prediction', marker='x', color='r')

    # predicted border of reachable set
    xmin_pred = xmin_pred[idx].detach()
    xmax_pred = xmax_pred[idx].detach()
    alpha_pred = alpha_pred[idx].detach()
    yp_pred = yp_pred[idx].detach()
    a1_pred = a1_pred[idx].detach()
    a2_pred = a2_pred[idx].detach()

    n = 100
    reach_pts = torch.zeros((2*n - 2, 2))
    xs = torch.linspace(xmin_pred.item(), xmax_pred.item(), n)
    ys, _ = reach_ellipses_torch(X=xs, param=(xmin_pred, xmax_pred, alpha_pred, yp_pred, a1_pred, a2_pred))
    reach_pts[:n, 0] = xs
    reach_pts[:n, 1] = ys
    reach_pts[n:, 0] = xs[1:-1]
    reach_pts[n:, 1] = -ys[1:-1]
    # sort reachset points by angle
    reach_pts = reach_pts.detach().numpy()
    center = np.mean(reach_pts, axis=0)
    angles = np.arctan2(reach_pts[:, 1] - center[1], reach_pts[:, 0] - center[0])
    idx = np.argsort(angles)
    reach_pts = reach_pts[idx, :]
    # plot reachset border
    poly = plt.Polygon(reach_pts, fill=None, edgecolor='b', linestyle='--', label='Reachset')
    ax.add_patch(poly)
    
    ax.set_aspect('equal')
    plt.xlim(-fov_r*1.2, fov_r*1.2)
    plt.ylim(-fov_r*1.2, fov_r*1.2)
    plt.xlabel('X [m]]')
    plt.ylabel('Y [m]')
    plt.grid()
    plt.legend()

    if return_fig:
        return fig
    else:
        plt.show()


def test_model(model, test_loader, writer, itr, device='cpu', num_test=1):
    model.eval()
    running_loss_dict = {
        'total': 0.,
        'feasibility': 0.,
        'points': 0.,
        'bound': 0.,
        'jump': 0.
    }
    assert num_test > 0

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            # transfer data to device
            for j in range(len(data)):
                data[j] = data[j].to(device)
            

            # NN forward pass
            ic_ = data[0]
            nn_output = model(ic_)

            # compute feasible reachset
            feasible_data = data[7]
            if torch.sum(feasible_data) > 0:  # if there is at least one feasible reachset
                x_data, y_data, xmin_data, xmax_data, x_pred, y_pred, xmin_pred, xmax_pred, jump_at_xp, params_pred = feasible_reachset(data, nn_output)

            # --------------------
            # compute loss
            # --------------------
            feasible_pred = nn_output[:, 0]
            fov_radius = data[6]
            loss, feasible_loss, points_loss, bound_loss, jump_loss = compute_loss(x_data, y_data, xmin_data, xmax_data, x_pred, y_pred, xmin_pred, xmax_pred, feasible_data, feasible_pred, fov_radius, jump_at_xp)

            fig = visualize_prediction(data, feasible_pred, params_pred, x_pred, y_pred, return_fig=True)

            # ------------------- #
            #    Monitor loss     #
            # ------------------- #

            running_loss_dict = {
                'total': running_loss_dict.get('total', 0) + loss.item(),
                'feasibility': running_loss_dict.get('feasibility', 0) + feasible_loss.item(),
                'points': running_loss_dict.get('points', 0) + points_loss.item(),
                'bound': running_loss_dict.get('bound', 0) + bound_loss.item(),  # 'bound' is a reserved keyword in python, so we use 'jump' instead of 'bound
                'jump': running_loss_dict.get('jump', 0) + jump_loss.item()
            }

            if writer is not None:
                writer.add_figure('Val/prediction', fig, global_step=itr)

            if i >= num_test-1:
                break

        if writer is not None:
            # Tensorboard
            writer.add_scalar('Val/total-loss', running_loss_dict['total']/num_test, itr)
            for key, value in running_loss_dict.items():
                writer.add_scalar(f'Val/{key}-loss', value/num_test, itr)


def feasible_reachset(data, nn_output):
    # unpack data
    _, x_data, y_data, xmin_data, xmax_data, alt, _, feasible_data = data

    # unpack nn output
    _, xmin_pred_, xmax_pred_, alpha_pred_, yp_pred_, a1_pred_, a2_pred_ = nn_output.T

    # Compute predicted reachable set
    assert torch.sum(feasible_data) > 0, 'No feasible data in batch'
    # 1. screen out infeasible data
    #   ground truth (physical dimensions)
    x_data = x_data[feasible_data]
    y_data = y_data[feasible_data]
    xmin_data = xmin_data[feasible_data].reshape(-1, 1)
    xmax_data = xmax_data[feasible_data].reshape(-1, 1)
    alt = alt[feasible_data]
    #   prediction (normalized)
    xmin_pred_ = xmin_pred_[feasible_data]
    xmax_pred_ = xmax_pred_[feasible_data]
    alpha_pred_ = alpha_pred_[feasible_data]
    yp_pred_ = yp_pred_[feasible_data]
    a1_pred_ = a1_pred_[feasible_data]
    a2_pred_ = a2_pred_[feasible_data]
    # 2. compute predicted points
    #  convert prediction to physical dimensions
    params_pred = inverse_transform_reachsetparam(xmin_pred_, xmax_pred_, alpha_pred_, yp_pred_, a1_pred_, a2_pred_, alt)
    #  compute predicted points
    xmin_pred, xmax_pred, _, _, _, _ = params_pred
    xmin_pred = xmin_pred.reshape(-1, 1)
    xmax_pred = xmax_pred.reshape(-1, 1)
    _x_data = (x_data - xmin_data) / (xmax_data - xmin_data)
    x_pred = xmin_pred + (xmax_pred - xmin_pred) * _x_data
    y_pred, jump_at_xp = reach_ellipses_torch(X=x_pred, param=params_pred)

    return x_data, y_data, xmin_data, xmax_data, x_pred, y_pred, xmin_pred, xmax_pred, jump_at_xp, params_pred


def compute_loss(x_data, y_data, xmin_data, xmax_data, x_pred, y_pred, xmin_pred, xmax_pred, feasible_data, feasible_pred, fov_radius, jump_at_xp):
    mse = nn.MSELoss()

    # compute feasibility loss
    feasible_loss = mse(feasible_pred, feasible_data.double())

    if torch.sum(feasible_data) > 0:
        # 1. normalize x and y
        fov_radius = fov_radius[feasible_data].reshape(-1, 1)
        x_data_ = x_data / fov_radius
        y_data_ = y_data / fov_radius
        xmin_data_, xmax_data_ = xmin_data / fov_radius, xmax_data / fov_radius
        x_pred_ = x_pred / fov_radius
        y_pred_ = y_pred / fov_radius
        xmin_pred_, xmax_pred_ = xmin_pred / fov_radius, xmax_pred / fov_radius
        # 2. concatenate x and y
        xy_data_ = torch.cat((x_data_, y_data_), dim=1)
        xy_pred_ = torch.cat((x_pred_, y_pred_), dim=1)
        # 3. compute loss
        points_loss = mse(xy_pred_, xy_data_)
        bound_loss = mse(xmin_pred_, xmin_data_) + mse(xmax_pred_, xmax_data_)

        # compute jump loss
        jump_at_xp_ = jump_at_xp / fov_radius
        jump_loss = torch.mean(jump_at_xp_)
    else:
        points_loss = torch.tensor(0.)
        bound_loss = torch.tensor(0.)
        jump_loss = torch.tensor(0.)

    # compute total loss
    loss = feasible_loss + points_loss + bound_loss + 0.05 * jump_loss

    return loss, feasible_loss, points_loss, bound_loss, jump_loss


if __name__ == '__main__':
    device = 'cpu'
    main(device=device)






