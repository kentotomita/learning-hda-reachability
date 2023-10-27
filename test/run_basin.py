import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import sys
import matplotlib.pyplot as plt

sys.path.append('../')
from src import lcvx as lc
from config import slr_config
from src.learning import MLP
from src.nn_guidance import *


def visualize_results(u_, tgo_next, x0, dt, sfmap, model, rocket):

    x = x0
    for i in range(u_.shape[0]):
        u = inverse_transform_u(u_[i], torch.tensor(rocket.rho1), torch.tensor(rocket.rho2), torch.tensor(rocket.pa))
        x = dynamics(x, u, dt, torch.tensor(rocket.g), torch.tensor(rocket.alpha))

    mean_safety, reach_mask = ic2mean_safety(x, tgo_next, model, sfmap, border_sharpness=1)
    print(f'x: {x}, tgo: {tgo_next}, mean safety: {mean_safety}')
    
    visualize_nn_reachset(reach_mask, sfmap, nskip=10)





def main():
     # Parameters
    rocket, _ = slr_config()
    x0 = torch.from_numpy(np.array([0, 0, 1500, 0., 30., -40., np.log(rocket.mwet)])).double()
    tgo = 60.
    dt = 1.0
    n_horizon = 1

    # load safety map
    sfmap, _ = make_simple_sfmap(x_range=(-500, 500), y_range=(-500, 500), n_points=1000)
    #visualize_sfmap(sfmap, nskip=10)


    # Load NN model
    model = MLP(
        input_dim=5,  # alt, vx, vz, z, tgo
        output_dim=6,  # a1, a2, b1, b2, xmin, xmax
        hidden_layers=[32, 64, 32],
        activation_fn=nn.ReLU(),
        output_activation=nn.Sigmoid()
    )
    model.load_state_dict(torch.load('../out/models/20230730_141132/model_final.pth'))
    model.load_state_dict(torch.load('../out/models/20230730_141132/model_30000.pth'))
    model.eval()

    mean_safety, reach_mask = ic2mean_safety(x0, tgo, model, sfmap, border_sharpness=1)
    print(f'x: {x0}, tgo: {tgo}, mean safety: {mean_safety}')
    visualize_nn_reachset(reach_mask, sfmap, nskip=10)


    # Make objective function
    def f(u_):
        u_ = u_.reshape(n_horizon, 2)
        mean_safety, _ = u_2mean_safety(
            u_=u_,
            tgo_next=torch.tensor(tgo - n_horizon * dt).requires_grad_(True),
            x0=x0.requires_grad_(True),
            dt=dt,
            rocket=rocket,
            sfmap=sfmap.requires_grad_(True),
            model=model,
            border_sharpness=1
        )
        return -mean_safety

    # Initialize variables
    u_ = Variable(torch.zeros(n_horizon * 2), requires_grad=True)

    # evaluate initial guess
    obj0 = f(u_)
    print(f"Initial objective: {obj0}")
    tgo_next = tgo - n_horizon * dt
    visualize_results(u_.reshape(n_horizon, 2), tgo_next, x0, dt, sfmap, model, rocket)


    # Optimize
    #u_opt = gradient_descent(f, u_, bounds=[0., 1.], lr=100., num_epochs=1000, verbose=True)
    #u_opt = newton_method(f, u_, bounds=[0., 1.], lr=1., num_epochs=1000, verbose=True)
    u_opt = basinhopping_torch(f, u_, bounds=[(0., 1.)], niter=10, verbose=True)
    u_opt = u_opt.reshape(n_horizon, 2)

    # Evaluate optimal solution
    obj_opt = f(u_opt)
    print(f"Optimal objective: {obj_opt}")
    visualize_results(u_.reshape(n_horizon, 2), tgo_next, x0, dt, sfmap, model, rocket)


if __name__ == '__main__':
    main()