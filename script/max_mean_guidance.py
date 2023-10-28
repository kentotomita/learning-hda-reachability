import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Variable
import numpy as np
import sys
import matplotlib.pyplot as plt

sys.path.append("../")
from src import lcvx as lc
from src.nn_guidance import (
    u_2mean_safety,
    minimize_f_torch,
    transform_u,
    visualize_nn_reachset,
    make_simple_sfmap,
)
from config import slr_config
from src.learning import MLP


def solve_soft_landing(rocket, N, x0, tgo):
    lcvx = lc.LCvxMinFuel(rocket, N)
    prob = lcvx.problem(x0=x0, tf=tgo)
    prob.solve(verbose=False)
    sol = lc.get_vars(prob, ["X", "U"])
    X_sol = sol["X"]
    U_sol = sol["U"]
    r, v, z, u, sigma = lcvx.recover_variables(X_sol, U_sol)
    return r, v, z, u, sigma


def check_dtype(model, sfmap, x0, tgo):
    print("Model Dtype:")
    for name, param in model.named_parameters():
        print(f"  Name: {name} | Type: {param.dtype}")

    print("Safety map Dtype:")
    print(f"  Type: {sfmap.dtype}")

    print("Initial state Dtype:")
    print(f"  Type: {x0.dtype}")

    print("Time to go Dtype:")
    print(f"  Type: {type(tgo)}")


def main():
    # Parameters
    rocket, _ = slr_config()
    x0 = np.array([0, 0, 1500, 5.0, 20.0, -30.0, np.log(rocket.mwet)])
    tgo = 60.0
    dt = 1.0
    N = int(tgo / dt)
    n_horizon = 1
    tgo_scale = 100

    # Load safety map
    sfmap, _ = make_simple_sfmap(
        x_range=(-500, 500), y_range=(-500, 500), n_points=1000, dtype="float64"
    )

    # Load NN model
    model = MLP(
        input_dim=5,  # alt, vx, vz, z, tgo
        output_dim=6,  # a1, a2, b1, b2, xmin, xmax
        hidden_layers=[32, 64, 32],
        activation_fn=nn.ReLU(),
        output_activation=nn.Sigmoid(),
    )
    # model.load_state_dict(torch.load('../out/model.pth'))
    model.load_state_dict(torch.load("../out/models/20230730_141132/model_final.pth"))
    model.eval()

    # Initial guess
    r, v, z, u, sigma = solve_soft_landing(rocket, N, x0, tgo)
    u = u * np.exp(z[:-1])  # N/kg -> N
    U_ = torch.zeros((n_horizon, 3))
    for i in range(n_horizon):
        U_[i, :] = transform_u(
            torch.from_numpy(u[:, i]).float(),
            rho1=torch.tensor(rocket.rho1),
            rho2=torch.tensor(rocket.rho2),
            pa=torch.tensor(rocket.pa),
        )
    tgo_next = tgo - dt * n_horizon
    tgo_next_ = tgo_next / tgo_scale
    print("Initial guess: U_: {} | tgo_next_: {}".format(U_, tgo_next_))

    print("Check data type:")
    check_dtype(model, sfmap, x0, tgo)

    # Design variables
    var = Variable(
        torch.cat((U_.reshape(-1), torch.tensor([tgo_next_])), dim=0),
        requires_grad=True,
    )

    # Parameters
    x0 = torch.from_numpy(x0).double().requires_grad_(True)
    tgo = torch.tensor(tgo).requires_grad_(True)
    sfmap = sfmap.requires_grad_(True)
    bounds = [(0.0, 1.0) for _ in range(n_horizon * 3)] + [(0.0, 1.0)]

    # Define objective function and its derivative
    def f(var):
        U_ = var[:-1].reshape(n_horizon, 3)
        tgo_next_ = var[-1]
        tgo_next = tgo_next_ * tgo_scale
        mean_safety, _ = u_2mean_safety(
            u_=U_,
            tgo_next=tgo_next,
            x0=x0,
            dt=dt,
            rocket=rocket,
            sfmap=sfmap,
            model=model,
            border_sharpness=1,
        )

        return -mean_safety

    # Optimize
    for i in range(1):
        # results before optimization
        visualize_resutls(
            var,
            x0,
            tgo,
            dt,
            rocket,
            sfmap,
            model,
            n_horizon=n_horizon,
            tgo_scale=tgo_scale,
        )

        # solve
        var_opt = minimize_f_torch(f, var, bounds=bounds, disp=True, niter=10)

        visualize_resutls(
            var_opt,
            x0,
            tgo,
            dt,
            rocket,
            sfmap,
            model,
            n_horizon=n_horizon,
            tgo_scale=tgo_scale,
        )


def visualize_resutls(
    var, x0, tgo, dt, rocket, sfmap, model, n_horizon=1, tgo_scale=100
):
    U_ = var[:-1].reshape(n_horizon, 3)
    tgo_next_ = var[-1]
    tgo_next = tgo_next_ * tgo_scale
    print("x0: {}".format(x0), "tgo: {}".format(tgo))
    mean_safety, reach_mask = u_2mean_safety(
        U_, tgo_next, x0, dt, rocket, sfmap, model, border_sharpness=10
    )
    print(
        "Mean safety: {:.4f} | U_: {} | tgo_next: {}".format(
            mean_safety.item(), U_, tgo_next.item()
        )
    )

    # Plot
    visualize_nn_reachset(reach_mask, sfmap, nskip=10)


if __name__ == "__main__":
    main()
