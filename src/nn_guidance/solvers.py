import torch
from torch.autograd import Variable
from torch.optim import Adam
from scipy.optimize import basinhopping


def gradient_descent_multiple_starts(
    f, initial_guesses, bounds, learning_rate=0.01, num_epochs=1000
):
    """Minimize function f using gradient descent with multiple initial guesses

    Args:
        f (callable): function compatible with Pytorch AD
        initial_guesses (list): list of initial guesses
        bounds (list): bounds for x
        learning_rate (float, optional): learning rate. Defaults to 0.01.
        num_epochs (int, optional): number of epochs. Defaults to 1000.

    Returns:
        torch.Tensor: best solution

    """
    best_x = None
    best_obj = float("inf")

    for x0 in initial_guesses:
        # Apply gradient descent with the current initial guess
        x = gradient_descent(f, x0, bounds, learning_rate, num_epochs)

        # Compute objective function value for the obtained solution
        obj = f(x).item()

        # Update the best solution if current one is better
        if obj < best_obj:
            best_obj = obj
            best_x = x

    return best_x


def adam_optimization(f, x0, bounds, lr=0.01, num_epochs=1000, tol=1e-6, verbose=False):
    """
    Performs optimization using the Adam optimizer.

    Args:
        f (function): The objective function to minimize.
        x0 (torch.Tensor): The initial guess for the design variables.
        bounds (list): A list of two scalars [lower_bound, upper_bound] indicating the bounds for the design variables.
        learning_rate (float, optional): The learning rate for the Adam optimizer. Default is 0.01.
        num_epochs (int, optional): The number of iterations for the Adam optimizer. Default is 1000.
        verbose (bool, optional): Whether to print progress messages. Default is False.

    Returns:
        torch.Tensor: The optimal design variables that minimize the objective function.
    """
    x = Variable(x0.clone(), requires_grad=True)
    lb, ub = torch.full_like(x, bounds[0]), torch.full_like(
        x, bounds[1]
    )  # create lower and upper bound tensors
    best_x = x0.clone()
    best_f_value = float("inf")

    # Initialize Adam optimizer
    optimizer = Adam([x], lr=lr)

    for epoch in range(num_epochs):
        optimizer.zero_grad()  # clear previous gradients

        f_value = f(x)
        f_value.backward()  # compute the gradient
        optimizer.step()  # update parameters

        # Apply bounds
        with torch.no_grad():
            x.data = torch.max(torch.min(x, ub), lb)

        # Check if the objective function value has increased
        if f_value > best_f_value + tol:
            print(
                f"Terminating at epoch {epoch + 1} due to increase in objective function value."
            )
            break

        # Update the best solution if current solution is better
        if f_value < best_f_value:
            best_x = x.clone()
            best_f_value = f_value

        if verbose and epoch % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{num_epochs}, Current Objective Function Value = {f_value.item()}"
            )

    return best_x


def newton_method(f, x0, bounds, lr=0.01, num_epochs=1000, verbose=False, eps=1e-5):
    """
    Performs optimization using Newton's method.

    Args:
        f (function): The objective function to minimize.
        x0 (torch.Tensor): The initial guess for the design variables.
        bounds (list): A list of two scalars [lower_bound, upper_bound] indicating the bounds for the design variables.
        lr (float, optional): The learning rate. Default is 0.01.
        num_epochs (int, optional): The number of iterations for the gradient descent. Default is 1000.
        verbose (bool, optional): Whether to print progress messages. Default is False.
        eps (float, optional): A small number for finite difference approximation. Default is 1e-5.

    Returns:
        torch.Tensor: The optimal design variables that minimize the objective function.
    """
    x = Variable(x0.clone(), requires_grad=True)
    lb, ub = torch.full_like(x, bounds[0]), torch.full_like(
        x, bounds[1]
    )  # create lower and upper bound tensors
    best_x = x0.clone()
    best_f_value = float("inf")

    for epoch in range(num_epochs):
        f_value = f(x)

        # Update the best solution if current solution is better
        if f_value < best_f_value:
            best_x = x.clone()
            best_f_value = f_value

        if x.grad is not None:
            x.grad.zero_()  # clear previous gradients

        f_value.backward()  # compute the gradient

        # Compute approximate Hessian via finite differences
        grad1 = x.grad.clone()
        x.data += eps
        f_value = f(x)
        f_value.backward()
        grad2 = x.grad.clone()
        hessian_approx = (grad2 - grad1) / eps

        with torch.no_grad():
            if verbose and epoch % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{num_epochs}, Current Objective Function Value = {f_value.item()}"
                )
            x -= lr * grad2 / (hessian_approx + 1e-10)  # Update parameters

            # Apply bounds
            x.data = torch.max(torch.min(x, ub), lb)

    return best_x


def gradient_descent(f, x0, bounds, lr=0.01, num_epochs=1000, verbose=False):
    """
    Performs gradient descent optimization.

    Args:
        f (function): The objective function to minimize.
        x0 (torch.Tensor): The initial guess for the design variables.
        bounds (list): A list of two scalars [lower_bound, upper_bound] indicating the bounds for the design variables.
        learning_rate (float, optional): The learning rate for gradient descent. Default is 0.01.
        num_epochs (int, optional): The number of iterations for the gradient descent. Default is 1000.
        verbose (bool, optional): Whether to print progress messages. Default is False.

    Returns:
        torch.Tensor: The optimal design variables that minimize the objective function.
    """
    x = Variable(x0.clone(), requires_grad=True)
    lb, ub = torch.full_like(x, bounds[0]), torch.full_like(
        x, bounds[1]
    )  # create lower and upper bound tensors
    best_x = x0.clone()
    best_f_value = float("inf")

    for epoch in range(num_epochs):
        f_value = f(x)

        # Update the best solution if current solution is better
        if f_value < best_f_value:
            best_x = x.clone()
            best_f_value = f_value

        if x.grad is not None:
            x.grad.zero_()  # clear previous gradients

        f_value.backward()  # compute the gradient

        with torch.no_grad():
            x -= lr * x.grad.detach()  # Detach gradients before in-place update
            # Apply bounds
            x.data = torch.max(torch.min(x, ub), lb)  # Directly update the data of x

        if verbose and epoch % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{num_epochs}, Current Objective Function Value = {f_value.item()}"
            )

    return best_x


def basinhopping_torch(
    f: callable,
    x0: torch.Tensor,
    bounds: list = None,
    niter: int = 100,
    verbose: bool = False,
):
    """Minimize function f using scipy.optimize.basinhopping

    Args:
        f (callable): function compatible with Pytorch AD
        x0 (torch.Tensor): initial guess. Must be a 1D array
        bounds (list): bounds for x, tuple of (min, max) pairs for each element in x. Example: [(0,1), (0,1), (0,1)] or [(0, 1)] for abbreviate form
        niter (int, optional): number of iterations. Defaults to 100.
    """
    shape = x0.shape
    assert len(shape) == 1, "x0 must be a 1D array"

    # Convert bounds to a list of tuples
    if bounds is not None and len(bounds) == 1:
        bounds = bounds * shape[0]

    def f_np(x):
        x_torch = torch.from_numpy(x).double()
        x_torch.requires_grad = False
        return f(x_torch).item()

    def df_np(x):
        x_torch = torch.from_numpy(x).double()
        x_torch.requires_grad = True
        y_torch = f(x_torch)
        y_torch.backward()
        return x_torch.grad.detach().numpy()

    if bounds is None:
        res = basinhopping(
            f_np,
            x0.detach().numpy(),
            niter=niter,
            minimizer_kwargs={"method": "L-BFGS-B", "jac": df_np},
            disp=verbose,
        )
    else:
        res = basinhopping(
            f_np,
            x0.detach().numpy(),
            niter=niter,
            minimizer_kwargs={"method": "L-BFGS-B", "jac": df_np, "bounds": bounds},
            disp=verbose,
        )
    return torch.Tensor(res.x)
