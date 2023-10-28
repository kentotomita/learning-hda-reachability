"""Functions for saving and loading optimized trajectory data."""
import cvxpy as cp
import numpy as np
import json

from .problem import LCvxProblem
from .cvxpy_helper import get_vars


def save_results(
    prob: cp.Problem,
    lcvx_obj: LCvxProblem,
    tf: float,
    filename: str,
    params: dict = None,
):
    """Save the results of a solved LCvxProblem to a JSON file.

    Args:
        prob (cp.Problem): The solved CVXPY problem.
        lcvx_obj (LCvxProblem): The LCvxProblem object.
        tf (float): Time of flight.
        filename (str): The name of the file to save the data to.
        params (dict, optional): Additional parameters to save. Defaults to None.
    """

    if prob.status == "optimal":
        # Get the solution
        sol = get_vars(prob, ["X", "U"])
        X_sol = sol["X"]
        U_sol = sol["U"]

        # Recover the variables
        r, v, z, u, sigma = lcvx_obj.recover_variables(X_sol, U_sol)
        m = np.exp(z)
        U = u.T * m[:-1].reshape(-1, 1)  # (N, 3)
        t = np.linspace(0, tf, lcvx_obj.N + 1)

        # Save the data
        data = {
            "status": prob.status,
            "tf": tf,
            "t": t.tolist(),  # (N+1, )
            "r": r.T.tolist(),  # (N+1, 3)
            "v": v.T.tolist(),  # (N+1, 3)
            "m": m.tolist(),  # (N+1, )
            "U": U.tolist(),  # (N, 3)
        }

    # If the problem is infeasible or unbounded, save the status and tf
    else:
        data = {
            "status": prob.status,
            "tf": tf,
            "t": None,
            "r": None,
            "v": None,
            "m": None,
            "U": None,
        }

    # Add any additional parameters
    if params is not None:
        for key, value in params.items():
            data[key] = value

    # Save the data
    with open(filename, "w") as f:
        json.dump(data, f)


def load_results(filename: str):
    """Load the results of a solved LCvxProblem from a JSON file."""

    with open(filename, "r") as f:
        data = json.load(f)

    # Get the data
    status = data["status"]
    tf = data["tf"]
    t = np.array(data["t"])
    r = np.array(data["r"])
    v = np.array(data["v"])
    m = np.array(data["m"])
    U = np.array(data["U"])

    # Get any additional parameters
    if set(data.keys()) != set({"status", "tf", "t", "r", "v", "m", "U"}):
        params = {
            key: value
            for key, value in data.items()
            if key not in {"status", "tf", "t", "r", "v", "m", "U"}
        }
    else:
        params = None

    return status, tf, t, r, v, m, U, params
