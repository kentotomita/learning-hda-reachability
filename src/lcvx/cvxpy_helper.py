"""Helper functions for CVXPY."""

import cvxpy as cp
from cvxpy import Problem

def set_params(problem: Problem, param: dict):
    """Set the optimization problem parameters.

    Args:
        problem (Problem): The optimization problem.
        param (dict): The parameters to set.

    Example:
        >>> from cvxpy_helper import set_params
        >>> from cvxpy import Problem, Parameter, Variable, Minimize, norm
        >>> import numpy as np
        >>> x = Parameter(2, name='x')
        >>> y = Variable(2)
        >>> objective = Minimize(norm(x - y))
        >>> constraints = [x + y == 1]
        >>> problem = Problem(objective, constraints)
        >>> param = {'x': np.array([1, 2])}
        >>> problem = set_params(problem, param)
        >>> problem.solve()
        5.656854249492381

    """
    for key, value in param.items():
        for p in problem.parameters():
            if p.name() == key:
                p.value = value
    return problem


def get_vars(problem: Problem, varnames: list):
    """Get the variables from the optimization problem.
    
    Args:
        problem (Problem): The optimization problem.
        varnames (list): The variable names.

    Returns:
        A list of variables.
        
    Example:
        >>> from cvxpy_helper import get_vars
        >>> from cvxpy import Problem, Parameter, Variable, Minimize, norm
        >>> import numpy as np
        >>> x = Parameter(2, name='x')
        >>> y = Variable(2)
        >>> objective = Minimize(norm(x - y))
        >>> constraints = [x + y == 1]
        >>> problem = Problem(objective, constraints)
        >>> param = {'x': np.array([1, 2])}
        >>> problem = set_params(problem, param)
        >>> problem.solve()
        >>> varnames = ['x', 'y']
        >>> vars = get_vars(problem, varnames)
        >>> vars
        [array([1., 2.]), array([0., 0.])]
    """

    out = {}
    for var in problem.variables():
        for name in varnames:
            if var.name() == name:
                out[name] = var.value
    return out