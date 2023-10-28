import numpy as np


def golden(
    f: callable, a: float, b: float, tol: float = 1e-6, max_iter: int = 100
) -> float:
    """Golden section search

    Args:
        f: Function to minimize
        a: Lower bound
        b: Upper bound
        tol: Tolerance
        max_iter: Maximum number of iterations

    Returns:
        Minimum of f

    Reference:
        - https://en.wikipedia.org/wiki/Golden-section_search
        - https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.golden.html
    """
    # Define golden ratio
    gr = (np.sqrt(5) - 1) / 2

    # Initial points
    x1 = a + (1 - gr) * (b - a)
    x2 = a + gr * (b - a)

    # Initial function values
    f1 = f(x1)
    f2 = f(x2)

    # Iteration counter
    i = 0

    # Golden section search
    while (b - a) > tol and i < max_iter:
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + (1 - gr) * (b - a)
            f1 = f(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + gr * (b - a)
            f2 = f(x2)
        i += 1

    return (a + b) / 2
