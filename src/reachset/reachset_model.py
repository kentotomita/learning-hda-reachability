import torch
import numpy as np


def reach_ellipses_torch(X: torch.Tensor, param: torch.Tensor):
    """Compute y coordinates of ellipse given x coordinates and ellipse parameters.

    Args:
        X (torch.Tensor): x coordinates
        param (torch.Tensor): ellipse parameters: xmin, xmax, alpha, yp, a1, a2
            - xmin, xmax: x coordinates of ellipse boundaries
            - alpha: xp = alpha * (xmax - xmin) + xmin
            - yp: y coordinate of ellipse center
            - a1, a2: semi-major axes of ellipses

    Returns:
        Y_positive (torch.Tensor): y coordinates of ellipse
        jump_at_xp (torch.Tensor): The difference of y coordinates of two ellipse at xp
    """

    eps = 1e-8  # small constant to prevent numerical instability
    xmin, xmax, alpha, yp, a1, a2 = param

    assert torch.all(a1 > alpha * (xmax - xmin) / 2)
    assert torch.all(a2 > (1 - alpha) * (xmax - xmin) / 2)

    xc1 = xmin + a1
    xc2 = xmax - a2
    xp = alpha * (xmax - xmin) + xmin

    if X.dim() > 1:
        xp = torch.reshape(xp, (-1, 1))
        xc1 = torch.reshape(xc1, (-1, 1))
        xc2 = torch.reshape(xc2, (-1, 1))
        a1 = torch.reshape(a1, (-1, 1))
        a2 = torch.reshape(a2, (-1, 1))
        yp = torch.reshape(yp, (-1, 1))
        xmin = torch.reshape(xmin, (-1, 1))
        xmax = torch.reshape(xmax, (-1, 1))

    b1 = torch.sqrt(
        torch.clamp(yp**2 * (1 - (xp - xc1) ** 2 / (a1**2 + eps)), min=eps)
    )
    b2 = torch.sqrt(
        torch.clamp(yp**2 * (1 - (xp - xc2) ** 2 / (a2**2 + eps)), min=eps)
    )

    Y_positive = torch.where(
        (xmin <= X) & (X < xp),
        torch.sqrt(
            torch.clamp(b1**2 * (1 - (X - xc1) ** 2 / (a1**2 + eps)), min=eps)
        ),
        0.0,
    )
    Y_positive = torch.where(
        (xp <= X) & (X <= xmax),
        torch.sqrt(
            torch.clamp(b2**2 * (1 - (X - xc2) ** 2 / (a2**2 + eps)), min=eps)
        ),
        Y_positive,
    )
    Y_positive = torch.where(X > xmax, 0.0, Y_positive)
    Y_positive = torch.where(X < xmin, 0.0, Y_positive)

    jump_at_xp = torch.sqrt(
        torch.clamp(
            (
                torch.sqrt(
                    torch.clamp(
                        b1**2 * (1 - (xp - xc1) ** 2 / (a1**2 + eps)), min=eps
                    )
                )
                - torch.sqrt(
                    torch.clamp(
                        b2**2 * (1 - (xp - xc2) ** 2 / (a2**2 + eps)), min=eps
                    )
                )
            )
            ** 2,
            min=eps,
        )
    )

    return Y_positive, jump_at_xp


def reach_ellipses_np(X: np.array, param: np.array):
    xmin, xmax, alpha, yp, a1, a2 = param

    assert a1 > alpha * (xmax - xmin) / 2
    assert a2 > (1 - alpha) * (xmax - xmin) / 2

    xc1 = xmin + a1
    xc2 = xmax - a2
    xp = alpha * (xmax - xmin) + xmin

    b1 = np.sqrt(yp**2 * (1 - (xp - xc1) ** 2 / a1**2))
    b2 = np.sqrt(yp**2 * (1 - (xp - xc2) ** 2 / a2**2))

    Y_positive = np.where(
        X < xp,
        np.sqrt(b1**2 * (1 - (X - xc1) ** 2 / a1**2)),
        np.sqrt(b2**2 * (1 - (X - xc2) ** 2 / a2**2)),
    )
    return Y_positive
