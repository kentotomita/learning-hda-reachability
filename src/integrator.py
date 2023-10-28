import numpy as np
from typing import Callable, Tuple


def sim_feedback(
    f: Callable,
    u: Callable,
    x0: np.ndarray,
    t_span: Tuple,
    dt: float,
    dt_g: float,
    event: Callable = None,
):
    """Simulate system dynamics with feedback control.

    Args:
        f: System dynamics with control input (dx/dt = f(t, x, u)).
        u: Feedback control law (u = u(t, x)).
        x0: Initial value of the state vector x.
        t_span: Tuple containing the initial and final values of the independent variable t.
        dt: Integration time step.
        dt_g: Guidance time step.
        events: Optional function representing the event detection conditions (event(t, x) = 0).

    Returns:
        Tuple containing the integration time vector, state vector, and control input vector.
    """
    assert (
        dt <= dt_g
    ), "Integration time step must be less than or equal to guidance time step."

    t0, tf = t_span

    t = np.arange(t0, tf + dt, dt)  # integration time vector
    tg = np.arange(t0, tf + dt_g, dt_g)  # guidance time vector
    X = np.zeros((len(t), len(x0)))  # state vector
    U = np.zeros((len(t), len(u(t0, x0))))  # control input vector

    # Initializations
    X[0] = x0  # initial state
    ig = 0  # guidance time index
    u_ = u(t0, x0)  # control input at guidance time
    for i in range(len(t) - 1):
        t_ = t[i]  # integration time
        x_ = X[i]  # state at integration time
        if t_ >= tg[ig + 1]:
            # print('Updating control input at t = {:.2f} s'.format(t_))
            ig += 1
            u_ = u(t_, x_)  # control input at guidance time

        # print('t = {:.2f} s, x = {:.2f} m, u = {:.2f} m/s^2'.format(t_, X[i, 2], U[i, 2]))
        k1 = f(t_, x_, u_)
        k2 = f(t_ + dt / 2, x_ + k1 * dt / 2, u_)
        k3 = f(t_ + dt / 2, x_ + k2 * dt / 2, u_)
        k4 = f(t_ + dt, x_ + k3 * dt, u_)
        X[i + 1] = X[i] + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        U[i] = u_

        if (
            event is not None
            and i > 0
            and event(t[i], X[i]) * event(t[i + 1], X[i + 1]) <= 0
        ):
            break

    return t[: i + 1], X[: i + 1], U[: i + 1]


def rk4(f, y0, t0, tf, dt, event=None):
    """
    Fourth-order Runge-Kutta solver for an ODE with event detection and multivariable states,
    compatible with Numba's JIT compilation.

    Args:
        f: Function representing the first derivative of y (dy/dt = f(t, y)).
        y0: Initial value of the dependent variable y (NumPy array).
        t0: Initial value of the independent variable t.
        tf: Final value of the independent variable t.
        dt: Step size for the independent variable t.
        event: Optional function representing the event detection condition (event(t, y) = 0).

    Returns:
        A tuple containing a NumPy array with the independent variable values (t) and
        a 2D NumPy array with the corresponding dependent variable values (y).
    """
    t = np.arange(t0, tf + dt, dt)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0

    for i in range(1, len(t)):
        k1 = f(t[i - 1], y[i - 1])
        k2 = f(t[i - 1] + dt / 2, y[i - 1] + k1 * dt / 2)
        k3 = f(t[i - 1] + dt / 2, y[i - 1] + k2 * dt / 2)
        k4 = f(t[i - 1] + dt, y[i - 1] + k3 * dt)
        y[i] = y[i - 1] + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        if event is not None and event(t[i], y[i]) * event(t[i - 1], y[i - 1]) <= 0:
            break

    return t[: i + 1], y[: i + 1]
