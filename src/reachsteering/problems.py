"""Define guidance optimization problems for reachability steering."""

import numpy as np
import pygmo as pg
from numba import jit, float64

class MyUDP():

    def fitness(self, x):
        """Compute fitness for given decision vector x
        Args:
            x (np.array-like): decision vector
        """
        raise NotImplementedError
    
    def get_nec(self):
        """Return number of equality constraints"""
        raise NotImplementedError

    def get_nic(self):
        """Return number of inequality constraints"""
        raise NotImplementedError

    def get_bounds(self):
        """Return decision vector bounds
        
        Return:
            (tuple): tuple containing:
                lb (np.array-like): lower bound
                ub (np.array-like): upper bound
        """
        raise NotImplementedError
		
    def gradient(self, x):
        """Compute gradient of fitness function for given decision vector x"""
        raise NotImplementedError


class MinFuel(MyUDP):
    """Minimum Fuel problem"""

    def __init__(self, rocket, N, x0, tgo):
        """Initialize the problem

        Args:
            rocket (Rocket): rocket (lander) model
            x0 (np.array-like): initial state
            tgo (float): time-to-go
        """
        self.rocket = rocket
        self.N = N
        self.x0 = x0
        self.tgo = tgo
        self.dt = tgo / N
        self.t = np.linspace(0, tgo, N + 1)
        
    def fitness(self, u_):
        """Compute fitness for given decision vector x

        Args:
            u_ (np.array-like): decision vector
        """
        # Unnormalize control sequence
        u = u_.reshape(self.N, 3) * self.rocket.rho2

        # Propagate dynamics
        r, v, z = _propagate_state(self.x0, u, self.N, self.dt, self.rocket.g, self.rocket.alpha)

        # Compute constraints
        cstr_eq, cstr_ineq = _get_cstr(r, v, z, u, self.N, self.rocket.rho1, self.rocket.rho2, self.rocket.pa, self.rocket.gsa, self.rocket.vmax)

        # Compute fitness
        obj = _minfuel(u, self.N, self.rocket.alpha, self.dt)
        return [obj] + list(cstr_eq) + list(cstr_ineq)
        
    def get_nec(self):
        """Return number of equality constraints"""
        return 4

    def get_nic(self):
        """Return number of inequality constraints"""
        return 5 * self.N + 3

    def get_bounds(self):
        """Return decision vector bounds

        Return:
            (tuple): tuple containing:
                lb (np.array-like): lower bound
                ub (np.array-like): upper bound
        """
        return -np.ones(3 * self.N), np.ones(3 * self.N)

    def gradient(self, x):
        """Compute gradient of fitness function for given decision vector x"""
        return pg.estimate_gradient_h(lambda x: self.fitness(x), x)
    

# TODO: 
# - Make sparse version of MinFuel; states are also decision variables
# - Make sparse version of ReachSteering; states are also decision variables
# - Gradient of reachable-safety is only evaluated for x(step_reachmax). 

class ReachSteering(MyUDP):
    """Reachability steering problem"""

    def __init__(self, rocket, N, x0, tgo, nn_reach):
        """Initialize the problem

        Args:
            rocket (Rocket): rocket (lander) model
            x0 (np.array-like): initial state
            tgo (float): time-to-go
            nn_reach (torch.nn.Module): neural network model for reachable set evaluation
        """
        self.rocket = rocket
        self.N = N
        self.x0 = x0
        self.tgo = tgo
        self.nn_reach = nn_reach
        self.dt = tgo / N
        self.t = np.linspace(0, tgo, N + 1)

    def fitness(self, u_, return_obj=True):
        """Compute fitness for given decision vector x

        Args:
            u_ (np.array-like): decision vector
        """
        # Unnormalize control sequence
        u = u_.reshape(self.N, 3) * self.rocket.rho2

        # Propagate dynamics
        r, v, z = _propagate_state(self.x0, u, self.N, self.dt, self.rocket.g, self.rocket.alpha)

        # Compute constraints
        cstr_eq, cstr_ineq = _get_cstr(r, v, z, u, self.N, self.rocket.rho1, self.rocket.rho2, self.rocket.pa, self.rocket.gsa, self.rocket.vmax)

        if return_obj:
            # Compute fitness
            obj = _minfuel(u, self.N, self.rocket.alpha, self.dt)
            return [obj] + list(cstr_eq) + list(cstr_ineq)
        else:
            return list(cstr_eq) + list(cstr_ineq)

    def get_nec(self):
        """Return number of equality constraints"""
        return 4

    def get_nic(self):
        """Return number of inequality constraints"""
        return 5 * self.N + 3

    def get_bounds(self):
        """Return decision vector bounds

        Return:
            (tuple): tuple containing:
                lb (np.array-like): lower bound
                ub (np.array-like): upper bound
        """
        return -np.ones(3 * self.N), np.ones(3 * self.N)



@jit(nopython=True)
def _dynamics(r, v, z, u, dt, g, alpha):
    dt22 = dt ** 2 / 2.0
    mass = np.exp(z)
    a = u / mass + g

    # Compute next state
    r_next = r + dt * v + dt22 * a
    v_next = v + dt * a
    z_next = z - dt * alpha * np.linalg.norm(u) / mass

    return r_next, v_next, z_next

@jit(nopython=True)
def _propagate_state(x0, u, N, dt, g, alpha):
    r = np.zeros((N + 1, 3))
    v = np.zeros((N + 1, 3))
    z = np.zeros(N + 1)

    r[0] = x0[:3]
    v[0] = x0[3:6]
    z[0] = x0[6]

    for i in range(N):
        r[i + 1], v[i + 1], z[i + 1] = _dynamics(r[i], v[i], z[i], u[i], dt, g, alpha)
    
    return r, v, z

@jit(nopython=True)
def _get_cstr(r, v, z, u, N, rho1, rho2, pa, gsa, vmax):
    cstr_eq = np.zeros(4)
    cstr_ineq = np.zeros(5 * N + 3)

    # Equality constraints
    cstr_eq[0] = r[N, 2]
    cstr_eq[1:] = v[N, :]

    # Inequality constraints
    i_ieq = 0
    # Thrust bounds
    for i in range(N):
        cstr_ineq[i_ieq] = rho1 - np.linalg.norm(u[i])
        cstr_ineq[i_ieq + 1] = np.linalg.norm(u[i]) - rho2
        i_ieq += 2

    # Pointing angle constraint
    for i in range(N):
        cstr_ineq[i_ieq] = np.linalg.norm(u[i]) * np.cos(pa) - u[i, 2]
        i_ieq += 1

    # Glide slope constraint
    for i in range(N+1):
        cstr_ineq[i_ieq] = np.linalg.norm(r[i, :2] - r[-1, :2]) * np.tan(gsa) - (r[i, 2] - r[-1, 2])
        i_ieq += 1

    # Velocity constraint
    for i in range(N):
        cstr_ineq[i_ieq] = np.linalg.norm(v[i]) - vmax
        i_ieq += 1

    return cstr_eq, cstr_ineq

@jit(nopython=True)
def _minfuel(u, N, alpha, dt):
    fuel = 0.0
    for i in range(N):
        fuel += alpha * np.linalg.norm(u[i]) * dt
    return fuel
