"""Define guidance optimization problems for reachability steering."""

import numpy as np
import pygmo as pg
from numba import jit, float64
from ..landers import Lander


class MinFuel():
    """Minimum Fuel problem where control sequence is decision variable"""

    def __init__(self, lander, N, x0, tgo):
        """Initialize the problem

        Args:
            lander (lander): lander (lander) model
            x0 (np.array-like): initial state
            tgo (float): time-to-go
        """
        self.lander = lander
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
        u = u_.reshape(self.N, 3) * self.lander.rho2

        # Propagate dynamics
        r, v, z = _propagate_state(self.x0, u, self.N, self.dt, self.lander.g, self.lander.alpha)

        # Compute constraints
        cstr_eq = []
        cstr_eq += [r[-1, 2]]  # Final altitude = 0
        cstr_eq += list(v[-1, :])  # Final velocity = 0
        cstr_ineq = _get_cstr(r, v, u, self.N, self.lander.rho1, self.lander.rho2, self.lander.pa, self.lander.gsa, self.lander.vmax)

        # Compute fitness
        obj = _minfuel(u, self.N, self.lander.alpha, self.dt)
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
    

class MinFuelStateCtrl():
    """Minimum Fuel problem where state and control sequences are decision variables"""
    def __init__(self, lander: Lander, N: int, x0: np.ndarray, tgo: float):
        """Initialize the problem

        Args:
            lander (lander): lander (lander) model
            x0 (np.array-like): initial state
            tgo (float): time-to-go
        """
        self.lander = lander
        self.N = N
        self.x0 = x0
        self.tgo = tgo
        self.dt = tgo / N
        self.t = np.linspace(0, tgo, N + 1)

        assert x0[6] > lander.mdry, "Initial mass must be greater than dry mass"
        
        # Scaling factors for normalization
        self.LU = lander.LU
        self.TU = lander.TU
        self.MU = lander.MU

        # normalized lander parameters
        self.alpha_ = lander.alpha / (self.TU / self.LU)  # alpha [s/m]
        self.rho1_ = lander.rho1 / (self.MU * self.LU / self.TU ** 2)  # rho1 [kg m/s^2]
        self.rho2_ = lander.rho2 / (self.MU * self.LU / self.TU ** 2)  # rho2 [kg m/s^2]
        self.vmax_ = lander.vmax / (self.LU / self.TU)  # vmax [m/s]
        self.g_ = lander.g / (self.LU / self.TU ** 2)  # g [m/s^2]

    def unpack_decision_vector(self, x):
        """Unpack decision vector into states and controls

        Args:
            x (np.array-like): decision vector
        """
        # Unpack decision vector
        r_ = x[:3 * (self.N + 1)].reshape(self.N + 1, 3)
        v_ = x[3 * (self.N + 1):6 * (self.N + 1)].reshape(self.N + 1, 3)
        m_ = x[6 * (self.N + 1):7 * (self.N + 1)]
        u_ = x[7 * (self.N + 1):].reshape(self.N, 3)
        return r_, v_, m_, u_
    
    def dimensionalize(self, r_, v_, m_, u_):
        r = r_ * self.LU
        v = v_ * self.LU / self.TU
        m = m_ * self.MU
        u = u_ * self.MU * self.LU / self.TU ** 2
        return r, v, m, u
        
    def fitness(self, x):
        """Compute fitness for given decision vector x

        Args:
            x (np.array-like): decision vector
        """
        # Unpack decision vector
        r_, v_, m_, u_ = self.unpack_decision_vector(x)

        # Dynamics constraints
        cstr_eq_dyn = self._dynamics_cstr(r_, v_, m_, u_, self.dt / self.TU, self.g_, self.alpha_, self.N)
        
        # Compute constraints
        cstr_ineq = _get_cstr(r_, v_, u_, self.N, self.rho1_, self.rho2_, self.lander.pa, self.lander.gsa, self.vmax_)

        # Compute fitness
        obj = _minfuel(u_, self.N, self.alpha_, self.dt / self.TU)

        return [obj] + list(cstr_eq_dyn) + list(cstr_ineq)
        
    def get_nec(self):
        """Return number of equality constraints"""
        n_eq_dyn = 7 * self.N
        return n_eq_dyn

    def get_nic(self):
        """Return number of inequality constraints"""
        n_ineq_operation = 5 * self.N + 3
        return n_ineq_operation

    def get_bounds(self):
        """Return decision vector bounds, Boundary constraints are imposed here.

        Return:
            (tuple): tuple containing:
                lb (np.array-like): lower bound
                ub (np.array-like): upper bound
        """
        r_min = -np.ones((self.N + 1, 3))
        r_max = np.ones((self.N + 1, 3))
        r_min[:, 2] = 0.0
        r_max[:, 2] = 2.0

        v_min = -np.ones((self.N + 1, 3))
        v_max = np.ones((self.N + 1, 3))

        m_min = np.ones(self.N + 1) * self.lander.mdry / self.MU
        m_max = np.ones(self.N + 1)

        u_max = np.ones((self.N, 3))
        u_min = np.zeros((self.N, 3))
        thrust_max = self.lander.rho2 / (self.MU * self.LU / self.TU ** 2)
        thrust_min = self.lander.rho1 / (self.MU * self.LU / self.TU ** 2)
        u_min[:, :2] = -thrust_max * np.sin(self.lander.pa)
        u_max[:, :2] = thrust_max * np.sin(self.lander.pa)
        u_min[:, 2] = thrust_min * np.cos(self.lander.pa)
        u_max[:, 2] = thrust_max

        # Boundary constraints
        r0_ = self.x0[:3] / self.LU
        v0_ = self.x0[3:6] / self.LU * self.TU
        m0_ = self.x0[6] / self.MU
        # Initial state
        r_min[0, :] = r0_
        r_max[0, :] = r0_
        v_min[0, :] = v0_
        v_max[0, :] = v0_
        m_min[0] = m0_
        m_max[0] = m0_
        # Final state
        r_min[-1, 2] = 0.0
        r_max[-1, 2] = 0.0
        v_min[-1, :] = 0.0
        v_max[-1, :] = 0.0

        lb = np.hstack((r_min.flatten(), v_min.flatten(), m_min.flatten(), u_min.flatten()))
        ub = np.hstack((r_max.flatten(), v_max.flatten(), m_max.flatten(), u_max.flatten()))
        return lb, ub

    #def gradient(self, x):
    #    """Compute gradient of fitness function for given decision vector x"""
    #    return pg.estimate_gradient_h(lambda x: self.fitness(x), x)

    def _dynamics_cstr(self, r, v, m, u, dt, g, alpha, N):
        """Equality constraints for the dynamics of the lander
        
        Args:
            r (np.array): position
            v (np.array): velocity
            m (np.array): mass
            u (np.array): control
            dt (float): time step
            g (np.array): gravity
            alpha (float): fuel consumption rate
            N (int): number of time steps

        Returns:
            np.array: equality constraints, (N * 7,)
        """
        cstr_eq = np.zeros(N * 7)
        for i in range(N):
            a = u[i] / m[i] + g
            for j in range(3):
                # Position dynamics
                cstr_eq[i * 7 + j] = r[i, j] + dt * v[i, j] + dt ** 2 / 2.0 * a[j] - r[i + 1, j]
                # Velocity dynamics
                cstr_eq[i * 7 + 3 + j] = v[i, j] + dt * a[j] - v[i + 1, j]
            # Mass dynamics
            cstr_eq[i * 7 + 6] = (m[i] - dt * alpha * np.linalg.norm(u[i])) - m[i + 1]

        return cstr_eq
    

    

# TODO: 
# - Make sparse version of MinFuel; states are also decision variables
# - Make sparse version of ReachSteering; states are also decision variables
# - Gradient of reachable-safety is only evaluated for x(step_reachmax). 

class ReachSteering():
    """Reachability steering problem"""

    def __init__(self, lander, N, x0, tgo, nn_reach):
        """Initialize the problem

        Args:
            lander (lander): lander (lander) model
            x0 (np.array-like): initial state
            tgo (float): time-to-go
            nn_reach (torch.nn.Module): neural network model for reachable set evaluation
        """
        self.lander = lander
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
        u = u_.reshape(self.N, 3) * self.lander.rho2

        # Propagate dynamics
        r, v, z = _propagate_state(self.x0, u, self.N, self.dt, self.lander.g, self.lander.alpha)

        # Compute constraints
        cstr_eq, cstr_ineq = _get_cstr(r, v, z, u, self.N, self.lander.rho1, self.lander.rho2, self.lander.pa, self.lander.gsa, self.lander.vmax)

        if return_obj:
            # Compute fitness
            obj = _minfuel(u, self.N, self.lander.alpha, self.dt)
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
def _dynamics_cstr(r, v, z, u, dt, g, alpha, N):
    """Equality constraints for the dynamics of the lander
    
    Args:
        r (np.array): position
        v (np.array): velocity
        z (np.array): log of mass
        u (np.array): control
        dt (float): time step
        g (np.array): gravity
        alpha (float): fuel consumption rate
        N (int): number of time steps

    Returns:
        np.array: equality constraints, (N * 7,)
    """
    cstr_eq = np.zeros(N * 7)
    for i in range(N):
        mass = np.exp(z[i])
        a = u[i] / mass + g
        for j in range(3):
            # Position dynamics
            cstr_eq[i * 7 + j] = r[i, j] + dt * v[i, j] + dt ** 2 / 2.0 * a[j] - r[i + 1, j]
            # Velocity dynamics
            cstr_eq[i * 7 + 3 + j] = v[i, j] + dt * a[j] - v[i + 1, j]
        # Mass dynamics
        cstr_eq[i * 7 + 6] = (z[i] - dt * alpha * np.linalg.norm(u[i]) / mass) - z[i + 1]

    return cstr_eq

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
def _get_cstr(r, v, u, N, rho1, rho2, pa, gsa, vmax):
    cstr_ineq = np.zeros(5 * N + 3)

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

    return cstr_ineq

@jit(nopython=True)
def _minfuel(u, N, alpha, dt):
    fuel = 0.0
    for i in range(N):
        fuel += alpha * np.linalg.norm(u[i]) * dt
    return fuel
