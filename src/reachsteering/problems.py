"""Define guidance optimization problems for reachability steering."""

import numpy as np
import pygmo as pg
from numba import jit, float64
from torch.nn import Module
import torch
from .objectives import ic2mean_safety_npy
from ..landers import Lander


class Pdl:
    """Base class for powered descent and landing."""

    def __init__(self, lander: Lander, N: int, x0: np.ndarray, tgo: float, 
                 grad_implemented: bool=False, normalize: bool=True):
        """Initialize the problem
        
        Args:
            lander (lander): lander model
            x0 (np.array-like): initial state; [x, y, z, vx, vy, vz, m]
            tgo (float): time-to-go
            grad_implemented (bool): whether gradient is implemented 
            normalize (bool): whether to normalize the problem
        """
        self.lander = lander
        self.N = N
        self.x0 = x0
        self.tgo = tgo
        self.dt = tgo / N

        # Check initial condition
        assert tgo > 0, "Time-to-go must be positive"
        assert x0[6] >= lander.mdry and x0[6] <= lander.mwet, "Initial mass must be between dry and wet mass"

        # Gradient implementation
        self.grad_implemented = grad_implemented
        if grad_implemented:
            self.gradient = self._gradient

        # Scaling factors for normalization
        if normalize:
            self.LU = lander.LU  # length unit
            self.TU = lander.TU  # time unit
            self.MU = lander.MU  # mass unit
        else:
            self.LU = 1.0
            self.TU = 1.0
            self.MU = 1.0

        # normalized parameters
        self.dt_ = self.dt / self.TU  # dt [s]
        self.alpha_ = lander.alpha / (self.TU / self.LU)  # alpha [s/m]
        self.rho1_ = lander.rho1 / (self.MU * self.LU / self.TU ** 2)  # rho1 [kg m/s^2]
        self.rho2_ = lander.rho2 / (self.MU * self.LU / self.TU ** 2)  # rho2 [kg m/s^2]
        self.vmax_ = lander.vmax / (self.LU / self.TU)  # vmax [m/s]
        self.g_ = lander.g / (self.LU / self.TU ** 2)  # g [m/s^2]

        # normalize initial state
        self.x0_ = np.zeros(7)
        self.x0_[:3] = x0[:3] / self.LU
        self.x0_[3:6] = x0[3:6] / (self.LU / self.TU)
        self.x0_[6] = x0[6] / self.MU

    def _gradient(self, x):
        """Compute gradient of fitness function for given decision vector x"""
        return pg.estimate_gradient_h(lambda x: self.fitness(x), x)
    

class PdlCtrl(Pdl):
    """Base class for powered descent and landing with control sequence as decision variable."""

    def __init__(self, lander: Lander, N: int, x0: np.ndarray, tgo: float, 
                 grad_implemented: bool=False, normalize: bool=True):
        """Initialize the problem
        
        Args:
            lander (lander): lander model
            x0 (np.array-like): initial state; [x, y, z, vx, vy, vz, m]
            tgo (float): time-to-go
            grad_implemented (bool): whether gradient is implemented 
            normalize (bool): whether to normalize the problem
        """
        super().__init__(lander, N, x0, tgo, grad_implemented, normalize)

        # gamma is the angle between the thrust vector and the vertical axis
        # phi is the angle between the thrust vector projected on the horizontal plane and the x-axis
        # The following bounds are used for normalize decision vector in the range [0, 1]
        self.gamma_lb = 0.0
        self.gamma_ub = self.lander.pa
        self.phi_lb = -np.pi
        self.phi_ub = np.pi

    def construct_x(self, u):
        """Construct decision vector given control sequence
        
        Args:
            u (np.array-like): control sequence, (N, 3)

        Returns:
            x (np.array): decision vector, (3 * N,)
        """
        u_ = u / (self.MU * self.LU / self.TU ** 2)
        u_norm = np.linalg.norm(u_, axis=1)
        throttle = (u_norm - self.rho1_) / (self.rho2_ - self.rho1_)
        gamma = np.arccos(u_[:, 2] / u_norm)
        phi = np.arctan2(u_[:, 1], u_[:, 0])

        # normalize decision vector
        gamma_ = (gamma - self.gamma_lb) / (self.gamma_ub - self.gamma_lb)
        phi_ = (phi - self.phi_lb) / (self.phi_ub - self.phi_lb)
        x = np.hstack((throttle, gamma_, phi_))
        return x

    def construct_thrust(self, x):
        """Construct thrust vector from decision vector

        Args:
            x (np.array-like): decision vector
        """
        # unpack decision vector
        throttle = x[: self.N]
        gamma_ = x[self.N: 2 * self.N]
        phi_ = x[2 * self.N:]

        # recover decision vector
        gamma = self.gamma_lb + (self.gamma_ub - self.gamma_lb) * gamma_
        phi = self.phi_lb + (self.phi_ub - self.phi_lb) * phi_

        # construct normalized thrust vector
        u_ = np.zeros((self.N, 3))
        u_norm_ = self.rho1_ + (self.rho2_ - self.rho1_) * throttle
        u_[:, 0] = u_norm_ * np.sin(gamma) * np.cos(phi)
        u_[:, 1] = u_norm_ * np.sin(gamma) * np.sin(phi)
        u_[:, 2] = u_norm_ * np.cos(gamma)
        return u_

    def construct_trajectory(self, x):
        u_ = self.construct_thrust(x)
        r_, v_, m_ = _propagate_state(self.x0_, u_, self.N, self.dt_, self.g_, self.alpha_)

        u = u_ * self.MU * self.LU / self.TU ** 2
        r = r_ * self.LU
        v = v_ * self.LU / self.TU
        m = m_ * self.MU
        return r, v, m, u

    def get_bounds(self):
        """Return decision vector bounds

        Return:
            (tuple): tuple containing:
                lb (np.array-like): lower bound
                ub (np.array-like): upper bound
        """
        return np.zeros(3 * self.N), np.ones(3 * self.N)


class PdlStateCtrl(PdlCtrl):
    """Base class for powered descent and landing with state and control sequences as decision variables."""

    def __init__(self, lander: Lander, N: int, x0: np.ndarray, tgo: float, 
                 grad_implemented: bool=False, normalize: bool=True):
        """Initialize the problem
        
        Args:
            lander (lander): lander model
            x0 (np.array-like): initial state; [x, y, z, vx, vy, vz, m]
            tgo (float): time-to-go
            grad_implemented (bool): whether gradient is implemented 
            normalize (bool): whether to normalize the problem
        """
        super().__init__(lander, N, x0, tgo, grad_implemented, normalize)

    def construct_x(self, X, u):
        """Construct decision vector given state and control sequences
        
        Args:
            X (np.array-like): state sequence, (N + 1, 7); [x, y, z, vx, vy, vz, m]
            u (np.array-like): control sequence, (N, 3)

        Returns:
            x (np.array): decision vector, (7 * N,)
        """
        r, v, m = X[:, :3], X[:, 3:6], X[:, 6]
        r_ = r / self.LU
        v_ = v / (self.LU / self.TU)
        m_ = m / self.MU
        x_u = super().construct_x(u)

        x = np.hstack((r_.flatten(), v_.flatten(), m_.flatten(), x_u))
        return x
    
    def unpack_x(self, x):
        """Unpack decision vector into state and control sequences
        
        Args:
            x (np.array-like): decision vector, (7 * N,)

        Returns:
            X (np.array-like): state sequence, (N + 1, 7); [x, y, z, vx, vy, vz, m]
            u (np.array-like): control sequence, (N, 3)
        """
        r_ = x[:3 * (self.N + 1)].reshape(self.N + 1, 3)
        v_ = x[3 * (self.N + 1):6 * (self.N + 1)].reshape(self.N + 1, 3)
        m_ = x[6 * (self.N + 1):7 * (self.N + 1)]
        u_ = super().construct_thrust(x[7 * (self.N + 1):])

        return r_, v_, m_, u_
    
    def construct_trajectory(self, x):
        r_, v_, m_, u_ = self.unpack_x(x)
        r = r_ * self.LU
        v = v_ * self.LU / self.TU
        m = m_ * self.MU
        u = u_ * self.MU * self.LU / self.TU ** 2
        return r, v, m, u
    
    def get_bounds(self):
        """Return decision vector bounds

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
        m_max = np.ones(self.N + 1) * self.lander.mwet / self.MU

        x_u_lb, x_u_ub = super().get_bounds()

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

        lb = np.hstack((r_min.flatten(), v_min.flatten(), m_min.flatten(), x_u_lb))
        ub = np.hstack((r_max.flatten(), v_max.flatten(), m_max.flatten(), x_u_ub))
        return lb, ub
    
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


class MinFuelCtrl(PdlCtrl):
    """Minimum Fuel problem where control sequence is decision variable"""

    def __init__(self, lander: Lander, N: int, x0: np.ndarray, tgo: float,
                 grad_implemented: bool=False, normalize: bool=True):
        """Initialize the problem
        
        Args:
            lander (lander): lander model
            x0 (np.array-like): initial state; [x, y, z, vx, vy, vz, m]
            tgo (float): time-to-go
            grad_implemented (bool): whether gradient is implemented 
            normalize (bool): whether to normalize the problem
        """
        super().__init__(lander, N, x0, tgo, grad_implemented, normalize)

    def fitness(self, x):
        """Compute fitness for given decision vector x

        Args:
            x (np.array-like): decision vector
        """
        u_ = self.construct_thrust(x)

        # Propagate dynamics
        r_, v_, m_ = _propagate_state(self.x0_, u_, self.N, self.dt_, self.g_, self.alpha_)

        # Terminal state constraints
        cstr_eq = [r_[-1, 2]]  # Final altitude = 0
        cstr_eq += list(v_[-1, :])  # Final velocity = 0

        cstr_ineq_terminal = [self.lander.mdry / self.MU - m_[-1]]  # Dry mass <= final mass

        # Operational constraints
        cstr_ineq = _get_cstr(r_, v_, self.N, self.lander.gsa, self.vmax_)

        # Compute fitness
        obj = _minfuel(u_, self.N, self.alpha_, self.dt_)
        return [obj] + list(cstr_eq) + cstr_ineq_terminal + list(cstr_ineq)
        
    def get_nec(self):
        """Return number of equality constraints"""
        n_softlanding = 4  # final altitude and velocity are zero
        return n_softlanding

    def get_nic(self):
        """Return number of inequality constraints"""
        n_operation = 2 * self.N + 1  # constraints from _get_cstr
        n_terminal = 1  # Dry mass <= final mass
        return n_operation + n_terminal


class MinFuelStateCtrl(PdlStateCtrl):
    """Minimum Fuel problem where state and control sequences are decision variables"""
    def __init__(self, lander: Lander, N: int, x0: np.ndarray, tgo: float,
                 grad_implemented: bool=False, normalize: bool=True):
        """Initialize the problem
        
        Args:
            lander (lander): lander model
            x0 (np.array-like): initial state; [x, y, z, vx, vy, vz, m]
            tgo (float): time-to-go
            grad_implemented (bool): whether gradient is implemented 
            normalize (bool): whether to normalize the problem
        """
        super().__init__(lander, N, x0, tgo, grad_implemented, normalize)
        
    def fitness(self, x):
        """Compute fitness for given decision vector x

        Args:
            x (np.array-like): decision vector
        """
        # Unpack decision vector
        r_, v_, m_, u_ = self.unpack_x(x)

        # Dynamics constraints
        cstr_eq_dyn = self._dynamics_cstr(r_, v_, m_, u_, self.dt_, self.g_, self.alpha_, self.N)
        
        # Compute constraints
        cstr_ineq = _get_cstr(r_, v_, self.N, self.lander.gsa, self.vmax_)

        # Compute fitness
        obj = _minfuel(u_, self.N, self.alpha_, self.dt_)

        return [obj] + list(cstr_eq_dyn) + list(cstr_ineq)
        
    def get_nec(self):
        """Return number of equality constraints"""
        n_eq_dyn = 7 * self.N
        return n_eq_dyn

    def get_nic(self):
        """Return number of inequality constraints"""
        n_operation = 2 * self.N + 1
        return n_operation


class ReachSteeringCtrl(MinFuelCtrl):
    """Reachability steering problem where control sequence is decision variable"""
    def __init__(self, lander: Lander, N: int, x0: np.ndarray, tgo: float,
                 sfmap: torch.Tensor, nn_reach: Module, kmax: int, border_sharpness: float,
                 grad_implemented: bool=False, normalize: bool=True):
        """Initialize the problem
        
        Args:
            lander (lander): lander model
            x0 (np.array-like): initial state; [x, y, z, vx, vy, vz, m]
            tgo (float): time-to-go
            sfmap (np.array-like): safety map, [[x, y, safety], ...]]
            nn_reach (torch.nn.Module): neural network model for reachability steering
            kmax (int): time step at which reachable safety is maximized
            border_sharpness (float): sharpness of the border of the safety map
            grad_implemented (bool): whether gradient is implemented 
            normalize (bool): whether to normalize the problem
        """
        super().__init__(lander, N, x0, tgo, grad_implemented, normalize)
        self.kmax = kmax
        self.sfmap = sfmap
        self.nn_reach = nn_reach
        self.border_sharpness = border_sharpness

    def fitness(self, x):
        """Compute fitness for given decision vector x

        Args:
            x (np.array-like): decision vector
        """
        u_ = self.construct_thrust(x)

        # Propagate dynamics
        r_, v_, m_ = _propagate_state(self.x0_, u_, self.N, self.dt_, self.g_, self.alpha_)

        # Terminal state constraints
        cstr_eq = [r_[-1, 2]]  # Final altitude = 0
        cstr_eq += list(v_[-1, :])  # Final velocity = 0

        cstr_ineq_terminal = [self.lander.mdry / self.MU - m_[-1]]  # Dry mass <= final mass

        # Operational constraints
        cstr_ineq = _get_cstr(r_, v_, self.N, self.lander.gsa, self.vmax_)

        # Compute fitness
        r = r_ * self.LU
        v = v_ * self.LU / self.TU
        m = m_ * self.MU
        nn_input = np.hstack((r[self.kmax, :].flatten(), v[self.kmax, :].flatten(), m[self.kmax]))
        safety, _ = ic2mean_safety_npy(
            lander=self.lander,
            x0=nn_input,
            tgo=self.tgo-self.dt * self.kmax,
            model=self.nn_reach,
            sfmap=self.sfmap,
            border_sharpness=self.border_sharpness,
            )
        return [-safety] + list(cstr_eq) + cstr_ineq_terminal + list(cstr_ineq)


@jit(nopython=True)
def _dynamics(r, v, m, u, dt, g, alpha):
    dt22 = dt ** 2 / 2.0
    a = u / m + g

    # Compute next state
    r_next = r + dt * v + dt22 * a
    v_next = v + dt * a
    m_next = m - dt * alpha * np.linalg.norm(u)

    return r_next, v_next, m_next

@jit(nopython=True)
def _propagate_state(x0, u, N, dt, g, alpha):
    r = np.zeros((N + 1, 3))
    v = np.zeros((N + 1, 3))
    m = np.zeros(N + 1)

    r[0] = x0[:3]
    v[0] = x0[3:6]
    m[0] = x0[6]

    for i in range(N):
        r[i + 1], v[i + 1], m[i + 1] = _dynamics(r[i], v[i], m[i], u[i], dt, g, alpha)
    
    return r, v, m

@jit(nopython=True)
def _get_cstr(r, v, N, gsa, vmax):
    cstr_ineq = np.zeros(2 * N + 1)

    # Inequality constraints
    i_ieq = 0

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
