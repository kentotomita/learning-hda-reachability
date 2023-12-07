"""Optimization problems for powered descent via lossless convexification."""
import numpy as np
import cvxpy as cp

from typing import Tuple, List, Union

from .linearized_dynamics import discrete_sys
from ..landers import Lander

__all__ = [
    "LCvxProblem",
    "LCvxMinFuel",
    "LCvxReachability",
    "LCvxControllability",
    "LcVxControllabilityVxz",
    "LCvxReachabilityRxy"
]


class LCvxProblem:
    """Base class for guidance optimization problems for powered descent via lossless convexification."""

    def __init__(
        self,
        lander: Lander,
        N: int,
        parameterize_x0: bool = False,
        parameterize_tf: bool = False,
    ):
        """Initialize the LCvx class.
        Args:
            lander: lander object.
            N: Number of discretization points.
            parameterize_x0: If True, then the initial state is a parameter. Default is False.
            parameterize_tf: If True, then the final time is a parameter. Default is False.
        """

        self.lander = lander
        self.N = N

        self.parameterize_x0 = parameterize_x0
        self.parameterize_tf = parameterize_tf

        # Scaling factors for normalization
        self.R_ref = lander.LU
        self.Acc_ref = self.lander.g_
        self.T_ref = np.sqrt(self.R_ref / self.Acc_ref)
        self.V_ref = self.R_ref / self.T_ref
        self.Z_max = np.log(self.lander.mwet)
        self.Z_min = np.log(self.lander.mdry)
        self.Sigma_ref = self.Acc_ref

    def _parameters(self, x0: np.ndarray, tf: np.ndarray):
        """Define the problem parameters.

        Returns:
            x0: Initial state; [rx, ry, rz, vx, vy, vz, z=log(mass)]
            tf: Final time.
            dt: Time step.
            t: Time vector.
            dt22: Time step squared divided by 2; dt^2/2. Only needed for parameterized tf.
        """
        if self.parameterize_x0:
            x0 = cp.Parameter(shape=7, name="x0")
        else:
            assert (
                x0 is not None
            ), "Initial state must be provided unless parameterized."

        if self.parameterize_tf:
            print(
                "WARNING: Parameterizing tf does not result in DPP for current CVXPY version."
            )
            tf = cp.Parameter(nonneg=True, name="tf")
            dt22 = cp.Parameter(name="dt22")  # time step squared divided by 2; dt^2/2
            dt = tf / self.N
            t = [i * dt for i in range(self.N + 1)]
        else:
            assert tf is not None, "Final time must be provided."
            dt22 = None
            dt = tf / self.N
            t = np.array([i * dt for i in range(self.N + 1)])

        return x0, tf, dt, t, dt22

    def problem(self) -> cp.Problem:
        """Define the optimization problem."""
        raise NotImplementedError

    def _objective(self):
        """Define the objective of the problem."""
        raise NotImplementedError

    def _boundary_cstr(self):
        """Define the boundary constraints of the problem."""
        return []

    def _lcvx_constraints(
        self, params: Tuple, vars: Tuple[cp.Variable]
    ) -> List[cp.Problem]:
        """Define constraints of the powered descent problem **except for the boundary conditions**.

        Args:
            params: Tuple of problem parameters.
            vars: Tuple of problem variables.

        Returns:
            A list of constraints.
        """
        # Unpac parameters
        x0, tf, dt, t, dt22 = params

        # Unpack variables
        r, v, z, u, sigma = vars

        cstr = []

        # Dynamics
        cstr += self._dynamics_cstr(vars=vars, dt=dt, dt22=dt22)

        # Thrust bounds (Convexified; Approximation)
        cstr += self._thrust_bounds_cstr(vars=vars, t=t)

        # Thrust bounds LCvx
        cstr += [cp.norm(u, axis=0) <= sigma]

        # Pointing constraint
        if self.lander.pa is not None:
            cstr += [u[2, :] >= sigma * np.cos(self.lander.pa)]

        # Glide slope constraint
        if self.lander.gsa is not None:
            cstr += [
                cp.norm(r[:2, i] - r[:2, -1], p=2) * np.tan(self.lander.gsa)
                <= r[2, i] - r[2, -1]
                for i in range(self.N + 1)
            ]

        # Velocity constraint
        if self.lander.vmax is not None:
            cstr += [cp.norm(v, p=2, axis=0) <= self.lander.vmax]

        return cstr

    def _dynamics_cstr(
        self,
        vars: Tuple[cp.Variable],
        dt: float or cp.Parameter,
        dt22: cp.Parameter = None,
    ):
        """Dynamics constraints for the convex relaxed landing problem."""
        r, v, z, u, sigma = vars

        cstr = []

        # If tf is a parameter, avoid using Numpy arrays
        if self.parameterize_tf:
            for k in range(self.N):
                for i in range(3):
                    cstr += [
                        r[i, k + 1]
                        == r[i, k]
                        + dt * v[i, k]
                        + dt22 * u[i, k]
                        + dt22 * self.lander.g[i]
                    ]
                    cstr += [
                        v[i, k + 1] == v[i, k] + dt * u[i, k] + dt * self.lander.g[i]
                    ]
                cstr += [z[k + 1] == z[k] - dt * self.lander.alpha * sigma[k]]

        # Else, use Numpy arrays
        else:
            A, B, p = discrete_sys(dt, self.lander.g_, self.lander.alpha)
            for k in range(self.N):
                xk = cp.hstack([r[:, k], v[:, k], z[k]])
                x_next = cp.hstack([r[:, k + 1], v[:, k + 1], z[k + 1]])
                uk = cp.hstack([u[:, k], sigma[k]])
                cstr += [x_next == A @ xk + B @ uk + p]

        return cstr

    def _thrust_bounds_cstr(
        self, vars: Tuple[cp.Variable], t: np.array or List[cp.Expression]
    ):
        """Thrust bounds constraints for the convex relaxed landing problem."""
        _, _, z, _, sigma = vars

        cstr = []
        # If t is a parameter expression, avoid using Numpy function
        if self.parameterize_tf:
            for i in range(self.N):
                z1 = cp.log(
                    self.lander.mwet - self.lander.alpha * self.lander.rho2 * t[i]
                )  # mass trajectory lower bound
                dz = z[i] - z1
                mu1 = self.lander.rho1 * cp.exp(-z1)
                mu2 = self.lander.rho2 * cp.exp(-z1)
                cstr += [mu1 * (1 - dz + 1 / 2 * dz**2) <= sigma[i]]
                cstr += [sigma[i] <= mu2 * (1 - dz)]

        # If t is NumPy array, use Numpy functions
        else:
            z1 = np.log(
                self.lander.mwet - self.lander.alpha * self.lander.rho2 * t
            )  # mass trajectory lower bound
            dz = z - z1
            mu1 = self.lander.rho1 * np.exp(-z1)
            mu2 = self.lander.rho2 * np.exp(-z1)
            cstr += [
                mu1[i] * (1 - dz[i] + 1 / 2 * dz[i] ** 2) <= sigma[i]
                for i in range(self.N)
            ]
            cstr += [sigma[i] <= mu2[i] * (1 - dz[i]) for i in range(self.N)]

        return cstr

    def recover_variables(self, X: cp.Variable, U: cp.Variable) -> tuple:
        """Recover the original variables from the scaled ones.

        Returns:
            r: Position vector (m), shape=(3, N+1)
            v: Velocity vector (m/s), shape=(3, N+1)
            z: Mass vector (log(kg)), shape=(N+1,)
            u: Thrust vector (m/s^2), shape=(3, N)
            sigma: Thrust magnitude slack variable (m/s^2), shape=(N,)
        """

        r = self.R_ref * X[:3, :]
        v = self.V_ref * X[3:6, :]
        z = self.Z_min + (self.Z_max - self.Z_min) * X[6, :]
        u = self.Acc_ref * U[:3, :]
        sigma = self.Sigma_ref * U[3, :]

        return r, v, z, u, sigma


class LCvxMinFuel(LCvxProblem):
    def __init__(
        self,
        lander: Lander,
        N: int,
        fixed_target: bool = False,
        parameterize_tf: bool = False,
        parameterize_x0: bool = False,
    ):
        """Landing problem for minimum fuel consumption.

        Args:
            lander: Lander model.
            N: Number of discretization points.
            fixed_target: If True, the target position is fixed at the origin.
            parameterize_tf: If True, the final time is a parameter.
            parameterize_x0: If True, the initial state is a parameter.
        """
        super().__init__(
            lander, N, parameterize_tf=parameterize_tf, parameterize_x0=parameterize_x0
        )
        self.fixed_target = fixed_target

    def problem(self, x0: np.ndarray = None, tf: np.ndarray = None) -> cp.Problem:
        """Define the optimization problem.

        Args:
            x0: Initial state.
            tf: Final time.

        Returns:
            A cvxpy problem.
        """
        # Problem parameters
        x0, tf, dt, t, dt22 = self._parameters(x0, tf)

        # Problem variables
        X = cp.Variable((7, self.N + 1), name="X")  # state variables
        U = cp.Variable((4, self.N), name="U")  # control variables

        # Recovered variables
        r, v, z, u, sigma = self.recover_variables(X, U)

        # Problem constraints
        cstr = self._lcvx_constraints(
            params=(x0, tf, dt, t, dt22), vars=(r, v, z, u, sigma)
        )
        cstr += self._boundary_cstr(vars=(r, v, z, u, sigma), x0=x0)

        # Problem objective
        obj = cp.Minimize(cp.sum(sigma) * dt)

        return cp.Problem(obj, cstr)

    def _boundary_cstr(self, vars: Tuple[cp.Variable], x0: np.ndarray or cp.Parameter):
        """Boundary conditions

        Args:
            vars: Tuple of cvxpy variables: r, v, m, u, sigma.
            x0: Initial state.
        """
        r, v, z, _, _ = vars

        # Initial conditions
        cstr = [r[:, 0] == x0[:3]]  # initial position
        cstr.append(v[:, 0] == x0[3:6])  # initial velocity
        cstr.append(z[0] == x0[6])  # initial log(mass)

        # Final conditions
        if self.fixed_target:
            cstr.append(
                r[:, -1] == np.zeros(3)
            )  # target position is fixed at the origin
        else:
            cstr.append(r[2, -1] == 0)  # target altitude is zero
        cstr += [v[:, -1] == np.zeros(3)]  # soft landing; v = 0 at final time
        cstr.append(
            z[-1] >= np.log(self.lander.mdry)
        )  # final log(mass) >= log(dry mass)

        return cstr


class LCvxReachability(LCvxProblem):
    """Reachability problems; given a bounded initial state set, compute the reachable set. 
    The final altitude and the final velocity are fixed at zero."""

    def __init__(self, lander: Lander, N: int, directional_cstr: Union[bool, List[bool]] = False):
        """Landing problem for computing a reachable set at a specific step.

        Args:
            lander: Lander model.
            N: Number of discretization points.
            directional_cstr: Directional constraints for each state.
        """
        super().__init__(lander, N)
        self.directional_cstr = directional_cstr
    
    def problem(
        self,
        x0_bounds: Tuple[np.ndarray],
        tf: float,
        maxk: int,
        xc: np.ndarray = None,
    ) -> cp.Problem:
        """Define the reachability problem.
        Args:
            x0_bounds: Initial state bounds.
            tf: Final time.
            maxk: Step at which the maximization is performed.
            xc: Center state from which range is maximized. Shape: (7,) for x, y, z, vx, vy, vz, log(mass).
        """
        
        # Problem parameters
        dt = tf / self.N
        t = np.array([i * dt for i in range(self.N + 1)])
        c = cp.Parameter(7, name="c")  # maximum state direction

        # Problem variables
        X = cp.Variable((7, self.N + 1), name="X")  # state variables
        U = cp.Variable((4, self.N), name="U")  # control variables

        # Recovered variables
        r, v, z, u, sigma = self.recover_variables(X, U)

        # Problem constraints
        cstr = self._lcvx_constraints(
            params=(None, tf, dt, t, None), vars=(r, v, z, u, sigma)
        )
        cstr += self._boundary_cstr(vars=(r, v, z, u, sigma), x0_bounds=x0_bounds)

        # Problem objective
        if xc is None:
            xc = np.zeros(7)
        xk = [
            r[0, maxk], r[1, maxk], r[2, maxk],
            v[0, maxk], v[1, maxk], v[2, maxk],
            z[maxk],
        ]
        xk = [xk[i] - xc[i] for i in range(7)]  # state at maximization step relative to the center state
        
        if self.directional_cstr:  # state vector from the given center has to be parallel to the maximum state direction
            xk_masked = [xk[i] for i in range(7) if self.directional_cstr[i]]
            c_masked = [c[i] for i in range(7) if self.directional_cstr[i]]
            for i in range(len(xk_masked)):
                cstr += [
                    xk_masked[0] * c_masked[i] - xk_masked[i] * c_masked[0] == 0
                ]

        obj = cp.Maximize(cp.sum([c[i] * xk[i] for i in range(7)]))

        return cp.Problem(obj, cstr)

    def _boundary_cstr(self, vars: Tuple[cp.Variable], x0_bounds: Tuple[List]=None, x0=None):
        """Boundary conditions. Inital states are bounded by definition. Final states are fixed for altitude and velocity.
        
        Args:
            vars: Tuple of cvxpy variables: r, v, m, u, sigma.
            x0_bounds: Initial state bounds.
        """
        r, v, z, _, _ = vars

        # Initial conditions
        if x0_bounds is not None:
            x0_min, x0_max = x0_bounds
            cstr  = [r[i, 0] >= x0_min[i] for i in range(3)]
            cstr += [r[i, 0] <= x0_max[i] for i in range(3)]
            cstr += [v[i, 0] >= x0_min[i + 3] for i in range(3)]
            cstr += [v[i, 0] <= x0_max[i + 3] for i in range(3)]
            cstr += [z[0] >= x0_min[6]]
            cstr += [z[0] <= x0_max[6]]
        elif x0 is not None:
            cstr  = [r[i, 0] == x0[i] for i in range(3)]
            cstr += [v[i, 0] == x0[i + 3] for i in range(3)]
            cstr += [z[0] == x0[6]]
        else:
            raise ValueError("Initial state or its bounds must be provided.")

        # Terminal conditions
        cstr.append(r[2, -1] == 0)  # target altitude is zero
        cstr += [v[:, -1] == np.zeros(3)]  # soft landing; v = 0 at final time
        cstr.append(
            z[-1] >= np.log(self.lander.mdry)
        )  # final log(mass) >= log(dry mass)

        return cstr
    

class LCvxReachabilityRxy(LCvxReachability):
    """Reachability problem class for range in x-y plane."""

    def __init__(self, lander: Lander, N: int, directional_cstr: bool = True):
        super().__init__(lander, N)
        self.directional_cstr = directional_cstr

    def problem(self, tf: float):
        """Define the reachability problem.
        Args:
            x0_bounds: Initial state bounds.
            tf: Final time.
        """
        # Problem parameters
        dt = tf / self.N
        t = np.array([i * dt for i in range(self.N + 1)])
        alt0 = cp.Parameter(nonneg=True, name="alt")
        vx0 = cp.Parameter(name="vx")
        vz0 = cp.Parameter(name="vz")
        z_mass0 = cp.Parameter(nonneg=True, name="z_mass")  # log(mass) at the initial time
        c = cp.Parameter(2, name="c")  # maximum range direction in x-y plane
        c_xc_arr = cp.Parameter((2, 2), name="c_xc_arr")  # product of center state and maximum state direction

        # Problem variables
        X = cp.Variable((7, self.N + 1), name="X")
        U = cp.Variable((4, self.N), name="U")

        # Recovered variables
        r, v, z, u, sigma = self.recover_variables(X, U)

        # Problem constraints
        x0 = [0.0, 0.0, alt0, vx0, 0.0, vz0, z_mass0]
        cstr = self._lcvx_constraints(
            params=(None, tf, dt, t, None), vars=(r, v, z, u, sigma)
        )
        cstr += self._boundary_cstr(vars=(r, v, z, u, sigma), x0=x0)

        # Problem objective
        obj = cp.Maximize(c[0] * r[0, -1] - c_xc_arr[0, 0] + c[1] * r[1, -1] - c_xc_arr[1, 1])

        # Directional constraint
        if self.directional_cstr:
            cstr += [(c[0] * r[1, -1] - c_xc_arr[0, 1]) - (c[1] * r[0, -1] - c_xc_arr[1, 0]) == 0]  # c x (v - vc) = 0

        prob = cp.Problem(obj, cstr)

        # If Disciplined Parametrized Programming (DPP), solving it repeatedly for different 
        # values of the parameters can be much faster than repeatedly solving a new problem.
        assert prob.is_dpp(), "Problem is not DPP."

        return prob


class LCvxControllability(LCvxProblem):
    """Controllability problem class; given a bounded terminal state set, compute the feasible initial state set."""

    def __init__(self, lander: Lander, N: int):
        """Landing problem for computing a controllable set.

        Args:
            lander: lander model.
            N: Number of discretization points.
        """
        super().__init__(lander, N)
    
    def problem(
        self,
        xf_bounds: Tuple[np.ndarray],
        tf: float,
        x0_paramed: Union[bool, List[bool]] = [False, False, True, False, False, False, False],
        c_paramed: Union[bool, List[bool]] = True,
        xc_paramed: Union[bool, List[bool]] = False,
        directional_cstr: Union[bool, List[bool]] = False,
        xc: np.ndarray = None,
    ) -> cp.Problem:
        """Define the reachability problem.
        Args:
            xf_bounds: Terminal state bounds.
            tf: Final time.
            x0_paramed: If True, then the initial state are parameters. Default is True for altitude.
            c_paramed: If True, then the maximum state direction is a parameter. Default is True.
            xc_paramed: If True, then the center state is a parameter. Default is False.
            directional_cstr: Directional constraints for each state.
            xc: Center state from which range is maximized. Shape: (7,) for x, y, z, vx, vy, vz, log(mass).
        """
        # Problem parameters
        dt = tf / self.N
        t = np.array([i * dt for i in range(self.N + 1)])
        # Parameterization for maximum state direction
        c = list(np.zeros(7))
        if c_paramed is not False:
            c_ = cp.Parameter(sum(c_paramed), name="c")
            j = 0
            for i in range(7):
                if c_paramed[i]:
                    c[i] = c_[j]
                    j += 1

        # Parameterization for initial state
        if x0_paramed is not False:
            x0 = cp.Parameter(sum(x0_paramed), name="x0")

        # Parameterization for center state
        xc = list(np.zeros(7))
        if xc_paramed is not False:
            xc_ = cp.Parameter(sum(xc_paramed), name="xc")
            j = 0
            for i in range(7):
                if xc_paramed[i]:
                    xc[i] = xc_[j]
                    j += 1

        # Problem variables
        X = cp.Variable((7, self.N + 1), name="X")  # state variables
        U = cp.Variable((4, self.N), name="U")  # control variables

        # Recovered variables
        r, v, z, u, sigma = self.recover_variables(X, U)

        # Problem constraints
        cstr = self._lcvx_constraints(
            params=(None, tf, dt, t, None), vars=(r, v, z, u, sigma)
        )
        cstr += self._boundary_cstr(vars=(r, v, z, u, sigma), xf_bounds=xf_bounds, x0_paramed=x0_paramed, x0=x0)

        # Problem objective
        if xc is None:
            xc = np.zeros(7)
        x0 = [
            r[0, 0], r[1, 0], r[2, 0],
            v[0, 0], v[1, 0], v[2, 0],
            z[0],
        ]
        x0 = [x0[i] - xc[i] for i in range(7)]  # initial state relative to the center state
        
        if directional_cstr:  # state vector from the given center has to be parallel to the maximum state direction
            x0_masked = [x0[i] for i in range(7) if directional_cstr[i]]
            c_masked = [c[i] for i in range(7) if directional_cstr[i]]
            for i in range(len(x0_masked)):
                cstr += [
                    x0_masked[0] * c_masked[i] - x0_masked[i] * c_masked[0] == 0
                ]

        obj = cp.Maximize(cp.sum([c[i] * x0[i] for i in range(7)]))
        prob = cp.Problem(obj, cstr)

        # If Disciplined Parametrized Programming (DPP), solving it repeatedly for different 
        # values of the parameters can be much faster than repeatedly solving a new problem.
        assert prob.is_dpp(), "Problem is not DPP."

        return prob

    def _boundary_cstr(self, vars: Tuple[cp.Variable], xf_bounds: Tuple[np.ndarray], x0_paramed: Union[bool, List[bool]], x0: cp.Parameter=None):
        """Boundary conditions. Terminal states are bounded by definition. Initial states are fixed for altitude.
        
        Args:
            vars: Tuple of cvxpy variables: r, v, m, u, sigma.
            xf_bounds: Terminal state bounds.
            x0_paramed: If True, then the initial state are parameters.
            x0: Initial state.
        """
        r, v, z, _, _ = vars

        # Terminal conditions
        xf_min, xf_max = xf_bounds
        cstr = [r[:, -1] >= xf_min[:3]]  # position
        cstr.append(r[:, -1] <= xf_max[:3])
        cstr.append(v[:, -1] >= xf_min[3:6])  # velocity
        cstr.append(v[:, -1] <= xf_max[3:6])
        cstr.append(z[-1] >= xf_min[6])  # log(mass)
        cstr.append(z[-1] <= xf_max[6])
        cstr.append(
            z[-1] >= np.log(self.lander.mdry)
        )  # final log(mass) >= log(dry mass)
        
        # Initial conditions
        cstr.append(z[0] <= np.log(self.lander.mwet))  # initial log(mass) <= log(wet mass)
        if x0_paramed is not None:
            j = 0
            for i, paramed in enumerate(x0_paramed):
                if paramed:
                    if 0 <= i <= 2:  # position
                        cstr += [r[i, 0] == x0[j]]
                    elif 3 <= i <= 5:  # velocity
                        cstr += [v[i - 3, 0] == x0[j]]
                    elif i == 6:  # log(mass)
                        cstr += [z[0] == x0[j]]
                    else:
                        raise ValueError(f"Undefined parameter: {i}")
                    j += 1
        return cstr


class LcVxControllabilityVxz(LCvxControllability):
    """Controllability problem class for velocity in x-z plane."""

    def __init__(self, lander: Lander, N: int):
        super().__init__(lander, N)

    def problem(self, xf_bounds: Tuple[np.ndarray], tf: float):
        """Define the reachability problem.
        Args:
            xf_bounds: Terminal state bounds.
            tf: Final time.
        """
        # Problem parameters
        dt = tf / self.N
        t = np.array([i * dt for i in range(self.N + 1)])
        alt0 = cp.Parameter(nonneg=True, name="alt")
        z_mass = cp.Parameter(nonneg=True, name="z_mass")  # log(mass) at the initial time
        c = cp.Parameter(2, name="c")
        c_xc_arr = cp.Parameter((2, 2), name="c_xc_arr")  # product of center state and maximum state direction

        # Problem variables# Problem variables
        X = cp.Variable((7, self.N + 1), name="X")  # state variables
        U = cp.Variable((4, self.N), name="U")  # control variables

        # Recovered variables
        r, v, z, u, sigma = self.recover_variables(X, U)

        # Problem constraints
        cstr = self._lcvx_constraints(
            params=(None, tf, dt, t, None), vars=(r, v, z, u, sigma)
        )
        cstr += self._boundary_cstr(
            vars=(r, v, z, u, sigma),
            xf_bounds=xf_bounds,
            x0_paramed=[False, False, True, False, False, False, True],
            x0=[alt0, z_mass])

        # Problem objective
        obj = cp.Maximize(c[0] * v[0, 0] - c_xc_arr[0, 0] + c[1] * v[2, 0] - c_xc_arr[1, 1])

        # Directional constraint
        cstr += [(c[0] * v[2, 0] - c_xc_arr[0, 1]) - (c[1] * v[0, 0] - c_xc_arr[1, 0]) == 0]  # c x (v - vc) = 0

        prob = cp.Problem(obj, cstr)

        # If Disciplined Parametrized Programming (DPP), solving it repeatedly for different 
        # values of the parameters can be much faster than repeatedly solving a new problem.
        assert prob.is_dpp(), "Problem is not DPP."

        return prob
