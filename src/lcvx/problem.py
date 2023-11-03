import numpy as np
import cvxpy as cp

from typing import Tuple, List, Union

from .discretization import zoh
from .rocket import Rocket

__all__ = [
    "LCvxProblem",
    "LCvxMinFuel",
    "LCvxReachability",
]


class LCvxProblem:
    """Base class for guidance optimization problems for powered descent via lossless convexification."""

    def __init__(
        self,
        rocket: Rocket,
        N: int,
        parameterize_x0: bool = False,
        parameterize_tf: bool = False,
    ):
        """Initialize the LCvx class.
        Args:
            rocket: Rocket object.
            N: Number of discretization points.
            parameterize_x0: If True, then the initial state is a parameter. Default is False.
            parameterize_tf: If True, then the final time is a parameter. Default is False.
        """

        self.rocket = rocket
        self.N = N

        self.parameterize_x0 = parameterize_x0
        self.parameterize_tf = parameterize_tf

        # Scaling factors for normalization
        self.R_ref = 1000
        self.Acc_ref = abs(self.rocket.g[2])
        self.T_ref = np.sqrt(self.R_ref / self.Acc_ref)
        self.V_ref = self.R_ref / self.T_ref
        # Z_ref = np.log(self.rocket.mwet)
        self.Z_max = np.log(self.rocket.mwet)
        self.Z_min = np.log(self.rocket.mdry)
        self.Sigma_ref = self.Acc_ref

    def _parameters(self, x0: np.ndarray, tf: np.ndarray):
        """Define the problem parameters.

        Returns:
            x0: Initial state.
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

    def problem(self, x0: np.ndarray = None, tf: np.ndarray = None) -> cp.Problem:
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
        if self.rocket.pa is not None:
            cstr += [u[2, :] >= sigma * np.cos(self.rocket.pa)]

        # Glide slope constraint
        if self.rocket.gsa is not None:
            cstr += [
                cp.norm(r[:2, i] - r[:2, -1], p=2) * np.tan(self.rocket.gsa)
                <= r[2, i] - r[2, -1]
                for i in range(self.N + 1)
            ]

        # Velocity constraint
        if self.rocket.vmax is not None:
            cstr += [cp.norm(v, p=2, axis=0) <= self.rocket.vmax]

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
                        + dt22 * self.rocket.g[i]
                    ]
                    cstr += [
                        v[i, k + 1] == v[i, k] + dt * u[i, k] + dt * self.rocket.g[i]
                    ]
                cstr += [z[k + 1] == z[k] - dt * self.rocket.alpha * sigma[k]]

        # Else, use Numpy arrays
        else:
            A, B, p = zoh(
                A=self.rocket.Ac, B=self.rocket.Bc, p=self.rocket.pc, dt=dt
            )  # discretize dynamics
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
                    self.rocket.mwet - self.rocket.alpha * self.rocket.rho2 * t[i]
                )  # mass trajectory lower bound
                dz = z[i] - z1
                mu1 = self.rocket.rho1 * cp.exp(-z1)
                mu2 = self.rocket.rho2 * cp.exp(-z1)
                cstr += [mu1 * (1 - dz + 1 / 2 * dz**2) <= sigma[i]]
                cstr += [sigma[i] <= mu2 * (1 - dz)]

        # If t is NumPy array, use Numpy functions
        else:
            z1 = np.log(
                self.rocket.mwet - self.rocket.alpha * self.rocket.rho2 * t
            )  # mass trajectory lower bound
            dz = z - z1
            mu1 = self.rocket.rho1 * np.exp(-z1)
            mu2 = self.rocket.rho2 * np.exp(-z1)
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
        rocket: Rocket,
        N: int,
        fixed_target: bool = False,
        parameterize_tf: bool = False,
        parameterize_x0: bool = False,
    ):
        """Landing problem for minimum fuel consumption.

        Args:
            rocket: Rocket model.
            N: Number of discretization points.
            fixed_target: If True, the target position is fixed at the origin.
            parameterize_tf: If True, the final time is a parameter.
            parameterize_x0: If True, the initial state is a parameter.
        """
        super().__init__(
            rocket, N, parameterize_tf=parameterize_tf, parameterize_x0=parameterize_x0
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
            z[-1] >= np.log(self.rocket.mdry)
        )  # final log(mass) >= log(dry mass)

        return cstr


class LCvxReachability(LCvxProblem):
    """Reachability problems; given a bounded initial state set, compute the reachable set. 
    The final altitude and the final velocity are fixed at zero."""

    def __init__(self, rocket: Rocket, N: int, maxk: int, directional_cstr: Union[bool, List[bool]] = False):
        """Landing problem for computing a reachable set at a specific step.

        Args:
            rocket: Rocket model.
            N: Number of discretization points.
            maxk: Step at which the maximization is performed.
            directional_cstr: Directional constraints for each state.
        """
        super().__init__(rocket, N)
        self.maxk = maxk
        self.directional_cstr = directional_cstr

    def _boundary_cstr(self, vars: Tuple[cp.Variable], x0_bounds: Tuple[np.ndarray]):
        """Boundary conditions. Inital states are bounded by definition. Final states are fixed for altitude and velocity.
        
        Args:
            vars: Tuple of cvxpy variables: r, v, m, u, sigma.
            x0_bounds: Initial state bounds.
        """
        r, v, z, _, _ = vars
        x0_min, x0_max = x0_bounds

        # Initial conditions
        cstr = [r[:, 0] >= x0_min[:3]]  # initial position
        cstr.append(r[:, 0] <= x0_max[:3])
        cstr.append(v[:, 0] >= x0_min[3:6])  # initial velocity
        cstr.append(v[:, 0] <= x0_max[3:6])
        cstr.append(z[0] >= x0_min[6])  # initial log(mass)
        cstr.append(z[0] <= x0_max[6])

        # Final conditions
        cstr.append(r[2, -1] == 0)  # target altitude is zero
        cstr += [v[:, -1] == np.zeros(3)]  # soft landing; v = 0 at final time
        cstr.append(
            z[-1] >= np.log(self.rocket.mdry)
        )  # final log(mass) >= log(dry mass)

        return cstr
    
    def problem(
        self,
        x0_bounds: Tuple[np.ndarray],
        tf: float,
        xc: np.ndarray = None,
    ) -> cp.Problem:
        """Define the reachability problem.
        Args:
            x0_bounds: Initial state bounds.
            tf: Final time.
            xc: Center state from which range is maximized. Shape: (7,) for x, y, z, vx, vy, vz, log(mass).
        """
        assert xc is not None or self.inner is False, "center state must be provided if reachability is computed for the inner approximation."
        
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
            r[0, self.maxk], r[1, self.maxk], r[2, self.maxk],
            v[0, self.maxk], v[1, self.maxk], v[2, self.maxk],
            z[self.maxk],
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

