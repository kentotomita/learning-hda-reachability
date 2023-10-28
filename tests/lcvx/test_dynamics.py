import unittest
import numpy as np
from scipy.integrate import solve_ivp
import src.lcvx as lc
from src.dynamics import pd_3dof_eom


class TestLCVxDynamics(unittest.TestCase):
    """Tests for the linearlized dynamics for the LCVX formulation"""

    def setUp(self):
        # Simulation parameters
        self.rocket = lc.Rocket(
            g_=3.7114,  # Gravitational acceleration (m/s^2)
            mdry=1505.0,  # Dry mass (kg)
            mwet=1905.0,  # Wet mass (kg)
            Isp=225.0,
            rho1=4972.0,  # Minimum thrust (N)
            rho2=13260.0,  # Maximum thrust (N)
        )
        self.x0 = np.array(
            [2000.0, 500.0, 1500.0, -100.0, 50.0, -75.0, self.rocket.mwet]
        )
        self.tmax = 100.0  # Maximum time (s)
        self.u_ref = np.array([1.0, 0.1, 4.0])  # Reference control

        # compute reference trajectory
        fun = lambda t, x: pd_3dof_eom(x, self.u_ref, self.rocket.g, self.rocket.alpha)
        event = lambda t, x: x[2]
        event.terminal = True
        sol = solve_ivp(fun, (0, self.tmax), self.x0, events=event)
        self.t = sol.t
        self.x = sol.y
        self.u = np.tile(self.u_ref, (sol.t.size, 1)).T

    def test_relaxed_dynamics_c(self):
        """Continuous dynamics"""

        # Linearized system
        Ac, Bc, pc = lc.continuous_sys(self.rocket.g_, self.rocket.alpha)

        def f(t, x):
            u_norm = np.linalg.norm(self.u_ref)
            mass = np.exp(x[-1])
            u_ref = np.hstack(
                (self.u_ref / mass, u_norm / mass)
            )  # Eqs (25) of Acikmese-2007
            dx = Ac @ x.reshape(-1, 1) + Bc @ u_ref.reshape(-1, 1) + pc.reshape(-1, 1)
            return dx.reshape(-1)

        def event(t, x):
            return x[2]

        event.terminal = True
        x0_ = np.hstack((self.x0[:-1], np.log(self.x0[-1])))

        sol = solve_ivp(f, (0, self.tmax), x0_, events=event)
        xf = np.hstack((sol.y[:-1, -1], np.exp(sol.y[-1, -1])))

        # Compare final states and time with reference
        self.assertAlmostEqual(sol.t[-1], self.t[-1])
        np.testing.assert_array_almost_equal(xf, self.x[:, -1], decimal=3)
