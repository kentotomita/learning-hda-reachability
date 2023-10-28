import unittest
import numpy as np
import src.lcvx as lc


class TestLCVxMinFuel(unittest.TestCase):
    def setUp(self):
        # Simulation parameters
        self.rocket = lc.Rocket(
            g_=3.7114,  # Gravitational acceleration (m/s^2)
            mdry=1505.0,  # Dry mass (kg)
            mwet=1905.0,  # Wet mass (kg)
            Isp=225.0,
            rho1=4972.0,  # Minimum thrust (N)
            rho2=13260.0,  # Maximum thrust (N)
            gsa=25 * np.pi / 180,
            pa=40 * np.pi / 180,
            vmax=None,
        )
        self.x0 = np.array(
            [2000.0, 500.0, 1500.0, -100.0, 50.0, -75.0, np.log(self.rocket.mwet)]
        )
        self.tf = 75.0
        self.N = 55
        self.dt = self.tf / self.N

    def test_noparam(self):
        """Test problem with no parameterization"""
        # Define problem

        lcvx_obj = lc.LCvxMinFuel(
            rocket=self.rocket,
            N=self.N,
            fixed_target=True,
        )
        prob = lcvx_obj.problem(x0=self.x0, tf=self.tf)
        prob.solve(verbose=False)

        # Assert problem is solved
        self.assertTrue(prob.status == "optimal")

    def test_param_x0(self):
        """Test problem with parameterization of x0"""
        # Define problem
        lcvx_obj = lc.LCvxMinFuel(
            rocket=self.rocket,
            N=self.N,
            parameterize_x0=True,
            fixed_target=True,
        )
        prob = lcvx_obj.problem(tf=self.tf)
        lc.set_params(prob, {"x0": self.x0})
        prob.solve(verbose=False)

        # Assert problem is solved
        self.assertTrue(prob.status == "optimal")
