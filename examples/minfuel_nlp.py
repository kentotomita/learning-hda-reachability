import pickle
import cvxpy as cp
import pygmo as pg
from pygmo_plugins_nonfree import snopt7
import numpy as np
import sys
sys.path.append('../')
from src.reachsteering import MinFuelStateCtrl
from src.visualization import *
import src.lcvx as lc


def main():
        with open('../saved/controllable_set/lander.pkl', 'rb') as f:
                lander = pickle.load(f)

        N = 60
        tf = 60.0
        alt = 1500.0
        mass = 1800.0
        x0 = np.array([0, 0, alt, -20.0, 0, -55.0, np.log(mass)])

        lcvx = lc.LCvxMinFuel(
                lander=lander,
                N=N,
                parameterize_x0=False,
                parameterize_tf=False,
                fixed_target=False,
        )
        prob = lcvx.problem(x0=x0, tf=tf)
        prob.solve(solver=cp.ECOS, verbose=True)

        sol = lc.get_vars(prob, ["X", "U"])
        X_sol = sol["X"]
        U_sol = sol["U"]
        r, v, z, u, sigma = lcvx.recover_variables(X_sol, U_sol)
        m = np.exp(z)
        u = u.T * m[:-1].reshape(-1, 1)
        r_ = r.T / lander.LU
        v_ = v.T / (lander.LU / lander.TU)
        m_ = m.T / lander.MU
        u_ = u / (lander.MU * lander.LU / lander.TU ** 2)
        x_udp_0 = np.hstack((r_.flatten(), v_.flatten(), m_.flatten(), u_.flatten()))

        x0[6] = mass
        udp = MinFuelStateCtrl(lander, N, x0, tf)
        prob = pg.problem(udp)

        uda = snopt7(screen_output=False, library="C:/Users/ktomita3/libsnopt7/snopt7.dll", minor_version=7)
        uda.set_integer_option("Major Iteration Limit", 1000)
        uda.set_numeric_option("Major optimality tolerance", 1e-4)
        uda.set_numeric_option("Major feasibility tolerance", 1E-6)
        #uda = pg.ipopt()
        algo = pg.algorithm(uda)

        algo.set_verbosity(100)

        print(algo)

        pop = pg.population(prob, 0)
        pop.push_back(x_udp_0)
        print(pop)

        result = algo.evolve(pop)
        print(result)

        r_, v_, m_, u_ = udp.unpack_decision_vector(result.champion_x)
        r, v, m, U = udp.dimensionalize(r_, v_, m_, u_)

        X = np.hstack((r, v, m.reshape(-1, 1)))
        t = np.linspace(0, tf, N + 1)
        # Plot results
        plot_3sides(t[:-1], X, U, uskip=1, gsa=lander.gsa)

        plot_vel(t, X, lander.vmax)

        plot_mass(t, X, lander.mdry, lander.mwet)

        plot_thrust_mag(t[:-1], U, lander.rho2, lander.rho1)

        plot_pointing(t[:-1], U, lander.pa)

        print("Done!")
        return


if __name__ == "__main__":
        main()

