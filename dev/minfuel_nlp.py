import pickle
import cvxpy as cp
import pygmo as pg
from pygmo_plugins_nonfree import snopt7
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from src.reachsteering import MinFuel
from src.visualization import *
import src.lcvx as lc


def main():
        with open('../saved/controllable_set/lander.pkl', 'rb') as f:
                lander = pickle.load(f)

        N = 60
        tf = 60.0
        alt = 1500.0
        mass = 1800.0
        x0 = np.array([0, 0, alt, -30.0, 0, -55.0, np.log(mass)])

        lcvx = lc.LCvxMinFuel(
                rocket=lander,
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
        X = np.hstack((r.T, v.T, m.reshape(-1, 1)))
        U = u.T * m[:-1].reshape(-1, 1)

        udp = MinFuel(lander, N, x0, tf)
        prob = pg.problem(udp)

        uda = snopt7(screen_output=False, library="C:/Users/ktomita3/libsnopt7/snopt7.dll", minor_version=7)
        uda.set_integer_option("Major Iteration Limit", 1000)
        #uda = pg.ipopt()
        algo = pg.algorithm(uda)

        #algo.extract(snopt7).set_integer_option("Major Iteration Limit", 1000)
        #algo.extract(snopt7).set_numeric_option("Major feasibility tolerance", 1E-10)
        algo.set_verbosity(1)

        print(algo)

        pop = pg.population(prob, 0)
        u_0 = U.flatten() / lander.rho2
        pop.push_back(u_0)

        result = algo.evolve(pop)
        print(result)

        U = result.champion_x.reshape(-1, 3) * lander.rho2
        r, v, z = np.zeros((N + 1, 3)), np.zeros((N + 1, 3)), np.zeros(N + 1)
        r[0] = x0[:3]
        v[0] = x0[3:6]
        z[0] = x0[6]
        from src.reachsteering.problems import _dynamics
        for i in range(N):
                r[i + 1], v[i + 1], z[i + 1] = _dynamics(r[i], v[i], z[i], U[i], tf/N, lander.g, lander.alpha)

        X = np.hstack((r, v, np.exp(z).reshape(-1, 1)))
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

