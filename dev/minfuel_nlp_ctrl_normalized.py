import pickle
import cvxpy as cp
import pygmo as pg
from pygmo_plugins_nonfree import snopt7
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from src.reachsteering import MinFuelCtrl, MinFuelStateCtrl
from src.visualization import *
import src.lcvx as lc

def solve_lcvx(x0, tf, lander, N):

        m0 = x0[6]
        assert m0 > lander.mdry and m0 < lander.mwet
        x0_log_mass = np.copy(x0)
        x0_log_mass[6] = np.log(x0[6])

        lcvx = lc.LCvxMinFuel(
                lander=lander,
                N=N,
                parameterize_x0=False,
                parameterize_tf=False,
                fixed_target=False,
        )
        prob = lcvx.problem(x0=x0_log_mass, tf=tf)
        prob.solve(solver=cp.ECOS, verbose=True)

        sol = lc.get_vars(prob, ["X", "U"])
        X_sol = sol["X"]
        U_sol = sol["U"]
        r, v, z, u, _ = lcvx.recover_variables(X_sol, U_sol)
        m = np.exp(z)
        X = np.hstack((r.T, v.T, m.reshape(-1, 1)))
        U = u.T * m[:-1].reshape(-1, 1)
        return X, U


def main():
        with open('../saved/controllable_set/lander.pkl', 'rb') as f:
                lander = pickle.load(f)

        N = 60
        tf = 60.0
        alt = 1500.0
        mass = 1900.0
        x0 = np.array([0, 0, alt, -20.0, 0, -45.0, mass])

        X, U = solve_lcvx(x0, tf, lander, N)

        ctrl_only = True
        give_initial = True


        if ctrl_only:
            udp = MinFuelCtrl(lander, N, x0, tf, grad_implemented=False)
            x0_udp = udp.construct_x(U)
        else:
            udp = MinFuelStateCtrl(lander, N, x0, tf, grad_implemented=False)
            x0_udp = udp.construct_x(X, U)
        prob = pg.problem(udp)

        uda = snopt7(screen_output=False, library="C:/Users/ktomita3/libsnopt7/snopt7.dll", minor_version=7)
        ftol = 1e-4
        ctol = 1e-6
        uda.set_integer_option("Major Iteration Limit", 1000)
        uda.set_numeric_option("Major optimality tolerance", ftol)
        uda.set_numeric_option("Major feasibility tolerance", ctol)
        uda.set_numeric_option('Minor feasibility tolerance', ctol)
        #uda = pg.ipopt()
        algo = pg.algorithm(uda)

        algo.set_verbosity(100)
        
        print(algo)

        if give_initial:
            pop = pg.population(prob, 0)
            pop.push_back(x0_udp)
        else:
            pop = pg.population(prob, 1)
        print(pop)

        result = algo.evolve(pop)
        print(result)

        fitness = np.array(udp.fitness(result.champion_x))
        # print index of all constraints that are not satisfied with its value
        indices = np.where(fitness > 0)[0]
        print(indices)
        print(fitness[indices])
        
        r, v, m, U = udp.construct_trajectory(result.champion_x)

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

