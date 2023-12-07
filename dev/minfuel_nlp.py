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

# visualize
m = np.exp(z)
X = np.hstack((r.T, v.T, m.reshape(-1, 1)))
U = u.T * m[:-1].reshape(-1, 1)

prob = pg.problem(MinFuel(lander, N, x0, tf, False))
prob.c_tol = [1e-6] * (prob.get_nec() + prob.get_nic())

uda = snopt7(screen_output=False, library="C:/Users/ktomita3/libsnopt7/snopt7.dll", minor_version=7)
uda.set_integer_option("Major Iteration Limit", 1)
#uda = pg.ipopt()
algo = pg.algorithm(uda)
#algo.extract(snopt7).set_integer_option("Major Iteration Limit", 1000)
#algo.extract(snopt7).set_numeric_option("Major feasibility tolerance", 1E-10)
#algo.set_verbosity(1)

print(algo)


pop = pg.population(prob, 0)
u_0 = U.flatten() / lander.rho2
#pop.set_x(0, u_0)
pop.push_back(u_0)

print("Im here")
result = algo.evolve(pop)
print(result)
print("Im here2")

