"""Compute the parameterization of the discrete system matrices of the relaxed landing problem.


We can confirm the following:

Ad = np.array([
        [1., 0., 0., dt, 0., 0., 0.],
        [0., 1., 0., 0., dt, 0., 0.],
        [0., 0., 1., 0., 0., dt, 0.],
        [0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 0., 1.]
    ])
Bd = np.array([
        [dt**2/2,   0.,         0.,         0.],
        [0.,        dt**2/2,    0.,         0.],
        [0.,        0.,         dt**2/2,    0.],
        [dt,        0.,         0.,         0.],
        [0.,        dt,         0.,         0.],
        [0.,        0.,         dt,         0.],
        [0.,        0.,         0.,         -alpha*dt]
    ])
pd = np.array([[0., 0., -dt**2*g/2, 0., 0., -dt*g, 0.]]).T

"""

from sympy import Matrix, exp, Symbol, BlockMatrix, ZeroMatrix
import numpy as np
import sys
from scipy.linalg import expm
sys.path.append('..')

from src.lcvx import Rocket

def main():
    # Simulation parameters
    t0 = 0.  # Initial time (s)
    tmax = 100.  # Maximum time (s)
    dt = 0.001  # Time step (s)
    dt_g = 1.    # Guidance update time step (s)
    g = np.array([0, 0, -3.7114])  # Gravitational acceleration (m/s^2)
    Tmax = 13260. # Maximum thrust (N)
    Tmin = 4972. # Minimum thrust (N)
    mdry = 1505. # Dry mass (kg)
    mwet = 1905. # Wet mass (kg)
    x0 = np.array([2000., 500., 1500., -100., 30., -75., mwet])  # Initial state vector (m, m, m, m/s, m/s, m/s, kg)
    xf = np.array([0., 0., 0., 0., 0., 0.])  # Final state vector (m, m, m, m/s, m/s, m/s)

    rocket = Rocket(
        g=g,
        mdry=mdry,
        mwet=mwet,
        Isp=225.,
        rho1=Tmin,
        rho2=Tmax,
        gsa=4 * np.pi / 180,
        pa=40 * np.pi / 180,
        vmax=800*1e3/3600
    )


    dt = Symbol('dt', real=True, nonnegative=True)
    alpha = Symbol('alpha', real=True, nonnegative=True)
    g = Symbol('g', real=True, nonnegative=True)

    A = Matrix([
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ])
    B = Matrix([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, -alpha]
        ])
    p = Matrix([[0], [0], [0], [0], [0], [-g], [0]])

    M0 = BlockMatrix([
        [A, B, p],
        [ZeroMatrix(5, 7), ZeroMatrix(5, 4), ZeroMatrix(5, 1)]
    ])
    M1 = M0 * dt
    EM = Matrix(M1).exp()
    print(f"EM = {EM}")


    em, Ad, Bd, pd = zoh(A=rocket.Ac, B=rocket.Bc, p=rocket.pc, dt=1)
    print(f"em = {em}")
    print(f"Ad = {Ad}")
    print(f"Bd = {Bd}")
    print(f"pd = {pd}")


def zoh(A: np.ndarray, B: np.ndarray, dt: float, p: np.ndarray):

    n = A.shape[0]  # number of states
    m = B.shape[1]  # number of inputs

    p = p.reshape((n, 1))

    # build exponential matrix
    em = np.block([
    [A, B, p],
    [np.zeros((m + 1, n + m + 1))]
    ])
    em = expm(em * dt)

    # extract discrete-time matrices
    Ad = em[:n, :n]
    Bd = em[:n, n:n + m]
    pd = em[:n, n + m]

    return em, Ad, Bd, pd


if __name__ == "__main__":
    main()

