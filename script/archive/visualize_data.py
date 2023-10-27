import os
import sys
sys.path.append('../')

import src.lcvx as lc
from src.visualization import *


if __name__=="__main__":

    N = 100

    n = 3
    m_range = np.linspace(1805, 1905, n)
    rz_range = np.linspace(1000, 2000, n)
    vz_range = np.linspace(-100, -30, n)
    vx_range = np.linspace(0, 100, n)
    tf_range = np.linspace(50, 80, n)
    c_list = [np.array([np.cos(theta), np.sin(theta), 0.]) for theta in np.linspace(0, np.pi, n)]


    for tfi, tf in enumerate(tf_range):
        for ci, c in enumerate(c_list):
            k = 0
            for vz in vz_range:
                for vx in vx_range:
                    
                    fname = os.path.join('../out', f'tf_{tfi}_c_{ci}_{k}.json')

                    if os.path.exists(fname):
                        status, tf, t, r, v, m, U, params = lc.load_results(fname)
                    
                        if status == 'optimal':
                            X = np.hstack([r, v, m.reshape(-1, 1)])
                            
                            print(f'Problem parameters  tf: {tf: .1f}, c: {c}, vz: {vz: .1f}, vx: {vx: .1f}')
                            plot_thrust_mag(t[:-1], U, Tmax=params['Tmax'], Tmin=params['Tmin'])
                            plot_3sides(t[:-1], X, U, uskip=1, gsa=params['gsa'])

                            #plot_vel(t, X, params['vmax'])

                            #plot_mass(t, X, params['mdry'], params['mwet'])

                            plot_pointing(t[:-1], U, params['pa'])

                    
                    k += 1 

    