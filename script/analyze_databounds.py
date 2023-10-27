import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from numba import njit

sys.path.append('../')
from src.reachset import vx_bound, vz_bound, mass_bound, tgo_bound

def visualize_convex_hull_2d(xs, ys, xlabel, ylabel):
    """
    Visualize the 2D convex hull of the given vertices.

    Args:
        vertices (numpy.ndarray): An array of 2D points representing the vertices.
    """
    vertices = np.vstack([xs, ys]).T
    hull = ConvexHull(vertices)

    plt.figure()
    for simplex in hull.simplices:
        plt.plot(vertices[simplex, 0], vertices[simplex, 1], 'k-')

    plt.plot(vertices[:,0], vertices[:,1], 'o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    # Limit axes for better visualization
    plt.xlim([np.min(vertices[:,0])-1, np.max(vertices[:,0])+1])
    plt.ylim([np.min(vertices[:,1])-1, np.max(vertices[:,1])+1])

    plt.grid()
    plt.show()


if __name__=="__main__":

    vx_bounds = np.load('../out/vx_data.npy')
    vz_bounds = np.load('../out/vz_data.npy')
    mass_bounds = np.load('../out/mass_data.npy')

    print(vx_bounds.shape)
    print(vz_bounds.shape)
    print(mass_bounds.shape)

    alt_min = 0.
    alt_max = 1500.
    alts = np.linspace(alt_min, alt_max, 1000)

    vx_bounds = vx_bounds[(vx_bounds[:, 1] > alt_min) & (vx_bounds[:, 1] < alt_max)]
    vz_bounds = vz_bounds[(vz_bounds[:, 1] > alt_min) & (vz_bounds[:, 1] < alt_max)]
    mass_bounds = mass_bounds[(mass_bounds[:, 1] > alt_min) & (mass_bounds[:, 1] < alt_max)]
    tgo_bounds = np.vstack([vz_bounds[:, 1:], mass_bounds[:, 1:], vx_bounds[:, 1:]])

    """
    visualize_convex_hull_2d(vx_bounds[:, 1], vx_bounds[:, 0], 'Altitude (m)', 'Vx (m/s)')
    visualize_convex_hull_2d(vz_bounds[:, 1], vz_bounds[:, 0], 'Altitude (m)', 'Vz (m/s)')
    visualize_convex_hull_2d(mass_bounds[:, 1], mass_bounds[:, 0], 'Altitude (m)', 'Mass (kg)')
    visualize_convex_hull_2d(tgo_bounds[:, 0], tgo_bounds[:, 1], 'Altitude (m)', 'Tgo (s)')
    """




    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    axs[0, 0].scatter(tgo_bounds[:, 0], tgo_bounds[:, 1], s=1)
    ymin, ymax = tgo_bound(alts)
    axs[0, 0].plot(alts, ymin, 'r--')
    axs[0, 0].plot(alts, ymax, 'r--')
    axs[0, 0].set_xlabel('Altitude (m)')
    axs[0, 0].set_ylabel('Tgo (s)')

    axs[0, 1].scatter(vx_bounds[:, 1], vx_bounds[:, 0], s=1)
    ymin, ymax = vx_bound(alts)
    axs[0, 1].plot(alts, ymin, 'r--')
    axs[0, 1].plot(alts, ymax, 'r--')
    axs[0, 1].set_xlabel('Altitude (m)')
    axs[0, 1].set_ylabel('Vx (m/s)')

    axs[1, 0].scatter(vz_bounds[:, 1], vz_bounds[:, 0], s=1)    
    ymin, ymax = vz_bound(alts)
    axs[1, 0].plot(alts, ymin, 'r--')
    axs[1, 0].plot(alts, ymax, 'r--')
    axs[1, 0].set_xlabel('Altitude (m)')
    axs[1, 0].set_ylabel('Vz (m/s)')

    axs[1, 1].scatter(mass_bounds[:, 1], mass_bounds[:, 0], s=1)
    ymin, ymax = mass_bound(alts)
    axs[1, 1].plot(alts, ymin, 'r--')
    axs[1, 1].plot(alts, ymax, 'r--')
    axs[1, 1].set_xlabel('Altitude (m)')
    axs[1, 1].set_ylabel('Mass (kg)')
    plt.tight_layout()
    plt.show()

