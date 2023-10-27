import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib

sys.path.append('../')
from src.reachset import read_reachset
from src.reachset import vx_bound, vz_bound, mass_bound, tgo_bound

fov = 15 * np.pi / 180


matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Times New Roman'
matplotlib.rcParams['mathtext.it'] = 'Times New Roman:italic'
matplotlib.rcParams['mathtext.bf'] = 'Times New Roman:bold'
matplotlib.rcParams['font.size'] = 16


if __name__=="__main__":
    _, _, reachset = read_reachset('../out/20230803_021318/reachset_sample_10000.json')

    print(reachset.shape)

    feas_list = []
    alt_list = []
    vx_list = []
    vz_list = []
    mass_list = []
    tgo_list = []

    for data in reachset:
        # x0 = x, y, z, vx, vy, vz, m
        alt_list.append(data['x0'][2])
        vx_list.append(data['x0'][3])
        vz_list.append(data['x0'][5])
        mass_list.append(np.exp(data['x0'][6]))
        tgo_list.append(data['tgo'][0])

        rf = np.array(data['rf'])
        if np.all(np.isnan(rf)):
            feas_list.append(0)
        else:
            feas_list.append(1)

    # make numpy array
    alt_list = np.array(alt_list)
    vx_list = np.array(vx_list)
    vz_list = np.array(vz_list)
    mass_list = np.array(mass_list)
    tgo_list = np.array(tgo_list)
    feas_list = np.array(feas_list)

    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    marker_size = 12
    feasible_marker = 'o'
    infeasible_marker = 'x'
    feasible_color = 'b'
    infeasible_color = 'r'
    alpha = 0.2
    linewidth = 0.5

    axs[0, 0].scatter(alt_list[feas_list==0], tgo_list[feas_list==0], s=marker_size, label='Infeasible', color=infeasible_color, marker=infeasible_marker, alpha=alpha, linewidth=linewidth)
    axs[0, 0].scatter(alt_list[feas_list==1], tgo_list[feas_list==1], s=marker_size, label='Feasible', color=feasible_color, marker=feasible_marker, alpha=alpha, linewidth=linewidth)
    axs[0, 0].set_xlabel('Altitude (m)')
    axs[0, 0].set_ylabel('Tgo (s)')

    axs[0, 1].scatter(alt_list[feas_list==0], vx_list[feas_list==0], s=marker_size, label='Infeasible', color=infeasible_color, marker=infeasible_marker, alpha=alpha, linewidth=linewidth)
    axs[0, 1].scatter(alt_list[feas_list==1], vx_list[feas_list==1], s=marker_size, label='Feasible', color=feasible_color, marker=feasible_marker, alpha=alpha, linewidth=linewidth)
    axs[0, 1].set_xlabel('Altitude (m)')
    axs[0, 1].set_ylabel('Vx (m/s)')

    axs[1, 0].scatter(alt_list[feas_list==0], vz_list[feas_list==0], s=marker_size, label='Infeasible', color=infeasible_color, marker=infeasible_marker, alpha=alpha, linewidth=linewidth)
    axs[1, 0].scatter(alt_list[feas_list==1], vz_list[feas_list==1], s=marker_size, label='Feasible', color=feasible_color, marker=feasible_marker, alpha=alpha, linewidth=linewidth)
    axs[1, 0].set_xlabel('Altitude (m)')
    axs[1, 0].set_ylabel('Vz (m/s)')

    axs[1, 1].scatter(alt_list[feas_list==0], mass_list[feas_list==0], s=marker_size, label='Infeasible', color=infeasible_color, marker=infeasible_marker, alpha=alpha, linewidth=linewidth)
    axs[1, 1].scatter(alt_list[feas_list==1], mass_list[feas_list==1], s=marker_size, label='Feasible', color=feasible_color, marker=feasible_marker, alpha=alpha, linewidth=linewidth)
    axs[1, 1].set_xlabel('Altitude (m)')
    axs[1, 1].set_ylabel('Mass (kg)')

    # grid for all subplots
    for ax in axs.flat:
        ax.grid(True)
    leg = plt.legend()
    for handle in leg.legendHandles:
        handle.set_alpha(1)
    plt.tight_layout()

    plt.savefig('../out/reachset_data.png', dpi=300)
    plt.savefig('../out/reachset_data.pdf')

    """
    # 3d plot of alt, vx, vz
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(alt_list[feas_list==0], vx_list[feas_list==0], vz_list[feas_list==0], s=marker_size, label='Infeasible', color=infeasible_color, marker=infeasible_marker, alpha=alpha, linewidth=linewidth)
    ax.scatter(alt_list[feas_list==1], vx_list[feas_list==1], vz_list[feas_list==1], s=marker_size, label='Feasible', color=feasible_color, marker=feasible_marker, alpha=alpha, linewidth=linewidth)
    ax.set_xlabel('Altitude (m)')
    ax.set_ylabel('Vx (m/s)')
    ax.set_zlabel('Vz (m/s)')
    ax.grid(True)
    leg = plt.legend()
    for handle in leg.legendHandles:
        handle.set_alpha(1)
    plt.tight_layout()
    plt.show()
    """

    


    # print('Number of reachsets: ', len(reachset))
    # print number of reachset and rate of infeasible
    num_invalid = 0
    for data in reachset:
        rf = np.array(data['rf'])
        if np.all(np.isnan(rf)):
            num_invalid += 1

    print('Number of reachsets: ', len(reachset))
    print('Number of invalid reachsets: ', num_invalid)