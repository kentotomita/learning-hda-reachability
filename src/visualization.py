"""Visualization functions"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"


def plot_3sides(t: np.ndarray, X: np.ndarray, U: np.ndarray, gsa: float=None, uskip: int = 1):
    """Plot trajectory from three sides

    Args:
        t: Time vector (s)
        X: State vector history, (m, m, m, m/s, m/s, m/s, kg)
        U: Control vector history (N, N, N)
        gsa: Glide slope angle constraint (rad)
        uskip: Control vector skip
    """
    # Unpack state vector
    x, y, z, vx, vy, vz, m = X.T

    # Unpack control vector
    ux, uy, uz = U.T

    # Plot state vector
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs[0, 0].plot(x, y, c='k')
    axs[0, 0].grid(True)
    axs[0, 0].set_ylabel('Position y (m)')
    axs[0, 0].set_xlabel('Position x (m)')
    axs[1, 0].plot(x, z, c='k')
    axs[1, 0].grid(True)
    axs[1, 0].set_xlabel('Position x (m)')
    axs[1, 0].set_ylabel('Position z (m)')
    axs[1, 1].plot(y, z, c='k')
    axs[1, 1].grid(True)
    axs[1, 1].set_xlabel('Position y (m)')
    axs[1, 1].set_ylabel('Position z (m)')

    # Plot control vector on the trajectory plot
    aw = 1e-1  # arrow width
    ahw = 1e-1  # arrow head width
    ahl = 1e-1  # arrow head length
    aa = 0.8   # arrow alpha
    scale = 1e-2
    for i in range(len(t)):
        if i % uskip == 0:
            axs[0, 0].arrow(x[i], y[i], ux[i] * scale, uy[i] * scale, color='r', width=aw, alpha=aa, head_width=ahw, head_length=ahl)
            axs[1, 0].arrow(x[i], z[i], ux[i] * scale, uz[i] * scale, color='r', width=aw, alpha=aa, head_width=ahw, head_length=ahl)
            axs[1, 1].arrow(y[i], z[i], uy[i] * scale, uz[i] * scale, color='r', width=aw, alpha=aa, head_width=ahw, head_length=ahl)
    
    # Plot glide slope angle constraint
    if gsa is not None:
        xmin, xmax = axs[0, 0].get_xlim()
        ymin, ymax = axs[0, 0].get_ylim()
        zmin, zmax = axs[1, 1].get_ylim()
        x_range = np.linspace(xmin, xmax, 100)
        y_range = np.linspace(ymin, ymax, 100)
        z_range = np.linspace(zmin, zmax, 100)
        axs[1, 0].plot(x_range, np.tan(gsa) * np.abs(x_range - x[-1]), 'k--', linewidth=0.5, alpha=0.2)
        axs[1, 1].plot(y_range, np.tan(gsa) * np.abs(y_range - y[-1]), 'k--', linewidth=0.5, alpha=0.2)
        axs[1, 0].fill_between(x_range, np.tan(gsa) * np.abs(x_range - x[-1]), color='gray', alpha=0.5)
        axs[1, 1].fill_between(y_range, np.tan(gsa) * np.abs(y_range - y[-1]), color='gray', alpha=0.5)
        
    plt.show()


def plot_vel(t: np.ndarray, X: np.ndarray, vmax: float=None):
    # Unpack state vector
    x, y, z, vx, vy, vz, m = X.T

    v_norm = np.sqrt(vx**2 + vy**2 + vz**2)

    # Plot velocity
    fig, axs = plt.subplots(1, 1, figsize=(12, 8))
    axs.plot(t, vx, label='vx')
    axs.plot(t, vy, label='vy')
    axs.plot(t, vz, label='vz')
    axs.plot(t, v_norm, label='v_norm')
    if vmax is not None:
        axs.plot(t, np.ones_like(t) * vmax, 'k--', label='vmax')
        y = np.ones_like(t) * vmax
        y2 = vmax * 1.1
        axs.fill_between(t, y, y2, where=(y<y2), color='gray', alpha=0.5)
    axs.grid(True)
    axs.set_xlabel('Time (s)')
    axs.set_ylabel('Velocity (m/s)')
    axs.legend()
    plt.show()


def plot_mass(t: np.ndarray, X: np.ndarray, mdry: float, mwet: float):
    """Plot mass history
    Args:
        t: Time vector (s)
        X: State vector history, (m, m, m, m/s, m/s, m/s, kg)
        mdry: Dry mass (kg)
        mwet: Wet mass (kg)
    """
    ymin = mdry - 0.2 * (mwet - mdry)
    ymax = mwet + 0.2 * (mwet - mdry)
    y = np.linspace(ymin, ymax, 500)
    _, y = np.meshgrid(t, y)

    m = X[:, 6]
    fig, ax = plt.subplots(1, 1)
    ax.plot(t, m, c='k')
    ax.plot(t, np.ones_like(t) * mdry, 'k--')
    ax.plot(t, np.ones_like(t) * mwet, 'k--')
    ax.imshow(y <= mwet, extent=[0, t[-1], ymin, ymax], origin='lower', cmap='gray', aspect='auto', alpha=0.2)
    ax.imshow(y >= mdry, extent=[0, t[-1], ymin, ymax], origin='lower', cmap='gray', aspect='auto', alpha=0.2)
    ax.grid(True)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Mass (kg)')
    plt.show()


def plot_thrust_mag(t: np.ndarray, U: np.ndarray, Tmax: float, Tmin: float):
    """Plot thrust magnitude history
    Args:
        t: Time vector (s)
        U: Control vector history (N, N, N)
        Tmax: Maximum thrust (N)
        Tmin: Minimum thrust (N)
    """
    Tmax *= 1e-3
    Tmin *= 1e-3

    ymin = Tmin - 0.2 * (Tmax - Tmin)
    ymax = Tmax + 0.2 * (Tmax - Tmin)
    y = np.linspace(ymin, ymax, 500)
    _, y = np.meshgrid(t, y)

    umag = np.linalg.norm(U, axis=1) * 1e-3
    print(umag.shape)
    fig, ax = plt.subplots(1, 1)
    ax.plot(t, umag, c='k')
    ax.grid(True)
    ax.plot(t, np.ones_like(t) * Tmax, 'k--', label='Tmax', linewidth=0.5, alpha=1.0)
    ax.plot(t, np.ones_like(t) * Tmin, 'k--', label='Tmin', linewidth=0.5, alpha=1.0)
    ax.imshow(y <= Tmax, extent=[0, t[-1], ymin, ymax], origin='lower', cmap='gray', aspect='auto', alpha=0.2)
    ax.imshow(y >= Tmin, extent=[0, t[-1], ymin, ymax], origin='lower', cmap='gray', aspect='auto', alpha=0.2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Thrust (kN)')
    plt.show()


def plot_pointing(t: np.ndarray, U: np.ndarray, pa: float):
    """Plot pointing angle history
    
    Args:
        t: Time vector (s)
        U: Control vector history (N, N, N)
        pa: Pointing angle constraint(rad)
    """

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Plot pointing angle
    pa_max_deg = np.rad2deg(pa)
    pa_deg = np.rad2deg(np.arctan2(np.sqrt(U[:, 0]**2 + U[:, 1]**2), U[:, 2]))
    ax.plot(t, pa_deg, c='k', label='Pointing angle')
    ax.plot(t, np.ones_like(t) * pa_max_deg, 'k--', alpha=0.2)

    y = np.ones_like(t) * pa_max_deg
    y2 = pa_max_deg * 1.1
    ax.fill_between(t, y, y2, where=(y<y2), color='gray', alpha=0.5)

    ax.grid(True)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Pointing angle (deg)')
    ax.legend()
    plt.show()


def plot_slack(t: np.ndarray, u: np.ndarray, sigma: np.ndarray):
    """Plot slack variable history
    
    Args:
        t: Time vector (s)
        u: Control variable history (3, N)
        sigma: Slack variable history (N,)
    """
    assert u.shape[0] == 3
    assert u.shape[1] == sigma.shape[0]
    if len(t) > u.shape[1]:
        t = t[:u.shape[1]]

    u_norm = np.linalg.norm(u, axis=0)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot(t, u_norm, label='u_norm')
    ax.plot(t, sigma, label='sigma')
    ax.legend()
    ax.set_xlabel('t')
    plt.show()
