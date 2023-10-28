"""Visualization tools for LCVX."""
import numpy as np
import matplotlib.pyplot as plt


def plot_slack_var(t: np.ndarray, U_sol: np.ndarray):
    u_norm = np.linalg.norm(U_sol[:3, :], axis=0)
    sigma = U_sol[3, :]

    # plot results
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot(t[:-1], u_norm, label="u_norm")
    ax.plot(t[:-1], sigma, label="sigma")
    ax.legend()
    ax.set_xlabel("t")
    plt.show()
