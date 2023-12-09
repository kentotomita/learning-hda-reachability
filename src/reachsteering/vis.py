import matplotlib.pyplot as plt
import numpy as np


def visualize_nn_reachset(reach_mask: np.ndarray, sfmap: np.ndarray):
    """Visualize soft landing reachable set overlaid on safety map.
    
    Args:
        reach_mask (np.ndarray): reachability mask; scalar value indicates the reachability; 0 is not reachable, 1 is reachable.
        sfmap (np.ndarray): safety map; shape (N, 3); each row is [x, y, safety].
    """

    n = int(np.sqrt(sfmap.shape[0]))

    x = sfmap[:, 0].reshape(n, n)
    y = sfmap[:, 1].reshape(n, n)
    safety = sfmap[:, 2].reshape(n, n)
    reach_mask = reach_mask.reshape(n, n)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Safety Map
    safety_img = ax.pcolormesh(x, y, safety, shading='auto', cmap='gray')
    plt.colorbar(safety_img, ax=ax, orientation='vertical', label='Safety')

    # Reachability Set - as the overlay with transparency
    reachability_img = ax.pcolormesh(x, y, reach_mask, shading='auto', cmap='jet', alpha=0.5)
    plt.colorbar(reachability_img, ax=ax, orientation='vertical', label='Reachability')

    ax.set_title('Safety Map with Reachability Overlay')
    ax.set_xlabel('x, m')
    ax.set_ylabel('y, m')
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()