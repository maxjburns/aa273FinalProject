import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def plot_trajectories(states, measurements):
    """
    Plot the true trajectory and the measurements in 3D.
    
    Parameters:
    - states: numpy array of shape (N, state_dim) containing the true states
    - measurements: numpy array of shape (N, meas_dim) containing the measurements
    """
    fig = plt.figure(figsize=(14, 6))
    
    # Plot true trajectory in 3D (x, y, z)
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot(states[:, 0], states[:, 1], states[:, 2], label='True Trajectory', marker='o')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_zlabel('Z Position')
    ax1.set_title('True Trajectory (3D)')
    ax1.legend()
    ax1.grid()
    
    # Plot measurements in 3D (x, y, z)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot(measurements[:, 0], measurements[:, 1], measurements[:, 2], label='Measurements', marker='x')
    ax2.set_xlabel('Measured X Position')
    ax2.set_ylabel('Measured Y Position')
    ax2.set_zlabel('Measured Z Position')
    ax2.set_title('Measurements (3D)')
    ax2.legend()
    ax2.grid()
    set_axes_equal(ax1)
    set_axes_equal(ax2)
    
    plt.tight_layout()
    plt.show()

def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])