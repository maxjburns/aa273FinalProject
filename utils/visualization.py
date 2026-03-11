import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_xyz_trajectory(states):
    """
    Plot the XYZ trajectory from a Simulator rollout.
    """
    t = states[:,0]
    x = states[:,1]
    y = states[:,2]
    z = states[:,3]

    # ---- 3D trajectory ----
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(x, y, z, linewidth=2)
    ax.scatter(x[0], y[0], z[0], label="start")
    ax.scatter(x[-1], y[-1], z[-1], label="end")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Trajectory")
    ax.legend()
    set_axes_equal(ax)

    plt.show()

    # ---- time plots ----
    fig, axs = plt.subplots(3,1, figsize=(10,8), sharex=True)

    axs[0].plot(t, x)
    axs[0].set_ylabel("x")

    axs[1].plot(t, y)
    axs[1].set_ylabel("y")

    axs[2].plot(t, z)
    axs[2].set_ylabel("z")
    axs[2].set_xlabel("time (s)")

    fig.suptitle("Position vs Time")
    plt.show()

def plot_camera_measured_trajectory(measurements):
    """
    Plot the trajectory measured by the camera (scaled position).
    """
    cam_pos = measurements[:, 7:10]
    
    x = cam_pos[:,0]
    y = cam_pos[:,1]
    z = cam_pos[:,2]

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(x, y, z, linewidth=2, label="camera trajectory")
    ax.scatter(x[0], y[0], z[0], label="start")
    ax.scatter(x[-1], y[-1], z[-1], label="end")

    ax.set_xlabel("X (scaled)")
    ax.set_ylabel("Y (scaled)")
    ax.set_zlabel("Z (scaled)")
    ax.set_title("Camera Measured Trajectory")
    ax.legend()

    set_axes_equal(ax)

    plt.show()

def plot_imu_readings(measurements):
    """
    Plot IMU measured accelerations and angular velocities (body frame).
    """
    t = measurements[:,0]
    w = measurements[:,1:4]
    acc = measurements[:,4:7]

    fig, axs = plt.subplots(2,1, figsize=(10,8), sharex=True)

    # --- Angular velocity subplot ---
    axs[0].plot(t, w[:,0], label='wx')
    axs[0].plot(t, w[:,1], label='wy')
    axs[0].plot(t, w[:,2], label='wz')
    axs[0].set_ylabel("Angular Velocity [rad/s]")
    axs[0].legend()
    axs[0].set_title("IMU Angular Velocity (Body Frame)")

    # --- Acceleration subplot ---
    axs[1].plot(t, acc[:,0], label='ax')
    axs[1].plot(t, acc[:,1], label='ay')
    axs[1].plot(t, acc[:,2], label='az')
    axs[1].set_ylabel("Acceleration [m/s²]")
    axs[1].set_xlabel("Time [s]")
    axs[1].legend()
    axs[1].set_title("IMU Acceleration (Body Frame)")

    plt.tight_layout()
    plt.show()

def plot_optimized_trajectory(t, result, pose_keys):
    """
    Plots the optimized camera trajectory in 3D and x/y/z vs time.

    :param t: time vector
    :param result: gtsam.Values from optimizer.optimize()
    :param pose_keys: list of pose keys used in the factor graph
    """
    # extract trajectory
    traj = np.array([result.atPose3(k).translation() for k in pose_keys])  # (N,3)
    x, y, z = traj[:,0], traj[:,1], traj[:,2]

    # 3D trajectory plot
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, '-o', color='blue', label='Optimized trajectory')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Optimized Camera Trajectory')
    ax.legend()
    ax.grid(True)
    set_axes_equal(ax)
    plt.show()

    # x/y/z vs time
    fig, axs = plt.subplots(3,1, figsize=(10,8), sharex=True)
    axs[0].plot(t, x, color='r')
    axs[0].set_ylabel("X")
    axs[0].grid(True)

    axs[1].plot(t, y, color='g')
    axs[1].set_ylabel("Y")
    axs[1].grid(True)

    axs[2].plot(t, z, color='b')
    axs[2].set_ylabel("Z")
    axs[2].set_xlabel("Time (s)")
    axs[2].grid(True)

    fig.suptitle("Position vs Time")
    plt.show()

def plot_optimized_velocity(t, result, vel_keys):
    """
    Plot optimized velocity vs time.

    :param t: time vector (length N)
    :param result: gtsam.Values from optimizer.optimize()
    :param vel_keys: list of velocity keys used in the factor graph
    """

    # extract velocity trajectory
    vel = np.array([result.atVector(k) for k in vel_keys])  # (N,3)

    vx = vel[:,0]
    vy = vel[:,1]
    vz = vel[:,2]

    fig, axs = plt.subplots(3, 1, figsize=(10,8), sharex=True)

    axs[0].plot(t, vx)
    axs[0].set_ylabel("v_x")
    axs[0].grid(True)

    axs[1].plot(t, vy)
    axs[1].set_ylabel("v_y")
    axs[1].grid(True)

    axs[2].plot(t, vz)
    axs[2].set_ylabel("v_z")
    axs[2].set_xlabel("time (s)")
    axs[2].grid(True)

    fig.suptitle("Optimized Velocity vs Time")
    plt.show()

def set_axes_equal(ax):
    """
    Make a 3D plot have equal scale so that spheres appear as spheres
    and cubes as cubes.
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


def plot_comparison(cam_times, states, dt_imu, result, pose_keys):
    """
    Compare ground truth trajectory from simulator with optimized trajectory.
    """

    # -----------------------------
    # Ground truth (sampled)
    # -----------------------------
    gt_indices = (cam_times / dt_imu).astype(int)
    gt_xyz = states[gt_indices, 1:4]

    # -----------------------------
    # Optimized trajectory
    # -----------------------------
    opt_xyz = np.array([
        result.atPose3(k).translation()
        for k in pose_keys
    ])

    # -----------------------------
    # Plot comparison
    # -----------------------------
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    labels = ['x', 'y', 'z']
    linewidth = 3

    for i in range(3):
        axs[i].plot(cam_times, gt_xyz[:, i],
                    label='Ground Truth',
                    linewidth=linewidth)

        axs[i].plot(cam_times, opt_xyz[:, i],
                    '--',
                    label='Optimized',
                    linewidth=linewidth)

        axs[i].set_title(f'{labels[i]} Position')
        axs[i].set_ylabel(f'{labels[i]} (m)')
        axs[i].grid(True)

    axs[-1].set_xlabel('Time (s)')
    axs[0].legend()

    fig.suptitle("Trajectory Comparison (Ground Truth vs Optimized)", fontsize=14)

    plt.tight_layout()
    plt.show()

def plot_bias_comparison(cam_times, states, dt_imu, result, bias_keys):
    """
    Plot true vs estimated accelerometer bias (x,y,z components) over time.
    """

    # -----------------------------
    # True bias from simulator
    # -----------------------------
    gt_indices = (cam_times / dt_imu).astype(int)
    true_bias = states[gt_indices, 17:20]   # accelerometer bias

    # -----------------------------
    # Estimated bias from optimizer
    # -----------------------------
    est_bias = np.array([
        result.atConstantBias(k).accelerometer()
        for k in bias_keys
    ])

    # -----------------------------
    # Plot
    # -----------------------------
    fig, axs = plt.subplots(3, 1, figsize=(10,8), sharex=True)

    labels = ['x', 'y', 'z']
    linewidth = 3

    for i in range(3):
        axs[i].plot(
            cam_times,
            true_bias[:, i],
            linewidth=linewidth,
            label="Ground Truth"
        )

        axs[i].plot(
            cam_times,
            est_bias[:, i],
            "--",
            linewidth=linewidth,
            label="Estimated"
        )

        axs[i].set_ylabel(f"Bias {labels[i]} (m/s²)")
        axs[i].set_title(f"Accelerometer Bias {labels[i]}")
        axs[i].grid(True)

    axs[-1].set_xlabel("Time (s)")
    axs[0].legend()

    fig.suptitle("Accelerometer Bias Comparison (Ground Truth vs Optimized)")
    plt.tight_layout()
    plt.show()

def plot_camera_vs_true(cam_times, cam_traj, states, dt_imu):
    """
    Plot unscaled camera trajectory vs true simulator trajectory.
    """

    # -----------------------------
    # True trajectory sampled at camera times
    # -----------------------------
    gt_indices = (cam_times / dt_imu).astype(int)
    gt_xyz = states[gt_indices, 1:4]

    # -----------------------------
    # Camera trajectory
    # -----------------------------
    cam_xyz = cam_traj[:,1:4]

    # -----------------------------
    # Plot
    # -----------------------------
    fig, axs = plt.subplots(3, 1, figsize=(10,8), sharex=True)

    labels = ['x', 'y', 'z']
    linewidth = 3

    for i in range(3):
        axs[i].plot(
            cam_times,
            gt_xyz[:, i],
            linewidth=linewidth,
            label="Ground Truth"
        )

        axs[i].plot(
            cam_times,
            cam_xyz[:, i],
            "g-",
            linewidth=linewidth,
            label="Camera (scaleless)"
        )

        axs[i].set_ylabel(f"{labels[i]} (m)")
        axs[i].set_title(f"{labels[i]} Position")
        axs[i].grid(True)

    axs[-1].set_xlabel("Time (s)")
    axs[0].legend()

    fig.suptitle("Camera Measurements vs True Trajectory")
    plt.tight_layout()
    plt.show()