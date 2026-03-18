import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.spatial.transform import Rotation as spR
import gtsam

def plot_xyz_trajectory(states):
    """
    Plot the XYZ trajectory from a Simulator rollout,
    expressed in the frame of the initial orientation.
    """
    t = states[:,0]

    pos = states[:,1:4]

    # initial pose
    p0 = pos[0]
    R0 = spR.from_quat(states[0,4:8])

    # transform into initial frame
    pos_rel = pos - p0
    pos_rel = R0.inv().apply(pos_rel)

    x = pos_rel[:,0]
    y = pos_rel[:,1]
    z = pos_rel[:,2]

    # ---- 3D trajectory ----
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(x, y, z, linewidth=2)
    ax.scatter(x[0], y[0], z[0], label="start")
    ax.scatter(x[-1], y[-1], z[-1], label="end")

    ax.set_xlabel("X (initial frame)")
    ax.set_ylabel("Y (initial frame)")
    ax.set_zlabel("Z (initial frame)")
    ax.set_title("3D Trajectory (Initial Frame)")
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

    fig.suptitle("Position vs Time (Initial Frame)")
    plt.show()

def plot_camera_measured_trajectory(measurements):
    """
    Plot the trajectory measured by the camera (scaled position)
    expressed in the frame of the initial orientation.
    """
    cam_pos = measurements[:, 7:10]

    # initial pose
    p0 = cam_pos[0]
    initial_orientation = spR.from_quat(measurements[0, 10:14])

    # transform positions into initial frame
    cam_pos_rel = cam_pos - p0
    cam_pos_rel = initial_orientation.inv().apply(cam_pos_rel)

    x = cam_pos_rel[:, 0]
    y = cam_pos_rel[:, 1]
    z = cam_pos_rel[:, 2]

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(x, y, z, linewidth=2, label="camera trajectory")
    ax.scatter(x[0], y[0], z[0], label="start")
    ax.scatter(x[-1], y[-1], z[-1], label="end")

    ax.set_xlabel("X (initial frame)")
    ax.set_ylabel("Y (initial frame)")
    ax.set_zlabel("Z (initial frame)")
    ax.set_title("Camera Measured Trajectory (Initial Frame)")
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

def plot_comparison(cam_times, states, dt_imu, result, pose_keys, T_ic):
    """
    Compare simulator camera trajectory against optimized IMU trajectory.

    Assumptions:
    - `states` contains the camera pose in world coordinates:
        states[:,1:4]   -> camera position
        states[:,4:8]   -> camera quaternion [x,y,z,w]
    - `result.atPose3(k)` returns the optimized IMU pose in world coordinates.
    - `T_ic` is the fixed IMU->camera transform.

    To remove arbitrary global frame offset, both trajectories are expressed
    relative to their own first camera pose:
        T_rel_k = T_0^{-1} * T_k
    """

    # -----------------------------
    # Ground-truth camera poses
    # -----------------------------
    gt_indices = np.clip(np.round(cam_times / dt_imu).astype(int), 0, len(states) - 1)

    gt_poses = []
    for idx in gt_indices:
        q = states[idx, 4:8]   # [x, y, z, w]
        p = states[idx, 1:4]
        gt_pose = gtsam.Pose3(
            gtsam.Rot3.Quaternion(q[3], q[0], q[1], q[2]),
            gtsam.Point3(p[0], p[1], p[2])
        )
        gt_poses.append(gt_pose)

    gt_poses = np.array(gt_poses, dtype=object)

    # -----------------------------
    # Optimized IMU poses -> camera poses
    # -----------------------------
    T_ic_pose = gtsam.Pose3(T_ic)

    opt_cam_poses = []
    for k in pose_keys:
        imu_pose = result.atPose3(k)          # T_wi
        cam_pose = imu_pose.compose(T_ic_pose)  # T_wc = T_wi * T_ic
        opt_cam_poses.append(cam_pose)

    opt_cam_poses = np.array(opt_cam_poses, dtype=object)

    # -----------------------------
    # Express both trajectories relative to first pose
    # -----------------------------
    gt0_inv = gt_poses[0].inverse()
    opt0_inv = opt_cam_poses[0].inverse()

    gt_rel_xyz = np.array([gt0_inv.compose(p).translation() for p in gt_poses])
    opt_rel_xyz = np.array([opt0_inv.compose(p).translation() for p in opt_cam_poses])

    # -----------------------------
    # If needed, interpolate optimizer trajectory to cam_times
    # -----------------------------
    if len(opt_rel_xyz) != len(cam_times):
        from scipy.interpolate import interp1d
        interp_func = interp1d(
            np.linspace(0.0, 1.0, len(opt_rel_xyz)),
            opt_rel_xyz.T,
            kind="linear",
            fill_value="extrapolate",
        )
        opt_rel_xyz = interp_func(np.linspace(0.0, 1.0, len(cam_times))).T

    # -----------------------------
    # Plot x/y/z comparison
    # -----------------------------
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ["x", "y", "z"]

    for i in range(3):
        axs[i].plot(
            cam_times,
            gt_rel_xyz[:, i],
            label="Ground Truth Camera",
            linewidth=2.5,
        )
        axs[i].plot(
            cam_times,
            opt_rel_xyz[:, i],
            "--",
            label="Optimized IMU -> Camera",
            linewidth=2.5,
        )
        axs[i].set_ylabel(f"{labels[i]} (m)")
        axs[i].set_title(f"{labels[i]} Position")
        axs[i].grid(True)

    axs[0].legend()
    axs[-1].set_xlabel("Time (s)")
    fig.suptitle("Relative Camera Motion Comparison", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # -----------------------------
    # Optional 3D comparison
    # -----------------------------
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        gt_rel_xyz[:, 0], gt_rel_xyz[:, 1], gt_rel_xyz[:, 2],
        label="Ground Truth Camera", linewidth=2.5
    )
    ax.plot(
        opt_rel_xyz[:, 0], opt_rel_xyz[:, 1], opt_rel_xyz[:, 2],
        "--", label="Optimized IMU -> Camera", linewidth=2.5
    )

    ax.scatter(gt_rel_xyz[0, 0], gt_rel_xyz[0, 1], gt_rel_xyz[0, 2], s=60, label="start")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("Relative 3D Motion Comparison")
    ax.legend()
    ax.grid(True)
    set_axes_equal(ax)
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

def plot_comparison_real(cam_times, camera_poses_mat, result, pose_keys, scale_factor=1.0, T_ic=None):
    """
    3D plot comparing the original camera trajectory (scaled) and optimized trajectory.

    Parameters
    ----------
    cam_times : array_like, shape (N,)
        Camera timestamps.
    camera_poses_mat : array_like, shape (N, 4, 4)
        Original camera poses in camera frame.
    result : gtsam.Values
        Optimizer result containing Pose3 estimates.
    pose_keys : list
        Keys corresponding to optimized poses.
    scale_factor : float
        Scale factor to apply to camera trajectory (e.g., from optimizer).
    T_ic : np.ndarray or None
        Optional 4x4 transformation matrix from IMU to camera frame.
        If provided, optimized trajectory will be transformed to camera frame.
    """

    cam_times = np.asarray(cam_times)
    camera_poses_mat = np.asarray(camera_poses_mat)

    if cam_times.ndim != 1:
        raise ValueError("cam_times must be a 1D array")
    if camera_poses_mat.ndim != 3 or camera_poses_mat.shape[1:] != (4, 4):
        raise ValueError("camera_poses_mat must have shape (N, 4, 4)")
    if len(cam_times) != len(camera_poses_mat):
        raise ValueError("len(cam_times) must match len(camera_poses_mat)")
    if len(pose_keys) == 0:
        raise ValueError("pose_keys cannot be empty")

    # Build scaled camera poses while preserving measured orientation.
    gt_poses = []
    t0 = camera_poses_mat[0, :3, 3]
    for pose_mat in camera_poses_mat:
        pose_local = pose_mat.copy()
        t = pose_local[:3, 3]
        pose_local[:3, 3] = (t - t0) * scale_factor + t0
        gt_poses.append(gtsam.Pose3(pose_local))

    # Convert optimized IMU poses to camera poses (right-compose T_ic: T_wc = T_wi * T_ic).
    T_ic_pose = gtsam.Pose3(T_ic) if T_ic is not None else None
    opt_poses = []
    for k in pose_keys:
        imu_pose = result.atPose3(k)
        cam_pose = imu_pose.compose(T_ic_pose) if T_ic_pose is not None else imu_pose
        opt_poses.append(cam_pose)

    # Compare relative motion to remove global frame offsets.
    gt0_inv = gt_poses[0].inverse()
    opt0_inv = opt_poses[0].inverse()

    gt_rel_xyz = np.array([gt0_inv.compose(p).translation() for p in gt_poses])
    opt_rel_xyz = np.array([opt0_inv.compose(p).translation() for p in opt_poses])

    if len(opt_rel_xyz) != len(cam_times):
        from scipy.interpolate import interp1d

        interp_func = interp1d(
            np.linspace(0.0, 1.0, len(opt_rel_xyz)),
            opt_rel_xyz.T,
            kind="linear",
            fill_value="extrapolate",
        )
        opt_rel_xyz = interp_func(np.linspace(0.0, 1.0, len(cam_times))).T

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        gt_rel_xyz[:, 0],
        gt_rel_xyz[:, 1],
        gt_rel_xyz[:, 2],
        label="Camera Trajectory (scaled)",
        color="blue",
        linewidth=2,
    )
    ax.plot(
        opt_rel_xyz[:, 0],
        opt_rel_xyz[:, 1],
        opt_rel_xyz[:, 2],
        label="Optimized IMU -> Camera",
        color="red",
        linewidth=2,
        linestyle="--",
    )

    ax.scatter(gt_rel_xyz[0, 0], gt_rel_xyz[0, 1], gt_rel_xyz[0, 2], s=50, color="blue")
    ax.scatter(opt_rel_xyz[0, 0], opt_rel_xyz[0, 1], opt_rel_xyz[0, 2], s=50, color="red")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("3D Trajectory Comparison (Relative Motion)")
    ax.legend()
    ax.grid(True)
    ax.view_init(elev=30, azim=-60)
    set_axes_equal(ax)

    plt.show()

def plot_merged_trajectory_3d(merged_data, show_timestamps=False, color_by_batch=False):
    """
    Plot the merged trajectory in 3D space with optional coloring and time information.
    
    Parameters
    ----------
    merged_data : dict
        Dictionary with keys:
        - "pi3_t": (N, 4, 4) camera poses
        - "camera_poses": (N,) timestamps
    show_timestamps : bool, optional
        If True, add text annotations for some key timestamps. Default: False
    color_by_batch : bool, optional
        If True, color different trajectory segments by their original batch.
        If False, use gradient coloring by time. Default: False
    
    Returns
    -------
    fig : matplotlib figure
        The figure object
    """
    poses = merged_data["pi3_t"]
    timestamps = merged_data["camera_poses"]
    
    # Extract positions
    positions = poses[:, 0:3, 3]
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]
    
    # Create figure
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color options
    if color_by_batch:
        # Define batch boundaries (approximate times)
        batch_times = {
            "0_150": (91.673, 99.123),
            "75_225": (95.423, 102.873),
            "150_300": (99.173, 106.623),
            "225_375": (102.923, 110.373),
            "282_432": (105.773, 113.223),
        }
        
        colors = {
            "0_150": "blue",
            "75_225": "green",
            "150_300": "red",
            "225_375": "purple",
            "282_432": "orange",
        }
        
        # Plot each segment with its batch color
        for batch_name, (t_start, t_end) in batch_times.items():
            mask = (timestamps >= t_start) & (timestamps <= t_end)
            if np.any(mask):
                ax.plot(x[mask], y[mask], z[mask], linewidth=2.5, 
                       label=batch_name, color=colors[batch_name], alpha=0.8)
    else:
        # Use color gradient based on time
        scatter = ax.scatter(x, y, z, c=timestamps, cmap='viridis', s=20, alpha=0.6)
        ax.plot(x, y, z, linewidth=1.5, alpha=0.4, color='gray')
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
        cbar.set_label("Time (seconds)")
    
    # Mark start and end points
    ax.scatter([x[0]], [y[0]], [z[0]], color='green', s=200, marker='o', 
              edgecolors='darkgreen', linewidth=2, label="Start", zorder=5)
    ax.scatter([x[-1]], [y[-1]], [z[-1]], color='red', s=200, marker='s', 
              edgecolors='darkred', linewidth=2, label="End", zorder=5)
    
    # Add timestamp annotations if requested
    if show_timestamps:
        # Show a few key timestamps
        key_indices = np.linspace(0, len(timestamps)-1, 5, dtype=int)
        for idx in key_indices:
            ax.text(x[idx], y[idx], z[idx], f"{timestamps[idx]:.1f}s", 
                   fontsize=8, alpha=0.7)
    
    # Labels and formatting
    ax.set_xlabel("X Position (m)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Y Position (m)", fontsize=11, fontweight='bold')
    ax.set_zlabel("Z Position (m)", fontsize=11, fontweight='bold')
    ax.set_title("Merged Camera Trajectory (3D View)", fontsize=13, fontweight='bold')
    
    # Set equal aspect ratio
    set_axes_equal(ax)
    
    # Legend
    if color_by_batch:
        ax.legend(loc='upper left', fontsize=10)
    else:
        ax.legend(loc='upper left', fontsize=10)
    
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_merged_trajectory_components(merged_data):
    """
    Plot position components (x, y, z) vs time for the merged trajectory.
    
    Parameters
    ----------
    merged_data : dict
        Dictionary with keys:
        - "pi3_t": (N, 4, 4) camera poses
        - "camera_poses": (N,) timestamps
    
    Returns
    -------
    fig : matplotlib figure
        The figure object
    """
    poses = merged_data["pi3_t"]
    timestamps = merged_data["camera_poses"]
    
    # Extract positions
    positions = poses[:, 0:3, 3]
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]
    
    # Create subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    # X position
    axs[0].plot(timestamps, x, linewidth=2, color='red', label='X')
    axs[0].fill_between(timestamps, x, alpha=0.2, color='red')
    axs[0].set_ylabel("X Position (m)", fontsize=11, fontweight='bold')
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(loc='upper left')
    
    # Y position
    axs[1].plot(timestamps, y, linewidth=2, color='green', label='Y')
    axs[1].fill_between(timestamps, y, alpha=0.2, color='green')
    axs[1].set_ylabel("Y Position (m)", fontsize=11, fontweight='bold')
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(loc='upper left')
    
    # Z position
    axs[2].plot(timestamps, z, linewidth=2, color='blue', label='Z')
    axs[2].fill_between(timestamps, z, alpha=0.2, color='blue')
    axs[2].set_ylabel("Z Position (m)", fontsize=11, fontweight='bold')
    axs[2].set_xlabel("Time (seconds)", fontsize=11, fontweight='bold')
    axs[2].grid(True, alpha=0.3)
    axs[2].legend(loc='upper left')
    
    fig.suptitle("Merged Trajectory - Position Components vs Time", 
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    return fig
