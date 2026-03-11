import numpy as np
from scipy.spatial.transform import Rotation as R

def generate_sinusoidal_action_sequence(num_steps, dt, noise_std=20):
    """
    action[:, 0:3] = linear accel in world frame
    action[:, 3:6] = angular accel in world frame

    Returns:
    - actions: (N,6) (accelerations in world frame)
    - xyz_traj: (N,3) world positions
    - v_traj: (N,3) world velocities
    - w_traj: (N,3) world angular velocities
    - world_orientations: scipy Rotation object
    """

    t = np.linspace(0, (num_steps - 1) * dt, num_steps)

    # ----------------------------
    # World position trajectory
    # ----------------------------
    xyz_traj = np.zeros((num_steps, 3))
    # forward motion with noise
    v_forward = 1.0  # m/s nominal forward velocity

    for i in range(1, num_steps):
        v_forward_i = v_forward + np.sin(2 * np.pi * 0.1 * t[i]) * 0.25  # add some sinusoidal variation to forward velocity
        xyz_traj[i, 0] = xyz_traj[i-1, 0] + v_forward_i * dt

    xyz_traj[:, 1] = 1.0 + 0.2 * np.sin(2 * np.pi * 0.5 * t+3*np.pi/4)
    #xyz_traj[0:5, 1] = 1.0

    xyz_traj[:, 2] = 1.0 + 0.2 * np.sin(2 * np.pi * 0.8 * t+np.pi/2)
    #xyz_traj[0:5, 2] = 1.0

    linear_world_vel = np.gradient(xyz_traj, dt, axis=0)
    linear_world_accel = np.gradient(linear_world_vel, dt, axis=0)

    # ----------------------------
    # World orientation trajectory
    # ----------------------------
    th_traj = np.zeros((num_steps, 3))
    # Example rotation (degrees)
    th_traj[:, 1] = 30 * np.sin(2 * np.pi * 0.2 * t)

    world_orientations = R.from_euler('xyz', th_traj, degrees=True)

    # Compute angular velocity in world frame
    angular_world_vel = np.gradient(th_traj, dt, axis=0)
    angular_world_vel = np.deg2rad(angular_world_vel)  # convert deg/s → rad/s
    angular_world_accel = np.gradient(angular_world_vel, dt, axis=0)
    
    # ----------------------------
    # Pack actions
    # ----------------------------
    actions = np.zeros((num_steps, 6))
    actions[:, 0:3] = linear_world_accel
    actions[:, 3:6] = angular_world_accel

    print(linear_world_accel)

    return actions, xyz_traj, linear_world_vel, angular_world_vel, world_orientations