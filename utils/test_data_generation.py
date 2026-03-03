import numpy as np
from scipy.spatial.transform import Rotation as R

def generate_sinusoidal_action_sequence(num_steps, dt, g, b_g, b_a, noise_std=0.1):
    """
    action[:, 0:3] = linear accel in body frame
    action[:, 3:6] = angular velocity in body frame (with gyro bias)

    Returns:
    - actions: (N,6)
    - xyz_traj: (N,3) world positions
    - world_orientations: scipy Rotation object
    """

    t = np.linspace(0, (num_steps - 1) * dt, num_steps)

    # ----------------------------
    # World position trajectory
    # ----------------------------
    xyz_traj = np.zeros((num_steps, 3))
    xyz_traj[:, 0] = t 
    xyz_traj[0:5, 0] = 0 

    xyz_traj[:, 1] = 1.0 + 0.2 * np.sin(2 * np.pi * 0.5 * t+3*np.pi/4)
    xyz_traj[0:5, 1] = 1.0

    xyz_traj[:, 2] = 1.0 + 0.2 * np.sin(2 * np.pi * 0.5 * t+np.pi/2)
    xyz_traj[0:5, 2] = 1.0

    linear_world_vel = np.gradient(xyz_traj, dt, axis=0)
    linear_world_accel = np.gradient(linear_world_vel, dt, axis=0)

    # Add gravity in world frame
    linear_world_accel += g

    # ----------------------------
    # World orientation trajectory
    # ----------------------------
    th_traj = np.zeros((num_steps, 3))
    # Example rotation (degrees)
    #th_traj[:, 1] = 30 * np.sin(2 * np.pi * 0.2 * t)

    world_orientations = R.from_euler('xyz', th_traj, degrees=True)

    # Compute angular velocity in world frame
    angular_world_vel = np.gradient(th_traj, dt, axis=0)
    angular_world_vel = np.deg2rad(angular_world_vel)  # convert deg/s → rad/s

    # ----------------------------
    # Convert to body frame
    # ----------------------------
    linear_body_accel = np.zeros_like(linear_world_accel)
    angular_body_vel = np.zeros_like(angular_world_vel)

    for k in range(num_steps):
        R_wb = world_orientations[k]

        # world → body
        linear_body_accel[k] = R_wb.inv().apply(linear_world_accel[k])
        angular_body_vel[k] = R_wb.inv().apply(angular_world_vel[k])

    # ----------------------------
    # Apply biases (correctly)
    # ----------------------------
    linear_body_accel += b_a          # accelerometer bias
    angular_body_vel += b_g           # gyro bias

    # Optional noise
    # linear_body_accel += np.random.normal(0, noise_std, linear_body_accel.shape)
    # angular_body_vel += np.random.normal(0, noise_std, angular_body_vel.shape)

    # ----------------------------
    # Pack actions
    # ----------------------------
    actions = np.zeros((num_steps, 6))
    actions[:, 0:3] = linear_body_accel
    actions[:, 3:6] = angular_body_vel
    #print(angular_body_vel)
    return actions, xyz_traj, world_orientations