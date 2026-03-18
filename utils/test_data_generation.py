import numpy as np
from scipy.spatial.transform import Rotation as R

def generate_sinusoidal_action_sequence(num_steps, dt, seed=None):

    rng = np.random.default_rng(seed)

    t = np.linspace(0, (num_steps - 1) * dt, num_steps)

    fwd_freq = rng.uniform(0.03, 0.15)    
    y_freq   = rng.uniform(0.2, 0.8)   
    z_freq   = rng.uniform(0.3, 1.2) 

    roll_freq  = rng.uniform(0.1, 0.3) 
    pitch_freq = rng.uniform(0.05, 0.21)
    yaw_freq   = rng.uniform(0.09, 0.26) 

    phase_y = 3 * np.pi / 4 + rng.uniform(-0.5, 0.5)
    phase_z = np.pi / 2     + rng.uniform(-0.5, 0.5)
    phase_pitch = 0.4       + rng.uniform(-0.3, 0.3)
    phase_yaw   = 0.8       + rng.uniform(-0.3, 0.3)

    # --- Position trajectory ---
    xyz_traj = np.zeros((num_steps, 3))
    v_forward = 1.0

    for i in range(1, num_steps):
        v_forward_i = v_forward + 0.25 * np.sin(2 * np.pi * fwd_freq * t[i])
        xyz_traj[i, 0] = xyz_traj[i - 1, 0] + v_forward_i * dt

    xyz_traj[:, 1] = 1.0 + 0.2 * np.sin(2 * np.pi * y_freq * t + phase_y)
    xyz_traj[:, 2] = 1.0 + 0.2 * np.sin(2 * np.pi * z_freq * t + phase_z)

    linear_world_vel = np.gradient(xyz_traj, dt, axis=0)
    linear_world_accel = np.gradient(linear_world_vel, dt, axis=0)

    # --- Orientation trajectory (degrees) ---
    th_traj = np.zeros((num_steps, 3))
    th_traj[:, 0] = 30.0 * np.sin(2 * np.pi * roll_freq * t)                 # roll
    th_traj[:, 1] = 20.0 * np.sin(2 * np.pi * pitch_freq * t + phase_pitch)  # pitch
    th_traj[:, 2] = 25.0 * np.sin(2 * np.pi * yaw_freq * t + phase_yaw)      # yaw

    world_orientations = R.from_euler("xyz", th_traj, degrees=True)

    # --- Angular velocity ---
    angular_body_vel = np.zeros((num_steps, 3))
    for i in range(num_steps - 1):
        dR_body = world_orientations[i].inv() * world_orientations[i + 1]
        angular_body_vel[i] = dR_body.as_rotvec() / dt
    angular_body_vel[-1] = angular_body_vel[-2]

    angular_body_accel = np.gradient(angular_body_vel, dt, axis=0)

    # --- Actions ---
    actions = np.zeros((num_steps, 6))
    actions[:, 0:3] = linear_world_accel
    actions[:, 3:6] = angular_body_accel

    return actions, xyz_traj, linear_world_vel, angular_body_vel, world_orientations