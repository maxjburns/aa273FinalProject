
from utils.simulator import Simulator, transition_function, camera_measurement_function
import numpy as np
from utils.test_data_generation import generate_sinusoidal_action_sequence
from utils.visualization import plot_trajectories
from scipy.spatial.transform import Rotation as rot_obj

# -----------------------
# Main script
# -----------------------
if __name__ == "__main__":

    # TODO: decide on the state, action, measurement system
    rot_mat = rot_obj.from_euler('xyz', [90, 0, 0], degrees=True)
    #p_0 = np.array([0.0, 0.0, 0.0])  # Initial position (x, y)
    v_0 = np.array([0.0, 0.0, 0.0])  # Initial position (x, y)
    #th_0 = np.array([1.0, 0.0, 0.0, 0.0]) # Initial orientation (theta)
    b_g = np.array([0.1, 1.1, -0.4]) # gyroscope bias
    b_a = np.array([-0.4, 1.0, 0.3]) # accelerometer bias
    g = rot_mat.apply(np.array([0, 0, -9.81])) # gravity vector
    true_scale = np.array([0.1]) # scale factor for vision measurements
    
    state_dim = 23
    action_dim = 6
    meas_dim = 7 # position and quat orientation
    Q = np.zeros((state_dim-1, state_dim-1))  # Process noise covariance
    #Q[0, 0] = 0.001
    #Q[1, 0] = 0.001
    #Q[2, 0] = 0.001 

    Q[6, 0] = 0.001
    Q[7, 0] = 0.001
    Q[8, 0] = 0.001 

    R = 0.01*np.eye(meas_dim - 1)  # Measurement noise covariance
    dt = 0.05
    
    T = 30
    n_steps = int(T /dt)
    actions, xyz_traj, world_orientations = generate_sinusoidal_action_sequence(n_steps, dt, g, b_g, b_a)
    
    # ----------------------------
    # Initial state
    # ----------------------------
    p_0 = xyz_traj[0].copy()
    th_0 = world_orientations[0].as_quat().copy()  # (x,y,z,w)

    s_0 = np.concatenate([p_0, th_0, v_0, b_g, b_a, g, true_scale])  # Example initial state (x, y, vx, vy, scale)
    
    sim = Simulator(s_0, 
                    Q=Q, R=R, 
                    transition_function=transition_function, 
                    measurement_function=camera_measurement_function, 
                    action_dim=action_dim, meas_dim=meas_dim,
                    dt=dt)

    #[print([round(float(actions[i][j]), 2) for j in range(actions.shape[1])]) for i in range(actions.shape[0])]
    #print("actions:", actions[:, 0])
    sim.step_k_times(actions)
    
    states = sim.get_all_states()
    measurements = sim.get_all_measurements()
    actions = sim.get_all_actions()
    #[print([round(float(states[i][j]), 2) for j in range(states.shape[1])]) for i in range(states.shape[0])]

    plot_trajectories(states, measurements)

    





