import numpy as np
from scipy.spatial.transform import Rotation as rot_obj

class Simulator:
    def __init__(self, s_0:np.ndarray, Q:np.ndarray, R:np.ndarray, transition_function, measurement_function,
                action_dim:int, meas_dim:int, dt=0.01):
        """
        s_0 --> initial state
        Q --> process noise covariance
        R --> measurement noise covariance
        dt --> time step
        transition_function --> function of form f(s, u, Q, dt, with_noise) that adds the next state 
        measurement_function --> function of form h(s, R, with_noise) that adds the current measurement
        
        u_hist[i] = u_i
        s_hist[i+1] = transition_function(s_hist[i], u_i, Q, dt, with_noise=True)
        y_hist[i+1] = measurement_function(s_hist[i+1], R, with_noise=True)

        So, s_hist and y_hist are the same length, u_hist is one shorter.
        """
        self.dt = dt
        self.Q = Q
        self.R = R
        self.transition_function = transition_function
        self.measurement_function = measurement_function

        self.t_hist = [0.0]
        self.u_hist = []
        self.s_hist = [s_0]
        self.y_hist = []

        self.state_dim = s_0.shape[0]
        self.action_dim = action_dim
        self.meas_dim = meas_dim

        # get the first observation
        self._measure()

        self.N = 1

    def step(self, u:np.ndarray):
        """
        step the simulator once, given action u.
        we only add the new state to state history
        """
        if u.shape[0] != self.action_dim:
            raise ValueError(f"Action dimension {u.shape[0]} does not match expected {self.action_dim}")
        
        s_cur = self.s_hist[-1]

        # add the action to action history
        self.u_hist.append(u)

        # apply the transition function
        s_new = self.transition_function(s_cur, u, self.Q, self.dt, with_noise=True)

        if type(s_new) != np.ndarray:
            raise ValueError(f"Transition function output is not a numpy array, got {type(s_new)}")
        
        if s_new.ndim != 1:
            raise ValueError(f"Transition function output is not a 1D array, got shape {s_new.shape}")
        
        if s_new.shape[0] != self.state_dim:
            raise ValueError(f"Transition function output dimension {s_new.shape[0]} does not match expected {self.state_dim}")
        
        self.s_hist.append(s_new)
        self.t_hist.append(self.t_hist[-1] + self.dt)
        self.N = len(self.s_hist)

        # take the measurement
        self._measure()        

    def _measure(self):
        """
        measure the current state, with noise if with_noise is True
        """
        s_cur = self.s_hist[-1]
        y = self.measurement_function(s_cur, self.R, with_noise=True)

        if type(y) != np.ndarray:
            raise ValueError(f"Measurement function output is not a numpy array, got {type(y)}")
        
        if y.ndim != 1:
            raise ValueError(f"Measurement function output is not a 1D array, got shape {y.shape}")
        
        if y.shape[0] != self.meas_dim:
            raise ValueError(f"Measurement dimension {y.shape[0]} does not match expected {self.meas_dim}")
        
        self.y_hist.append(y)

    def step_k_times(self, actions:np.ndarray):
        """
        step the simulator k times, once for each action u in actions.
        we only add the new states to state history
        """
        for u in actions:
            self.step(u)

    def noiseless_transition(self, i:int, u:np.ndarray):
        """
        get the noiseless transition at a given timestep.
        """
        s_cur = self.s_hist[i]
        s_next = self.transition_function(s_cur, u, self.Q, self.dt, with_noise=False)
        return s_next
    
    def noiseless_measurement(self, i:int):
        """
        get the noiseless measurement at a given timestep
        """
        s_cur = self.s_hist[i]
        y = self.measurement_function(s_cur, self.R, with_noise=False)
        return y
    
    def get_all_states(self):
        return np.array(self.s_hist)
    
    def get_all_measurements(self):
        return np.array(self.y_hist)
    
    def get_all_actions(self):
        return np.array(self.u_hist)

def transition_function(s:np.ndarray, u:np.ndarray, Q:np.ndarray, dt=0.01, with_noise=True):
    """
    state definition for this function is:
    [x, y, z,
    quat_x, quat_y, quat_z, quat_w,
    vx, vy, vz,
    wx, wy, wz,
    b_ax, b_ay, b_az,
    b_wx, b_wy, b_wz,
    gx, gy, gz,
    scale]

    action definition for this function is:
    [ax, ay, az,
    alpha_x, alpha_y, alpha_z]

    all are in camera frame
    """

    # --- unpack state ---
    p = s[0:3]
    q = s[3:7]
    v = s[7:10]
    b_w = s[10:13]
    b_a = s[13:16]
    g = s[16:19]
    scale = s[19]

    # --- unpack input ---
    a_meas = u[0:3]
    omega_meas = u[3:6]   # now angular velocity

    # --- process noise ---
    if with_noise:
        noise = np.random.multivariate_normal(np.zeros(Q.shape[0]), Q)
    else:
        noise = np.zeros(Q.shape[0])

    noise_p   = noise[0:3]
    noise_q   = noise[3:6]    # small-angle perturbation
    noise_v   = noise[6:9]
    noise_w   = noise[9:12]
    noise_bw  = noise[12:15]
    noise_ba  = noise[15:18]

    # -------------------------------------------------
    # Bias-corrected IMU measurements
    # -------------------------------------------------
    a_body = a_meas - b_a
    omega_body = omega_meas - b_w

    # -------------------------------------------------
    # Orientation update
    # -------------------------------------------------
    R_wb = rot_obj.from_quat(q)

    # Small rotation from angular velocity
    delta_theta = omega_body * dt
    delta_R = rot_obj.from_rotvec(delta_theta)

    R_new = R_wb * delta_R

    # Optional orientation noise (small-angle)
    if with_noise:
        R_noise = rot_obj.from_rotvec(noise_q)
        R_new = R_new * R_noise

    q_new = R_new.as_quat()

    # -------------------------------------------------
    # Store angular velocity state (now measured, not integrated)
    # -------------------------------------------------
    w_new = omega_body + noise_w

    # -------------------------------------------------
    # Linear acceleration to world frame
    # -------------------------------------------------
    a_world = R_wb.apply(a_body) - g

    # -------------------------------------------------
    # Velocity update
    # -------------------------------------------------
    v_new = v + a_world * dt + noise_v

    # -------------------------------------------------
    # Position update
    # -------------------------------------------------
    p_new = p + v * dt + 0.5 * a_world * dt**2 + noise_p

    # -------------------------------------------------
    # Bias random walk
    # -------------------------------------------------
    b_a_new = b_a + noise_ba * dt
    b_w_new = b_w + noise_bw * dt

    # -------------------------------------------------
    # Assemble new state
    # -------------------------------------------------
    s_new = np.zeros_like(s)

    s_new[0:3]   = p_new
    s_new[3:7]   = q_new
    s_new[7:10]  = v_new
    s_new[10:13] = b_w_new
    s_new[13:16] = b_a_new
    s_new[16:19] = g
    s_new[19]    = scale

    return s_new

def imu_measurement_function(s:np.ndarray, R:np.ndarray, with_noise=True):
    """
    state definition for this function is:
    [x, y, z,
    quat_x, quat_y, quat_z, quat_w,
    vx, vy, vz,
    b_ax, b_ay, b_az,
    b_wx, b_wy, b_wz,
    scale]
    positions and velocities are metric, so
    x*scale is the unscaled postion, vx*scale is the unscaled velocity, etc.

    
    measurements are scaled positions, and orientations from vision. Also are 
    preintegrated IMU measurements.

    measurement definition for this function is:
    [x_meas, y_meas, z_meas, 
    quat_x, quat_y, quat_z, quat_w 
    ]
    """
    y = 0

def camera_measurement_function(s:np.ndarray, R:np.ndarray, with_noise=True):
    """
    state definition for this function is:
    [x, y, z,
    quat_x, quat_y, quat_z, quat_w,
    vx, vy, vz,
    b_ax, b_ay, b_az,
    b_wx, b_wy, b_wz,
    scale]
    positions and velocities are metric, so
    x*scale is the unscaled postion as observed by the camera, vx*scale is the unscaled velocity, etc.

    measurement definition for this function is:
    [x_scaled, y_scaled, z_scaled, 
    quat_x, quat_y, quat_z, quat_w, 
    ]
    """
    y = np.array([s[0], 
                  s[1], 
                  s[2], 
                  s[3], 
                  s[4], 
                  s[5], 
                  s[6]])
    
    angles = rot_obj.from_quat(s[3:7]).as_euler('xyz', degrees=True)

    if with_noise:
        noise = np.random.multivariate_normal(np.zeros(R.shape[0]), R)

    y[0:3] += noise[0:3]
    angles += noise[3:6]

    y[0:3] *= s[-1]

    y[3:7] = rot_obj.from_euler('xyz', angles, degrees=True).as_quat()

    return y


