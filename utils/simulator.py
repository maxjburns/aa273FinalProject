import numpy as np
from scipy.spatial.transform import Rotation as rot_obj

class Simulator:
    def __init__(self, s_0:np.ndarray, Q:np.ndarray, R_imu:np.ndarray, R_cam:np.ndarray, transition_function, 
                 imu_measurement_function, camera_measurement_function,
                action_dim:int, meas_dim:int, dt=0.01, T_ic=np.eye(4)):
        """
        s_0 --> initial state
        Q --> process noise covariance
        R_imu --> IMU measurement noise covariance
        R_cam --> camera measurement noise covariance
        dt --> time step
        transition_function --> function of form f(s, u, Q, dt, with_noise) that adds the next state 
        imu_measurement_function --> function of form h_imu(s, R, with_noise) that adds the current IMU measurement
        camera_measurement_function --> function of form h_camera(s, R, with_noise) that adds the current camera measurement
        action_dim --> dimension of action space
        meas_dim --> dimension of measurement space

        u_hist[i] = u_i
        s_hist[i+1] = transition_function(s_hist[i], u_i, Q, dt, with_noise=True)
        y_hist[i+1, 0:6] = imu_measurement_function(s_hist[i], s_hist[i+1], R_imu, with_noise=True)
        y_hist[i+1, 6:13] = camera_measurement_function(s_hist[i+1], R_cam, with_noise=True)

        So, s_hist and y_hist are the same length, u_hist is one shorter.
        
        state definition for this class and any functions given to it is:
        [t, x, y, z,
        quat_x, quat_y, quat_z, quat_w,
        vx, vy, vz,
        wx, wy, wz,
        b_wx, b_wy, b_wz,
        b_ax, b_ay, b_az,
        gx, gy, gz,
        scale]

        action definition for this class is:
        [ax, ay, az,
        alpha_x, alpha_y, alpha_z]
        
        all states are tracked in the world frame. This includes acceleration inputs and all
        state variables except for biases.
        """
        self.dt = dt
        self.Q = Q
        self.R_imu = R_imu
        self.R_cam = R_cam
        self.transition_function = transition_function
        self.imu_measurement_function = imu_measurement_function
        self.camera_measurement_function = camera_measurement_function

        self.t_hist = [0.0]
        self.u_hist = []
        self.s_hist = [s_0]
        self.y_hist = []

        self.state_dim = s_0.shape[0]
        self.action_dim = action_dim
        self.meas_dim = meas_dim

        self.N = 1
        self.T_ic = T_ic

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
        if len(self.s_hist) < 2:
            raise ValueError(f"Need at least 2 states in history to take a measurement, got only {len(self.s_hist)}")
        s_prev = self.s_hist[-2]
        s_cur = self.s_hist[-1]
        y_imu = self.imu_measurement_function(s_prev, s_cur, self.R_imu, self.dt, T_ic=self.T_ic, with_noise=True)
        y_cam = self.camera_measurement_function(s_cur, self.R_cam, with_noise=True)

        if type(y_imu) != np.ndarray:
            raise ValueError(f"IMU measurement function output is not a numpy array, got {type(y_imu)}")
        
        if type(y_cam) != np.ndarray:
            raise ValueError(f"Camera measurement function output is not a numpy array, got {type(y_cam)}")
        
        if y_imu.ndim != 1:
            raise ValueError(f"IMU measurement function output is not a 1D array, got shape {y_imu.shape}")
        
        if y_cam.ndim != 1:
            raise ValueError(f"Camera measurement function output is not a 1D array, got shape {y_cam.shape}")

        self.y_hist.append(np.concatenate([s_cur[0:1], y_imu, y_cam]))

    def step_through_actions(self, actions:np.ndarray):
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
        if len(self.s_hist) < 2:
            raise ValueError(f"Need at least 2 states in history to take a measurement, got only {len(self.s_hist)}")
        s_cur = self.s_hist[i]
        s_prev = self.s_hist[i-1]
        y_imu = self.imu_measurement_function(s_prev, s_cur, self.R_imu, self.dt, T_ic=self.T_ic, with_noise=False)
        y_cam = self.camera_measurement_function(s_cur, self.R_cam, with_noise=False)
        y = np.concatenate([s_cur[0:1], y_imu, y_cam])
        return y
    
    def get_all_states(self):
        return np.array(self.s_hist)
    
    def get_all_measurements(self):
        return np.array(self.y_hist)
    
    def get_all_actions(self):
        return np.array(self.u_hist)

def transition_function(s:np.ndarray, u:np.ndarray, Q:np.ndarray, dt=0.01, with_noise=True):
    """
    given a state, control input, and noise covariance, return the next state.
    """

    # --- unpack state ---
    t = s[0]
    p = s[1:4]
    q = s[4:8]
    v = s[8:11]
    w = s[11:14]
    b_w = s[14:17]
    b_a = s[17:20]
    g = s[20:23]
    scale = s[23]

    # --- unpack input ---
    a = u[0:3]
    alpha = u[3:6] 

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

    # update orientation
    R_wb = rot_obj.from_quat(q)

    w_mid = w + 0.5 * alpha * dt
    delta_theta = w_mid * dt + noise_q
    delta_R = rot_obj.from_rotvec(delta_theta)

    R_new = R_wb * delta_R

    q_new = R_new.as_quat()

    # update angular velocity
    delta_w = alpha * dt
    w_new = w + delta_w + noise_w

    # update linear velocity
    v_new = v + a * dt + noise_v

    # update position
    p_new = p + v * dt + 0.5 * a * dt**2 + noise_p

    # update biases
    b_a_new = b_a + noise_ba
    b_w_new = b_w + noise_bw

    # return new state
    s_new = np.zeros_like(s)

    s_new[0]     = t + dt
    s_new[1:4]   = p_new
    s_new[4:8]  = q_new
    s_new[8:11] = v_new
    s_new[11:14] = w_new
    s_new[14:17] = b_w_new
    s_new[17:20] = b_a_new
    s_new[20:23] = g
    s_new[23]    = scale

    return s_new

def imu_measurement_function( s_prev:np.ndarray, s_cur:np.ndarray, R:np.ndarray, dt, T_ic=np.eye(4), with_noise=True):
    """
    we return fake IMU measurements based on the given states and noise covariance R
    return is:
    [wx, wy, wz,
    ax, ay, az]
    in IMU frame
    """
    vel_cur = s_cur[8:11]
    vel_prev = s_prev[8:11]
    b_w_cur = s_cur[14:17]
    b_a_cur = s_cur[17:20]
    orientation_cur = s_cur[4:8]
    orientation_prev = s_prev[4:8]
    g = s_cur[20:23]

    R_ic = T_ic[0:3, 0:3]   # IMU -> camera
    t_ic = T_ic[0:3, 3]     # IMU origin in camera frame
    R_ci = R_ic.T
    r_ic_c = -t_ic          # camera -> IMU vector in camera frame

    a_meas_w = (vel_cur - vel_prev) / dt + g
    R_cw = rot_obj.from_quat(orientation_cur).inv()
    a_meas_c = R_cw.apply(a_meas_w)

    R_cur = rot_obj.from_quat(orientation_cur)
    R_prev = rot_obj.from_quat(orientation_prev)
    dR =  R_prev.inv() * R_cur
    w_meas_c = dR.as_rotvec() / dt

    a_meas_c = a_meas_c + np.cross(w_meas_c, np.cross(w_meas_c, r_ic_c))

    w_meas_b = R_ci @ w_meas_c + b_w_cur
    a_meas_b = R_ci @ a_meas_c + b_a_cur

    if with_noise:
        noise = np.random.multivariate_normal(np.zeros(6), R)
    else:
        noise = np.zeros(6)

    w_meas_b += noise[0:3]
    a_meas_b += noise[3:6]

    return np.concatenate([w_meas_b, a_meas_b])


def camera_measurement_function(s:np.ndarray, R:np.ndarray, with_noise=True):
    """
    measurement definition for this function is:
    [x_scaled, y_scaled, z_scaled, 
    quat_x, quat_y, quat_z, quat_w, 
    ]
    """
    y = s[1:8].copy() # get position and orientation
    
    angles = rot_obj.from_quat(s[4:8]).as_euler('xyz', degrees=False)

    if with_noise:
        noise = np.random.multivariate_normal(np.zeros(R.shape[0]), R)
    else:
        noise = np.zeros(R.shape[0])

    y[0:3] += noise[0:3]
    angles += noise[3:6]

    y[0:3] *= s[-1]

    y[3:7] = rot_obj.from_euler('xyz', angles, degrees=False).as_quat()

    return y


