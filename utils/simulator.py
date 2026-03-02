import numpy as np


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
        self._measure(with_noise=True)        

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

    def step_k_times(self, u:np.ndarray, k:int):
        """
        step the simulator k times, given action u.
        we only add the new states to state history
        """
        for _ in range(k):
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

def transition_function(s:np.ndarray, u:np.ndarray, Q:np.ndarray, dt=0.01, with_noise=True):
    """
    TODO: implement the transition function for the simulator
    """
    pass

def measurement_function(s:np.ndarray, R:np.ndarray, with_noise=True):
    """
    TODO: implement the measurement function for the simulator
    """
    pass


