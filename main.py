from utils.optimization import test
from utils.simulator import Simulator, transition_function, measurement_function
import numpy as np



if __name__ == "__main__":
    s_0 = np.array([0.0, 0.0])  # Example initial state (x, y)
    Q = np.eye(2)  # Process noise covariance
    R = np.eye(2)  # Measurement noise covariance
    dt = 0.01

    sim = Simulator(s_0, Q=np.eye(2), R=np.eye(2), 
                    transition_function=transition_function, 
                    measurement_function=measurement_function, 
                    action_dim=2, meas_dim=2,
                    dt=dt)