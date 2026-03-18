

import numpy as np
from utils.visualization import plot_bias_comparison, plot_camera_vs_true, plot_comparison, plot_xyz_trajectory, plot_camera_measured_trajectory, plot_imu_readings, plot_optimized_trajectory, plot_optimized_velocity
from scipy.spatial.transform import Rotation as rot_obj
from utils.optimization import ScaleOptimizer
import time
from utils.data_retrieval import load_npz
from scipy.spatial.transform import Rotation as spR
# -----------------------
# Main script
# -----------------------
if __name__ == "__main__":

    plot_optimizer_output = True
    path = "datasets/pi3_output_0_150.npz"
    # provide an estimate of imu noise parameters. these include the drift of the
    # biases and the noise of any given measurement.
    noise_params = {
        "acc_noise_sig": 0.0011259800314265293,
        "gyro_noise_sig": 0.0001986250579148769,
        "acc_bias_sig": 0.00012779200872340579,
        "gyro_bias_sig": 4.518260343701138e-06,
        "cam_xyz_noise_sig": 0.0012,
        "cam_ang_noise_sig": 0.002,
    }
    

    state_dim = 24
    action_dim = 6
    meas_dim = 13  # position and quat orientation
    
    dt_imu = 0.001
    dt_cam = 0.05

    Q = np.zeros((state_dim-2, state_dim-2))  # Process noise covariance

    Q[12, 12] = (noise_params["gyro_bias_sig"] * np.sqrt(dt_imu))**2  # Gyro bias noise
    Q[13, 13] = (noise_params["gyro_bias_sig"] * np.sqrt(dt_imu))**2  # Gyro bias noise
    Q[14, 14] = (noise_params["gyro_bias_sig"] * np.sqrt(dt_imu))**2  # Gyro bias noise

    Q[15, 15] = (noise_params["acc_bias_sig"] * np.sqrt(dt_imu))**2  # Acc bias noise
    Q[16, 16] = (noise_params["acc_bias_sig"] * np.sqrt(dt_imu))**2  # Acc bias noise
    Q[17, 17] = (noise_params["acc_bias_sig"] * np.sqrt(dt_imu))**2  # Acc bias noise

    R_imu = np.zeros((6, 6))  # Measurement noise covariance

    R_imu[0, 0] = (noise_params["gyro_noise_sig"] / np.sqrt(dt_imu))**2
    R_imu[1, 1] = (noise_params["gyro_noise_sig"] / np.sqrt(dt_imu))**2
    R_imu[2, 2] = (noise_params["gyro_noise_sig"] / np.sqrt(dt_imu))**2
    R_imu[3, 3] = (noise_params["acc_noise_sig"] / np.sqrt(dt_imu))**2
    R_imu[4, 4] = (noise_params["acc_noise_sig"] / np.sqrt(dt_imu))**2
    R_imu[5, 5] = (noise_params["acc_noise_sig"] / np.sqrt(dt_imu))**2

    R_cam = np.zeros((6, 6))
    R_cam[0:3, 0:3] = np.eye(3) * (noise_params["cam_xyz_noise_sig"]**2)  # Position measurement noise
    R_cam[3:6, 3:6] = np.eye(3) * (noise_params["cam_ang_noise_sig"]**2)  # Orientation measurement noise

    T = 7.5 # time elapsed
    n_steps = int(T / dt_imu)
    

    T_ic = np.array([[-0.11065033, 0.99303542, -0.04046195, 0.00438529],
                     [0.79982103,  0.06480697, -0.59672974, -0.10066521,],
                     [-0.58995155, -0.09839066, -0.80142152,  0.07477948,],
                     [0.0, 0.0, 0.0, 1.0]]) # transformation from IMU to camera frame

    data = load_npz(path)
    pi3_t = data["camera_poses"]
    camera_poses_mat = data["pi3_t"]
    imu_data = data["imu_data"]

    plot_imu_readings(imu_data)

    print(pi3_t.shape, camera_poses_mat[:, 0:3, 0:3].shape, imu_data.shape)
    
    cam_traj_R = spR.from_matrix(camera_poses_mat[:, 0:3, 0:3]).as_quat()

    cam_traj_t = camera_poses_mat[:, 0:3, 3]  # (N,3) with (x,y,z)
    cam_traj = np.hstack((pi3_t[:, None], cam_traj_t, cam_traj_R))  # (N,8) with (t, x,y,z, quat_x, quat_y, quat_z, quat_w)


    # we need to downsample the camera measurements, because IMU should
    # operate at a higher frequency.
    x_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]) # dummy initial guess for scale optimizer

    # now we use the optimizer to estimate the true trajectory and scale factor, based on measurement data.
    optimizer = ScaleOptimizer(R_cam, noise_params, T_ic = T_ic)
    optimizer.init_factor_graph(cam_traj, imu_data, x_init)
    print(optimizer.factor_graph)
    # run the optimizer to try and estimate scale
    start_time = time.time()
    result = optimizer.optimize()
    end_time = time.time()
    print(f"Optimization time: {end_time - start_time}")

    if plot_optimizer_output:

        first_pose = result.atPose3(optimizer.pose_keys[0])
        R_world_body = first_pose.rotation().matrix()  # 3x3 rotation matrix
        gravity_world = R_world_body @ np.array([0, 0, -1])  # body Z-axis points down

        bias1 = result.atConstantBias(optimizer.bias_keys[0])

        print("Estimated gravity direction in world frame:", gravity_world)
        print("Estimated accelerometer bias:", bias1.accelerometer())
        print("Estimated gyroscope bias:", bias1.gyroscope())
        
        print("scale_opt:", 1/result.atVector(optimizer.scale_key)[0])

        plot_optimized_trajectory(cam_traj[:, 0], result, optimizer.pose_keys)
        plot_optimized_velocity(cam_traj[:, 0], result, optimizer.vel_keys)
        