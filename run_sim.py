
from utils.simulator import Simulator, transition_function, camera_measurement_function, imu_measurement_function
import numpy as np
from utils.test_data_generation import generate_sinusoidal_action_sequence
from utils.visualization import plot_bias_comparison, plot_camera_vs_true, plot_comparison, plot_xyz_trajectory, plot_camera_measured_trajectory, plot_imu_readings, plot_optimized_trajectory, plot_optimized_velocity
from scipy.spatial.transform import Rotation as rot_obj
from utils.optimization import ScaleOptimizer
import time
# -----------------------
# Main script
# -----------------------
if __name__ == "__main__":

    plot_sim_output = True
    plot_optimizer_output = False

    # provide an estimate of imu noise parameters. these include the drift of the
    # biases and the noise of any given measurement.
    noise_params = {
        "acc_noise_sig": 0.0011,
        "gyro_noise_sig": 0.00019,
        "acc_bias_sig": 0.0001,
        "gyro_bias_sig": 0.000001,
        "cam_xyz_noise_sig": 0.001,
        "cam_ang_noise_sig": 0.0001,
    }
    
    rot_mat = rot_obj.from_euler('xyz', [80, 45, 45], degrees=True)
    #b_g = np.array([0.1, 1.1, -0.4]) # gyroscope bias
    #b_a = np.array([-0.4, 1.0, 0.3]) # accelerometer bias
    
    out = []

    num_runs = 3
    for j in range(num_runs):
        b_g = np.random.uniform(low=[-0.15, -0.8, -0.2],
                                high=[0.15, 1.2, 0.6])   # gyroscope bias

        b_a = np.random.uniform(low=[-0.3, -0.5, -0.6],
                                high=[0.7, 0.1, 1.0])    # accelerometer bias

        g = np.array([0, 0, -9.81]) # gravity vector
        g = rot_mat.apply(g) # gravity vector
        #rot_mat = rot_mat.as_matrix()
        true_scale = np.array([np.random.uniform(0.25, 1.3)])
        
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

        actions, xyz_traj, linear_world_vel, angular_world_vel, world_orientations = generate_sinusoidal_action_sequence(n_steps, dt_imu)
        
        # ----------------------------
        # Initial state
        # ----------------------------
        p_0 = xyz_traj[0].copy()
        th_0 = world_orientations[0].as_quat().copy() 
        v_0 = linear_world_vel[0].copy()
        w_0 = angular_world_vel[0].copy()

        s_0 = np.concatenate([[0.0], p_0, th_0, v_0, w_0, b_g, b_a, g, true_scale]) 

        T_ic = np.eye(4)  # identity transform between camera and IMU frames
        T_ic[0, 3] = 0.05
        T_ic[1, 3] = 0.001
        T_ic[2, 3] = 0.002
        cam_imu_rot = rot_obj.from_euler('xyz', [45, 45, 0], degrees=True)
        T_ic[0:3, 0:3] = cam_imu_rot.as_matrix()

        sim = Simulator(s_0, 
                        Q=Q, R_imu=R_imu, R_cam=R_cam, 
                        transition_function=transition_function, 
                        imu_measurement_function=imu_measurement_function,
                        camera_measurement_function=camera_measurement_function, 
                        action_dim=action_dim, meas_dim=meas_dim,
                        dt=dt_imu,
                        T_ic = T_ic)

        sim.step_through_actions(actions)
        
        states = sim.get_all_states()
        measurements = sim.get_all_measurements()
        actions = sim.get_all_actions()
        
        if plot_sim_output:
            plot_xyz_trajectory(states)
            plot_camera_measured_trajectory(measurements)
            plot_imu_readings(measurements)

        # we need to downsample the camera measurements, because IMU should
        # operate at a higher frequency.
        start_idx = 2
        imu_data = measurements[start_idx:, 0:7]  # (N,7) with (wx, wy, wz, ax, ay, az)
        cam_traj = measurements[start_idx:, [0, 7, 8, 9, 10, 11, 12, 13]]  # (N,8) with (t, x,y,z, quat_x, quat_y, quat_z, quat_w)
        cam_traj = cam_traj[::int(dt_cam/dt_imu)]  # downsample by factor of 20.
        x_init = states[start_idx, 1:8]  # initial position and orientation

        # now we use the optimizer to estimate the true trajectory and scale factor, based on measurement data.
        optimizer = ScaleOptimizer(R_cam, noise_params, T_ic = np.linalg.inv(T_ic))
        optimizer.init_factor_graph(cam_traj, imu_data, x_init)
        print(optimizer.factor_graph)
        # run the optimizer to try and estimate scale
        start_time = time.time()
        result = optimizer.optimize()
        end_time = time.time()
        print(f"Optimization time: {end_time - start_time}")

        # ----------------------------
        # Extract results
        # ----------------------------
        #scale_opt = result.atVector(optimizer.scale_key)[0]  # scalar
        #scaled_traj = [scale_opt * result.atPose3(k).translation() for k in optimizer.pose_keys]

        out.append([((1/result.atVector(optimizer.scale_key)[0]) - true_scale[0])/true_scale[0], 
                    np.linalg.norm(result.atConstantBias(optimizer.bias_keys[0]).accelerometer() - b_a),
                    np.linalg.norm(result.atConstantBias(optimizer.bias_keys[0]).gyroscope() - b_g),
                    end_time - start_time])

        if plot_optimizer_output:

            first_pose = result.atPose3(optimizer.pose_keys[0])
            R_world_body = first_pose.rotation().matrix()  # 3x3 rotation matrix
            gravity_world = R_world_body @ np.array([0, 0, -1])  # body Z-axis points down

            bias1 = result.atConstantBias(optimizer.bias_keys[0])
            print("Estimated gravity direction in world frame:", gravity_world)
            print("Estimated accelerometer bias:", bias1.accelerometer())
            print("Estimated gyroscope bias:", bias1.gyroscope())
            #bias2 = result.atConstantBias(optimizer.bias_keys[1000])
            #print("Estimated accelerometer bias (later):", bias2.accelerometer())
            #print("Estimated gyroscope bias (later):", bias2.gyroscope())
            print("scale_opt:", 1/result.atVector(optimizer.scale_key)[0])
            plot_optimized_trajectory(cam_traj[:, 0], result, optimizer.pose_keys)
            plot_optimized_velocity(cam_traj[:, 0], result, optimizer.vel_keys)
            plot_comparison(
                cam_traj[:,0],
                states,
                dt_imu,
                result,
                optimizer.pose_keys,
                np.linalg.inv(T_ic)
            )
            plot_bias_comparison(
                cam_traj[:,0],
                states,
                dt_imu,
                result,
                optimizer.bias_keys
            )
            # plot_camera_vs_true(
            #     cam_traj[:,0],
            #     cam_traj,
            #     states,
            #     dt_imu
            # )

    out = np.array(out)
    # RMSE = sqrt(mean(square(errors), axis=0))
    rmse = np.sqrt(np.mean(out**2, axis=0))

    print("RMSE per column (SCALE, ACCEL, GYRO, TIME):")
    print(rmse)
