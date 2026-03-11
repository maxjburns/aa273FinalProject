import numpy as np
import gtsam
from functools import partial
from scipy.spatial.transform import Rotation as spR

class ScaleOptimizer:
    def __init__(self, R_cam, noise_params):
        """
        R_cam is vision measurement noise covariance
        noise_params is a dict with keys: acc_noise_sig, gyro_noise_sig, acc_bias_sig, gyro_bias_sig
        """
        
        self.R_cam = R_cam
        self.noise_params = noise_params
        # add imu factors, which are relative between frames
        self.init_preintegration_params(acc_noise_sig=noise_params["acc_noise_sig"], 
                                        gyro_noise_sig=noise_params["gyro_noise_sig"])
        


    def init_factor_graph(self, 
                          cam_data, 
                          accel_data,
                          x_init): 
        """
        
        """
        
        t_imu = accel_data[:, 0]
        w = accel_data[:, 1:4]
        a = accel_data[:, 4:7]

        t_cam = cam_data[:, 0]
        cam_traj = cam_data[:, 1:]  # (N,7) with (x,y,z, quat_x, quat_y, quat_z, quat_w)

        N_cam = t_cam.shape[0]
        N_imu = t_imu.shape[0]

        self.factor_graph = gtsam.NonlinearFactorGraph()
        self.values = gtsam.Values()

        self.pose_keys = [gtsam.symbol('x', k) for k in range(N_cam)]
        self.vel_keys = [gtsam.symbol('v', k) for k in range(N_cam)]
        self.bias_keys = [gtsam.symbol('b', k) for k in range(N_cam)]
        self.scale_key = gtsam.symbol('s', 0)

        # Set initial values for all variables
        self.values = gtsam.Values()

        for k in range(N_cam):
            # this is a static prior
            pose = gtsam.Pose3(
                gtsam.Rot3.Quaternion(cam_traj[k, 6], cam_traj[k, 3], cam_traj[k, 4], cam_traj[k, 5]),
                gtsam.Point3(cam_traj[k, 0], cam_traj[k, 1], cam_traj[k, 2])
            )
            #pose = gtsam.Pose3()  # dummy initial pose guess

            vel = np.zeros(3) 
            bias = gtsam.imuBias.ConstantBias() 

            # insert the values at this timestep.
            self.values.insert(self.pose_keys[k], pose)  # initial guess
            self.values.insert(self.vel_keys[k], vel)  # initial velocity guess
            self.values.insert(self.bias_keys[k], bias)  # initial bias guess

        # initial scale estimate
        self.values.insert(self.scale_key, np.array([1.0])) 

        # vision noise model
        self.vision_model = gtsam.noiseModel.Gaussian.Covariance(self.R_cam)

        # add constraints between the camera poses and the scale variable
        for cam_idx in range(N_cam - 1):
            self.factor_graph.add(
                gtsam.CustomFactor(
                    self.vision_model,
                    [self.pose_keys[cam_idx], self.pose_keys[cam_idx + 1], self.scale_key],
                    partial(vision_factor, cam_traj[cam_idx], cam_traj[cam_idx + 1]),
                )
            )
        # now we add the IMU factors, which constrain connections between camera frames with real scale
        for cam_idx in range(N_cam-1):
            dt_cam = t_cam[cam_idx+1] - t_cam[cam_idx]
            imu_idx_cur = np.searchsorted(t_imu, t_cam[cam_idx], side='left')
            imu_idx_next = np.searchsorted(t_imu, t_cam[cam_idx+1], side='left')

            Xi = self.pose_keys[cam_idx]
            Vi = self.vel_keys[cam_idx]
            Bi = self.bias_keys[cam_idx]

            Xj = self.pose_keys[cam_idx+1]
            Vj = self.vel_keys[cam_idx+1]
            Bj = self.bias_keys[cam_idx+1]

            bias_i = self.values.atConstantBias(Bi)
            pim_k = gtsam.PreintegratedImuMeasurements(self.preint_params, bias_i)

            for i in range(imu_idx_cur, imu_idx_next):
                dt_imu = t_imu[i+1] - t_imu[i]
                pim_k.integrateMeasurement(a[i], w[i], dt_imu)

            imu_factor = gtsam.ImuFactor(Xi, Vi, Xj, Vj, Bi, pim_k)

            sigma = np.sqrt(dt_cam) * np.hstack([
                self.noise_params["acc_bias_sig"]  * np.ones(3),
                self.noise_params["gyro_bias_sig"] * np.ones(3)
            ]) 

            bias_noise_model = gtsam.noiseModel.Diagonal.Sigmas(sigma)

            self.factor_graph.add(imu_factor)
            self.factor_graph.add(
                gtsam.BetweenFactorConstantBias(
                    Bi, Bj,
                    gtsam.imuBias.ConstantBias(),
                    bias_noise_model
                )
            )

        # add the prior for the initial position
        self.factor_graph.add(gtsam.PriorFactorPose3(
            self.pose_keys[0],
            gtsam.Pose3(
                gtsam.Rot3.Quaternion(x_init[6], x_init[3], x_init[4], x_init[5]),
                gtsam.Point3(x_init[0], x_init[1], x_init[2])
            ),
            gtsam.noiseModel.Diagonal.Sigmas(np.array([
                10.0, 10.0, 10.0,    # Loose on rotation (allow gravity alignment)
                0.01, 0.01, 0.01  # VERY loose on position (let scale determine)
            ]))
        ))

        # Add loose prior on scale to help convergence
        #scale_prior = gtsam.noiseModel.Isotropic.Sigma(1, 0.5)  # loose
        #self.factor_graph.add(
        #    gtsam.PriorFactorVector(
        #        self.scale_key,
        #        np.array([1.0]),
        #        scale_prior
        #    )
        #)


    def init_preintegration_params(self, acc_noise_sig, gyro_noise_sig):

        params = gtsam.PreintegrationParams.MakeSharedU(-9.81)
        params.setAccelerometerCovariance(np.eye(3) * acc_noise_sig**2)
        params.setGyroscopeCovariance(np.eye(3) * gyro_noise_sig**2)
        params.setIntegrationCovariance(np.eye(3) * 1e-8)

        self.preint_params = params

    def optimize(self):
        """
        run the optimizer on the factor graph
        """
        params = gtsam.LevenbergMarquardtParams()
        params.setVerbosity("ERROR")  # or try "TERMINATION", "VALUES"
        params.setMaxIterations(200)      # increase max iterations to 1000
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.factor_graph, self.values, params)
        result = optimizer.optimize()
        return result
    
def vision_factor(meas_i, meas_j, this, values, H=None):
    key_x_i, key_x_j, key_s = this.keys()

    pose_i: gtsam.Pose3 = values.atPose3(key_x_i)
    pose_j: gtsam.Pose3 = values.atPose3(key_x_j)
    s = values.atVector(key_s)[0]

    # estimated relative motion in metric frame
    t_i = np.array(pose_i.translation())
    t_j = np.array(pose_j.translation())
    dt_est = t_j - t_i

    R_i = pose_i.rotation()
    R_j = pose_j.rotation()
    R_ij_est = R_i.between(R_j)

    # measured relative motion in unscaled camera frame
    t_i_meas = meas_i[0:3]
    t_j_meas = meas_j[0:3]
    dt_meas = t_j_meas - t_i_meas

    q_i_meas = meas_i[3:7]
    q_j_meas = meas_j[3:7]
    R_i_meas = gtsam.Rot3.Quaternion(q_i_meas[3], q_i_meas[0], q_i_meas[1], q_i_meas[2])
    R_j_meas = gtsam.Rot3.Quaternion(q_j_meas[3], q_j_meas[0], q_j_meas[1], q_j_meas[2])
    R_ij_meas = R_i_meas.between(R_j_meas)

    # translation: scaled measured delta should match estimated delta
    r_t = dt_est - s * dt_meas

    # rotation: measured relative rotation should match estimated relative rotation
    R_err = R_ij_meas.between(R_ij_est)
    r_R = np.array(gtsam.Rot3.Logmap(R_err))

    error = np.hstack([r_t, r_R])

    if H is not None:
        Jr_inv = right_jacobian_inv_so3(r_R)

        # Pose tangent order in GTSAM: [rotation(3), translation(3)]
        H_xi = np.zeros((6, 6))
        H_xj = np.zeros((6, 6))
        H_s = np.zeros((6, 1))

        # translation residual jacobians
        H_xi[0:3, 3:6] = -np.eye(3)      # d(dt_est)/d(t_i)
        H_xj[0:3, 3:6] = np.eye(3)       # d(dt_est)/d(t_j)
        H_s[0:3, 0] = -dt_meas           # d(-s*dt_meas)/ds

        # rotation residual jacobians (first-order)
        H_xi[3:6, 0:3] = -Jr_inv
        H_xj[3:6, 0:3] = Jr_inv

        H[0] = H_xi
        H[1] = H_xj
        H[2] = H_s

    return error

def skew_symmetric(v):
    """Convert vector to skew-symmetric matrix"""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def right_jacobian_inv_so3(omega):
    """
    Inverse of right Jacobian of SO(3).
    Maps rotation vector perturbations to tangent space.
    """
    theta = np.linalg.norm(omega)
    
    if theta < 1e-6:
        # Small angle approximation
        return np.eye(3) + 0.5 * skew_symmetric(omega)
    
    # General case
    axis = omega / theta
    half_theta = 0.5 * theta
    
    K = skew_symmetric(axis)
    
    # Jr^{-1} = I + (θ/2) * K + (1 - θ*cot(θ/2)/2) * K²
    cot_half = 1.0 / np.tan(half_theta)
    
    return (np.eye(3) + 
            half_theta * K + 
            (1.0 - theta * cot_half / 2.0) * (K @ K))