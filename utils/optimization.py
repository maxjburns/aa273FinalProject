import gtsam
import numpy as np
from gtsam import Pose3, Point3

class VisionScaleFactor(gtsam.CustomFactor):
    def __init__(self, key_pose_i, key_pose_j, key_scale, t_vis, noise):
        keys = gtsam.KeyVector()
        keys.append(key_pose_i)
        keys.append(key_pose_j)
        keys.append(key_scale)
        super().__init__(noise, keys)

        self.t_vis = t_vis

    def evaluateError(self, values, jacobians=None):
        pose_i = values.atPose3(self.keys()[0])
        pose_j = values.atPose3(self.keys()[1])
        scale = values.atDouble(self.keys()[2])

        p_i = np.array(pose_i.translation())
        p_j = np.array(pose_j.translation())

        # residual = metric distance - scaled vision measurement
        r = p_j - p_i - scale * self.t_vis

        # optionally compute jacobians if required
        return r