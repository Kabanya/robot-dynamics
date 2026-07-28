import time
import numpy as np
import pinocchio as pin
from scipy.optimize import fmin_bfgs
import meshcat.transformations as tf

class Talos:
    def __init__(self, robot, model, data, left_foot_id, right_foot_id,
                 left_ank, right_ank, com_traj, footsteps, robot_base_height, base_foot_height, dt=0.05,
                 zmp_traj=None):

        self.robot = robot
        self.model = model
        self.data = data
        self.left_foot_id = left_foot_id
        self.right_foot_id = right_foot_id
        self.left_ank = left_ank
        self.right_ank = right_ank
        self.com_traj = com_traj
        self.footsteps = footsteps
        self.robot_base_height = robot_base_height
        self.base_foot_height = base_foot_height
        self.dt = dt
        self.torso_id = self.model.getFrameId("torso_2_link")
        self.zmp_traj = zmp_traj

    def animate(self, t_total):
        N = int(t_total / self.dt)
        q = self.robot.q0.copy()
        viz = self.robot.viz.viewer

        for i in range(N):
            t = i * self.dt
            com = self.com_traj(t)
            l_ank = self.left_ank(t)
            r_ank = self.right_ank(t)
            phase = self.footsteps.get_phase_type(t)
            support = 1 if phase == 'left' else 0
            swing = 1 - support

            q_new = self.IK_CoM_solve(
                support, swing,
                r_ank if swing else l_ank,
                l_ank if support == 0 else r_ank,
                np.array([com[0], com[1], self.robot_base_height]),
                q
            )
            q = q_new.copy()
            self.robot.display(q)

            # Visualization of ZMP and CoM
            if self.zmp_traj is not None:
                zmp = self.zmp_traj(t)
                viz["zmp"].set_transform(tf.translation_matrix([zmp[0], zmp[1], 0.0]))
            viz["com"].set_transform(tf.translation_matrix([com[0], com[1], com[2]]))

            time.sleep(self.dt * 0.5)


    def IK_CoM_solve(self, supporting_leg, swing_leg, swing_ankle, support_ankle, CoM, q_init):
        model = self.model
        robot = self.robot

        def IK_cost(x):
            pin.framesForwardKinematics(model, self.data, x)

            f_swing = self.data.oMf[self.right_foot_id if swing_leg == 1 else self.left_foot_id]
            f_support = self.data.oMf[self.right_foot_id if supporting_leg == 1 else self.left_foot_id]
            CoM_k = pin.centerOfMass(model, self.data, x)

            err = (
                np.linalg.norm(f_support.translation - support_ankle) +
                np.linalg.norm(CoM_k - CoM) +
                np.linalg.norm(f_swing.translation - swing_ankle)
            )

            swing_rot = f_swing.rotation
            support_rot = f_support.rotation

            world_x = np.array([1, 0, 0])
            world_y = np.array([0, 1, 0])
            world_z = np.array([0, 0, 1])

            swing_x = swing_rot[:, 0]
            swing_y = swing_rot[:, 1]
            swing_z = swing_rot[:, 2]

            support_x = support_rot[:, 0]
            support_y = support_rot[:, 1]
            support_z = support_rot[:, 2]

            orientation_weight = 0.5
            err += orientation_weight * (
                np.linalg.norm(swing_x - world_x) +
                np.linalg.norm(swing_z - world_z) +
                np.linalg.norm(support_x - world_x) +
                np.linalg.norm(support_z - world_z)
            )

            roll_weight = 0.3
            swing_y_horizontal = swing_y.copy()
            swing_y_horizontal[2] = 0
            swing_y_horizontal = swing_y_horizontal / (np.linalg.norm(swing_y_horizontal) + 1e-6)

            support_y_horizontal = support_y.copy()
            support_y_horizontal[2] = 0
            support_y_horizontal = support_y_horizontal / (np.linalg.norm(support_y_horizontal) + 1e-6)

            if swing_leg == 0:  # left
                target_y = world_y
            else:  # right
                target_y = -world_y

            if supporting_leg == 0:  # left
                support_target_y = world_y
            else:  # right
                support_target_y = -world_y

            err += roll_weight * (
                np.linalg.norm(swing_y_horizontal - target_y) +
                np.linalg.norm(support_y_horizontal - support_target_y)
            )

            # torso_2_link
            f_torso = self.data.oMf[self.torso_id]
            torso_rot = f_torso.rotation
            torso_x = torso_rot[:, 0]
            torso_z = torso_rot[:, 2]

            torso_weight = 0.3
            err += torso_weight * (
                np.linalg.norm(torso_x - world_x) +
                np.linalg.norm(torso_z - world_z)
            )
            regularization_weight = 0.1
            err += regularization_weight * np.linalg.norm(x - q_init)

            return err

        x0 = np.copy(q_init)
        opt_q = fmin_bfgs(IK_cost, x0, maxiter=100, disp=False)
        return opt_q