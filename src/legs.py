import numpy as np

from .footsteps import FootSteps

FOOT_LIFT_HEIGHT = 0.1

class LeftLeg(object):
    def __init__(self, footsteps: FootSteps, base_foot_height=0.0) -> None:
        self.footsteps = footsteps
        self.base_foot_height = base_foot_height

    def __call__(self, t):
        left = np.array(self.footsteps.get_left_position(t))
        left_next = np.array(self.footsteps.get_left_next_position(t))
        t_start = self.footsteps.get_phase_start(t)
        t_total = self.footsteps.get_phase_duration(t)
        k = (t - t_start) / t_total if t_total > 0 else 0.0

        phase_type = self.footsteps.get_phase_type(t)

        if phase_type == 'left':
            z_foot = self.base_foot_height + FOOT_LIFT_HEIGHT * np.sin(k * np.pi)
            act_left = left + (left_next - left) * k
        else:
            z_foot = self.base_foot_height
            act_left = left

        return np.array([act_left[0], act_left[1], z_foot])

class RightLeg(object):
    def __init__(self, footsteps: FootSteps, base_foot_height=0.0) -> None:
        self.footsteps = footsteps
        self.base_foot_height = base_foot_height

    def __call__(self, t):
        right = np.array(self.footsteps.get_right_position(t))
        right_next = np.array(self.footsteps.get_right_next_position(t))
        t_start = self.footsteps.get_phase_start(t)
        t_total = self.footsteps.get_phase_duration(t)
        k = (t - t_start) / t_total if t_total > 0 else 0.0

        phase_type = self.footsteps.get_phase_type(t)

        if phase_type == 'right':
            z_foot = self.base_foot_height + FOOT_LIFT_HEIGHT * np.sin(k * np.pi)
            act_right = right + (right_next - right) * k
        else:
            z_foot = self.base_foot_height
            act_right = right

        return np.array([act_right[0], act_right[1], z_foot])
