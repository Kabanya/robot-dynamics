import numpy as np

from .footsteps import FootSteps
from .support import move_inside_polygon, polygon_margin, support_polygon

class ZmpClass(object):
    def __init__(self, footsteps: FootSteps,
                 mode="baseline",
                 foot_length=0.22,
                 foot_width=0.12,
                 safety_margin=0.0,
                 bias=(0.0, 0.0),
                 smooth_double_support=False,
                 left_offset=(0.0, 0.0),
                 right_offset=(0.0, 0.0)):
        self.footsteps = footsteps
        self.mode = mode
        self.foot_length = foot_length
        self.foot_width = foot_width
        self.safety_margin = safety_margin
        self.bias = np.asarray(bias, dtype=float)
        self.smooth_double_support = smooth_double_support
        self.left_offset = np.asarray(left_offset, dtype=float)
        self.right_offset = np.asarray(right_offset, dtype=float)

    def __call__(self, t):
        foot = self.footsteps.get_phase_type(t)
        left_pos = self.footsteps.get_left_position(t) + self.left_offset
        right_pos = self.footsteps.get_right_position(t) + self.right_offset

        if foot == 'left':
            result = self._single_support_zmp(right_pos)
        elif foot == 'right':
            result = self._single_support_zmp(left_pos)
        else: # double support
            result = self._double_support_zmp(t, left_pos, right_pos)
        return result

    def polygon(self, t):
        return support_polygon(
            self.footsteps, t,
            foot_length=self.foot_length,
            foot_width=self.foot_width,
            left_offset=self.left_offset,
            right_offset=self.right_offset,
        )

    def margin(self, t):
        return polygon_margin(self(t), self.polygon(t))

    def _single_support_zmp(self, support_pos):
        support_pos = np.asarray(support_pos, dtype=float)
        if self.mode == "baseline":
            return support_pos.copy()

        half_length = max(0.0, self.foot_length / 2.0 - self.safety_margin)
        half_width = max(0.0, self.foot_width / 2.0 - self.safety_margin)
        bias = np.array([
            np.clip(self.bias[0], -half_length, half_length),
            np.clip(self.bias[1], -half_width, half_width),
        ])
        return support_pos + bias

    def _double_support_zmp(self, t, left_pos, right_pos):
        left_pos = np.asarray(left_pos, dtype=float)
        right_pos = np.asarray(right_pos, dtype=float)

        if self.mode == "baseline" or not self.smooth_double_support:
            point = (left_pos + right_pos) / 2.0
        else:
            point = self._smooth_double_support_point(t, left_pos, right_pos)

        if self.mode != "baseline":
            point = point + self.bias
            point = move_inside_polygon(point, self.polygon(t), self.safety_margin)
        return point

    def _smooth_double_support_point(self, t, left_pos, right_pos):
        idx = self.footsteps.get_index_from_time(t)
        start = self._nearest_support_center(idx - 1, -1)
        finish = self._nearest_support_center(idx + 1, 1)
        midpoint = (left_pos + right_pos) / 2.0
        start = midpoint if start is None else start
        finish = midpoint if finish is None else finish

        duration = self.footsteps.get_phase_duration(t)
        alpha = 0.0 if duration <= 0 else (t - self.footsteps.get_phase_start(t)) / duration
        alpha = np.clip(alpha, 0.0, 1.0)
        return (1.0 - alpha) * start + alpha * finish

    def _nearest_support_center(self, idx, step):
        while 0 <= idx < len(self.footsteps.flying_foot):
            phase = self.footsteps.flying_foot[idx]
            if phase == "left":
                return (
                    np.asarray(self.footsteps.right[idx], dtype=float)
                    + self.right_offset
                )
            if phase == "right":
                return (
                    np.asarray(self.footsteps.left[idx], dtype=float)
                    + self.left_offset
                )
            idx += step
        return None
