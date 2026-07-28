import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.footsteps import FootSteps
from src.support import polygon_margin, support_polygon
from src.zmp import ZmpClass


def sample_footsteps():
    footsteps = FootSteps([0.0, -0.1], [0.0, 0.1])
    footsteps.add_phase(0.2, "none")
    footsteps.add_phase(0.7, "left", [0.1, 0.1])
    footsteps.add_phase(0.2, "none")
    footsteps.add_phase(0.7, "right", [0.2, -0.1])
    return footsteps


def test_margin_sign():
    footsteps = sample_footsteps()
    polygon = support_polygon(footsteps, 0.3)
    assert polygon_margin([0.0, -0.1], polygon) > 0.0
    assert polygon_margin([1.0, 1.0], polygon) < 0.0


def test_double_support_contains_both_feet():
    footsteps = sample_footsteps()
    polygon = support_polygon(footsteps, 0.05)
    assert polygon_margin([0.0, -0.1], polygon) > 0.0
    assert polygon_margin([0.0, 0.1], polygon) > 0.0


def test_adaptive_zmp_stays_inside_polygon():
    footsteps = sample_footsteps()
    zmp = ZmpClass(
        footsteps,
        mode="adaptive",
        safety_margin=0.01,
        bias=(1.0, 1.0),
        smooth_double_support=True,
    )
    for t in [0.05, 0.3, 1.0]:
        assert zmp.margin(t) >= 0.0


def test_baseline_zmp_does_not_alias_footstep_position():
    footsteps = FootSteps(np.array([0.0, -0.1]), np.array([0.0, 0.1]))
    footsteps.add_phase(0.2, "none")
    footsteps.add_phase(0.7, "left", np.array([0.1, 0.1]))
    support_before = footsteps.get_right_position(0.3).copy()
    zmp = ZmpClass(footsteps)(0.3)
    zmp[0] = 1.0
    np.testing.assert_array_equal(footsteps.get_right_position(0.3), support_before)


if __name__ == "__main__":
    test_margin_sign()
    test_double_support_contains_both_feet()
    test_adaptive_zmp_stays_inside_polygon()
    test_baseline_zmp_does_not_alias_footstep_position()
    print("support tests passed")
