import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.gait import GaitParams, PreviewCoMClass, build_footsteps
from src.support import foot_rectangle, polygon_margin
from src.zmp import ZmpClass


def test_preview_com_is_finite():
    params = GaitParams(steps=2, com_dt=0.05)
    zmp = ZmpClass(build_footsteps(params))
    com = PreviewCoMClass(zmp, com_z_nominal=params.com_height, dt=params.com_dt)
    points = np.array([com(t) for t in np.linspace(0.0, zmp.footsteps.timetime[-1], 10)])
    assert np.all(np.isfinite(points))


def test_capture_point_margin_detects_outside_support():
    polygon = foot_rectangle([0.0, 0.0], length=0.22, width=0.12)
    assert polygon_margin([0.0, 0.0], polygon) > 0.0
    assert polygon_margin([1.0, 1.0], polygon) < 0.0


if __name__ == "__main__":
    test_preview_com_is_finite()
    test_capture_point_margin_detects_outside_support()
    print("planner mode tests passed")
