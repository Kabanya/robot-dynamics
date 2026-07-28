import math
import os
import sys

import example_robot_data
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.feasibility import (
    GaitSample,
    PhysicsSignature,
    RolloutResult,
    WholeBodyTrajectory,
    _inset_polygon,
    _polygon_centroid,
    _sole_polygon_offsets,
    load_robot_spec,
)
from src.support import polygon_margin


def test_support_polygon_inset_is_strictly_inside_the_mesh_projection():
    raw = np.array([
        [-0.10, -0.05],
        [0.10, -0.05],
        [0.10, 0.05],
        [-0.10, 0.05],
    ])
    inset = _inset_polygon(raw, 0.002)
    for point in inset:
        assert math.isclose(
            polygon_margin(point, raw),
            0.002,
            rel_tol=0.0,
            abs_tol=1e-12,
        )


def test_robot_specs_use_model_mass_frames_and_verified_sole_sizes():
    expected = {
        "talos": (
            "left_sole_link", "right_sole_link", "torso_2_link",
            ((0.095, 0.12), (0.06, 0.07)),
        ),
        "icub": (
            "l_sole", "r_sole", "chest",
            ((0.07, 0.09), (0.025, 0.04)),
        ),
    }
    for name, (left, right, torso, size_ranges) in expected.items():
        spec = load_robot_spec(name)
        robot = example_robot_data.load(
            "icub_reduced" if name == "icub" else name
        )
        assert spec.name == name
        assert spec.left_sole_frame == left
        assert spec.right_sole_frame == right
        assert spec.torso_frame == torso
        assert spec.mass == sum(inertia.mass for inertia in robot.model.inertias)
        assert size_ranges[0][0] <= spec.sole_half_length <= size_ranges[0][1]
        assert size_ranges[1][0] <= spec.sole_half_width <= size_ranges[1][1]
        for frame_id in (spec.left_sole_frame_id, spec.right_sole_frame_id):
            polygon = _sole_polygon_offsets(spec, frame_id)
            assert polygon.ndim == 2 and polygon.shape[0] > 4
            assert polygon.shape[1] == 2
            assert np.isfinite(polygon).all()
        for rotation in (spec.left_sole_rotation, spec.right_sole_rotation):
            np.testing.assert_allclose(rotation[:, 2], (0.0, 0.0, 1.0), atol=1e-10)
        assert spec.leg_length > 0.0
        assert spec.neutral_com_height > 0.0
        assert spec.collision_model.ngeoms == robot.collision_model.ngeoms
    talos = load_robot_spec("talos")
    np.testing.assert_allclose(
        np.ptp(
            _sole_polygon_offsets(talos, talos.left_sole_frame_id), axis=0
        ),
        (0.19767, 0.12013),
        atol=5e-5,
    )
    icub = load_robot_spec("icub")
    icub_polygon = _sole_polygon_offsets(icub, icub.left_sole_frame_id)
    np.testing.assert_allclose(
        np.ptp(icub_polygon, axis=0), (0.15583, 0.06071), atol=5e-5
    )
    centroid = _polygon_centroid(icub_polygon)
    assert 0.047 < centroid[0] < 0.049
    assert abs(centroid[1]) < 0.002
    reloaded = load_robot_spec("icub")
    np.testing.assert_array_equal(
        icub_polygon,
        _sole_polygon_offsets(reloaded, reloaded.left_sole_frame_id),
    )


def test_icub_uses_the_official_reduced_model_with_locked_neck():
    spec = load_robot_spec("icub")
    reduced = example_robot_data.load("icub_reduced")
    assert spec.model.nq == reduced.model.nq
    assert spec.model.nv == reduced.model.nv
    assert spec.mass == sum(inertia.mass for inertia in reduced.model.inertias)
    for joint_name in ("neck_pitch", "neck_roll", "neck_yaw"):
        assert not spec.model.existJointName(joint_name)


def test_robot_spec_natural_time_is_leg_length_over_gravity():
    spec = load_robot_spec("talos")
    assert spec.natural_time == math.sqrt(spec.leg_length / 9.81)


def test_gait_sample_dimensionalizes_for_each_robot():
    sample = GaitSample(
        step_length=0.25,
        step_width=1.0,
        single_support_duration=2.0,
        double_support_duration=0.5,
        com_height_scale=1.0,
        zmp_bias_x=0.25,
        zmp_bias_y=-0.25,
        friction=0.6,
        payload_fraction=0.05,
        timing_error_seconds=0.0,
        impulse=0.02,
        seed=7,
    )
    for name in ("talos", "icub"):
        spec = load_robot_spec(name)
        params = sample.to_gait_params(spec, steps=4, dt=0.01)
        assert params.step_length == sample.step_length * spec.leg_length
        assert params.step_width == sample.step_width * spec.neutral_step_width
        assert params.single_support_duration == sample.single_support_duration * spec.natural_time
        assert params.double_support_duration == sample.double_support_duration * spec.natural_time
        assert params.com_height == spec.neutral_com_height
        assert params.zmp_bias_x == sample.zmp_bias_x * spec.sole_half_length
        assert params.zmp_bias_y == sample.zmp_bias_y * spec.sole_half_width
        assert params.foot_length == 2.0 * spec.sole_half_length
        assert params.foot_width == 2.0 * spec.sole_half_width
        assert params.steps == 4
        assert params.com_dt == params.eval_dt == 0.01


def test_gait_sample_rejects_outside_id_and_ood_domains():
    base = dict(
        step_length=0.25,
        step_width=1.0,
        single_support_duration=2.0,
        double_support_duration=0.5,
        com_height_scale=1.0,
        zmp_bias_x=0.0,
        zmp_bias_y=0.0,
        friction=0.6,
        payload_fraction=0.05,
        timing_error_seconds=0.0,
        impulse=0.02,
        seed=1,
    )
    for changes in (
        {"step_length": 0.09},
        {"step_width": 0.84},
        {"friction": 0.39},
        {"payload_fraction": 0.11},
        {"timing_error_seconds": 0.03},
        {"impulse": 0.05},
        {"ood": True, "friction": 0.5},
        {"ood": True, "timing_error_seconds": 0.0},
        {"ood": True, "impulse": 0.03},
    ):
        try:
            GaitSample(**(base | changes))
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid sample: {changes}")


def _assert_array_is_immutable(array):
    try:
        array.flat[0] = 0.0
    except ValueError:
        return
    raise AssertionError("array mutation did not raise ValueError")


def test_result_dataclasses_freeze_array_data_and_normalize_sequences():
    time = np.array([1.0])
    trajectory = WholeBodyTrajectory(
        time=time,
        q=np.array([[1.0]]),
        v=np.array([[1.0]]),
        a=np.array([[1.0]]),
        left_foot=np.array([[1.0]]),
        right_foot=np.array([[1.0]]),
        com=np.array([[1.0]]),
        contact_modes=["double"],
        dt=0.01,
    )
    signature = PhysicsSignature(
        feature_names=["residual"],
        values=np.array([1.0]),
        dynamics_slack=np.array([1.0]),
        solver_status=["ok"],
    )
    result = RolloutResult(
        time=np.array([1.0]),
        q=np.array([[1.0]]),
        v=np.array([[1.0]]),
        contact_wrenches=np.array([[1.0]]),
        success=True,
        failure_reason="",
        active_contacts=np.array([[True, False]]),
        scheduled_contacts=np.array([[True, True]]),
    )
    time[0] = 2.0
    assert trajectory.time[0] == 1.0
    for array in (
        trajectory.time, trajectory.q, trajectory.v, trajectory.a,
        trajectory.left_foot, trajectory.right_foot, trajectory.com,
        signature.values, signature.dynamics_slack,
        result.time, result.q, result.v, result.contact_wrenches,
        result.active_contacts, result.scheduled_contacts,
    ):
        _assert_array_is_immutable(array)
    assert trajectory.contact_modes == ("double",)
    assert signature.feature_names == ("residual",)
    assert signature.solver_status == ("ok",)


if __name__ == "__main__":
    test_robot_specs_use_model_mass_frames_and_verified_sole_sizes()
    test_robot_spec_natural_time_is_leg_length_over_gravity()
    test_gait_sample_dimensionalizes_for_each_robot()
    test_gait_sample_rejects_outside_id_and_ood_domains()
    test_result_dataclasses_freeze_array_data_and_normalize_sequences()
    print("feasibility tests passed")
