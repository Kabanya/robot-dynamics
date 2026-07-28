import os
import sys
from dataclasses import replace

import numpy as np
import pinocchio as pin

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.feasibility import GaitSample, build_whole_body_trajectory, load_robot_spec


def _sample():
    return GaitSample(
        step_length=0.20,
        step_width=1.0,
        single_support_duration=1.4,
        double_support_duration=0.5,
        com_height_scale=0.95,
        zmp_bias_x=0.0,
        zmp_bias_y=0.0,
        friction=0.6,
        payload_fraction=0.0,
        timing_error_seconds=0.0,
        impulse=0.0,
        seed=3,
    )


def test_whole_body_trajectory_is_finite_bounded_and_on_each_robot_manifold():
    for name in ("talos", "icub"):
        spec = load_robot_spec(name)
        trajectory = build_whole_body_trajectory(spec, _sample(), steps=1, dt=0.1)
        assert trajectory.q.shape == (len(trajectory.time), spec.model.nq)
        assert trajectory.v.shape == trajectory.a.shape == (len(trajectory.time), spec.model.nv)
        for values in (trajectory.q, trajectory.v, trajectory.a):
            assert np.isfinite(values).all()
            assert not values.flags.writeable
        assert all(pin.isNormalized(spec.model, q) for q in trajectory.q)
        for joint in spec.model.joints[2:]:
            if joint.nq != joint.nv:
                continue
            indices = slice(joint.idx_q, joint.idx_q + joint.nq)
            assert np.all(trajectory.q[:, indices] >= spec.position_lower_limits[indices])
            assert np.all(trajectory.q[:, indices] <= spec.position_upper_limits[indices])
            lower = spec.position_lower_limits[indices]
            upper = spec.position_upper_limits[indices]
            finite = np.isfinite(lower) & np.isfinite(upper) & (upper > lower)
            original = spec.robot.q0[indices]
            at_bound = finite & (
                np.isclose(original, lower) | np.isclose(original, upper)
            )
            if np.any(at_bound):
                margin = 0.01 * (upper[at_bound] - lower[at_bound])
                assert np.all(trajectory.q[0, indices][at_bound] > lower[at_bound] + margin)
                assert np.all(trajectory.q[0, indices][at_bound] < upper[at_bound] - margin)


def test_whole_body_states_track_feet_com_and_upright_torso():
    for name in ("talos", "icub"):
        spec = load_robot_spec(name)
        trajectory = build_whole_body_trajectory(spec, _sample(), steps=1, dt=0.1)
        data = spec.model.createData()
        pin.forwardKinematics(spec.model, data, spec.robot.q0)
        pin.updateFramePlacements(spec.model, data)
        torso_rotation = data.oMf[spec.torso_frame_id].rotation.copy()
        for q, left, right, com in zip(
            trajectory.q, trajectory.left_foot, trajectory.right_foot, trajectory.com
        ):
            pin.forwardKinematics(spec.model, data, q)
            pin.updateFramePlacements(spec.model, data)
            actual_com = pin.centerOfMass(spec.model, data, q)
            np.testing.assert_allclose(
                data.oMf[spec.left_sole_frame_id].translation, left, atol=1e-4
            )
            np.testing.assert_allclose(
                data.oMf[spec.right_sole_frame_id].translation, right, atol=1e-4
            )
            np.testing.assert_allclose(actual_com, com, atol=1e-4)
            assert np.linalg.norm(
                pin.log3(torso_rotation @ data.oMf[spec.torso_frame_id].rotation.T)
            ) < 1e-4


def test_velocity_and_acceleration_project_manifold_finite_differences():
    spec = load_robot_spec("talos")
    sample = replace(
        _sample(), step_length=0.4, step_width=0.85, double_support_duration=0.2
    )
    trajectory = build_whole_body_trajectory(spec, sample, steps=1, dt=0.05)
    expected_velocity = np.asarray([
        pin.difference(spec.model, trajectory.q[index - 1], trajectory.q[index])
        / trajectory.dt
        for index in range(1, len(trajectory.time))
    ])
    corrections = trajectory.v[1:] - expected_velocity
    assert np.all(
        np.linalg.norm(corrections, axis=1)
        <= np.linalg.norm(expected_velocity, axis=1) + 1e-10
    )
    assert np.linalg.norm(trajectory.v) > 0.0
    assert np.linalg.norm(trajectory.a) > 0.0


def test_unreachable_ik_target_still_returns_a_valid_oracle_input():
    spec = load_robot_spec("icub")
    trajectory = build_whole_body_trajectory(
        spec,
        replace(_sample(), com_height_scale=1.05),
        steps=1,
        dt=0.01,
    )
    assert np.isfinite(trajectory.q).all()
    assert all(pin.isNormalized(spec.model, q) for q in trajectory.q)
    data = spec.model.createData()
    achieved = np.asarray([
        pin.centerOfMass(spec.model, data, q) for q in trajectory.q
    ])
    assert np.max(np.linalg.norm(achieved - trajectory.com, axis=1)) > 1e-3


def test_active_contact_reference_kinematics_are_constrained():
    for name in ("talos", "icub"):
        spec = load_robot_spec(name)
        trajectory = build_whole_body_trajectory(
            spec, _sample(), steps=1, dt=0.01
        )
        data = spec.model.createData()
        for q, v, a, mode in zip(
            trajectory.q,
            trajectory.v,
            trajectory.a,
            trajectory.contact_modes,
        ):
            frame_ids = {
                "left": (spec.left_sole_frame_id,),
                "right": (spec.right_sole_frame_id,),
                "touchdown": (
                    spec.left_sole_frame_id,
                    spec.right_sole_frame_id,
                ),
                "double": (
                    spec.left_sole_frame_id,
                    spec.right_sole_frame_id,
                ),
            }[mode]
            pin.forwardKinematics(spec.model, data, q, v, a)
            pin.updateFramePlacements(spec.model, data)
            for frame_id in frame_ids:
                assert np.linalg.norm(pin.getFrameVelocity(
                    spec.model, data, frame_id, pin.LOCAL_WORLD_ALIGNED
                ).vector) < 1e-8
                assert np.linalg.norm(pin.getFrameAcceleration(
                    spec.model, data, frame_id, pin.LOCAL_WORLD_ALIGNED
                ).vector) < 1e-7


def test_contact_modes_name_active_support_and_mark_touchdown():
    spec = load_robot_spec("talos")
    trajectory = build_whole_body_trajectory(spec, _sample(), steps=1, dt=0.1)
    assert "right" in trajectory.contact_modes  # legacy "left" phase is left swing
    touchdown = trajectory.contact_modes.index("touchdown")
    assert trajectory.contact_modes[touchdown - 1] == "right"
    assert trajectory.contact_modes.count("touchdown") == 1
    assert set(trajectory.contact_modes[touchdown + 1:]) <= {"double"}


def test_swing_height_scales_with_leg_length():
    for name in ("talos", "icub"):
        spec = load_robot_spec(name)
        trajectory = build_whole_body_trajectory(spec, _sample(), steps=1, dt=0.05)
        ground = min(trajectory.left_foot[:, 2].min(), trajectory.right_foot[:, 2].min())
        clearance = max(trajectory.left_foot[:, 2].max(), trajectory.right_foot[:, 2].max()) - ground
        assert 0.07 * spec.leg_length < clearance <= 0.08 * spec.leg_length + 1e-12


if __name__ == "__main__":
    test_whole_body_trajectory_is_finite_bounded_and_on_each_robot_manifold()
    test_whole_body_states_track_feet_com_and_upright_torso()
    test_velocity_and_acceleration_project_manifold_finite_differences()
    test_unreachable_ik_target_still_returns_a_valid_oracle_input()
    test_active_contact_reference_kinematics_are_constrained()
    test_contact_modes_name_active_support_and_mark_touchdown()
    test_swing_height_scales_with_leg_length()
    print("whole-body trajectory tests passed")
