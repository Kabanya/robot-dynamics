import os
import sys
from dataclasses import replace

import numpy as np
import pinocchio as pin

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.feasibility import (
    GaitSample,
    WholeBodyTrajectory,
    _make_contacts,
    _sole_polygon_offsets,
    _solve_inverse_dynamics,
    build_whole_body_trajectory,
    compute_physics_signature,
    load_robot_spec,
)
from src.support import polygon_margin


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


def _static_double_support(spec):
    q = spec.robot.q0.copy()
    data = spec.model.createData()
    pin.forwardKinematics(spec.model, data, q)
    pin.updateFramePlacements(spec.model, data)
    com = pin.centerOfMass(spec.model, data, q)
    return WholeBodyTrajectory(
        time=np.array([0.0]),
        q=q[None],
        v=np.zeros((1, spec.model.nv)),
        a=np.zeros((1, spec.model.nv)),
        left_foot=data.oMf[spec.left_sole_frame_id].translation[None],
        right_foot=data.oMf[spec.right_sole_frame_id].translation[None],
        com=com[None],
        contact_modes=("double",),
        dt=0.1,
    )


def _assert_local_inverse_dynamics(spec, q, v, a, frame_ids, sample, static=False):
    solved_acceleration, torques, wrenches, residual, status = _solve_inverse_dynamics(
        spec, q, v, a, frame_ids, sample.friction
    )
    assert status == "optimal"
    assert np.isfinite(residual)
    if static:
        assert residual < 1e-6

    data = spec.model.createData()
    mass_matrix = pin.crba(spec.model, data, q)
    mass_matrix = np.triu(mass_matrix) + np.triu(mass_matrix, 1).T
    target = mass_matrix @ solved_acceleration + pin.nonLinearEffects(
        spec.model, data, q, v
    )
    pin.computeJointJacobians(spec.model, data, q)
    pin.updateFramePlacements(spec.model, data)
    supplied = np.r_[np.zeros(6), torques]
    for frame_id, wrench in zip(frame_ids, wrenches):
        supplied += pin.getFrameJacobian(
            spec.model, data, frame_id, pin.LOCAL_WORLD_ALIGNED
        ).T @ wrench
    effort = np.maximum(np.abs(spec.effort_limits[6:]), 1.0)
    scale = np.r_[
        np.full(3, spec.mass * 9.81),
        np.full(3, spec.mass * 9.81 * spec.leg_length),
        effort,
    ]
    reconstructed = np.max(np.abs(target - supplied) / scale)
    assert reconstructed < 1e-7

    force_tolerance = 1e-7 * spec.mass * 9.81
    moment_tolerance = force_tolerance * spec.leg_length
    length_tolerance = 1e-7 * spec.leg_length
    torsion = sample.friction * min(spec.sole_half_length, spec.sole_half_width)
    for frame_id, (fx, fy, fz, mx, my, mz) in zip(frame_ids, wrenches):
        assert fz >= -force_tolerance
        assert abs(fx) + abs(fy) <= sample.friction * fz + force_tolerance
        assert abs(mz) <= torsion * fz + moment_tolerance
        if fz > force_tolerance:
            assert polygon_margin(
                np.array([-my / fz, mx / fz]),
                _sole_polygon_offsets(
                    spec, frame_id, data.oMf[frame_id].rotation
                ),
            ) >= -length_tolerance
    if static:
        np.testing.assert_allclose(
            wrenches[:, 2].sum(), spec.mass * 9.81, rtol=1e-2, atol=force_tolerance
        )


def test_static_inverse_dynamics_returns_local_constrained_wrenches_for_each_robot():
    sample = _sample()
    for name in ("talos", "icub"):
        spec = load_robot_spec(name)
        trajectory = _static_double_support(spec)
        _assert_local_inverse_dynamics(
            spec,
            trajectory.q[0],
            trajectory.v[0],
            trajectory.a[0],
            (spec.left_sole_frame_id, spec.right_sole_frame_id),
            sample,
            static=True,
        )


def test_moving_inverse_dynamics_matches_constraint_dynamics():
    sample = _sample()
    for name in ("talos", "icub"):
        spec = load_robot_spec(name)
        trajectory = build_whole_body_trajectory(
            spec, sample, steps=1, dt=0.01
        )
        start = next(
            index for index, mode in enumerate(trajectory.contact_modes)
            if mode in ("left", "right")
        )
        mode = trajectory.contact_modes[start]
        end = start
        while end < len(trajectory.time) and trajectory.contact_modes[end] == mode:
            end += 1
        index = start + (end - start) // 2
        q, v, desired = (
            trajectory.q[index],
            trajectory.v[index],
            trajectory.a[index],
        )
        frame_ids = (
            (spec.left_sole_frame_id,)
            if mode == "left"
            else (spec.right_sole_frame_id,)
        )
        placement_data = spec.model.createData()
        pin.forwardKinematics(spec.model, placement_data, q)
        pin.updateFramePlacements(spec.model, placement_data)
        targets = {
            frame_id: placement_data.oMf[frame_id].copy()
            for frame_id in frame_ids
        }
        for target in targets.values():
            target.translation[0] += 1e-4 * spec.leg_length

        solved, torque, expected_wrenches, _, status = _solve_inverse_dynamics(
            spec,
            q,
            v,
            desired,
            frame_ids,
            sample.friction,
            contact_targets=targets,
        )
        assert status == "optimal"
        assert np.all(np.abs(torque) <= np.abs(spec.effort_limits[6:]) + 1e-10)

        data = spec.model.createData()
        _, contact_models, contact_datas = _make_contacts(
            spec, spec.model, data, q, mode, targets
        )
        actual = np.asarray(pin.constraintDynamics(
            spec.model,
            data,
            q,
            v,
            np.r_[np.zeros(6), torque],
            contact_models,
            contact_datas,
            pin.ProximalSettings(1e-12, 1e-12, 1e-10, 20),
        ))
        actual_wrenches = np.asarray([
            contact_data.contact_force.vector for contact_data in contact_datas
        ])
        acceleration_scale = np.r_[
            np.full(3, 9.81),
            np.full(spec.model.nv - 3, 9.81 / spec.leg_length),
        ]
        wrench_scale = np.tile(
            np.r_[
                np.full(3, spec.mass * 9.81),
                np.full(3, spec.mass * 9.81 * spec.leg_length),
            ],
            (len(frame_ids), 1),
        )
        assert np.max(np.abs(actual - solved) / acceleration_scale) < 1e-7
        assert np.max(
            np.abs(actual_wrenches - expected_wrenches) / wrench_scale
        ) < 1e-7


def test_icub_generated_touchdown_has_local_constrained_wrenches():
    sample = _sample()
    spec = load_robot_spec("icub")
    trajectory = build_whole_body_trajectory(spec, sample, steps=1, dt=0.1)
    index = trajectory.contact_modes.index("touchdown")
    _assert_local_inverse_dynamics(
        spec,
        trajectory.q[index],
        trajectory.v[index],
        trajectory.a[index],
        (spec.left_sole_frame_id, spec.right_sole_frame_id),
        sample,
    )


def test_static_double_support_has_negligible_normalized_dynamics_slack():
    spec = load_robot_spec("talos")
    signature = compute_physics_signature(spec, _static_double_support(spec), _sample())
    assert signature.solver_status == ("optimal",)
    assert signature.dynamics_slack[0] < 1e-6


def test_low_friction_extreme_horizontal_acceleration_has_explicit_slack():
    spec = load_robot_spec("talos")
    trajectory = _static_double_support(spec)
    acceleration = trajectory.a.copy()
    acceleration[0, 0] = 10.0 * 9.81
    trajectory = replace(trajectory, a=acceleration)
    sample = replace(
        _sample(),
        friction=0.25,
        payload_fraction=0.10,
        timing_error_seconds=0.02,
        impulse=0.04,
        ood=True,
    )
    signature = compute_physics_signature(spec, trajectory, sample)
    assert signature.solver_status == ("optimal",)
    assert np.isfinite(signature.dynamics_slack[0])
    assert signature.dynamics_slack[0] > 1.0


def test_touchdown_feature_uses_preimpact_speed_at_shifted_timing():
    spec = load_robot_spec("talos")
    trajectory = build_whole_body_trajectory(
        spec, _sample(), steps=1, dt=0.01
    )
    early = compute_physics_signature(
        spec,
        trajectory,
        replace(_sample(), timing_error_seconds=-0.02),
    )
    feature = early.feature_names.index("touchdown.impact_proxy.max")
    assert early.raw_values[feature] > 1e-5


def test_signature_features_are_deterministic_and_finite():
    spec = load_robot_spec("talos")
    trajectory = build_whole_body_trajectory(spec, _sample(), steps=1, dt=0.1)
    first = compute_physics_signature(spec, trajectory, _sample())
    second = compute_physics_signature(spec, trajectory, _sample())
    assert first.feature_names == second.feature_names
    np.testing.assert_array_equal(first.values, second.values)
    np.testing.assert_array_equal(first.raw_values, second.raw_values)
    assert first.raw_values.shape == first.values.shape
    assert np.isfinite(first.values).all()
    assert np.isfinite(first.raw_values).all()
    assert np.isfinite(first.dynamics_slack).all()
    expected_names = tuple(
        f"{phase}.{channel}.{statistic}"
        for phase in ("single_support", "touchdown", "double_support")
        for channel in (
            "torque",
            "friction",
            "cop",
            "zmp",
            "joint_position",
            "joint_velocity",
            "ik_residual",
            "dynamics_slack",
            "impact_proxy",
            "solver_failure",
        )
        for statistic in ("min", "p05", "median", "p95", "max")
    )
    assert first.feature_names == expected_names
    assert not any("moment" in name for name in first.feature_names)


if __name__ == "__main__":
    test_static_inverse_dynamics_returns_local_constrained_wrenches_for_each_robot()
    test_moving_inverse_dynamics_matches_constraint_dynamics()
    test_icub_generated_touchdown_has_local_constrained_wrenches()
    test_static_double_support_has_negligible_normalized_dynamics_slack()
    test_low_friction_extreme_horizontal_acceleration_has_explicit_slack()
    test_touchdown_feature_uses_preimpact_speed_at_shifted_timing()
    test_signature_features_are_deterministic_and_finite()
    print("physics signature tests passed")
