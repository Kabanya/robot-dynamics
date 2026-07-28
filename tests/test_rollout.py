import os
import sys
from dataclasses import replace

import numpy as np
import pinocchio as pin

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import src.feasibility as feasibility
from src.feasibility import (
    GaitSample,
    WholeBodyTrajectory,
    _plant_model,
    _shift_contact_modes,
    build_whole_body_trajectory,
    load_robot_spec,
    rollout,
)


def _sample(**changes):
    values = dict(
        step_length=0.20,
        step_width=1.0,
        single_support_duration=1.4,
        double_support_duration=0.5,
        com_height_scale=1.0,
        zmp_bias_x=0.0,
        zmp_bias_y=0.0,
        friction=0.8,
        payload_fraction=0.0,
        timing_error_seconds=0.0,
        impulse=0.0,
        seed=11,
    )
    values.update(changes)
    return GaitSample(**values)


def _trajectory(spec, modes=("double",), dt=0.01, q=None):
    q = spec.robot.q0.copy() if q is None else np.asarray(q)
    data = spec.model.createData()
    pin.forwardKinematics(spec.model, data, q)
    pin.updateFramePlacements(spec.model, data)
    com = pin.centerOfMass(spec.model, data, q)
    count = len(modes)
    return WholeBodyTrajectory(
        time=np.arange(count) * dt,
        q=np.repeat(q[None], count, axis=0),
        v=np.zeros((count, spec.model.nv)),
        a=np.zeros((count, spec.model.nv)),
        left_foot=np.repeat(
            data.oMf[spec.left_sole_frame_id].translation[None], count, axis=0
        ),
        right_foot=np.repeat(
            data.oMf[spec.right_sole_frame_id].translation[None], count, axis=0
        ),
        com=np.repeat(com[None], count, axis=0),
        contact_modes=modes,
        dt=dt,
    )


def _constraint_wrenches_and_acceleration(spec, q, v, applied_torque, mode):
    model = spec.model
    data = model.createData()
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    frame_ids = {
        "left": (spec.left_sole_frame_id,),
        "right": (spec.right_sole_frame_id,),
        "double": (spec.left_sole_frame_id, spec.right_sole_frame_id),
        "touchdown": (spec.left_sole_frame_id, spec.right_sole_frame_id),
    }[mode]
    contact_models = pin.StdVec_RigidConstraintModel()
    contact_datas = pin.StdVec_RigidConstraintData()
    for frame_id in frame_ids:
        frame = model.frames[frame_id]
        contact = pin.RigidConstraintModel(
            pin.ContactType.CONTACT_6D,
            model,
            frame.parentJoint,
            frame.placement,
            0,
            data.oMf[frame_id].copy(),
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        contact_models.append(contact)
        contact_datas.append(contact.createData())
    pin.initConstraintDynamics(model, data, contact_models, contact_datas)
    tau = np.r_[np.zeros(6), applied_torque]
    acceleration = pin.constraintDynamics(
        model,
        data,
        q,
        v,
        tau,
        contact_models,
        contact_datas,
        pin.ProximalSettings(1e-12, 1e-12, 1e-10, 20),
    )
    return acceleration, frame_ids, np.asarray(
        [contact_data.contact_force.vector for contact_data in contact_datas]
    )


def test_rollout_is_deterministic_and_reports_fixed_left_right_local_wrenches():
    spec = load_robot_spec("talos")
    trajectory = _trajectory(spec, ("double", "double"), dt=0.002)
    first = rollout(spec, trajectory, _sample())
    second = rollout(spec, trajectory, _sample())
    for name in (
        "q",
        "v",
        "contact_wrenches",
        "torque_demand",
        "applied_torque",
        "normal_force_margin",
        "friction_margin",
        "cop_margin",
        "active_contacts",
        "scheduled_contacts",
    ):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
    assert first.success == second.success
    assert first.failure_reason == second.failure_reason
    assert first.failure_index == second.failure_index
    assert first.contact_wrenches.shape == (len(trajectory.time), 2, 6)
    assert first.active_contacts.shape == (len(trajectory.time), 2)
    assert first.scheduled_contacts.shape == (len(trajectory.time), 2)


def test_static_double_support_remains_stable_for_multiple_steps():
    for name in ("talos", "icub"):
        spec = load_robot_spec(name)
        # Seven seconds exceeds the longest six-step protocol trajectory.
        result = rollout(spec, _trajectory(spec, ("double",) * 700), _sample())
        assert result.success, (name, result.failure_reason, result.failure_index)
        assert np.max(np.abs(result.v)) < 2e-3
        assert np.max(
            np.abs(result.torque_demand) / spec.effort_limits[None, 6:]
        ) <= 1.01


def test_generated_conservative_six_step_rollout_succeeds_for_each_robot():
    sample = _sample(
        step_length=0.10,
        single_support_duration=2.8,
        double_support_duration=0.8,
        com_height_scale=0.90,
        zmp_bias_y=-0.25,
        impulse=0.01,
    )
    for name in ("talos", "icub"):
        spec = load_robot_spec(name)
        trajectory = build_whole_body_trajectory(
            spec, sample, steps=6, dt=0.01
        )
        result = rollout(spec, trajectory, sample)
        assert result.success, (name, result.failure_reason, result.failure_index)
        assert np.all(np.sum(result.active_contacts, axis=1) >= 1)
        assert np.all(np.sum(result.scheduled_contacts, axis=1) >= 1)
        data = spec.model.createData()
        pin.framesForwardKinematics(spec.model, data, spec.robot.q0)
        ground = np.mean([
            data.oMf[spec.left_sole_frame_id].translation[2],
            data.oMf[spec.right_sole_frame_id].translation[2],
        ])
        for q in result.q:
            pin.framesForwardKinematics(spec.model, data, q)
            assert min(
                data.oMf[spec.left_sole_frame_id].translation[2],
                data.oMf[spec.right_sole_frame_id].translation[2],
            ) >= ground - 1e-3 * spec.leg_length - 1e-12


def test_icub_rollout_accepts_cop_error_below_conservative_footprint_inset():
    spec = load_robot_spec("icub")
    sample = _sample(
        step_length=0.10,
        single_support_duration=2.8,
        double_support_duration=0.8,
        com_height_scale=0.90,
        zmp_bias_y=-0.25,
        payload_fraction=0.001,
    )
    result = rollout(
        spec,
        build_whole_body_trajectory(spec, sample, steps=6, dt=0.01),
        sample,
    )
    assert result.success, (result.failure_reason, result.failure_index)


def test_unilateral_liftoff_releases_a_separating_unloaded_contact():
    spec = load_robot_spec("icub")
    sample = GaitSample(
        step_length=0.10567180172540247,
        step_width=0.9811656006425619,
        single_support_duration=2.7938346873410045,
        double_support_duration=0.7947817261703313,
        com_height_scale=0.9118938408792019,
        zmp_bias_x=0.015858493894338608,
        zmp_bias_y=-0.23332611077465118,
        friction=0.6956493325531483,
        payload_fraction=0.02193792061880231,
        timing_error_seconds=0.019594247713685033,
        impulse=0.007934561222791672,
        seed=923736807,
    )
    trajectory = build_whole_body_trajectory(
        spec, sample, steps=2, dt=0.01
    )
    result = rollout(
        spec,
        trajectory,
        sample,
    )
    repeated = rollout(spec, trajectory, sample)

    assert result.success, (result.failure_reason, result.failure_index)
    np.testing.assert_array_equal(result.scheduled_contacts[19], (True, True))
    np.testing.assert_array_equal(result.active_contacts[19], (False, True))
    np.testing.assert_array_equal(
        result.active_contacts, repeated.active_contacts
    )
    np.testing.assert_array_equal(
        result.contact_wrenches, repeated.contact_wrenches
    )
    assert np.any(np.all(result.active_contacts[20:], axis=1))

    plant = _plant_model(spec, sample)
    acceleration = (result.v[20] - result.v[19]) / trajectory.dt
    tau = np.r_[np.zeros(6), result.applied_torque[19]]
    assert feasibility._dynamics_are_consistent(
        plant,
        plant.createData(),
        result.q[19],
        result.v[19],
        acceleration,
        tau,
        (spec.right_sole_frame_id,),
        result.contact_wrenches[19, 1:2],
    )
    force_tolerance = 1e-6 * sum(
        inertia.mass for inertia in plant.inertias
    ) * 9.81
    assert result.normal_force_margin[19, 1] >= -force_tolerance
    assert result.friction_margin[19, 1] >= -force_tolerance
    assert (
        result.cop_margin[19, 1]
        >= -feasibility._CONTINUOUS_COP_TOLERANCE_RATIO * spec.leg_length
    )

    data = plant.createData()
    pin.forwardKinematics(
        plant, data, result.q[19], result.v[19], acceleration
    )
    pin.updateFramePlacements(plant, data)
    normal_velocity = pin.getFrameVelocity(
        plant, data, spec.left_sole_frame_id, pin.LOCAL_WORLD_ALIGNED
    ).linear[2]
    normal_acceleration = pin.getFrameClassicalAcceleration(
        plant, data, spec.left_sole_frame_id, pin.LOCAL_WORLD_ALIGNED
    ).linear[2]
    assert (
        normal_velocity + trajectory.dt * normal_acceleration
        > feasibility._IMPACT_VELOCITY_TOLERANCE_RATIO
        * np.sqrt(9.81 * spec.leg_length)
    )
    gaps = []
    for q in result.q[19:21]:
        pin.framesForwardKinematics(plant, data, q)
        gaps.append(
            data.oMf[spec.left_sole_frame_id].translation[2]
            - trajectory.left_foot[19, 2]
        )
    assert min(gaps) >= -1e-3 * spec.leg_length
    assert (
        gaps[1]
        >= gaps[0]
        - feasibility._CONTINUOUS_LIFTOFF_GAP_TOLERANCE_RATIO
        * spec.leg_length
    )


def test_icub_impact_tolerance_accepts_small_payload_mismatch_with_sliding():
    spec = load_robot_spec("icub")
    assert feasibility._IMPACT_VELOCITY_TOLERANCE_RATIO == 1e-4
    assert (
        feasibility._IMPACT_VELOCITY_TOLERANCE_RATIO
        * np.sqrt(9.81 * spec.leg_length) * 0.01
        < 1e-4
    )
    base = _sample(
        step_length=0.10,
        single_support_duration=2.8,
        double_support_duration=0.8,
        com_height_scale=0.90,
        zmp_bias_y=-0.25,
    )
    results = []
    for payload_fraction in (0.01, 0.02):
        sample = replace(base, payload_fraction=payload_fraction)
        results.append(rollout(
            spec,
            build_whole_body_trajectory(spec, sample, steps=6, dt=0.01),
            sample,
        ))
    assert all(result.success for result in results), [
        (result.failure_reason, result.failure_index) for result in results
    ]


def test_effort_saturation_does_not_preempt_later_physical_failure():
    spec = load_robot_spec("icub")
    sample = _sample(
        step_length=0.10,
        single_support_duration=2.8,
        double_support_duration=0.8,
        com_height_scale=0.90,
        zmp_bias_y=-0.25,
        payload_fraction=0.05,
    )
    result = rollout(
        spec,
        build_whole_body_trajectory(spec, sample, steps=6, dt=0.01),
        sample,
    )
    assert result.failure_reason == "normal_force"
    assert result.failure_index > 3
    assert np.max(
        np.abs(result.torque_demand) / spec.effort_limits[None, 6:]
    ) > 1.01
    assert np.max(
        np.abs(result.applied_torque) / spec.effort_limits[None, 6:]
    ) <= 1.0 + 1e-12


def test_unreachable_scheduled_contact_is_classified_as_tracking():
    spec = load_robot_spec("icub")
    sample = _sample(
        step_length=0.10,
        single_support_duration=2.8,
        double_support_duration=0.8,
        com_height_scale=1.05,
        zmp_bias_y=-0.25,
    )
    result = rollout(
        spec,
        build_whole_body_trajectory(spec, sample, steps=6, dt=0.01),
        sample,
    )
    assert result.failure_reason == "tracking"
    assert result.failure_index == 0


def test_returned_world_aligned_wrenches_reconstruct_constraint_dynamics():
    spec = load_robot_spec("talos")
    trajectory = _trajectory(spec)
    result = rollout(spec, trajectory, _sample())
    acceleration, frame_ids, expected = _constraint_wrenches_and_acceleration(
        spec, result.q[0], result.v[0], result.applied_torque[0], "double"
    )
    np.testing.assert_allclose(result.contact_wrenches[0], expected, atol=1e-10)

    data = spec.model.createData()
    mass = pin.crba(spec.model, data, result.q[0])
    mass = np.triu(mass) + np.triu(mass, 1).T
    nonlinear = pin.nonLinearEffects(spec.model, data, result.q[0], result.v[0])
    pin.computeJointJacobians(spec.model, data, result.q[0])
    pin.updateFramePlacements(spec.model, data)
    supplied = np.r_[np.zeros(6), result.applied_torque[0]]
    for frame_id, wrench in zip(frame_ids, result.contact_wrenches[0]):
        supplied += pin.getFrameJacobian(
            spec.model, data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        ).T @ wrench
    np.testing.assert_allclose(mass @ acceleration + nonlinear, supplied, atol=1e-8)


def test_impact_solver_is_used_for_touchdown_but_not_external_pushes():
    spec = load_robot_spec("talos")
    original = feasibility.pin.impulseDynamics
    calls = []

    def recording_impulse(*args, **kwargs):
        calls.append(np.asarray(args[3]).copy())
        return original(*args, **kwargs)

    feasibility.pin.impulseDynamics = recording_impulse
    try:
        rollout(spec, _trajectory(spec, ("touchdown",)), _sample())
        assert len(calls) == 3
        calls.clear()
        wide_support = replace(spec, sole_half_length=10.0, sole_half_width=10.0)
        rollout(
            wide_support,
            _trajectory(spec, ("right",) * 5, dt=0.001),
            _sample(impulse=0.02),
        )
        assert not calls
    finally:
        feasibility.pin.impulseDynamics = original


def test_external_push_has_exact_one_step_momentum_integral():
    spec = load_robot_spec("talos")
    for dt in (0.001, 0.002):
        sample = _sample(payload_fraction=0.10, impulse=1e-4)
        trajectory = _trajectory(spec, ("right",) * 5, dt=dt)
        original = feasibility.pin.constraintDynamics

        def recorded_taus(run_sample):
            calls = []

            def recording(*args, **kwargs):
                calls.append(np.asarray(args[4]).copy())
                return original(*args, **kwargs)

            feasibility.pin.constraintDynamics = recording
            try:
                result = rollout(spec, trajectory, run_sample)
            finally:
                feasibility.pin.constraintDynamics = original
            return result, calls

        baseline, baseline_taus = recorded_taus(replace(sample, impulse=0.0))
        pushed, pushed_taus = recorded_taus(sample)
        assert baseline.success and pushed.success
        midpoint = 2
        plant = _plant_model(spec, sample)
        com_jacobian = pin.jacobianCenterOfMass(
            plant, plant.createData(), pushed.q[midpoint]
        )
        angle = np.random.default_rng(sample.seed).uniform(0.0, 2.0 * np.pi)
        direction = np.array([np.cos(angle), np.sin(angle), 0.0])
        plant_mass = sum(inertia.mass for inertia in plant.inertias)
        expected = com_jacobian.T @ (
            plant_mass * np.sqrt(9.81 * spec.leg_length)
            * sample.impulse / dt * direction
        )
        np.testing.assert_allclose(
            pushed_taus[midpoint] - baseline_taus[midpoint], expected,
            rtol=1e-11, atol=1e-11,
        )
        for index, tau in enumerate(pushed_taus):
            if index != midpoint:
                np.testing.assert_array_equal(tau[:6], np.zeros(6))


def test_touchdown_checks_every_active_contact_impulse():
    spec = load_robot_spec("talos")
    sample = _sample(
        step_length=0.10,
        single_support_duration=2.8,
        double_support_duration=0.8,
        com_height_scale=0.95,
    )
    original = feasibility._resolve_impact
    checked = []

    def recording_resolve(
        spec,
        sample,
        model,
        q,
        v,
        frame_ids,
    ):
        checked.append(tuple(frame_ids))
        return original(
            spec,
            sample,
            model,
            q,
            v,
            frame_ids,
        )

    feasibility._resolve_impact = recording_resolve
    try:
        result = rollout(
            spec,
            build_whole_body_trajectory(spec, sample, steps=1, dt=0.01),
            sample,
        )
    finally:
        feasibility._resolve_impact = original
    assert result.success, (result.failure_reason, result.failure_index)
    assert checked
    assert set(checked) == {(
        spec.left_sole_frame_id,
        spec.right_sole_frame_id,
    )}


def test_contact_mode_changes_preserve_anchors_for_continuing_supports():
    spec = load_robot_spec("talos")
    sample = _sample(
        step_length=0.10,
        single_support_duration=2.8,
        double_support_duration=0.8,
        com_height_scale=0.90,
    )
    original = feasibility._make_frame_contacts
    records = []

    def recording_make(
        spec, model, data, q, frame_ids, anchors=None, height_tolerance=None
    ):
        result = original(
            spec, model, data, q, frame_ids, anchors, height_tolerance
        )
        if anchors is not None:
            records.append((
                set(result[0]),
                {
                    frame_id: (anchor.translation.copy(), anchor.rotation.copy())
                    for frame_id, anchor in anchors.items()
                },
            ))
        return result

    feasibility._make_frame_contacts = recording_make
    try:
        result = rollout(
            spec,
            build_whole_body_trajectory(spec, sample, steps=1, dt=0.01),
            sample,
        )
    finally:
        feasibility._make_frame_contacts = original
    assert result.success, (result.failure_reason, result.failure_index)
    for (old_ids, old), (new_ids, new) in zip(records, records[1:]):
        for frame_id in old_ids & new_ids:
            np.testing.assert_array_equal(old[frame_id][0], new[frame_id][0])
            np.testing.assert_array_equal(old[frame_id][1], new[frame_id][1])


def test_touchdown_rejects_an_impact_that_requires_tensile_ground_impulse():
    spec = load_robot_spec("talos")
    trajectory = _trajectory(spec, ("touchdown",))
    upward_velocity = trajectory.v.copy()
    upward_velocity[0, 2] = 0.1

    result = rollout(
        spec, replace(trajectory, v=upward_velocity), _sample()
    )

    assert result.failure_reason == "normal_force"
    assert result.failure_index == 0


def test_touchdown_rejects_small_tensile_impulses_above_solver_noise():
    spec = load_robot_spec("talos")
    trajectory = _trajectory(spec, ("touchdown",))
    upward_velocity = trajectory.v.copy()
    upward_velocity[0, 2] = 0.001

    result = rollout(
        spec, replace(trajectory, v=upward_velocity), _sample()
    )

    assert result.failure_reason == "normal_force"
    assert result.failure_index == 0


def _single_sole_sliding_fixture():
    spec = load_robot_spec("talos")
    sample = _sample(friction=0.40)
    q = spec.robot.q0.copy()
    frame_ids = (spec.left_sole_frame_id,)
    data = spec.model.createData()
    mass = pin.crba(spec.model, data, q)
    mass = np.triu(mass) + np.triu(mass, 1).T
    pin.computeJointJacobians(spec.model, data, q)
    pin.updateFramePlacements(spec.model, data)
    jacobian = pin.getFrameJacobian(
        spec.model, data, frame_ids[0], pin.LOCAL_WORLD_ALIGNED
    )
    sticking_impulse = np.array([0.44, 0.0, 1.0, 0.0, 0.0, 0.0])
    preimpact_velocity = -np.linalg.solve(
        mass, jacobian.T @ sticking_impulse
    )
    return (
        spec,
        sample,
        q,
        frame_ids,
        data,
        mass,
        jacobian,
        sticking_impulse,
        preimpact_velocity,
    )


def _run_with_recorded_admm(callback, delegated_tolerance=None):
    original_solver = feasibility.pin.ADMMConstraintSolver
    original_point_contact = feasibility.pin.PointContactConstraintModel
    records = []
    constructed = []

    def recording_point_contact(*args, **kwargs):
        model = original_point_contact(*args, **kwargs)
        constructed.append(model)
        return model

    class RecordingSolver:
        def __init__(self, size):
            self.delegate = original_solver(size)

        def solve(
            self,
            delassus,
            free_velocity,
            constraint_models,
            constraint_datas,
            settings,
            result,
        ):
            if delegated_tolerance is not None:
                settings.absolute_complementarity_tol = delegated_tolerance
                settings.relative_complementarity_tol = delegated_tolerance
                settings.absolute_feasibility_tol = delegated_tolerance
                settings.relative_feasibility_tol = delegated_tolerance
            solved = self.delegate.solve(
                delassus,
                free_velocity,
                constraint_models,
                constraint_datas,
                settings,
                result,
            )
            point_models = constructed[-len(constraint_models):]
            records.append({
                "solved": solved,
                "tolerances": np.array([
                    settings.absolute_complementarity_tol,
                    settings.relative_complementarity_tol,
                    settings.absolute_feasibility_tol,
                    settings.relative_feasibility_tol,
                ]),
                "positions": np.asarray([
                    model.joint1_placement.translation.copy()
                    for model in point_models
                ]),
                "frictions": np.asarray([
                    model.getFriction() for model in point_models
                ]),
                "impulses": result.retrieveConstraintImpulses().reshape(-1, 3),
                "velocities": result.retrieveConstraintVelocities().reshape(-1, 3),
                "metrics": np.array([
                    result.complementarity,
                    result.primal_feasibility,
                    result.dual_feasibility,
                ]),
            })
            return solved

    feasibility.pin.ADMMConstraintSolver = RecordingSolver
    feasibility.pin.PointContactConstraintModel = recording_point_contact
    try:
        output = callback()
    finally:
        feasibility.pin.ADMMConstraintSolver = original_solver
        feasibility.pin.PointContactConstraintModel = original_point_contact
    return output, records


def _assert_valid_point_ncp(
    record, friction, impulse_tolerance, velocity_tolerance
):
    impulses = record["impulses"]
    velocities = record["velocities"]
    assert record["solved"]
    assert np.isfinite(record["positions"]).all()
    assert np.isfinite(impulses).all()
    assert np.isfinite(velocities).all()
    assert np.isfinite(record["metrics"]).all()
    assert np.max(np.abs(record["metrics"])) <= 1e-8
    assert np.all(impulses[:, 2] >= -impulse_tolerance)
    assert np.all(
        np.linalg.norm(impulses[:, :2], axis=1)
        <= friction * np.maximum(impulses[:, 2], 0.0) + impulse_tolerance
    )
    assert np.all(velocities[:, 2] >= -velocity_tolerance)
    assert np.all(np.abs(
        impulses[:, 2] * velocities[:, 2]
    ) <= 1e-8)


def test_single_sole_frictional_touchdown_slides():
    (
        spec,
        sample,
        q,
        frame_ids,
        data,
        mass,
        jacobian,
        sticking_impulse,
        preimpact_velocity,
    ) = _single_sole_sliding_fixture()

    impact_data = spec.model.createData()
    _, contact_models, contact_datas = feasibility._make_frame_contacts(
        spec, spec.model, impact_data, q, frame_ids
    )
    pin.impulseDynamics(
        spec.model,
        impact_data,
        q,
        preimpact_velocity,
        contact_models,
        contact_datas,
        0.0,
    )
    actual_sticking_impulse = np.asarray(
        contact_datas[0].contact_force.vector
    )
    np.testing.assert_allclose(
        actual_sticking_impulse, sticking_impulse, rtol=1e-10, atol=1e-10
    )
    assert feasibility._contact_constraint_violations(
        spec,
        sample,
        actual_sticking_impulse,
        frame_ids[0],
        data.oMf[frame_ids[0]].rotation,
        1e-8 * max(spec.mass * np.sqrt(9.81 * spec.leg_length), 1.0),
        1e-9 * spec.leg_length,
    ) == (False, True, False)

    (first, repeated), records = _run_with_recorded_admm(
        lambda: (
            feasibility._resolve_impact(
                spec, sample, spec.model, q, preimpact_velocity, frame_ids
            ),
            feasibility._resolve_impact(
                spec, sample, spec.model, q, preimpact_velocity, frame_ids
            ),
        )
    )
    post_velocity, selected, reason = first
    repeated_velocity, repeated_selected, repeated_reason = repeated

    velocity_tolerance = (
        feasibility._IMPACT_VELOCITY_TOLERANCE_RATIO
        * np.sqrt(9.81 * spec.leg_length)
    )
    impulse_tolerance = 1e-8 * max(
        spec.mass * np.sqrt(9.81 * spec.leg_length), 1.0
    )
    post_twist = jacobian @ post_velocity
    assert reason == ""
    assert selected == frame_ids
    assert repeated_reason == reason
    assert repeated_selected == selected
    np.testing.assert_array_equal(repeated_velocity, post_velocity)
    assert len(records) == 2
    expected_vertices = np.asarray([
        data.oMf[frame_ids[0]].act(np.r_[point, 0.0])
        for point in spec.left_sole_polygon
    ])
    assert len(records[0]["positions"]) == len(spec.left_sole_polygon)
    for record in records:
        np.testing.assert_array_equal(
            record["tolerances"], np.full(4, 5e-9)
        )
        np.testing.assert_allclose(
            record["positions"], expected_vertices, rtol=0.0, atol=1e-12
        )
        np.testing.assert_array_equal(
            record["frictions"],
            np.full(len(expected_vertices), sample.friction),
        )
        _assert_valid_point_ncp(
            record, sample.friction, impulse_tolerance, velocity_tolerance
        )
    np.testing.assert_array_equal(records[1]["impulses"], records[0]["impulses"])
    np.testing.assert_array_equal(
        records[1]["velocities"], records[0]["velocities"]
    )
    point_impulses = records[0]["impulses"]
    point_velocities = records[0]["velocities"]
    sliding = np.linalg.norm(point_velocities[:, :2], axis=1) > 1e-4
    saturated = np.abs(
        np.linalg.norm(point_impulses[:, :2], axis=1)
        - sample.friction * point_impulses[:, 2]
    ) <= impulse_tolerance
    opposing = np.sum(
        point_impulses[:, :2] * point_velocities[:, :2], axis=1
    ) < 0.0
    assert np.any(sliding & saturated & opposing)
    assert abs(post_twist[2]) <= velocity_tolerance
    assert np.linalg.norm(post_twist[:2]) > 1e-4
    aggregate_impulse = np.linalg.lstsq(
        jacobian.T,
        mass @ (post_velocity - preimpact_velocity),
        rcond=1e-12,
    )[0]
    momentum_scale = np.r_[
        np.full(3, max(spec.mass * np.sqrt(9.81 * spec.leg_length), 1.0)),
        np.full(
            spec.model.nv - 3,
            max(
                spec.mass * spec.leg_length
                * np.sqrt(9.81 * spec.leg_length),
                1.0,
            ),
        ),
    ]
    momentum_residual = (
        mass @ (post_velocity - preimpact_velocity)
        - jacobian.T @ aggregate_impulse
    ) / momentum_scale
    assert np.max(np.abs(momentum_residual)) < 1e-10
    assert feasibility._contact_constraint_violations(
        spec,
        sample,
        aggregate_impulse,
        frame_ids[0],
        data.oMf[frame_ids[0]].rotation,
        1e-8 * max(spec.mass * np.sqrt(9.81 * spec.leg_length), 1.0),
        1e-9 * spec.leg_length,
    ) == (False, False, False)


def test_impact_uses_the_same_cop_tolerance_as_continuous_contact():
    (
        spec,
        sample,
        q,
        frame_ids,
        _,
        _,
        _,
        _,
        preimpact_velocity,
    ) = _single_sole_sliding_fixture()
    original = feasibility._contact_constraint_violations
    tolerances = []

    def recording(*args):
        tolerances.append(args[-1])
        return original(*args)

    feasibility._contact_constraint_violations = recording
    try:
        _, selected, reason = feasibility._resolve_impact(
            spec, sample, spec.model, q, preimpact_velocity, frame_ids
        )
    finally:
        feasibility._contact_constraint_violations = original

    assert selected == frame_ids
    assert reason == ""
    expected = feasibility._CONTINUOUS_COP_TOLERANCE_RATIO * spec.leg_length
    assert tolerances == [expected, expected]


def test_sliding_impact_cop_uses_normal_pressure_not_tangential_lever_arm():
    (
        spec,
        sample,
        q,
        frame_ids,
        _,
        _,
        _,
        _,
        preimpact_velocity,
    ) = _single_sole_sliding_fixture()
    original = feasibility._contact_constraint_violations
    checked_wrenches = []

    def shifted_full_wrench(*args):
        wrench = np.asarray(args[2])
        violations = original(*args)
        checked_wrenches.append(wrench.copy())
        return (
            violations[0],
            violations[1],
            violations[2] or np.linalg.norm(wrench[:2]) > 0.0,
        )

    feasibility._contact_constraint_violations = shifted_full_wrench
    try:
        _, selected, reason = feasibility._resolve_impact(
            spec, sample, spec.model, q, preimpact_velocity, frame_ids
        )
    finally:
        feasibility._contact_constraint_violations = original

    assert selected == frame_ids
    assert reason == ""
    assert any(np.linalg.norm(wrench[:2]) > 0.0 for wrench in checked_wrenches)
    assert any(np.linalg.norm(wrench[:2]) == 0.0 for wrench in checked_wrenches)


def test_vertex_friction_cones_do_not_use_the_6d_torsion_proxy():
    (
        spec,
        sample,
        q,
        frame_ids,
        _,
        _,
        _,
        _,
        preimpact_velocity,
    ) = _single_sole_sliding_fixture()
    original = feasibility._contact_constraint_violations

    def reject_distributed_tangential_wrench(*args):
        violations = original(*args)
        return (
            violations[0],
            violations[1] or np.linalg.norm(np.asarray(args[2])[:2]) > 0.0,
            violations[2],
        )

    feasibility._contact_constraint_violations = (
        reject_distributed_tangential_wrench
    )
    try:
        _, selected, reason = feasibility._resolve_impact(
            spec, sample, spec.model, q, preimpact_velocity, frame_ids
        )
    finally:
        feasibility._contact_constraint_violations = original

    assert selected == frame_ids
    assert reason == ""


def test_grazing_sliding_contact_preserves_the_cone_apex():
    spec = load_robot_spec("talos")
    sample = _sample(friction=0.40)
    q = spec.robot.q0.copy()
    frame_ids = (spec.left_sole_frame_id,)
    preimpact_velocity = np.zeros(spec.model.nv)
    preimpact_velocity[0] = 0.01

    (post_velocity, selected, reason), records = _run_with_recorded_admm(
        lambda: feasibility._resolve_impact(
            spec, sample, spec.model, q, preimpact_velocity, frame_ids
        )
    )

    velocity_tolerance = (
        feasibility._IMPACT_VELOCITY_TOLERANCE_RATIO
        * np.sqrt(9.81 * spec.leg_length)
    )
    impulse_tolerance = 1e-8 * max(
        spec.mass * np.sqrt(9.81 * spec.leg_length), 1.0
    )
    assert reason == ""
    assert selected == frame_ids
    np.testing.assert_allclose(
        post_velocity, preimpact_velocity, rtol=0.0, atol=velocity_tolerance
    )
    assert len(records) == 1
    _assert_valid_point_ncp(
        records[0], sample.friction, impulse_tolerance, velocity_tolerance
    )
    assert np.max(np.abs(records[0]["impulses"])) <= impulse_tolerance
    assert np.all(np.abs(
        records[0]["velocities"][:, 2]
    ) <= velocity_tolerance)
    assert np.max(np.linalg.norm(
        records[0]["velocities"][:, :2], axis=1
    )) > 1e-4


def test_two_sole_impact_accepts_complementary_edge_sliding_deterministically():
    spec = load_robot_spec("talos")
    sample = _sample()
    q = spec.robot.q0.copy()
    frame_ids = (spec.left_sole_frame_id, spec.right_sole_frame_id)
    data = spec.model.createData()
    pin.computeJointJacobians(spec.model, data, q)
    pin.updateFramePlacements(spec.model, data)
    jacobian = np.vstack([
        pin.getFrameJacobian(
            spec.model, data, frame_id, pin.LOCAL_WORLD_ALIGNED
        )
        for frame_id in frame_ids
    ])
    contact_twist = np.array([
        0.0025765056454350797,
        -0.0006001705153789289,
        -0.0005726524169151736,
        0.0014983857281213036,
        0.005214922979779207,
        -0.0016023578453278912,
        0.000259749841234268,
        -0.00037817277472376905,
        -0.004629052567584539,
        -0.002778169630827856,
        0.0013302989608076914,
        0.0004553411335236713,
    ])
    preimpact_velocity = np.linalg.pinv(
        jacobian, rcond=1e-10
    ) @ contact_twist

    (first, repeated), records = _run_with_recorded_admm(
        lambda: (
            feasibility._resolve_impact(
                spec, sample, spec.model, q, preimpact_velocity, frame_ids
            ),
            feasibility._resolve_impact(
                spec, sample, spec.model, q, preimpact_velocity, frame_ids
            ),
        )
    )

    velocity_tolerance = (
        feasibility._IMPACT_VELOCITY_TOLERANCE_RATIO
        * np.sqrt(9.81 * spec.leg_length)
    )
    impulse_tolerance = 1e-8 * max(
        spec.mass * np.sqrt(9.81 * spec.leg_length), 1.0
    )
    assert first[2] == repeated[2] == ""
    assert first[1] == repeated[1] == frame_ids
    np.testing.assert_array_equal(first[0], repeated[0])
    assert len(records) == 2
    expected_vertices = []
    vertex_counts = []
    for frame_id, polygon in (
        (spec.left_sole_frame_id, spec.left_sole_polygon),
        (spec.right_sole_frame_id, spec.right_sole_polygon),
    ):
        vertex_counts.append(len(polygon))
        expected_vertices.extend(
            data.oMf[frame_id].act(np.r_[point, 0.0])
            for point in polygon
        )
    expected_vertices = np.asarray(expected_vertices)
    for record in records:
        np.testing.assert_allclose(
            record["positions"], expected_vertices, rtol=0.0, atol=1e-12
        )
        _assert_valid_point_ncp(
            record, sample.friction, impulse_tolerance, velocity_tolerance
        )
        split = vertex_counts[0]
        assert np.sum(record["impulses"][:split, 2]) > impulse_tolerance
        assert np.sum(record["impulses"][split:, 2]) > impulse_tolerance
    np.testing.assert_array_equal(records[1]["positions"], records[0]["positions"])
    np.testing.assert_array_equal(records[1]["impulses"], records[0]["impulses"])
    np.testing.assert_array_equal(
        records[1]["velocities"], records[0]["velocities"]
    )


def test_sliding_impact_rejects_large_absolute_admm_residuals():
    (
        spec,
        sample,
        q,
        frame_ids,
        _,
        _,
        _,
        _,
        preimpact_velocity,
    ) = _single_sole_sliding_fixture()

    (_, selected, reason), records = _run_with_recorded_admm(
        lambda: feasibility._resolve_impact(
            spec, sample, spec.model, q, preimpact_velocity, frame_ids
        ),
        delegated_tolerance=1e-6,
    )

    assert len(records) == 1
    assert np.max(np.abs(records[0]["metrics"])) > 1e-8
    assert selected == ()
    assert reason == "impact_dynamics"


def test_sliding_impact_accepts_admm_residuals_below_physical_checks():
    (
        spec,
        sample,
        q,
        frame_ids,
        _,
        _,
        _,
        _,
        preimpact_velocity,
    ) = _single_sole_sliding_fixture()

    (_, selected, reason), records = _run_with_recorded_admm(
        lambda: feasibility._resolve_impact(
            spec, sample, spec.model, q, preimpact_velocity, frame_ids
        ),
        delegated_tolerance=5e-9,
    )

    assert len(records) == 1
    assert 1e-9 < np.max(np.abs(records[0]["metrics"])) < 1e-8
    assert selected == frame_ids
    assert reason == ""


def test_sliding_impact_propagates_point_contact_api_errors():
    (
        spec,
        sample,
        q,
        frame_ids,
        _,
        _,
        _,
        _,
        preimpact_velocity,
    ) = _single_sole_sliding_fixture()
    original = feasibility.pin.PointContactConstraintModel

    def invalid_point_contact(*args, **kwargs):
        del args, kwargs
        raise TypeError("point-contact API mismatch")

    feasibility.pin.PointContactConstraintModel = invalid_point_contact
    try:
        try:
            feasibility._resolve_impact(
                spec, sample, spec.model, q, preimpact_velocity, frame_ids
            )
        except TypeError as error:
            assert str(error) == "point-contact API mismatch"
        else:
            assert False, "programming/API errors must propagate"
    finally:
        feasibility.pin.PointContactConstraintModel = original


def test_impact_resolves_yaw_with_vertex_friction_but_rejects_tensile_modes():
    spec = load_robot_spec("talos")
    q = spec.robot.q0.copy()
    frame_ids = (spec.left_sole_frame_id,)
    data = spec.model.createData()
    mass = pin.crba(spec.model, data, q)
    mass = np.triu(mass) + np.triu(mass, 1).T
    pin.computeJointJacobians(spec.model, data, q)
    pin.updateFramePlacements(spec.model, data)
    jacobian = pin.getFrameJacobian(
        spec.model, data, frame_ids[0], pin.LOCAL_WORLD_ALIGNED
    )
    forced_impulse = np.zeros(6)
    original = feasibility.pin.impulseDynamics

    def forced_impact(
        model, impact_data, configuration, velocity,
        contact_models, contact_datas, restitution,
    ):
        del impact_data, configuration, velocity, contact_models, restitution
        contact_datas[0].contact_force = pin.Force(forced_impulse)
        return np.zeros(model.nv)

    feasibility.pin.impulseDynamics = forced_impact
    try:
        _, selected, reason = feasibility._resolve_impact(
            spec, _sample(), spec.model, q, np.zeros(spec.model.nv), frame_ids
        )
        assert selected == frame_ids
        assert reason == ""

        for impulse, vertex_feasible in (
            (np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), False),
            (np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]), True),
            (np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]), False),
        ):
            forced_impulse[:] = impulse
            preimpact_velocity = -np.linalg.solve(
                mass, jacobian.T @ forced_impulse
            )
            _, selected, reason = feasibility._resolve_impact(
                spec,
                _sample(),
                spec.model,
                q,
                preimpact_velocity,
                frame_ids,
            )
            assert (selected == frame_ids and reason == "") == vertex_feasible
    finally:
        feasibility.pin.impulseDynamics = original


def test_low_friction_plant_wrench_causes_friction_failure():
    spec = load_robot_spec("talos")
    sample = _sample(
        friction=0.25,
        payload_fraction=0.10,
        timing_error_seconds=0.02,
        impulse=0.04,
        ood=True,
    )
    original_controller = feasibility._controller_torque
    torque = np.zeros(spec.model.nv - 6)
    torque[7] = spec.effort_limits[13]
    feasibility._controller_torque = lambda *args, **kwargs: (torque, torque)
    try:
        result = rollout(spec, _trajectory(spec), sample)
    finally:
        feasibility._controller_torque = original_controller
    assert result.failure_reason == "friction"
    assert result.failure_index == 0
    side = int(np.argmin(result.friction_margin[0]))
    assert result.friction_margin[0, side] < 0.0
    assert np.linalg.norm(result.contact_wrenches[0, side, :2]) > sample.friction * (
        result.contact_wrenches[0, side, 2]
    )


def test_contact_margin_rejects_torsion_and_true_sole_polygon_overrun():
    spec = load_robot_spec("icub")
    sample = _sample()
    frame_id = spec.left_sole_frame_id
    normal = spec.mass * 9.81
    radius = min(spec.sole_half_length, spec.sole_half_width)
    _, friction, _ = feasibility._contact_margins(
        spec,
        sample,
        np.array([
            0.0,
            0.0,
            normal,
            0.0,
            0.0,
            2.0 * sample.friction * radius * normal,
        ]),
        frame_id,
        spec.left_sole_rotation,
    )
    assert friction < 0.0
    _, _, cop = feasibility._contact_margins(
        spec,
        sample,
        np.array([0.0, 0.0, normal, 0.0, 0.08 * normal, 0.0]),
        frame_id,
        spec.left_sole_rotation,
    )
    assert cop < 0.0
    polygon = feasibility._sole_polygon_offsets(
        spec, frame_id, spec.left_sole_rotation
    )
    edges = np.roll(polygon, -1, axis=0) - polygon
    edge_index = int(np.argmax(np.linalg.norm(edges, axis=1)))
    edge = edges[edge_index]
    midpoint = (polygon[edge_index] + polygon[(edge_index + 1) % len(polygon)]) / 2.0
    outward = np.array([edge[1], -edge[0]]) / np.linalg.norm(edge)
    if np.dot(outward, np.mean(polygon, axis=0) - midpoint) > 0.0:
        outward = -outward
    assert feasibility._CONTINUOUS_COP_TOLERANCE_RATIO == 1e-4
    cop_tolerance = feasibility._CONTINUOUS_COP_TOLERANCE_RATIO * spec.leg_length
    for scale, expected in ((0.5, False), (2.0, True)):
        x, y = midpoint + scale * cop_tolerance * outward
        violations = feasibility._contact_constraint_violations(
            spec,
            sample,
            np.array([0.0, 0.0, normal, y * normal, -x * normal, 0.0]),
            frame_id,
            spec.left_sole_rotation,
            1e-6 * normal,
            cop_tolerance,
        )
        assert violations == (False, False, expected)


def test_low_load_transition_wrench_uses_sole_radius_moment_floor():
    spec = load_robot_spec("icub")
    wrench = np.array([
        -2.55e-7,
        -3.26e-5,
        2.894912075950319e-5,
        1.92e-7,
        -4.143e-6,
        6.13e-7,
    ])
    assert feasibility._contact_constraint_violations(
        spec,
        _sample(),
        wrench,
        spec.right_sole_frame_id,
        spec.right_sole_rotation,
        1e-6 * spec.mass * 9.81,
        feasibility._CONTINUOUS_COP_TOLERANCE_RATIO * spec.leg_length,
    ) == (False, False, False)


def test_zero_load_cop_moment_floor_rejects_immediately_outside_boundary():
    spec = load_robot_spec("icub")
    frame_id = spec.right_sole_frame_id
    rotation = spec.right_sole_rotation
    force_tolerance = 1e-6 * spec.mass * 9.81
    cop_tolerance = (
        feasibility._CONTINUOUS_COP_TOLERANCE_RATIO * spec.leg_length
    )
    sole_radius = float(np.max(np.linalg.norm(
        feasibility._sole_polygon_offsets(spec, frame_id, rotation), axis=1
    )))
    unit_wrench = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    unit_margin = feasibility._cop_wrench_margin(
        spec, unit_wrench, frame_id, rotation, normal=0.0
    )
    assert unit_margin < 0.0
    for scale, expected in ((0.99, False), (1.01, True)):
        wrench = unit_wrench.copy()
        wrench[3:5] *= (
            scale * force_tolerance * sole_radius / -unit_margin
        )
        assert feasibility._contact_constraint_violations(
            spec,
            _sample(),
            wrench,
            frame_id,
            rotation,
            force_tolerance,
            cop_tolerance,
        ) == (False, False, expected)


def test_terrain_penetration_accepts_only_one_step_upward_recovery():
    length = 0.5
    tolerance = 1e-3 * length
    velocity_tolerance = (
        feasibility._IMPACT_VELOCITY_TOLERANCE_RATIO * np.sqrt(9.81 * length)
    )
    cases = (
        (-tolerance, 0.0, False),
        (-2.0 * tolerance, tolerance / 0.01, False),
        (-2.0 * tolerance - 1e-12, 1.0, True),
        (-1.5 * tolerance, -1.0, True),
        (-1.5 * tolerance, velocity_tolerance, True),
        (-1.5 * tolerance, 0.4 * tolerance / 0.01, True),
        (-1.5 * tolerance, 0.5 * tolerance / 0.01, False),
    )
    for gap, normal_velocity, expected in cases:
        assert feasibility._terrain_penetration_is_failure(
            gap, normal_velocity, 0.01, length
        ) is expected


def test_terrain_penetration_context_exceptions_preserve_hard_rejects():
    length = 0.5
    tolerance = 1e-3 * length
    velocity_tolerance = (
        feasibility._IMPACT_VELOCITY_TOLERANCE_RATIO * np.sqrt(9.81 * length)
    )
    cases = (
        (-1.5 * tolerance, -1.0, True, False, False),
        (-2.0 * tolerance, -1.0, True, False, False),
        (-2.0 * tolerance - 1e-12, 1.0, True, False, True),
        (-1.5 * tolerance, 0.0, False, True, True),
        (-1.5 * tolerance, -1.0, False, True, True),
        (
            -1.5 * tolerance,
            velocity_tolerance,
            False,
            True,
            True,
        ),
        (
            -1.5 * tolerance,
            2.0 * velocity_tolerance,
            False,
            True,
            False,
        ),
        (
            -2.0 * tolerance - 1e-12,
            2.0 * velocity_tolerance,
            False,
            True,
            True,
        ),
    )
    for gap, velocity, impact_transition, active_contact, expected in cases:
        assert feasibility._terrain_penetration_is_failure(
            gap,
            velocity,
            0.01,
            length,
            impact_transition=impact_transition,
            active_contact=active_contact,
        ) is expected


def test_icub_transition_regressions_succeed_at_frozen_timestep():
    samples = (
        _sample(
            step_length=0.1423721275664866,
            step_width=1.0148975576274097,
            single_support_duration=2.707944435812533,
            double_support_duration=0.7629460867494344,
            com_height_scale=0.9691554799862205,
            zmp_bias_x=-0.03859863355755806,
            zmp_bias_y=-0.17337578637525441,
            seed=3661564547,
        ),
        _sample(
            step_length=0.12315681682899594,
            step_width=1.0840227518230678,
            single_support_duration=2.7789022165350614,
            double_support_duration=0.7195589575916529,
            com_height_scale=0.9455421503446997,
            zmp_bias_x=0.04567095814272762,
            zmp_bias_y=-0.18175988462753595,
            seed=3779867835,
        ),
    )
    spec = load_robot_spec("icub")
    for sample in samples:
        result = rollout(
            spec,
            build_whole_body_trajectory(spec, sample, steps=6, dt=0.01),
            sample,
        )
        assert result.success, (
            sample.seed, result.failure_reason, result.failure_index
        )


def test_zero_load_contact_rejects_tangential_force_and_moment_but_accepts_apex():
    spec = load_robot_spec("talos")
    trajectory = _trajectory(spec)
    forced_wrench = np.zeros(6)
    frame_ids = (spec.left_sole_frame_id, spec.right_sole_frame_id)
    original = feasibility.pin.constraintDynamics

    def forced_constraint_dynamics(
        model, data, q, v, tau, contact_models, contact_datas, settings
    ):
        del contact_models, settings
        for contact_data in contact_datas:
            contact_data.contact_force = pin.Force(forced_wrench)
        mass = pin.crba(model, data, q)
        mass = np.triu(mass) + np.triu(mass, 1).T
        nonlinear = pin.nonLinearEffects(model, data, q, v)
        pin.computeJointJacobians(model, data, q)
        pin.updateFramePlacements(model, data)
        supplied = tau.copy()
        for frame_id, contact_data in zip(frame_ids, contact_datas):
            supplied += pin.getFrameJacobian(
                model, data, frame_id, pin.LOCAL_WORLD_ALIGNED
            ).T @ contact_data.contact_force.vector
        return np.linalg.solve(mass, supplied - nonlinear)

    feasibility.pin.constraintDynamics = forced_constraint_dynamics
    try:
        for wrench, expected in (
            (np.zeros(6), ""),
            (np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0]), "friction"),
            (np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]), "cop"),
        ):
            forced_wrench[:] = wrench
            result = rollout(spec, trajectory, _sample())
            assert result.failure_reason == expected
    finally:
        feasibility.pin.constraintDynamics = original


def test_unsaturated_torque_request_is_diagnostic_when_applied_is_bounded():
    spec = load_robot_spec("talos")
    limits = spec.effort_limits.copy()
    limits[6:] = 1.0
    limited = replace(spec, effort_limits=limits)
    trajectory = _trajectory(spec, ("double",) * 5)
    references = trajectory.q.copy()
    tangent = np.zeros(spec.model.nv)
    tangent[6] = 0.001
    references[1:] = pin.integrate(spec.model, spec.robot.q0, tangent)
    trajectory = replace(trajectory, q=references)
    result = rollout(limited, trajectory, _sample())
    assert np.all(np.any(np.abs(result.torque_demand[:3]) > 1.01, axis=1))
    assert np.max(np.abs(result.applied_torque)) <= 1.0 + 1e-12
    assert result.success
    assert result.failure_reason == ""
    assert result.failure_index == -1
    assert result.peak_torque_joint == "leg_left_4_joint"
    np.testing.assert_allclose(
        result.peak_torque_ratio,
        np.max(np.abs(result.torque_demand) / limits[None, 6:]),
    )


def test_off_terrain_contacts_invalid_quaternions_and_mismatched_lengths_are_rejected():
    spec = load_robot_spec("talos")
    lifted = spec.robot.q0.copy()
    lifted[2] += 0.03
    air = rollout(spec, _trajectory(spec, q=lifted), _sample())
    assert air.failure_reason == "tracking"
    assert air.failure_index == 0

    shifted = spec.robot.q0.copy()
    shifted[0] += 0.03
    references = _trajectory(spec)
    displaced = rollout(
        spec,
        replace(_trajectory(spec, q=shifted), left_foot=references.left_foot),
        _sample(),
    )
    assert displaced.failure_reason == "tracking"
    assert displaced.failure_index == 0

    invalid = spec.robot.q0.copy()
    invalid[3:7] *= 2.0
    for trajectory in (
        _trajectory(spec, q=invalid),
        WholeBodyTrajectory(
            time=np.array([0.0, 0.01]),
            q=np.repeat(spec.robot.q0[None], 2, axis=0),
            v=np.zeros((2, spec.model.nv)),
            a=np.zeros((2, spec.model.nv)),
            left_foot=_trajectory(spec).left_foot,
            right_foot=_trajectory(spec).right_foot,
            com=_trajectory(spec).com,
            contact_modes=("double",),
            dt=0.01,
        ),
    ):
        try:
            rollout(spec, trajectory, _sample())
        except ValueError:
            pass
        else:
            raise AssertionError("invalid trajectory must be rejected")


def test_inactive_swing_foot_penetration_is_rejected():
    spec = load_robot_spec("talos")
    q = spec.robot.q0.copy()
    knee = spec.model.joints[spec.model.getJointId("leg_right_4_joint")]
    q[knee.idx_q] = 0.5578185

    data = spec.model.createData()
    pin.forwardKinematics(spec.model, data, spec.robot.q0)
    pin.updateFramePlacements(spec.model, data)
    ground = np.mean([
        data.oMf[spec.left_sole_frame_id].translation[2],
        data.oMf[spec.right_sole_frame_id].translation[2],
    ])
    trajectory = _trajectory(spec, ("left",), q=q)
    assert trajectory.left_foot[0, 2] >= ground - 1e-3 * spec.leg_length
    assert trajectory.right_foot[0, 2] < ground - 1e-3 * spec.leg_length

    result = rollout(spec, trajectory, _sample())

    assert not result.success
    assert result.failure_reason == "impact_dynamics"
    assert result.failure_index == 0


def test_icub_collision_exclusions_are_explicit_and_leave_real_pairs_enabled():
    spec = load_robot_spec("icub")
    assert len(spec.ignored_collision_pairs) == 32
    assert ("l_foot_0", "r_foot_0") not in spec.ignored_collision_pairs
    collision_data = pin.GeometryData(spec.collision_model)
    assert not pin.computeCollisions(
        spec.model,
        spec.model.createData(),
        spec.collision_model,
        collision_data,
        spec.robot.q0,
        False,
    )
    result = rollout(spec, _trajectory(spec), _sample())
    assert result.failure_reason != "collision"

    colliding = spec.robot.q0.copy()
    hip = spec.model.joints[spec.model.getJointId("l_hip_roll")]
    colliding[hip.idx_q] = spec.position_lower_limits[hip.idx_q]
    collision_data = pin.GeometryData(spec.collision_model)
    assert pin.computeCollisions(
        spec.model,
        spec.model.createData(),
        spec.collision_model,
        collision_data,
        colliding,
        True,
    )


def test_collision_invalid_mode_and_solver_exception_are_classified():
    spec = load_robot_spec("talos")
    original_collision = feasibility.pin.computeCollisions
    feasibility.pin.computeCollisions = lambda *args, **kwargs: True
    try:
        collision = rollout(spec, _trajectory(spec), _sample())
        assert collision.failure_reason == "collision"
        assert collision.failure_index == 0
    finally:
        feasibility.pin.computeCollisions = original_collision

    invalid = rollout(spec, _trajectory(spec, ("airborne",)), _sample())
    assert invalid.failure_reason == "dynamics"
    assert invalid.failure_index == 0

    original_dynamics = feasibility.pin.constraintDynamics
    feasibility.pin.constraintDynamics = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("solver failed")
    )
    try:
        dynamics = rollout(spec, _trajectory(spec), _sample())
        assert dynamics.failure_reason == "dynamics"
        assert dynamics.failure_index == 0
    finally:
        feasibility.pin.constraintDynamics = original_dynamics


def test_rollout_output_shapes_and_arrays_are_immutable():
    spec = load_robot_spec("talos")
    trajectory = _trajectory(spec, ("double", "right", "touchdown"), dt=0.005)
    result = rollout(spec, trajectory, _sample())
    count, joints = len(trajectory.time), spec.model.nv - 6
    assert result.time.shape == (count,)
    assert result.q.shape == (count, spec.model.nq)
    assert result.v.shape == (count, spec.model.nv)
    assert result.contact_wrenches.shape == (count, 2, 6)
    assert result.torque_demand.shape == result.applied_torque.shape == (count, joints)
    for margins in (
        result.normal_force_margin,
        result.friction_margin,
        result.cop_margin,
    ):
        assert margins.shape == (count, 2)
    for values in (
        result.time,
        result.q,
        result.v,
        result.contact_wrenches,
        result.torque_demand,
        result.applied_torque,
        result.normal_force_margin,
        result.friction_margin,
        result.cop_margin,
    ):
        assert not values.flags.writeable


def test_payload_changes_only_the_copied_plant_mass_at_the_torso():
    spec = load_robot_spec("talos")
    nominal_mass = sum(inertia.mass for inertia in spec.model.inertias)
    nominal_armature = spec.model.armature.copy()
    plant = _plant_model(spec, _sample(payload_fraction=0.10))
    assert sum(inertia.mass for inertia in plant.inertias) == nominal_mass * 1.10
    assert sum(inertia.mass for inertia in spec.model.inertias) == nominal_mass
    assert plant is not spec.model
    assert np.all(plant.armature[6:] > 0.0)
    np.testing.assert_array_equal(spec.model.armature, nominal_armature)


def test_timing_error_shifts_every_scheduled_transition_by_rounded_samples():
    modes = ("double", "right", "right", "touchdown", "double")
    assert _shift_contact_modes(modes, 1) == (
        "double",
        "double",
        "right",
        "right",
        "touchdown",
    )
    assert _shift_contact_modes(modes, -1) == (
        "right",
        "right",
        "touchdown",
        "double",
        "double",
    )


if __name__ == "__main__":
    test_rollout_is_deterministic_and_reports_fixed_left_right_local_wrenches()
    test_static_double_support_remains_stable_for_multiple_steps()
    test_generated_conservative_six_step_rollout_succeeds_for_each_robot()
    test_icub_rollout_accepts_cop_error_below_conservative_footprint_inset()
    test_icub_impact_tolerance_accepts_small_payload_mismatch_with_sliding()
    test_effort_saturation_does_not_preempt_later_physical_failure()
    test_unreachable_scheduled_contact_is_classified_as_tracking()
    test_returned_world_aligned_wrenches_reconstruct_constraint_dynamics()
    test_impact_solver_is_used_for_touchdown_but_not_external_pushes()
    test_external_push_has_exact_one_step_momentum_integral()
    test_touchdown_checks_every_active_contact_impulse()
    test_contact_mode_changes_preserve_anchors_for_continuing_supports()
    test_touchdown_rejects_an_impact_that_requires_tensile_ground_impulse()
    test_touchdown_rejects_small_tensile_impulses_above_solver_noise()
    test_single_sole_frictional_touchdown_slides()
    test_impact_uses_the_same_cop_tolerance_as_continuous_contact()
    test_sliding_impact_cop_uses_normal_pressure_not_tangential_lever_arm()
    test_vertex_friction_cones_do_not_use_the_6d_torsion_proxy()
    test_grazing_sliding_contact_preserves_the_cone_apex()
    test_two_sole_impact_accepts_complementary_edge_sliding_deterministically()
    test_sliding_impact_rejects_large_absolute_admm_residuals()
    test_sliding_impact_accepts_admm_residuals_below_physical_checks()
    test_sliding_impact_propagates_point_contact_api_errors()
    test_impact_resolves_yaw_with_vertex_friction_but_rejects_tensile_modes()
    test_low_friction_plant_wrench_causes_friction_failure()
    test_contact_margin_rejects_torsion_and_true_sole_polygon_overrun()
    test_low_load_transition_wrench_uses_sole_radius_moment_floor()
    test_zero_load_cop_moment_floor_rejects_immediately_outside_boundary()
    test_terrain_penetration_accepts_only_one_step_upward_recovery()
    test_terrain_penetration_context_exceptions_preserve_hard_rejects()
    test_icub_transition_regressions_succeed_at_frozen_timestep()
    test_zero_load_contact_rejects_tangential_force_and_moment_but_accepts_apex()
    test_unsaturated_torque_request_is_diagnostic_when_applied_is_bounded()
    test_off_terrain_contacts_invalid_quaternions_and_mismatched_lengths_are_rejected()
    test_inactive_swing_foot_penetration_is_rejected()
    test_icub_collision_exclusions_are_explicit_and_leave_real_pairs_enabled()
    test_collision_invalid_mode_and_solver_exception_are_classified()
    test_rollout_output_shapes_and_arrays_are_immutable()
    test_payload_changes_only_the_copied_plant_mass_at_the_torso()
    test_timing_error_shifts_every_scheduled_transition_by_rounded_samples()
    print("rollout tests passed")
