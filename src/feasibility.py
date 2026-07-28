"""Robot-neutral inputs and data containers for feasibility experiments."""

from dataclasses import dataclass, field
from functools import lru_cache
from math import isfinite, sqrt
from pathlib import Path
from time import perf_counter
import xml.etree.ElementTree as ET

import example_robot_data
import numpy as np
import pinocchio as pin
from scipy.linalg import solve_banded
from scipy.optimize import least_squares, linprog

from .support import convex_hull, polygon_margin


_CONTINUOUS_COP_TOLERANCE_RATIO = 1e-4
_CONTINUOUS_LIFTOFF_GAP_TOLERANCE_RATIO = 1e-5
_IMPACT_VELOCITY_TOLERANCE_RATIO = 1e-4


class _ContactReferenceError(ValueError):
    pass


_ROBOT_FRAMES = {
    "talos": {
        "left_sole": "left_sole_link",
        "right_sole": "right_sole_link",
        "left_hip": "leg_left_1_link",
        "right_hip": "leg_right_1_link",
        "torso": "torso_2_link",
        "sole_leveling_joints": (
            "leg_left_5_joint", "leg_left_6_joint",
            "leg_right_5_joint", "leg_right_6_joint",
        ),
        "sole_surface_tolerance": 0.002,
    },
    "icub": {
        "loader_name": "icub_reduced",
        "left_sole": "l_sole",
        "right_sole": "r_sole",
        "left_hip": "l_hip_1",
        "right_hip": "r_hip_1",
        "torso": "chest",
        "sole_leveling_joints": (
            "l_ankle_pitch", "l_ankle_roll",
            "r_ankle_pitch", "r_ankle_roll",
        ),
        "sole_surface_tolerance": 0.0001,
        "neutral_joints": (
            ("l_shoulder_roll", 0.5),
            ("r_shoulder_roll", 0.5),
        ),
        # example-robot-data does not ship an iCub SRDF. These pairs were
        # reviewed once against its neutral pose: they are adjacent/coarse
        # geometries that overlap or are separated by less than 5 mm. Keeping
        # the list explicit prevents a real collision from silently becoming
        # an exclusion merely because it occurs at the chosen initial state.
        "ignored_collision_pairs": (
            ("root_link_0", "torso_1_0"),
            ("root_link_0", "chest_0"),
            ("root_link_0", "l_forearm_0"),
            ("l_hip_1_0", "l_hip_2_0"),
            ("l_hip_1_0", "l_upper_leg_0"),
            ("l_hip_2_0", "l_upper_leg_0"),
            ("l_hip_2_0", "torso_1_0"),
            ("l_upper_leg_0", "torso_1_0"),
            ("l_upper_leg_0", "chest_0"),
            ("l_upper_leg_0", "l_lower_leg_0"),
            ("l_lower_leg_0", "r_lower_leg_0"),
            ("l_lower_leg_0", "l_ankle_1_0"),
            ("l_lower_leg_0", "l_foot_0"),
            ("l_ankle_1_0", "l_foot_0"),
            ("r_hip_1_0", "r_hip_2_0"),
            ("r_hip_1_0", "r_upper_leg_0"),
            ("r_hip_2_0", "r_upper_leg_0"),
            ("r_upper_leg_0", "r_lower_leg_0"),
            ("r_lower_leg_0", "r_ankle_1_0"),
            ("r_lower_leg_0", "r_foot_0"),
            ("r_ankle_1_0", "r_foot_0"),
            ("torso_1_0", "chest_0"),
            ("torso_1_0", "l_forearm_0"),
            ("chest_0", "l_upper_arm_0"),
            ("l_forearm_0", "l_hand_0"),
            ("l_shoulder_1_0", "neck_1_0"),
            ("l_shoulder_1_0", "neck_2_0"),
            ("l_shoulder_1_0", "head_0"),
            ("l_shoulder_2_0", "l_upper_arm_0"),
            ("r_shoulder_1_0", "r_shoulder_2_0"),
            ("r_shoulder_2_0", "r_upper_arm_0"),
            ("r_forearm_0", "r_hand_0"),
        ),
    },
}


@dataclass(frozen=True)
class RobotSpec:
    name: str
    robot: object
    model: object
    collision_model: object
    left_sole_frame: str
    right_sole_frame: str
    torso_frame: str
    left_sole_frame_id: int
    right_sole_frame_id: int
    torso_frame_id: int
    mass: float
    leg_length: float
    neutral_com_height: float
    position_lower_limits: np.ndarray
    position_upper_limits: np.ndarray
    velocity_limits: np.ndarray
    effort_limits: np.ndarray
    sole_half_length: float
    sole_half_width: float
    neutral_step_width: float
    left_sole_polygon: np.ndarray
    right_sole_polygon: np.ndarray
    left_sole_rotation: np.ndarray
    right_sole_rotation: np.ndarray
    ignored_collision_pairs: tuple[tuple[str, str], ...] = ()

    @property
    def natural_time(self):
        return sqrt(self.leg_length / 9.81)


@dataclass(frozen=True)
class GaitSample:
    step_length: float
    step_width: float
    single_support_duration: float
    double_support_duration: float
    com_height_scale: float
    zmp_bias_x: float
    zmp_bias_y: float
    friction: float
    payload_fraction: float
    timing_error_seconds: float
    impulse: float
    seed: int
    ood: bool = False

    def __post_init__(self):
        values = {
            "step_length": self.step_length,
            "step_width": self.step_width,
            "single_support_duration": self.single_support_duration,
            "double_support_duration": self.double_support_duration,
            "com_height_scale": self.com_height_scale,
            "zmp_bias_x": self.zmp_bias_x,
            "zmp_bias_y": self.zmp_bias_y,
            "friction": self.friction,
            "payload_fraction": self.payload_fraction,
            "timing_error_seconds": self.timing_error_seconds,
            "impulse": self.impulse,
        }
        if not all(isfinite(value) for value in values.values()):
            raise ValueError("gait sample values must be finite")
        for name, value, low, high in (
            ("step_length", self.step_length, 0.10, 0.40),
            ("step_width", self.step_width, 0.85, 1.15),
            ("single_support_duration", self.single_support_duration, 1.4, 2.8),
            ("double_support_duration", self.double_support_duration, 0.2, 0.8),
            ("com_height_scale", self.com_height_scale, 0.90, 1.05),
            ("zmp_bias_x", self.zmp_bias_x, -0.25, 0.25),
            ("zmp_bias_y", self.zmp_bias_y, -0.25, 0.25),
        ):
            if not low <= value <= high:
                raise ValueError(f"{name} must be in [{low}, {high}]")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not isinstance(self.ood, bool):
            raise ValueError("ood must be a bool")
        if self.ood:
            valid_friction = 0.25 <= self.friction <= 0.40 or 0.80 <= self.friction <= 0.90
            valid_timing = 0.02 <= abs(self.timing_error_seconds) <= 0.04
            limits = (("friction", valid_friction), ("payload_fraction", 0.10 <= self.payload_fraction <= 0.15),
                      ("timing_error_seconds", valid_timing), ("impulse", 0.04 <= self.impulse <= 0.08))
        else:
            limits = (("friction", 0.4 <= self.friction <= 0.8), ("payload_fraction", 0.0 <= self.payload_fraction <= 0.10),
                      ("timing_error_seconds", -0.02 <= self.timing_error_seconds <= 0.02), ("impulse", 0.0 <= self.impulse <= 0.04))
        for name, valid in limits:
            if not valid:
                raise ValueError(f"{name} is outside the {'OOD' if self.ood else 'ID'} domain")

    def to_gait_params(self, spec, steps=6, dt=0.01):
        from .gait import GaitParams

        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
            raise ValueError("steps must be a positive integer")
        if not isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be positive and finite")
        return GaitParams(
            step_length=self.step_length * spec.leg_length,
            step_width=self.step_width * spec.neutral_step_width,
            single_support_duration=self.single_support_duration * spec.natural_time,
            double_support_duration=self.double_support_duration * spec.natural_time,
            initial_double_support=self.double_support_duration * spec.natural_time,
            final_double_support=self.double_support_duration * spec.natural_time,
            steps=steps,
            com_height=self.com_height_scale * spec.neutral_com_height,
            zmp_bias_x=self.zmp_bias_x * spec.sole_half_length,
            zmp_bias_y=self.zmp_bias_y * spec.sole_half_width,
            foot_length=2.0 * spec.sole_half_length,
            foot_width=2.0 * spec.sole_half_width,
            com_dt=dt,
            eval_dt=dt,
        )


@dataclass(frozen=True)
class WholeBodyTrajectory:
    time: np.ndarray
    q: np.ndarray
    v: np.ndarray
    a: np.ndarray
    left_foot: np.ndarray
    right_foot: np.ndarray
    com: np.ndarray
    contact_modes: tuple[str, ...]
    dt: float

    def __post_init__(self):
        for name in ("time", "q", "v", "a", "left_foot", "right_foot", "com"):
            object.__setattr__(self, name, _readonly(getattr(self, name)))
        object.__setattr__(self, "contact_modes", tuple(self.contact_modes))


@dataclass(frozen=True)
class PhysicsSignature:
    feature_names: tuple[str, ...]
    values: np.ndarray
    dynamics_slack: np.ndarray
    solver_status: tuple[str, ...]
    raw_values: np.ndarray = field(default_factory=lambda: np.empty(0))

    def __post_init__(self):
        object.__setattr__(self, "feature_names", tuple(self.feature_names))
        object.__setattr__(self, "values", _readonly(self.values))
        object.__setattr__(self, "dynamics_slack", _readonly(self.dynamics_slack))
        object.__setattr__(self, "solver_status", tuple(self.solver_status))
        object.__setattr__(self, "raw_values", _readonly(self.raw_values))


@dataclass(frozen=True)
class RolloutResult:
    time: np.ndarray
    q: np.ndarray
    v: np.ndarray
    contact_wrenches: np.ndarray
    success: bool
    failure_reason: str
    torque_demand: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    applied_torque: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    normal_force_margin: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    friction_margin: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    cop_margin: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    active_contacts: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=bool))
    scheduled_contacts: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=bool))
    failure_index: int = -1
    runtime: float = 0.0
    peak_torque_joint: str = ""
    peak_torque_ratio: float | None = None

    def __post_init__(self):
        for name in (
            "time", "q", "v", "contact_wrenches", "torque_demand",
            "applied_torque", "normal_force_margin", "friction_margin", "cop_margin",
            "active_contacts", "scheduled_contacts",
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name)))

    @property
    def runtime_seconds(self):
        return self.runtime


def _frame_id(model, name):
    frame_id = model.getFrameId(name)
    if frame_id == model.nframes:
        raise ValueError(f"robot model does not contain frame {name!r}")
    return frame_id


def _readonly(values):
    result = np.array(values, copy=True)
    result.flags.writeable = False
    return result


def _collision_model_with_exclusions(collision_model, excluded_names):
    collision_model = pin.GeometryModel(collision_model)
    for first_name, second_name in excluded_names:
        if not (
            collision_model.existGeometryName(first_name)
            and collision_model.existGeometryName(second_name)
        ):
            raise ValueError(
                f"collision exclusion references missing geometry "
                f"{first_name!r}, {second_name!r}"
            )
        pair = pin.CollisionPair(
            collision_model.getGeometryId(first_name),
            collision_model.getGeometryId(second_name),
        )
        if not collision_model.existCollisionPair(pair):
            raise ValueError(
                f"collision exclusion is not a configured pair "
                f"{first_name!r}, {second_name!r}"
            )
        collision_model.removeCollisionPair(pair)
    return collision_model


@lru_cache(maxsize=None)
def _collada_up_axis(mesh_path):
    if Path(mesh_path).suffix.lower() != ".dae":
        return "Z_UP"
    root = ET.parse(mesh_path).getroot()
    return next(
        (
            element.text.strip()
            for element in root.iter()
            if element.tag.rsplit("}", 1)[-1] == "up_axis" and element.text
        ),
        "Z_UP",
    )


def _normalize_collision_mesh_axes(collision_model):
    y_up_to_z_up = pin.SE3(
        np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ]),
        np.zeros(3),
    )
    for geometry_object in collision_model.geometryObjects:
        up_axis = _collada_up_axis(geometry_object.meshPath)
        if up_axis == "Y_UP":
            geometry_object.placement = geometry_object.placement * y_up_to_z_up
        elif up_axis != "Z_UP":
            raise ValueError(f"unsupported COLLADA up axis {up_axis!r}")


def _sole_polygon_from_collision_geometry(
    model, collision_model, frame_id, surface_tolerance
):
    sole = model.frames[frame_id]
    horizontal_triangles = []
    for geometry_object in collision_model.geometryObjects:
        if geometry_object.parentJoint != sole.parentJoint:
            continue
        geometry = geometry_object.geometry
        if not hasattr(geometry, "vertices") or not hasattr(
            geometry, "tri_indices"
        ):
            raise ValueError("sole collision geometry must be a triangle mesh")
        transform = sole.placement.inverse() * geometry_object.placement
        vertices = np.asarray([
            transform.act(vertex) for vertex in np.asarray(geometry.vertices())
        ])
        for index in range(geometry.num_tris):
            triangle = geometry.tri_indices(index)
            points = vertices[[triangle[offset] for offset in range(3)]]
            normal = np.cross(points[1] - points[0], points[2] - points[0])
            norm = np.linalg.norm(normal)
            if norm and abs(normal[2]) / norm >= 0.99:
                horizontal_triangles.append((float(np.mean(points[:, 2])), points))
    if not horizontal_triangles:
        raise ValueError("sole joint has no collision geometry")
    lowest = min(height for height, _ in horizontal_triangles)
    points = np.vstack([
        triangle
        for height, triangle in horizontal_triangles
        if height <= lowest + surface_tolerance
    ])
    polygon = convex_hull(points[:, :2])
    if (
        polygon.shape[0] < 4
        or not np.isfinite(polygon).all()
        or np.any(np.ptp(polygon, axis=0) <= 0.0)
    ):
        raise ValueError("sole collision geometry has no valid support polygon")
    return _inset_polygon(polygon, surface_tolerance)


def _polygon_centroid(polygon):
    polygon = np.asarray(polygon, dtype=float)
    following = np.roll(polygon, -1, axis=0)
    cross = (
        polygon[:, 0] * following[:, 1]
        - following[:, 0] * polygon[:, 1]
    )
    area_scale = np.sum(cross)
    if len(polygon) < 3 or abs(area_scale) <= np.finfo(float).eps:
        raise ValueError("support polygon has zero area")
    return np.sum(
        (polygon + following) * cross[:, None], axis=0
    ) / (3.0 * area_scale)


def _inset_polygon(polygon, margin):
    center = _polygon_centroid(polygon)
    radius = polygon_margin(center, polygon)
    if not 0.0 <= margin < radius:
        raise ValueError("support polygon inset must be smaller than its inradius")
    return center + (polygon - center) * (1.0 - margin / radius)


def _rotated_polygon(polygon, rotation):
    points = np.column_stack((np.asarray(polygon, dtype=float), np.zeros(
        len(polygon)
    )))
    return convex_hull((np.asarray(rotation) @ points.T).T[:, :2])


def _sole_polygon_offsets(spec, frame_id, rotation=None):
    if frame_id == spec.left_sole_frame_id:
        polygon = spec.left_sole_polygon
        default_rotation = spec.left_sole_rotation
    elif frame_id == spec.right_sole_frame_id:
        polygon = spec.right_sole_polygon
        default_rotation = spec.right_sole_rotation
    else:
        raise ValueError("frame is not a configured sole")
    return _rotated_polygon(
        polygon, default_rotation if rotation is None else rotation
    )


def _sole_polygon(spec, frame_id, translation, rotation=None):
    return _sole_polygon_offsets(spec, frame_id, rotation) + np.asarray(
        translation, dtype=float
    )[:2]


def _level_neutral_soles(model, data, q, frame_ids, joint_names):
    indices = []
    for name in joint_names:
        joint = model.joints[model.getJointId(name)]
        if joint.nq != 1:
            raise ValueError(f"sole-leveling joint {name!r} must be scalar")
        indices.append(joint.idx_q)
    if len(indices) != 2 * len(frame_ids):
        raise ValueError("sole leveling requires pitch/roll joints for each foot")

    reference = np.asarray(q, dtype=float).copy()

    def residual(values):
        candidate = reference.copy()
        candidate[indices] = values
        pin.framesForwardKinematics(model, data, candidate)
        return np.asarray([
            data.oMf[frame_id].rotation[:2, 2] for frame_id in frame_ids
        ]).ravel()

    solution = least_squares(
        residual,
        reference[indices],
        bounds=(
            model.lowerPositionLimit[indices],
            model.upperPositionLimit[indices],
        ),
        ftol=1e-14,
        xtol=1e-14,
        gtol=1e-14,
        max_nfev=100,
    )
    if not solution.success or np.max(np.abs(residual(solution.x))) > 1e-10:
        raise ValueError("neutral sole frames cannot be leveled inside joint limits")
    q[indices] = solution.x


def load_robot_spec(name):
    if name not in _ROBOT_FRAMES:
        raise ValueError("name must be exactly 'talos' or 'icub'")
    frames = _ROBOT_FRAMES[name]
    robot = example_robot_data.load(frames.get("loader_name", name))
    _normalize_collision_mesh_axes(robot.collision_model)
    model, data = robot.model, robot.data
    for joint_name, value in frames.get("neutral_joints", ()):
        joint = model.joints[model.getJointId(joint_name)]
        if joint.nq != 1 or not (
            model.lowerPositionLimit[joint.idx_q]
            <= value
            <= model.upperPositionLimit[joint.idx_q]
        ):
            raise ValueError(f"invalid neutral value for {joint_name!r}")
        robot.q0[joint.idx_q] = value
    left_sole_id = _frame_id(model, frames["left_sole"])
    right_sole_id = _frame_id(model, frames["right_sole"])
    torso_id = _frame_id(model, frames["torso"])
    left_hip_id = _frame_id(model, frames["left_hip"])
    right_hip_id = _frame_id(model, frames["right_hip"])
    _level_neutral_soles(
        model,
        data,
        robot.q0,
        (left_sole_id, right_sole_id),
        frames["sole_leveling_joints"],
    )
    pin.framesForwardKinematics(model, data, robot.q0)
    left_sole = data.oMf[left_sole_id].translation
    right_sole = data.oMf[right_sole_id].translation
    leg_length = float(
        (np.linalg.norm(data.oMf[left_hip_id].translation - left_sole)
         + np.linalg.norm(data.oMf[right_hip_id].translation - right_sole)) / 2.0
    )
    sole_height = (left_sole[2] + right_sole[2]) / 2.0
    neutral_com_height = float(pin.centerOfMass(model, data, robot.q0)[2] - sole_height)
    mass = float(sum(inertia.mass for inertia in model.inertias))
    regularization = 1e-8 * mass * leg_length * leg_length
    model.armature[6:] = np.maximum(model.armature[6:], regularization)
    left_polygon = _sole_polygon_from_collision_geometry(
        model,
        robot.collision_model,
        left_sole_id,
        frames["sole_surface_tolerance"],
    )
    right_polygon = _sole_polygon_from_collision_geometry(
        model,
        robot.collision_model,
        right_sole_id,
        frames["sole_surface_tolerance"],
    )
    left_rotation = data.oMf[left_sole_id].rotation.copy()
    right_rotation = data.oMf[right_sole_id].rotation.copy()
    left_sizes = np.ptp(
        _rotated_polygon(left_polygon, left_rotation), axis=0
    ) / 2.0
    right_sizes = np.ptp(
        _rotated_polygon(right_polygon, right_rotation), axis=0
    ) / 2.0
    if not np.allclose(left_sizes, right_sizes, rtol=1e-3, atol=1e-5):
        raise ValueError("left and right sole collision dimensions disagree")
    sole_half_length, sole_half_width = np.mean(
        (left_sizes, right_sizes), axis=0
    )
    ignored_collision_pairs = tuple(frames.get("ignored_collision_pairs", ()))
    collision_model = _collision_model_with_exclusions(
        robot.collision_model, ignored_collision_pairs
    )
    return RobotSpec(
        name=name,
        robot=robot,
        model=model,
        collision_model=collision_model,
        left_sole_frame=frames["left_sole"],
        right_sole_frame=frames["right_sole"],
        torso_frame=frames["torso"],
        left_sole_frame_id=left_sole_id,
        right_sole_frame_id=right_sole_id,
        torso_frame_id=torso_id,
        mass=mass,
        leg_length=leg_length,
        neutral_com_height=neutral_com_height,
        position_lower_limits=_readonly(model.lowerPositionLimit),
        position_upper_limits=_readonly(model.upperPositionLimit),
        velocity_limits=_readonly(model.velocityLimit),
        effort_limits=_readonly(model.effortLimit),
        sole_half_length=sole_half_length,
        sole_half_width=sole_half_width,
        neutral_step_width=float(abs(left_sole[1] - right_sole[1])),
        left_sole_polygon=_readonly(left_polygon),
        right_sole_polygon=_readonly(right_polygon),
        left_sole_rotation=_readonly(left_rotation),
        right_sole_rotation=_readonly(right_rotation),
        ignored_collision_pairs=ignored_collision_pairs,
    )


def _clamp_joint_bounds(model, q):
    """Clamp Euclidean joints without touching free-flyer quaternion coordinates."""
    q = pin.normalize(model, q)
    for joint in model.joints[2:]:
        if joint.nq != joint.nv:
            continue
        indices = slice(joint.idx_q, joint.idx_q + joint.nq)
        low = model.lowerPositionLimit[indices]
        high = model.upperPositionLimit[indices]
        values = q[indices]
        q[indices] = np.where(np.isfinite(low), np.maximum(values, low), values)
        q[indices] = np.where(np.isfinite(high), np.minimum(q[indices], high), q[indices])
    return pin.normalize(model, q)


def _interior_neutral(model, q, fraction=0.02):
    q = pin.normalize(model, np.asarray(q, dtype=float).copy())
    for joint in model.joints[2:]:
        if joint.nq != joint.nv:
            continue
        indices = slice(joint.idx_q, joint.idx_q + joint.nq)
        lower = model.lowerPositionLimit[indices]
        upper = model.upperPositionLimit[indices]
        finite = np.isfinite(lower) & np.isfinite(upper) & (upper > lower)
        margin = fraction * (upper[finite] - lower[finite])
        values = q[indices]
        values[finite] = np.minimum(
            np.maximum(values[finite], lower[finite] + margin),
            upper[finite] - margin,
        )
        q[indices] = values
    return pin.normalize(model, q)


def _posture_joint_ids(model, frame_ids):
    task_joints = set()
    for frame_id in frame_ids:
        joint_id = model.frames[frame_id].parentJoint
        while joint_id:
            task_joints.add(joint_id)
            joint_id = model.parents[joint_id]
    return tuple(
        joint_id
        for joint_id in range(2, model.njoints)
        if joint_id not in task_joints
    )


def _bounded_manifold_ik(spec, q, left_target, right_target, com_target,
                         foot_rotations, torso_rotation, locked_joint_ids,
                         max_evaluations=60):
    """Solve one bounded, warm-started IK sample in the model tangent space."""
    model, data = spec.model, spec.model.createData()
    reference = pin.normalize(model, np.asarray(q, dtype=float).copy())
    locked = {
        index
        for joint_id in locked_joint_ids
        for index in range(
            model.joints[joint_id].idx_v,
            model.joints[joint_id].idx_v + model.joints[joint_id].nv,
        )
    }
    allowed = np.asarray([
        index for index in range(model.nv) if index not in locked
    ])
    allowed_lookup = {index: offset for offset, index in enumerate(allowed)}
    lower = np.full(len(allowed), -np.inf)
    upper = np.full(len(allowed), np.inf)
    for joint in model.joints[2:]:
        if joint.nq != joint.nv:
            continue
        for offset in range(joint.nv):
            velocity_index = joint.idx_v + offset
            if velocity_index not in allowed_lookup:
                continue
            position_index = joint.idx_q + offset
            bound_index = allowed_lookup[velocity_index]
            low = model.lowerPositionLimit[position_index] - reference[position_index]
            high = model.upperPositionLimit[position_index] - reference[position_index]
            if np.isfinite(low) and np.isfinite(high) and high > low:
                lower[bound_index] = min(low + 1e-10, -1e-12)
                upper[bound_index] = max(high - 1e-10, 1e-12)

    def expand(tangent):
        full = np.zeros(model.nv)
        full[allowed] = tangent
        return full

    def residual(tangent):
        configuration = pin.normalize(
            model, pin.integrate(model, reference, expand(tangent))
        )
        pin.forwardKinematics(model, data, configuration)
        pin.updateFramePlacements(model, data)
        errors = []
        for frame_id, target, rotation in (
            (spec.left_sole_frame_id, left_target, foot_rotations[0]),
            (spec.right_sole_frame_id, right_target, foot_rotations[1]),
        ):
            placement = data.oMf[frame_id]
            errors.extend(
                (target - placement.translation) / spec.leg_length
            )
            errors.extend(2.0 * pin.log3(rotation @ placement.rotation.T))
        torso = data.oMf[spec.torso_frame_id]
        errors.extend(
            0.5 * pin.log3(torso_rotation @ torso.rotation.T)
        )
        errors.extend(
            0.5 * (
                com_target - pin.centerOfMass(model, data, configuration)
            ) / spec.leg_length
        )
        errors.extend(1e-3 * tangent)
        return np.asarray(errors)

    solution = least_squares(
        residual,
        np.zeros(len(allowed)),
        bounds=(lower, upper),
        max_nfev=max_evaluations,
        ftol=1e-9,
        xtol=1e-9,
        gtol=1e-9,
    )
    q = _clamp_joint_bounds(
        model, pin.integrate(model, reference, expand(solution.x))
    )
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    hard_errors = []
    for frame_id, target, rotation in (
        (spec.left_sole_frame_id, left_target, foot_rotations[0]),
        (spec.right_sole_frame_id, right_target, foot_rotations[1]),
    ):
        placement = data.oMf[frame_id]
        hard_errors.extend(
            (target - placement.translation) / spec.leg_length
        )
        hard_errors.extend(
            2.0 * pin.log3(rotation @ placement.rotation.T)
        )
    hard_errors.extend(
        0.5 * pin.log3(
            torso_rotation @ data.oMf[spec.torso_frame_id].rotation.T
        )
    )
    hard_errors.extend(
        0.5 * (
            com_target - pin.centerOfMass(model, data, q)
        ) / spec.leg_length
    )
    error_norm = float(np.linalg.norm(hard_errors))
    return q, error_norm


def _smooth_lipm_reference(zmp, height, dt, start=None, end=None):
    if len(zmp) < 2:
        return zmp.copy()
    ratio = height / (9.81 * dt * dt)
    banded = np.zeros((3, len(zmp)))
    banded[1] = 1.0 + 2.0 * ratio
    banded[0, 1:] = -ratio
    banded[2, :-1] = -ratio
    banded[1, (0, -1)] = 1.0 + ratio
    rhs = np.asarray(zmp, dtype=float).copy()
    if start is not None:
        banded[1, 0] = 1.0
        banded[0, 1] = 0.0
        rhs[0] = start
    if end is not None:
        banded[1, -1] = 1.0
        banded[2, -2] = 0.0
        rhs[-1] = end
    return np.column_stack([
        solve_banded((1, 1), banded, rhs[:, axis]) for axis in range(2)
    ])


def _foot_reference(footsteps, t, side, ground_height, swing_height):
    current = np.asarray(getattr(footsteps, f"get_{side}_position")(t), dtype=float)
    phase = footsteps.get_phase_type(t)
    if phase != side:
        return np.r_[current, ground_height]
    target = np.asarray(getattr(footsteps, f"get_{side}_next_position")(t), dtype=float)
    duration = footsteps.get_phase_duration(t)
    alpha = 0.0 if duration <= 0.0 else (t - footsteps.get_phase_start(t)) / duration
    alpha = float(np.clip(alpha, 0.0, 1.0))
    blend = alpha ** 3 * (10.0 - 15.0 * alpha + 6.0 * alpha * alpha)
    return np.r_[(1.0 - blend) * current + blend * target,
                 ground_height + 64.0 * swing_height
                 * alpha ** 3 * (1.0 - alpha) ** 3]


def _active_frame_ids(spec, mode):
    if mode == "left":
        return (spec.left_sole_frame_id,)
    if mode == "right":
        return (spec.right_sole_frame_id,)
    if mode in ("double", "touchdown"):
        return (spec.left_sole_frame_id, spec.right_sole_frame_id)
    raise ValueError(f"unknown contact mode {mode!r}")


def _contact_kinematics(model, q, v, frame_ids):
    data = model.createData()
    pin.forwardKinematics(model, data, q, v, np.zeros(model.nv))
    pin.computeJointJacobians(model, data, q)
    pin.updateFramePlacements(model, data)
    jacobian = np.vstack([
        pin.getFrameJacobian(model, data, frame_id, pin.LOCAL_WORLD_ALIGNED)
        for frame_id in frame_ids
    ])
    drift = np.concatenate([
        pin.getFrameAcceleration(
            model, data, frame_id, pin.LOCAL_WORLD_ALIGNED
        ).vector
        for frame_id in frame_ids
    ])
    return jacobian, drift


def _project_contact_velocity(model, q, v, frame_ids):
    if not frame_ids:
        return np.asarray(v, dtype=float).copy()
    jacobian, _ = _contact_kinematics(model, q, v, frame_ids)
    return np.asarray(v, dtype=float) - np.linalg.lstsq(
        jacobian, jacobian @ v, rcond=1e-10
    )[0]


def _contact_consistent_derivatives(spec, q, modes, dt):
    model = spec.model
    velocity = np.zeros((len(q), model.nv))
    if len(q) > 1:
        velocity[1:] = np.asarray([
            pin.difference(model, q[index - 1], q[index]) / dt
            for index in range(1, len(q))
        ])
        velocity[0] = velocity[1]
    for index, mode in enumerate(modes):
        velocity[index] = _project_contact_velocity(
            model, q[index], velocity[index], _active_frame_ids(spec, mode)
        )

    acceleration = np.zeros_like(velocity)
    if len(q) > 1:
        acceleration[1:] = np.diff(velocity, axis=0) / dt
        acceleration[0] = acceleration[1]
    for index, mode in enumerate(modes):
        jacobian, drift = _contact_kinematics(
            model, q[index], velocity[index], _active_frame_ids(spec, mode)
        )
        acceleration[index] -= np.linalg.lstsq(
            jacobian,
            jacobian @ acceleration[index] + drift,
            rcond=1e-10,
        )[0]
    return velocity, acceleration


def build_whole_body_trajectory(spec, sample, steps=6, dt=0.01):
    """Build robot-neutral references and manifold-valid whole-body states."""
    from .gait import build_footsteps
    from .zmp import ZmpClass

    params = sample.to_gait_params(spec, steps=steps, dt=dt)
    footsteps = build_footsteps(params)
    time = np.arange(0.0, footsteps.timetime[-1], dt)
    zmp_trajectory = ZmpClass(
        footsteps,
        mode="adaptive",
        foot_length=params.foot_length,
        foot_width=params.foot_width,
        bias=(params.zmp_bias_x, params.zmp_bias_y),
        smooth_double_support=True,
        left_offset=_polygon_centroid(_sole_polygon_offsets(
            spec, spec.left_sole_frame_id
        )),
        right_offset=_polygon_centroid(_sole_polygon_offsets(
            spec, spec.right_sole_frame_id
        )),
    )
    zmp = np.asarray([zmp_trajectory(t) for t in time])

    model, neutral_data = spec.model, spec.model.createData()
    neutral = _interior_neutral(model, spec.robot.q0)
    pin.forwardKinematics(model, neutral_data, neutral)
    pin.updateFramePlacements(model, neutral_data)
    left_neutral = neutral_data.oMf[spec.left_sole_frame_id]
    right_neutral = neutral_data.oMf[spec.right_sole_frame_id]
    torso_rotation = neutral_data.oMf[spec.torso_frame_id].rotation.copy()
    ground_height = float((left_neutral.translation[2] + right_neutral.translation[2]) / 2.0)
    swing_height = 0.08 * spec.leg_length

    left = np.asarray([
        _foot_reference(footsteps, t, "left", ground_height, swing_height) for t in time
    ])
    right = np.asarray([
        _foot_reference(footsteps, t, "right", ground_height, swing_height) for t in time
    ])
    neutral_com = pin.centerOfMass(model, neutral_data, neutral)
    neutral_stance_center = (
        left_neutral.translation[:2] + right_neutral.translation[:2]
    ) / 2.0
    com_offset = neutral_com[:2] - neutral_stance_center
    stance_centers = (left[:, :2] + right[:, :2]) / 2.0
    desired_com_xy = _smooth_lipm_reference(
        zmp,
        params.com_height,
        dt,
        start=stance_centers[0] + com_offset,
        end=stance_centers[-1] + com_offset,
    )
    desired_com = np.column_stack((
        desired_com_xy,
        np.full(len(time), ground_height + params.com_height),
    ))

    modes = []
    for t in time:
        legacy_mode = footsteps.get_phase_type(t)
        mode = {"left": "right", "right": "left", "none": "double"}[legacy_mode]
        if mode == "double" and modes and modes[-1] in ("left", "right"):
            mode = "touchdown"
        modes.append(mode)

    q = np.empty((len(time), model.nq))
    previous = neutral
    rotations = (left_neutral.rotation.copy(), right_neutral.rotation.copy())
    posture_joint_ids = _posture_joint_ids(
        model,
        (
            spec.left_sole_frame_id,
            spec.right_sole_frame_id,
            spec.torso_frame_id,
        ),
    )
    for index in range(len(time)):
        previous, ik_error = _bounded_manifold_ik(
            spec,
            previous,
            left[index],
            right[index],
            desired_com[index],
            rotations,
            torso_rotation,
            posture_joint_ids,
        )
        if not np.isfinite(ik_error) or not np.isfinite(previous).all():
            raise RuntimeError(f"whole-body IK failed at sample {index}")
        q[index] = previous
    v, a = _contact_consistent_derivatives(spec, q, modes, dt)
    return WholeBodyTrajectory(
        time, q, v, a, left, right, desired_com, tuple(modes), dt
    )


def _contact_inequality_rows(
    spec,
    frame_ids,
    rotations,
    friction,
    variable_count,
    contact_start,
):
    rows = []
    torsional_friction = friction * min(
        spec.sole_half_length, spec.sole_half_width
    )
    for contact_index, (frame_id, rotation) in enumerate(
        zip(frame_ids, rotations)
    ):
        start = contact_start + 6 * contact_index
        for x_sign in (-1.0, 1.0):
            for y_sign in (-1.0, 1.0):
                row = np.zeros(variable_count)
                row[start] = x_sign
                row[start + 1] = y_sign
                row[start + 2] = -friction
                rows.append(row)
        polygon = _sole_polygon_offsets(spec, frame_id, rotation)
        for first, second in zip(polygon, np.roll(polygon, -1, axis=0)):
            edge = second - first
            row = np.zeros(variable_count)
            row[start + 2] = edge[0] * first[1] - edge[1] * first[0]
            row[start + 3] = -edge[0]
            row[start + 4] = -edge[1]
            rows.append(row)
        for sign in (-1.0, 1.0):
            row = np.zeros(variable_count)
            row[start + 2] = -torsional_friction
            row[start + 5] = sign
            rows.append(row)
    return rows


def _solve_inverse_dynamics(
    spec,
    q,
    v,
    desired_acceleration,
    frame_ids,
    friction,
    *,
    enforce_effort_limits=True,
    contact_targets=None,
):
    """Track acceleration subject to exact rigid-contact inverse dynamics."""
    model, data = spec.model, spec.model.createData()
    mass_matrix = pin.crba(model, data, q)
    mass_matrix = np.triu(mass_matrix) + np.triu(mass_matrix, 1).T
    nonlinear = pin.nonLinearEffects(model, data, q, v)
    pin.computeJointJacobians(model, data, q)
    pin.forwardKinematics(model, data, q, v, np.zeros(model.nv))
    pin.updateFramePlacements(model, data)

    n_tau = model.nv - 6
    n_mechanical = n_tau + 6 * len(frame_ids)
    generalized = np.zeros((model.nv, n_mechanical))
    generalized[6:, :n_tau] = np.eye(n_tau)
    for index, frame_id in enumerate(frame_ids):
        start = n_tau + 6 * index
        generalized[:, start:start + 6] = pin.getFrameJacobian(
            model, data, frame_id, pin.LOCAL_WORLD_ALIGNED
        ).T

    contact_jacobian = np.vstack([
        pin.getFrameJacobian(model, data, frame_id, pin.LOCAL_WORLD_ALIGNED)
        for frame_id in frame_ids
    ])
    contact_rhs = -np.concatenate([
        pin.getFrameAcceleration(
            model, data, frame_id, pin.LOCAL_WORLD_ALIGNED
        ).vector
        for frame_id in frame_ids
    ])
    if contact_targets is not None:
        omega = 4.0 * sqrt(9.81 / spec.leg_length)
        corrections = []
        for frame_id in frame_ids:
            local_error = pin.log6(
                contact_targets[frame_id].inverse() * data.oMf[frame_id]
            )
            rotation = contact_targets[frame_id].rotation
            placement_error = np.r_[
                rotation @ local_error.linear,
                rotation @ local_error.angular,
            ]
            velocity_error = pin.getFrameVelocity(
                model, data, frame_id, pin.LOCAL_WORLD_ALIGNED
            ).vector
            corrections.append(
                omega * omega * placement_error + 2.0 * omega * velocity_error
            )
        contact_rhs -= np.concatenate(corrections)

    bodyweight = spec.mass * 9.81
    effort_limits = np.abs(np.asarray(spec.effort_limits[6:], dtype=float))
    acceleration_scale = np.r_[
        np.full(3, 9.81),
        np.full(model.nv - 3, 9.81 / spec.leg_length),
    ]
    acceleration_start = 0
    mechanical_start = model.nv
    positive_slack_start = mechanical_start + n_mechanical
    negative_slack_start = positive_slack_start + model.nv
    n_primary = negative_slack_start + model.nv

    dynamics = np.zeros((model.nv, n_primary))
    dynamics[:, acceleration_start:mechanical_start] = mass_matrix
    dynamics[:, mechanical_start:positive_slack_start] = -generalized
    contact = np.zeros((contact_jacobian.shape[0], n_primary))
    contact[:, acceleration_start:mechanical_start] = contact_jacobian
    tracking = np.zeros((model.nv, n_primary))
    tracking[:, acceleration_start:mechanical_start] = np.eye(model.nv)
    tracking[:, positive_slack_start:negative_slack_start] = -np.diag(
        acceleration_scale
    )
    tracking[:, negative_slack_start:] = np.diag(acceleration_scale)
    equality = np.vstack((dynamics, contact, tracking))
    equality_rhs = np.concatenate((
        -nonlinear,
        contact_rhs,
        np.asarray(desired_acceleration, dtype=float),
    ))

    bounds = [(None, None)] * model.nv
    for limit in effort_limits:
        bounds.append(
            (-float(limit), float(limit))
            if enforce_effort_limits and np.isfinite(limit)
            else (None, None)
        )
    normal_floor = 1e-4 * bodyweight if len(frame_ids) == 1 else 0.0
    for _ in frame_ids:
        bounds.extend(((None, None), (None, None), (normal_floor, None),
                       (None, None), (None, None), (None, None)))
    bounds.extend([(0.0, None)] * (2 * model.nv))

    friction_capacity = friction * (1.0 - 1e-3)
    inequalities = np.asarray(_contact_inequality_rows(
        spec,
        frame_ids,
        [data.oMf[frame_id].rotation for frame_id in frame_ids],
        friction_capacity,
        n_primary,
        mechanical_start + n_tau,
    ))
    zeros = np.zeros(len(inequalities))

    objective = np.zeros(n_primary)
    objective[positive_slack_start:] = 1.0
    try:
        primary = linprog(
            objective,
            A_ub=inequalities,
            b_ub=zeros,
            A_eq=equality,
            b_eq=equality_rhs,
            bounds=bounds,
            method="highs",
        )
    except (TypeError, ValueError):
        return None, None, None, None, "primary_failed"
    if not primary.success:
        return None, None, None, None, f"primary_failed_{primary.status}"

    # Fix optimal tracking and remove arbitrary torque/wrench choices.
    n_second = n_primary + n_mechanical
    second_equality = np.pad(equality, ((0, 0), (0, n_mechanical)))
    second_inequalities = [np.pad(row, (0, n_mechanical)) for row in inequalities]
    slack_row = np.zeros(n_second)
    slack_row[positive_slack_start:n_primary] = 1.0
    second_inequalities.append(slack_row)
    second_rhs = [
        *zeros,
        max(float(primary.fun), 0.0)
        + 1e-7 * max(1.0, abs(float(primary.fun))),
    ]

    magnitude_scale = [*np.maximum(effort_limits, 1.0)]
    contact_force_scale = bodyweight / len(frame_ids)
    for _ in frame_ids:
        magnitude_scale.extend((contact_force_scale,) * 3)
        magnitude_scale.extend((contact_force_scale * spec.leg_length,) * 3)
    for index, scale in enumerate(magnitude_scale):
        for sign in (-1.0, 1.0):
            row = np.zeros(n_second)
            row[mechanical_start + index] = sign
            row[n_primary + index] = -1.0
            second_inequalities.append(row)
            second_rhs.append(0.0)
    second_objective = np.zeros(n_second)
    second_objective[n_primary:] = (
        1.0 + 1e-7 * np.arange(n_mechanical)
    ) / np.asarray(magnitude_scale)
    try:
        secondary = linprog(
            second_objective,
            A_ub=np.asarray(second_inequalities),
            b_ub=np.asarray(second_rhs),
            A_eq=second_equality,
            b_eq=equality_rhs,
            bounds=[*bounds, *([(0.0, None)] * n_mechanical)],
            method="highs",
        )
    except (TypeError, ValueError):
        return None, None, None, None, "secondary_failed"
    if not secondary.success:
        return None, None, None, None, f"secondary_failed_{secondary.status}"

    acceleration = secondary.x[:model.nv]
    mechanical = secondary.x[
        mechanical_start:mechanical_start + n_mechanical
    ]
    tracking_slack = float(np.max(
        np.abs(acceleration - desired_acceleration) / acceleration_scale
    ))
    torques = mechanical[:n_tau]
    wrenches = mechanical[n_tau:].reshape(len(frame_ids), 6)
    return acceleration, torques, wrenches, tracking_slack, "optimal"


def compute_physics_signature(spec, trajectory, sample):
    """Compute deterministic phase-wise inverse-dynamics feasibility features."""
    phases = ("single_support", "touchdown", "double_support")
    channels = (
        "torque", "friction", "cop", "zmp", "joint_position", "joint_velocity",
        "ik_residual", "dynamics_slack", "impact_proxy", "solver_failure",
    )
    statistics = ("min", "p05", "median", "p95", "max")
    cap = 1e3
    values = {channel: np.zeros(len(trajectory.time)) for channel in channels}
    raw_values = {channel: np.zeros(len(trajectory.time)) for channel in channels}
    dynamics_slack = np.zeros(len(trajectory.time))
    statuses = []
    model = spec.model
    bodyweight = spec.mass * 9.81
    com_acceleration = np.zeros((len(trajectory.time), 2))
    if len(trajectory.time) >= 3:
        com_velocity = np.gradient(
            trajectory.com[:, :2], trajectory.time, axis=0, edge_order=2
        )
        com_acceleration = np.gradient(
            com_velocity, trajectory.time, axis=0, edge_order=2
        )

    for index, (q, v, a, mode) in enumerate(zip(
        trajectory.q, trajectory.v, trajectory.a, trajectory.contact_modes
    )):
        if mode == "left":
            contacts = (spec.left_sole_frame_id,)
        elif mode == "right":
            contacts = (spec.right_sole_frame_id,)
        elif mode in ("double", "touchdown"):
            contacts = (spec.left_sole_frame_id, spec.right_sole_frame_id)
        else:
            raise ValueError(f"unknown contact mode {mode!r}")

        if mode == "left":
            support = _sole_polygon(
                spec,
                spec.left_sole_frame_id,
                trajectory.left_foot[index],
            )
            ground_height = trajectory.left_foot[index, 2]
        elif mode == "right":
            support = _sole_polygon(
                spec,
                spec.right_sole_frame_id,
                trajectory.right_foot[index],
            )
            ground_height = trajectory.right_foot[index, 2]
        else:
            support = convex_hull(np.vstack((
                _sole_polygon(
                    spec,
                    spec.left_sole_frame_id,
                    trajectory.left_foot[index],
                ),
                _sole_polygon(
                    spec,
                    spec.right_sole_frame_id,
                    trajectory.right_foot[index],
                ),
            )))
            ground_height = np.mean((
                trajectory.left_foot[index, 2],
                trajectory.right_foot[index, 2],
            ))
        com_height = max(trajectory.com[index, 2] - ground_height, 1e-6)
        reference_zmp = (
            trajectory.com[index, :2]
            - com_height * com_acceleration[index] / 9.81
        )
        values["zmp"][index] = (
            -polygon_margin(reference_zmp, support) / spec.sole_half_width
        )
        raw_values["zmp"][index] = -polygon_margin(reference_zmp, support)

        data = model.createData()
        pin.forwardKinematics(model, data, q)
        pin.computeJointJacobians(model, data, q)
        pin.updateFramePlacements(model, data)
        actual_com = pin.centerOfMass(model, data, q)
        values["ik_residual"][index] = max(
            np.linalg.norm(data.oMf[spec.left_sole_frame_id].translation
                           - trajectory.left_foot[index]),
            np.linalg.norm(data.oMf[spec.right_sole_frame_id].translation
                           - trajectory.right_foot[index]),
            np.linalg.norm(actual_com - trajectory.com[index]),
        ) / spec.leg_length
        raw_values["ik_residual"][index] = (
            values["ik_residual"][index] * spec.leg_length
        )

        position_violations = []
        raw_position_violations = []
        for joint in model.joints[2:]:
            if joint.nq != joint.nv:
                continue
            joint_slice = slice(joint.idx_q, joint.idx_q + joint.nq)
            lower = np.asarray(spec.position_lower_limits[joint_slice])
            upper = np.asarray(spec.position_upper_limits[joint_slice])
            finite = np.isfinite(lower) & np.isfinite(upper) & (upper > lower)
            if np.any(finite):
                center = (lower[finite] + upper[finite]) / 2.0
                radius = (upper[finite] - lower[finite]) / 2.0
                position_violations.extend(
                    np.abs((q[joint_slice][finite] - center) / radius) - 1.0
                )
                raw_position_violations.extend(np.maximum(
                    q[joint_slice][finite] - upper[finite],
                    lower[finite] - q[joint_slice][finite],
                ))
        values["joint_position"][index] = max(position_violations, default=0.0)
        raw_values["joint_position"][index] = max(
            raw_position_violations, default=0.0
        )

        velocity_limits = np.asarray(spec.velocity_limits[6:], dtype=float)
        finite_velocity = np.isfinite(velocity_limits) & (velocity_limits > 0.0)
        values["joint_velocity"][index] = (
            float(np.max(np.abs(v[6:][finite_velocity])
                         / velocity_limits[finite_velocity] - 1.0))
            if np.any(finite_velocity) else 0.0
        )
        raw_values["joint_velocity"][index] = (
            float(np.max(
                np.abs(v[6:][finite_velocity]) - velocity_limits[finite_velocity]
            ))
            if np.any(finite_velocity) else 0.0
        )

        if mode == "touchdown":
            previous = trajectory.contact_modes[index - 1] if index else "double"
            new_contacts = (
                (spec.right_sole_frame_id,) if previous == "left" else
                (spec.left_sole_frame_id,) if previous == "right" else contacts
            )
            impact_index = int(np.clip(
                index + round(sample.timing_error_seconds / trajectory.dt),
                1,
                len(trajectory.time) - 1,
            ))
            impact_q = trajectory.q[impact_index]
            preimpact_velocity = pin.difference(
                model,
                trajectory.q[impact_index - 1],
                impact_q,
            ) / trajectory.dt
            impact_data = model.createData()
            pin.forwardKinematics(model, impact_data, impact_q)
            pin.computeJointJacobians(model, impact_data, impact_q)
            pin.updateFramePlacements(model, impact_data)
            contact_speed = max(
                np.linalg.norm(pin.getFrameJacobian(
                    model, impact_data, frame_id, pin.LOCAL_WORLD_ALIGNED
                )[:3] @ preimpact_velocity)
                for frame_id in new_contacts
            )
            values["impact_proxy"][index] = (
                contact_speed / sqrt(9.81 * spec.leg_length) + sample.impulse
            )
            raw_values["impact_proxy"][index] = contact_speed + (
                sample.impulse * sqrt(9.81 * spec.leg_length)
            )

        solved_acceleration, torques, wrenches, residual, status = _solve_inverse_dynamics(
            spec, q, v, a, contacts, sample.friction
        )
        statuses.append(status)
        if status != "optimal":
            for channel in ("torque", "friction", "cop", "dynamics_slack"):
                values[channel][index] = cap
            raw_values["torque"][index] = cap * max(
                float(np.max(np.abs(spec.effort_limits[6:]))), 1.0
            )
            raw_values["friction"][index] = cap * bodyweight
            raw_values["cop"][index] = cap * spec.leg_length
            raw_values["dynamics_slack"][index] = cap * 9.81 / spec.leg_length
            values["solver_failure"][index] = 1.0
            raw_values["solver_failure"][index] = 1.0
            dynamics_slack[index] = cap
            continue

        effort_limits = np.abs(np.asarray(spec.effort_limits[6:], dtype=float))
        positive_effort = effort_limits > 0.0
        torque_violation = np.zeros_like(torques)
        torque_violation[positive_effort] = (
            np.abs(torques[positive_effort]) / effort_limits[positive_effort] - 1.0
        )
        torque_violation[~positive_effort] = np.where(
            np.abs(torques[~positive_effort]) <= 1e-10, 0.0, cap
        )
        values["torque"][index] = float(np.max(torque_violation))
        raw_values["torque"][index] = float(np.max(
            np.abs(torques) - effort_limits
        ))

        normal_scale = bodyweight / len(contacts)
        torsional_friction = sample.friction * min(
            spec.sole_half_length, spec.sole_half_width
        )
        friction_values, cop_values = [], []
        raw_friction_values, raw_cop_values = [], []
        for frame_id, wrench in zip(contacts, wrenches):
            fx, fy, fz, mx, my, mz = wrench
            friction_values.extend((
                (abs(fx) + abs(fy) - sample.friction * fz)
                / (sample.friction * normal_scale),
                -fz / normal_scale,
                (abs(mz) - torsional_friction * fz)
                / (torsional_friction * normal_scale),
            ))
            cop_margin = (
                polygon_margin(
                    np.array([-my / fz, mx / fz]),
                    _sole_polygon_offsets(
                        spec, frame_id, data.oMf[frame_id].rotation
                    ),
                )
                if fz > 0.0 else -cap * spec.leg_length
            )
            cop_values.append(-cop_margin / spec.sole_half_width)
            raw_friction_values.extend((
                abs(fx) + abs(fy) - sample.friction * fz,
                -fz,
                abs(mz) / min(spec.sole_half_length, spec.sole_half_width)
                - sample.friction * fz,
            ))
            raw_cop_values.append(-cop_margin)
        values["friction"][index] = max(friction_values)
        values["cop"][index] = max(cop_values)
        values["dynamics_slack"][index] = residual
        raw_values["friction"][index] = max(raw_friction_values)
        raw_values["cop"][index] = max(raw_cop_values)
        raw_values["dynamics_slack"][index] = float(np.max(
            np.abs(solved_acceleration - a)
        ))
        dynamics_slack[index] = residual

    phase_labels = np.asarray([
        "single_support" if mode in ("left", "right") else
        "double_support" if mode == "double" else mode
        for mode in trajectory.contact_modes
    ])
    feature_names, feature_values, raw_feature_values = [], [], []
    percentiles = (0, 5, 50, 95, 100)
    for phase in phases:
        mask = phase_labels == phase
        for channel in channels:
            aggregate = (
                np.percentile(np.clip(values[channel][mask], -cap, cap), percentiles)
                if np.any(mask) else np.zeros(len(statistics))
            )
            raw_aggregate = (
                np.percentile(raw_values[channel][mask], percentiles)
                if np.any(mask) else np.zeros(len(statistics))
            )
            for statistic, value, raw_value in zip(
                statistics, aggregate, raw_aggregate
            ):
                feature_names.append(f"{phase}.{channel}.{statistic}")
                feature_values.append(value)
                raw_feature_values.append(raw_value)
    return PhysicsSignature(
        tuple(feature_names),
        np.asarray(feature_values),
        dynamics_slack,
        tuple(statuses),
        np.asarray(raw_feature_values),
    )


def _plant_model(spec, sample):
    """Return the payload-perturbed plant without mutating the nominal model."""
    model = pin.Model(spec.model)
    payload_mass = sample.payload_fraction * spec.mass
    if payload_mass:
        torso = model.frames[spec.torso_frame_id]
        model.appendBodyToJoint(
            torso.parentJoint,
            pin.Inertia(payload_mass, np.zeros(3), np.zeros((3, 3))),
            torso.placement,
        )
    return model


def _shift_contact_modes(modes, sample_shift):
    modes = tuple(modes)
    if not modes:
        return modes
    return tuple(
        modes[min(max(index - sample_shift, 0), len(modes) - 1)]
        for index in range(len(modes))
    )


def _make_frame_contacts(
    spec, model, data, q, frame_ids, anchors=None, height_tolerance=None
):
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    frame_ids = tuple(frame_ids)
    height_tolerance = (
        1e-3 * spec.leg_length
        if height_tolerance is None
        else float(height_tolerance)
    )
    contact_models = pin.StdVec_RigidConstraintModel()
    contact_datas = pin.StdVec_RigidConstraintData()
    for frame_id in frame_ids:
        frame = model.frames[frame_id]
        target = data.oMf[frame_id].copy() if anchors is None else anchors[frame_id]
        if anchors is not None:
            height_error = abs(data.oMf[frame_id].translation[2] - target.translation[2])
            horizontal_error = np.linalg.norm(
                data.oMf[frame_id].translation[:2] - target.translation[:2]
            )
            rotation_error = np.linalg.norm(
                pin.log3(target.rotation @ data.oMf[frame_id].rotation.T)
            )
            if (
                height_error > height_tolerance
                or horizontal_error > 0.02 * spec.leg_length
                or rotation_error > np.deg2rad(5.0)
            ):
                raise _ContactReferenceError(
                    "scheduled contact is inconsistent with flat terrain"
                )
        contact = pin.RigidConstraintModel(
            pin.ContactType.CONTACT_6D,
            model,
            frame.parentJoint,
            frame.placement,
            0,
            target,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        contact_omega = 4.0 * sqrt(9.81 / spec.leg_length)
        contact.setBaumgarteCorrectorParameters(pin.BaumgarteCorrectorParameters(
            contact_omega * contact_omega, 2.0 * contact_omega
        ))
        contact.name = "left" if frame_id == spec.left_sole_frame_id else "right"
        contact_models.append(contact)
        contact_datas.append(contact.createData())
    pin.initConstraintDynamics(model, data, contact_models, contact_datas)
    return frame_ids, contact_models, contact_datas


def _make_contacts(spec, model, data, q, mode, anchors=None):
    return _make_frame_contacts(
        spec, model, data, q, _active_frame_ids(spec, mode), anchors
    )


def _first_single_support_midpoint(modes):
    for start, mode in enumerate(modes):
        if mode not in ("left", "right"):
            continue
        end = start + 1
        while end < len(modes) and modes[end] == mode:
            end += 1
        return start + (end - start) // 2
    return None


def _controller_torque(
    spec,
    q,
    v,
    q_ref,
    v_ref,
    a_ref,
    frame_ids,
    friction,
    contact_targets,
):
    omega = sqrt(9.81 / spec.leg_length)
    desired_acceleration = a_ref + (
        20.0 * omega * omega * pin.difference(spec.model, q, q_ref)
        + 8.0 * omega * (v_ref - v)
    )
    _, demand, _, _, status = _solve_inverse_dynamics(
        spec,
        q,
        v,
        desired_acceleration,
        frame_ids,
        friction,
        enforce_effort_limits=False,
        contact_targets=contact_targets,
    )
    if status != "optimal":
        raise RuntimeError(f"controller inverse dynamics failed: {status}")
    effort_limits = np.abs(np.asarray(spec.effort_limits[6:], dtype=float))
    if np.all(np.abs(demand) <= effort_limits):
        return demand, demand
    _, applied, _, _, status = _solve_inverse_dynamics(
        spec,
        q,
        v,
        desired_acceleration,
        frame_ids,
        friction,
        contact_targets=contact_targets,
    )
    if status != "optimal":
        raise RuntimeError(f"saturated inverse dynamics failed: {status}")
    return demand, applied


def _contact_margins(spec, sample, wrench, frame_id, rotation):
    fx, fy, fz, mx, my, mz = wrench
    normal = float(fz)
    supported_normal = max(normal, 0.0)
    radius = min(spec.sole_half_length, spec.sole_half_width)
    friction = float(min(
        sample.friction * supported_normal - np.hypot(fx, fy),
        sample.friction * supported_normal - abs(mz) / radius,
    ))
    cop_wrench = _cop_wrench_margin(
        spec, wrench, frame_id, rotation, supported_normal
    )
    cop = (
        cop_wrench / supported_normal
        if supported_normal > 0.0
        else 0.0 if cop_wrench >= 0.0 else -np.inf
    )
    return normal, friction, cop


def _cop_wrench_margin(spec, wrench, frame_id, rotation, normal=None):
    _, _, fz, mx, my, _ = wrench
    normal = max(float(fz), 0.0) if normal is None else float(normal)
    polygon = _sole_polygon_offsets(spec, frame_id, rotation)
    margins = []
    for first, second in zip(polygon, np.roll(polygon, -1, axis=0)):
        edge = second - first
        edge_length = np.linalg.norm(edge)
        offset = edge[0] * first[1] - edge[1] * first[0]
        margins.append(
            (edge[0] * mx + edge[1] * my - offset * normal) / edge_length
        )
    return float(min(margins))


def _contact_constraint_violations(
    spec,
    sample,
    wrench,
    frame_id,
    rotation,
    force_tolerance,
    cop_tolerance,
):
    normal, friction, _ = _contact_margins(
        spec, sample, wrench, frame_id, rotation
    )
    supported_normal = max(normal, 0.0)
    cop_wrench = _cop_wrench_margin(
        spec, wrench, frame_id, rotation, supported_normal
    )
    sole_radius = float(np.max(np.linalg.norm(
        _sole_polygon_offsets(spec, frame_id, rotation), axis=1
    )))
    cop_moment_tolerance = (
        supported_normal * cop_tolerance + force_tolerance * sole_radius
    )
    return (
        normal < -force_tolerance,
        friction < -force_tolerance,
        cop_wrench < -cop_moment_tolerance,
    )


def _terrain_penetration_is_failure(
    gap,
    normal_velocity,
    dt,
    leg_length,
    *,
    impact_transition=False,
    active_contact=False,
):
    tolerance = 1e-3 * leg_length
    if gap >= -tolerance:
        return False
    if gap < -2.0 * tolerance:
        return True
    velocity_tolerance = (
        _IMPACT_VELOCITY_TOLERANCE_RATIO * sqrt(9.81 * leg_length)
    )
    if impact_transition:
        return False
    if active_contact:
        return bool(normal_velocity <= velocity_tolerance)
    return bool(
        normal_velocity <= velocity_tolerance
        or gap + dt * normal_velocity < -tolerance
    )


def _dynamics_are_consistent(model, data, q, v, acceleration, tau, frame_ids, wrenches):
    mass_matrix = pin.crba(model, data, q)
    mass_matrix = np.triu(mass_matrix) + np.triu(mass_matrix, 1).T
    target = mass_matrix @ acceleration + pin.nonLinearEffects(model, data, q, v)
    pin.computeJointJacobians(model, data, q)
    pin.updateFramePlacements(model, data)
    supplied = tau.copy()
    for frame_id, wrench in zip(frame_ids, wrenches):
        supplied += pin.getFrameJacobian(
            model, data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        ).T @ wrench
    scale = max(float(np.max(np.abs(target))), 1.0)
    return np.max(np.abs(target - supplied)) <= 1e-7 * scale


def _resolve_impact(
    spec,
    sample,
    model,
    q,
    v,
    frame_ids,
):
    frame_ids = tuple(frame_ids)
    impulse_tolerance = 1e-8 * max(
        spec.mass * sqrt(9.81 * spec.leg_length), 1.0
    )
    velocity_tolerance = (
        _IMPACT_VELOCITY_TOLERANCE_RATIO
        * sqrt(9.81 * spec.leg_length)
    )
    cop_tolerance = _CONTINUOUS_COP_TOLERANCE_RATIO * spec.leg_length
    kinematic_data = model.createData()
    pin.computeJointJacobians(model, kinematic_data, q)
    pin.updateFramePlacements(model, kinematic_data)
    jacobians = {
        frame_id: pin.getFrameJacobian(
            model, kinematic_data, frame_id, pin.LOCAL_WORLD_ALIGNED
        ).copy()
        for frame_id in frame_ids
    }
    preimpact_normal_velocities = [
        float((jacobians[frame_id] @ v)[2]) for frame_id in frame_ids
    ]

    mass_matrix = pin.crba(model, kinematic_data, q)
    mass_matrix = np.triu(mass_matrix) + np.triu(mass_matrix, 1).T
    momentum_scale = np.r_[
        np.full(3, max(spec.mass * sqrt(9.81 * spec.leg_length), 1.0)),
        np.full(
            model.nv - 3,
            max(
                spec.mass * spec.leg_length
                * sqrt(9.81 * spec.leg_length),
                1.0,
            ),
        ),
    ]
    feasible = []
    for mask in range(1, 1 << len(frame_ids)):
        active = tuple(
            frame_id for index, frame_id in enumerate(frame_ids)
            if mask & (1 << index)
        )
        impact_data = model.createData()
        active_ids, active_models, active_datas = _make_frame_contacts(
            spec, model, impact_data, q, active
        )
        try:
            velocity = np.asarray(pin.impulseDynamics(
                model, impact_data, q, v, active_models, active_datas, 0.0
            )).copy()
        except Exception:
            continue
        impulses = [
            np.asarray(contact_data.contact_force.vector).copy()
            for contact_data in active_datas
        ]
        if any(impulse[2] < -impulse_tolerance for impulse in impulses):
            continue
        if any(
            np.max(np.abs(jacobians[frame_id] @ velocity))
            > velocity_tolerance
            for frame_id in active_ids
        ):
            continue
        if any(
            (jacobians[frame_id] @ velocity)[2] < -velocity_tolerance
            for frame_id in frame_ids if frame_id not in active_ids
        ):
            continue
        supplied_impulse = sum(
            (
                jacobians[frame_id].T @ impulse
                for frame_id, impulse in zip(active_ids, impulses)
            ),
            start=np.zeros(model.nv),
        )
        momentum_residual = (
            mass_matrix @ (velocity - v) - supplied_impulse
        ) / momentum_scale
        if np.max(np.abs(momentum_residual)) > 1e-10:
            continue
        violations = [
            _contact_constraint_violations(
                spec,
                sample,
                impulse,
                frame_id,
                kinematic_data.oMf[frame_id].rotation,
                impulse_tolerance,
                cop_tolerance,
            )
            for frame_id, impulse in zip(active_ids, impulses)
        ]
        if any(any(contact_violations) for contact_violations in violations):
            continue
        kinetic_energy = float(velocity @ mass_matrix @ velocity) / 2.0
        feasible.append((kinetic_energy, -len(active_ids), active_ids, velocity))
    if not feasible:
        reason = (
            "normal_force"
            if all(value > velocity_tolerance
                   for value in preimpact_normal_velocities)
            else "impact_dynamics"
        )
        if reason == "normal_force":
            return np.asarray(v).copy(), (), reason
        try:
            point_models = pin.StdVec_ConstraintModel()
            point_datas = pin.StdVec_ConstraintData()
            point_jacobians = []
            point_owners = []
            for frame_id in frame_ids:
                frame = model.frames[frame_id]
                frame_placement = kinematic_data.oMf[frame_id]
                polygon = (
                    spec.left_sole_polygon
                    if frame_id == spec.left_sole_frame_id
                    else spec.right_sole_polygon
                )
                for point in polygon:
                    world_point = frame_placement.act(np.r_[point, 0.0])
                    world_placement = pin.SE3(np.eye(3), world_point)
                    joint_placement = (
                        kinematic_data.oMi[frame.parentJoint].inverse()
                        * world_placement
                    )
                    contact = pin.PointContactConstraintModel(
                        model,
                        0,
                        world_placement,
                        frame.parentJoint,
                        joint_placement,
                    )
                    contact.setFriction(sample.friction)
                    contact_data = contact.createData()
                    contact.calc(model, kinematic_data, contact_data)
                    point_models.append(contact)
                    point_datas.append(contact_data)
                    point_jacobians.append(np.asarray(
                        contact.jacobian(model, kinematic_data, contact_data)
                    ).copy())
                    point_owners.append((frame_id, world_point))
            point_jacobian = np.vstack(point_jacobians)
            mass_response = np.linalg.solve(mass_matrix, point_jacobian.T)
            delassus_matrix = point_jacobian @ mass_response
            delassus_matrix = (delassus_matrix + delassus_matrix.T) / 2.0
            free_point_velocity = point_jacobian @ v
            solver_tolerance = 1e-8
            iteration_tolerance = solver_tolerance / 2.0
            settings = pin.ADMMSolverSettings()
            settings.absolute_complementarity_tol = iteration_tolerance
            settings.relative_complementarity_tol = iteration_tolerance
            settings.absolute_feasibility_tol = iteration_tolerance
            settings.relative_feasibility_tol = iteration_tolerance
            settings.max_iterations = 10000
            settings.solve_ncp = True
            settings.stat_record = False
            settings.warmstart_rho_with_previous_result = False
            result = pin.ADMMSolverResult()
            converged = pin.ADMMConstraintSolver(
                len(free_point_velocity)
            ).solve(
                pin.DelassusOperatorDense(delassus_matrix),
                free_point_velocity,
                point_models,
                point_datas,
                settings,
                result,
            )
            impulses = result.retrieveConstraintImpulses()
            velocity = np.asarray(v) + mass_response @ impulses
        except (np.linalg.LinAlgError, FloatingPointError, RuntimeError):
            return np.asarray(v).copy(), (), reason

        point_velocities = point_jacobian @ velocity
        solver_residuals = np.array([
            result.complementarity,
            result.primal_feasibility,
            result.dual_feasibility,
        ])
        if (
            not converged
            or not result.converged
            or not np.isfinite(np.r_[
                delassus_matrix.ravel(),
                free_point_velocity,
                impulses,
                velocity,
                point_velocities,
                solver_residuals,
            ]).all()
            or np.max(np.abs(solver_residuals)) > solver_tolerance
            or np.any(point_velocities.reshape(-1, 3)[:, 2]
                      < -velocity_tolerance)
        ):
            return np.asarray(v).copy(), (), reason
        point_impulses = impulses.reshape(-1, 3)
        if any(
            impulse[2] < -impulse_tolerance
            or np.linalg.norm(impulse[:2])
            > sample.friction * max(impulse[2], 0.0) + impulse_tolerance
            for impulse in point_impulses
        ):
            return np.asarray(v).copy(), (), reason

        wrenches = {frame_id: np.zeros(6) for frame_id in frame_ids}
        pressure_wrenches = {frame_id: np.zeros(6) for frame_id in frame_ids}
        for (frame_id, world_point), impulse in zip(
            point_owners, point_impulses
        ):
            offset = world_point - kinematic_data.oMf[frame_id].translation
            wrench = wrenches[frame_id]
            wrench[:3] += impulse
            wrench[3:] += np.cross(offset, impulse)
            normal_impulse = np.array([0.0, 0.0, max(impulse[2], 0.0)])
            pressure_wrench = pressure_wrenches[frame_id]
            pressure_wrench[:3] += normal_impulse
            pressure_wrench[3:] += np.cross(offset, normal_impulse)
        for frame_id in frame_ids:
            wrench = wrenches[frame_id]
            supported_normal = max(wrench[2], 0.0)
            pressure_cop_violation = _contact_constraint_violations(
                spec,
                sample,
                pressure_wrenches[frame_id],
                frame_id,
                kinematic_data.oMf[frame_id].rotation,
                impulse_tolerance,
                cop_tolerance,
            )[2]
            if (
                wrench[2] < -impulse_tolerance
                or np.linalg.norm(wrench[:2])
                > sample.friction * supported_normal + impulse_tolerance
                or pressure_cop_violation
            ):
                return np.asarray(v).copy(), (), reason
        supplied_impulse = point_jacobian.T @ impulses
        momentum_residual = (
            mass_matrix @ (velocity - v) - supplied_impulse
        ) / momentum_scale
        energy_scale = max(spec.mass * 9.81 * spec.leg_length, 1.0)
        if (
            np.max(np.abs(momentum_residual)) >= 1e-10
            or float(velocity @ mass_matrix @ velocity)
            > float(v @ mass_matrix @ v) + 2e-10 * energy_scale
        ):
            return np.asarray(v).copy(), (), reason
        active_ids = tuple(
            frame_id for frame_id in frame_ids
            if (
                wrenches[frame_id][2] > impulse_tolerance
                or any(
                    owner_id == frame_id
                    and abs(impulse[2]) <= impulse_tolerance
                    and abs(point_velocity[2]) <= velocity_tolerance
                    for (owner_id, _), impulse, point_velocity in zip(
                        point_owners, point_impulses,
                        point_velocities.reshape(-1, 3),
                    )
                )
            )
        )
        if not active_ids:
            return np.asarray(v).copy(), (), reason
        return velocity, active_ids, ""
    _, _, active_ids, velocity = min(feasible, key=lambda item: item[:3])
    return velocity, active_ids, ""


def _position_exceeds_limits(spec, model, q):
    for joint in model.joints[2:]:
        if joint.nq != joint.nv:
            continue
        indices = slice(joint.idx_q, joint.idx_q + joint.nq)
        lower = np.asarray(spec.position_lower_limits[indices])
        upper = np.asarray(spec.position_upper_limits[indices])
        finite = np.isfinite(lower) & np.isfinite(upper) & (upper > lower)
        tolerance = 0.01 * (upper[finite] - lower[finite])
        if np.any(q[indices][finite] < lower[finite] - tolerance) or np.any(
            q[indices][finite] > upper[finite] + tolerance
        ):
            return True
    return False


def _validate_rollout_trajectory(spec, trajectory):
    count = len(trajectory.time)
    expected = {
        "q": (count, spec.model.nq),
        "v": (count, spec.model.nv),
        "a": (count, spec.model.nv),
        "left_foot": (count, 3),
        "right_foot": (count, 3),
        "com": (count, 3),
    }
    if count == 0 or len(trajectory.contact_modes) != count:
        raise ValueError("trajectory time and contact modes must be non-empty and aligned")
    if not isfinite(trajectory.dt) or trajectory.dt <= 0.0:
        raise ValueError("trajectory dt must be positive and finite")
    if count > 1 and (
        np.any(np.diff(trajectory.time) <= 0.0)
        or not np.allclose(np.diff(trajectory.time), trajectory.dt)
    ):
        raise ValueError("trajectory time must be strictly increasing at dt")
    for name, shape in expected.items():
        values = np.asarray(getattr(trajectory, name))
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError(f"trajectory {name} must have finite shape {shape}")
    if not all(pin.isNormalized(spec.model, q, 1e-8) for q in trajectory.q):
        raise ValueError("trajectory contains a non-normalized configuration")


def rollout(spec, trajectory, sample):
    """Run the payload-perturbed closed-loop constrained plant at trajectory.dt."""
    started = perf_counter()
    _validate_rollout_trajectory(spec, trajectory)
    count = len(trajectory.time)
    model = _plant_model(spec, sample)
    data = model.createData()
    collision_data = pin.GeometryData(spec.collision_model)
    modes = _shift_contact_modes(
        trajectory.contact_modes, round(sample.timing_error_seconds / trajectory.dt)
    )
    impulse_index = _first_single_support_midpoint(modes)
    rng = np.random.default_rng(sample.seed)
    angle = rng.uniform(0.0, 2.0 * np.pi)
    impulse_direction = np.array([np.cos(angle), np.sin(angle), 0.0])
    plant_mass = float(sum(inertia.mass for inertia in model.inertias))
    force_tolerance = 1e-6 * plant_mass * 9.81
    cop_tolerance = _CONTINUOUS_COP_TOLERANCE_RATIO * spec.leg_length

    q = np.asarray(trajectory.q[0], dtype=float).copy()
    v = np.asarray(trajectory.v[0], dtype=float).copy()
    q_history = np.repeat(q[None], count, axis=0)
    v_history = np.repeat(v[None], count, axis=0)
    joint_count = model.nv - 6
    torque_demand = np.zeros((count, joint_count))
    applied_torque = np.zeros_like(torque_demand)
    contact_wrenches = np.zeros((count, 2, 6))
    active_contacts = np.zeros((count, 2), dtype=bool)
    scheduled_contacts = np.zeros((count, 2), dtype=bool)
    normal_margin = np.full((count, 2), np.inf)
    friction_margin = np.full((count, 2), np.inf)
    cop_margin = np.full((count, 2), np.inf)
    effort_limits = np.abs(np.asarray(spec.effort_limits[6:], dtype=float))
    velocity_limits = np.abs(np.asarray(spec.velocity_limits[6:], dtype=float))
    persistence = {"torque": 0, "joint_position": 0, "joint_velocity": 0}
    required_samples = max(2, int(np.ceil(0.02 / trajectory.dt - 1e-12)) + 1)
    failure_reason = ""
    failure_index = -1
    peak_torque_joint = ""
    peak_torque_ratio = None
    previous_scheduled_mode = None
    early_touchdown = False
    swing_was_airborne = False
    suppressed_contacts = set()
    frame_ids = ()
    contact_models = pin.StdVec_RigidConstraintModel()
    contact_datas = pin.StdVec_RigidConstraintData()
    anchors = {}

    neutral_data = spec.model.createData()
    pin.forwardKinematics(spec.model, neutral_data, spec.robot.q0)
    pin.updateFramePlacements(spec.model, neutral_data)
    neutral_ground = float(np.mean([
        neutral_data.oMf[spec.left_sole_frame_id].translation[2],
        neutral_data.oMf[spec.right_sole_frame_id].translation[2],
    ]))
    neutral_rotations = {
        spec.left_sole_frame_id:
            neutral_data.oMf[spec.left_sole_frame_id].rotation.copy(),
        spec.right_sole_frame_id:
            neutral_data.oMf[spec.right_sole_frame_id].rotation.copy(),
    }
    neutral_pelvis_height = float(spec.robot.q0[2] - neutral_ground)

    for index, scheduled_mode in enumerate(modes):
        q_history[index] = q
        v_history[index] = v
        try:
            scheduled_ids = _active_frame_ids(spec, scheduled_mode)
            scheduled_contacts[index] = (
                spec.left_sole_frame_id in scheduled_ids,
                spec.right_sole_frame_id in scheduled_ids,
            )
            if (
                scheduled_mode in ("left", "right")
                and scheduled_mode != previous_scheduled_mode
            ):
                early_touchdown = False
                swing_was_airborne = False
            mode = scheduled_mode
            if scheduled_mode in ("left", "right"):
                if early_touchdown:
                    mode = "double"
                else:
                    swing_frame = (
                        spec.right_sole_frame_id
                        if scheduled_mode == "left"
                        else spec.left_sole_frame_id
                    )
                    pin.forwardKinematics(model, data, q, v)
                    pin.updateFramePlacements(model, data)
                    swing_velocity = pin.getFrameVelocity(
                        model, data, swing_frame, pin.LOCAL_WORLD_ALIGNED
                    ).linear
                    swing_was_airborne |= (
                        data.oMf[swing_frame].translation[2]
                        > neutral_ground + 0.005 * spec.leg_length
                    )
                    if (
                        swing_was_airborne
                        and
                        data.oMf[swing_frame].translation[2]
                        <= neutral_ground + 1e-6 * spec.leg_length
                        and swing_velocity[2]
                        < -1e-6 * sqrt(9.81 * spec.leg_length)
                    ):
                        mode = "touchdown"
                        early_touchdown = True
            elif scheduled_mode == "touchdown" and early_touchdown:
                mode = "double"
            previous_scheduled_mode = scheduled_mode

            desired_ids = _active_frame_ids(spec, mode)
            suppressed_contacts.intersection_update(desired_ids)
            pin.forwardKinematics(model, data, q, v)
            pin.updateFramePlacements(model, data)
            gaps = {
                frame_id:
                data.oMf[frame_id].translation[2] - neutral_ground
                for frame_id in (
                    spec.left_sole_frame_id,
                    spec.right_sole_frame_id,
                )
            }
            normal_velocities = {
                frame_id: pin.getFrameVelocity(
                    model, data, frame_id, pin.LOCAL_WORLD_ALIGNED
                ).linear[2]
                for frame_id in gaps
            }
            recontacts = {
                frame_id
                for frame_id in suppressed_contacts
                if (
                    gaps[frame_id] <= 1e-6 * spec.leg_length
                    and normal_velocities[frame_id]
                    < -1e-6 * sqrt(9.81 * spec.leg_length)
                )
            }
            impact_contacts = set(recontacts)
            if mode == "touchdown":
                impact_contacts.update(set(desired_ids) - set(frame_ids))
            if any(
                _terrain_penetration_is_failure(
                    gaps[frame_id],
                    normal_velocities[frame_id],
                    trajectory.dt,
                    spec.leg_length,
                    impact_transition=frame_id in impact_contacts,
                    active_contact=frame_id in frame_ids,
                )
                for frame_id in (
                    spec.left_sole_frame_id,
                    spec.right_sole_frame_id,
                )
            ):
                failure_reason = "impact_dynamics"
                failure_index = index
                break
            suppressed_contacts.difference_update(recontacts)
            candidate_ids = tuple(
                frame_id for frame_id in desired_ids
                if frame_id not in suppressed_contacts
            )
            impact_required = mode == "touchdown" or bool(recontacts)
            new_anchors = {}
            reference_positions = {
                spec.left_sole_frame_id: trajectory.left_foot[index],
                spec.right_sole_frame_id: trajectory.right_foot[index],
            }
            for frame_id in candidate_ids:
                if frame_id in anchors and frame_id in frame_ids:
                    new_anchors[frame_id] = anchors[frame_id]
                    continue
                xy = (
                    data.oMf[frame_id].translation[:2]
                    if impact_required else reference_positions[frame_id][:2]
                )
                new_anchors[frame_id] = pin.SE3(
                    neutral_rotations[frame_id],
                    np.r_[xy, neutral_ground],
                )
            if impact_required:
                v, selected_ids, failure_reason = _resolve_impact(
                    spec,
                    sample,
                    model,
                    q,
                    v,
                    candidate_ids,
                )
                v_history[index] = v
                if failure_reason:
                    failure_index = index
                    break
                suppressed_contacts.update(
                    set(candidate_ids) - set(selected_ids)
                )
                candidate_ids = selected_ids
                new_anchors = {
                    frame_id: new_anchors[frame_id]
                    for frame_id in candidate_ids
                }
            if not candidate_ids:
                failure_reason = "normal_force"
                failure_index = index
                break
            if candidate_ids != frame_ids:
                anchors = new_anchors
                frame_ids, contact_models, contact_datas = _make_frame_contacts(
                    spec,
                    model,
                    data,
                    q,
                    candidate_ids,
                    anchors,
                    (
                        2e-3 * spec.leg_length
                        if impact_required
                        else None
                    ),
                )
            demand, applied = _controller_torque(
                spec,
                q,
                v,
                trajectory.q[index],
                trajectory.v[index],
                trajectory.a[index],
                frame_ids,
                sample.friction,
                anchors,
            )
            torque_demand[index] = demand
            applied_torque[index] = applied
            tau = np.r_[np.zeros(6), applied]
            dynamics_tau = tau
            if index == impulse_index and sample.impulse > 0.0:
                impulse_data = model.createData()
                com_jacobian = pin.jacobianCenterOfMass(model, impulse_data, q)
                magnitude = (
                    plant_mass * sqrt(9.81 * spec.leg_length) * sample.impulse
                    / trajectory.dt
                )
                dynamics_tau = tau + com_jacobian.T @ (
                    magnitude * impulse_direction
                )
            settings = pin.ProximalSettings(1e-12, 1e-12, 1e-10, 20)
            acceleration = np.asarray(pin.constraintDynamics(
                model,
                data,
                q,
                v,
                dynamics_tau,
                contact_models,
                contact_datas,
                settings,
            )).copy()
            wrenches = np.asarray([
                contact_data.contact_force.vector.copy() for contact_data in contact_datas
            ])
            if not all(np.isfinite(values).all() for values in (
                q, v, demand, applied, acceleration, wrenches
            )) or not _dynamics_are_consistent(
                model, data, q, v, acceleration, dynamics_tau, frame_ids, wrenches
            ):
                raise FloatingPointError("invalid constrained dynamics result")

            absolute_demand = np.abs(demand)
            torque_ratios = np.divide(
                absolute_demand,
                effort_limits,
                out=np.where(absolute_demand > 0.0, np.inf, 0.0),
                where=effort_limits > 0.0,
            )
            column = int(np.argmax(torque_ratios))
            ratio = float(torque_ratios[column])
            if peak_torque_ratio is None or ratio > peak_torque_ratio:
                velocity_index = column + 6
                joint_id = next(
                    joint_id
                    for joint_id, joint in enumerate(model.joints)
                    if joint.idx_v <= velocity_index < joint.idx_v + joint.nv
                )
                local_index = velocity_index - model.joints[joint_id].idx_v
                peak_torque_joint = model.names[joint_id]
                if model.joints[joint_id].nv > 1:
                    peak_torque_joint += f"[{local_index}]"
                peak_torque_ratio = ratio

            constraint_violations = [
                _contact_constraint_violations(
                    spec,
                    sample,
                    wrench,
                    frame_id,
                    data.oMf[frame_id].rotation,
                    force_tolerance,
                    cop_tolerance,
                )
                for frame_id, wrench in zip(frame_ids, wrenches)
            ]
            if len(frame_ids) > 1:
                separation_speed = (
                    _IMPACT_VELOCITY_TOLERANCE_RATIO
                    * sqrt(9.81 * spec.leg_length)
                )
                penetration_tolerance = 1e-3 * spec.leg_length
                for dropped_id, violations in zip(
                    frame_ids, constraint_violations
                ):
                    if not any(violations):
                        continue
                    trial_ids = tuple(
                        frame_id for frame_id in frame_ids
                        if frame_id != dropped_id
                    )
                    trial_anchors = {
                        frame_id: anchors[frame_id] for frame_id in trial_ids
                    }
                    trial_data = model.createData()
                    (
                        trial_ids,
                        trial_models,
                        trial_datas,
                    ) = _make_frame_contacts(
                        spec,
                        model,
                        trial_data,
                        q,
                        trial_ids,
                        trial_anchors,
                    )
                    trial_acceleration = np.asarray(pin.constraintDynamics(
                        model,
                        trial_data,
                        q,
                        v,
                        dynamics_tau,
                        trial_models,
                        trial_datas,
                        settings,
                    )).copy()
                    trial_wrenches = np.asarray([
                        contact_data.contact_force.vector.copy()
                        for contact_data in trial_datas
                    ])
                    if (
                        not np.isfinite(trial_acceleration).all()
                        or not np.isfinite(trial_wrenches).all()
                        or not _dynamics_are_consistent(
                            model,
                            trial_data,
                            q,
                            v,
                            trial_acceleration,
                            dynamics_tau,
                            trial_ids,
                            trial_wrenches,
                        )
                    ):
                        continue
                    trial_violations = [
                        _contact_constraint_violations(
                            spec,
                            sample,
                            wrench,
                            frame_id,
                            trial_data.oMf[frame_id].rotation,
                            force_tolerance,
                            cop_tolerance,
                        )
                        for frame_id, wrench in zip(
                            trial_ids, trial_wrenches
                        )
                    ]
                    if any(any(values) for values in trial_violations):
                        continue
                    pin.forwardKinematics(
                        model, trial_data, q, v, trial_acceleration
                    )
                    pin.updateFramePlacements(model, trial_data)
                    normal_velocity = pin.getFrameVelocity(
                        model, trial_data, dropped_id, pin.LOCAL_WORLD_ALIGNED
                    ).linear[2]
                    normal_acceleration = pin.getFrameClassicalAcceleration(
                        model, trial_data, dropped_id, pin.LOCAL_WORLD_ALIGNED
                    ).linear[2]
                    gap = (
                        trial_data.oMf[dropped_id].translation[2]
                        - neutral_ground
                    )
                    next_velocity = v + trajectory.dt * trial_acceleration
                    next_q = pin.integrate(
                        model, q, trajectory.dt * next_velocity
                    )
                    next_data = model.createData()
                    pin.framesForwardKinematics(model, next_data, next_q)
                    next_gap = (
                        next_data.oMf[dropped_id].translation[2]
                        - neutral_ground
                    )
                    if (
                        gap < -penetration_tolerance
                        or next_gap < -penetration_tolerance
                        or next_gap
                        < gap
                        - _CONTINUOUS_LIFTOFF_GAP_TOLERANCE_RATIO
                        * spec.leg_length
                        or normal_velocity
                        + trajectory.dt * normal_acceleration
                        <= separation_speed
                    ):
                        continue
                    suppressed_contacts.add(dropped_id)
                    anchors = trial_anchors
                    data = trial_data
                    frame_ids = trial_ids
                    contact_models = trial_models
                    contact_datas = trial_datas
                    acceleration = trial_acceleration
                    wrenches = trial_wrenches
                    constraint_violations = trial_violations
                    break

            for frame_id, wrench in zip(frame_ids, wrenches):
                side = 0 if frame_id == spec.left_sole_frame_id else 1
                contact_wrenches[index, side] = wrench
                normal_margin[index, side], friction_margin[index, side], cop_margin[
                    index, side
                ] = _contact_margins(
                    spec,
                    sample,
                    wrench,
                    frame_id,
                    data.oMf[frame_id].rotation,
                )

            torque_excess = np.any(
                np.where(effort_limits > 0.0,
                         np.abs(applied) > 1.01 * effort_limits,
                         np.abs(applied) > 0.0)
            )
            finite_velocity = np.isfinite(velocity_limits) & (velocity_limits > 0.0)
            velocity_excess = np.any(
                np.abs(v[6:][finite_velocity]) > 1.01 * velocity_limits[finite_velocity]
            )
            excesses = {
                "torque": torque_excess,
                "joint_position": _position_exceeds_limits(spec, model, q),
                "joint_velocity": velocity_excess,
            }
            for name, exceeds in excesses.items():
                persistence[name] = persistence[name] + 1 if exceeds else 0

            active_sides = [
                0 if frame_id == spec.left_sole_frame_id else 1 for frame_id in frame_ids
            ]
            active_contacts[index, active_sides] = True
            if any(values[0] for values in constraint_violations):
                failure_reason = "normal_force"
            if not failure_reason and any(
                values[1] for values in constraint_violations
            ):
                failure_reason = "friction"
            elif not failure_reason and any(
                values[2] for values in constraint_violations
            ):
                failure_reason = "cop"
            elif persistence["torque"] >= required_samples:
                failure_reason = "torque"
            elif persistence["joint_position"] >= required_samples:
                failure_reason = "joint_position"
            elif persistence["joint_velocity"] >= required_samples:
                failure_reason = "joint_velocity"
            elif pin.computeCollisions(
                model, data, spec.collision_model, collision_data, q, True
            ):
                failure_reason = "collision"
            else:
                base_rotation = pin.XYZQUATToSE3(q[:7]).rotation
                roll, pitch = pin.rpy.matrixToRpy(base_rotation)[:2]
                if (
                    q[2] - neutral_ground < 0.65 * neutral_pelvis_height
                    or abs(roll) > np.pi / 6.0
                    or abs(pitch) > np.pi / 6.0
                ):
                    failure_reason = "fall"
                else:
                    actual_com = pin.centerOfMass(model, data, q)
                    if np.linalg.norm(actual_com - trajectory.com[index]) > 0.12 * spec.leg_length:
                        failure_reason = "tracking"
            if failure_reason:
                failure_index = index
                break

            if index + 1 < count:
                v = v + trajectory.dt * acceleration
                q = pin.integrate(model, q, trajectory.dt * v)
                if not pin.isNormalized(model, q, 1e-8):
                    raise FloatingPointError("integrator returned invalid configuration")
        except _ContactReferenceError:
            failure_reason = "tracking"
            failure_index = index
            break
        except Exception:
            failure_reason = "dynamics"
            failure_index = index
            break

    if failure_index >= 0 and failure_index + 1 < count:
        q_history[failure_index + 1:] = q
        v_history[failure_index + 1:] = v
    return RolloutResult(
        time=trajectory.time,
        q=q_history,
        v=v_history,
        contact_wrenches=contact_wrenches,
        success=failure_index < 0,
        failure_reason=failure_reason,
        torque_demand=torque_demand,
        applied_torque=applied_torque,
        normal_force_margin=normal_margin,
        friction_margin=friction_margin,
        cop_margin=cop_margin,
        active_contacts=active_contacts,
        scheduled_contacts=scheduled_contacts,
        failure_index=failure_index,
        runtime=perf_counter() - started,
        peak_torque_joint=peak_torque_joint,
        peak_torque_ratio=peak_torque_ratio,
    )
