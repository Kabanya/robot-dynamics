import csv
import os
import time
from dataclasses import asdict, dataclass, replace

import example_robot_data
import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin
from scipy.linalg import solve_discrete_are
from scipy.optimize import minimize

from .com import CoMClass
from .footsteps import FootSteps
from .legs import LeftLeg, RightLeg
from .support import polygon_margin
from .talos import Talos
from .zmp import ZmpClass

PLANNER_MODES = (
    "baseline_fixed",
    "support_polygon",
    "smooth_ds",
    "zmp_bias",
    "phase_adaptive",
    "full_adaptive",
    "preview_control",
    "dcm_heuristic",
)


@dataclass(frozen=True)
class GaitParams:
    step_length: float = 0.10
    step_width: float = 0.20
    single_support_duration: float = 0.70
    double_support_duration: float = 0.10
    initial_double_support: float = 0.30
    final_double_support: float = 0.50
    steps: int = 6
    com_height: float = 0.8766814
    zmp_bias_x: float = 0.0
    zmp_bias_y: float = 0.0
    foot_length: float = 0.22
    foot_width: float = 0.12
    zmp_safety_margin: float = 0.015
    com_dt: float = 0.02
    eval_dt: float = 0.20


@dataclass(frozen=True)
class GaitPerturbation:
    com_offset_x: float = 0.0
    com_offset_y: float = 0.0
    foot_noise_x: float = 0.0
    foot_noise_y: float = 0.0
    com_height_scale: float = 1.0


@dataclass(frozen=True)
class TrajectoryMetrics:
    max_ik_error: float
    mean_ik_error: float
    max_com_error: float
    mean_com_error: float
    max_foot_error: float
    mean_foot_error: float
    max_torso_orientation_error: float
    min_zmp_margin: float
    min_normalized_zmp_margin: float
    max_capture_point_violation: float
    max_com_jerk: float
    max_support_transition_jump: float
    max_moment_norm: float
    ik_failure_count: int
    constraint_violation_count: int
    optimizer_evals: int
    runtime_sec: float
    feasible: bool
    failure_reason: str

    def as_dict(self):
        return asdict(self)


@dataclass
class PlanResult:
    initial_params: GaitParams
    adapted_params: GaitParams
    metrics: TrajectoryMetrics
    mode: str
    success: bool
    message: str
    initial_metrics: TrajectoryMetrics | None = None
    footsteps: FootSteps | None = None
    zmp_traj: ZmpClass | None = None
    com_traj: CoMClass | None = None
    left_ank: LeftLeg | None = None
    right_ank: RightLeg | None = None


@dataclass
class RobotContext:
    robot: object
    model: object
    data: object
    left_foot_id: int
    right_foot_id: int
    base_foot_height: float


@dataclass(frozen=True)
class ExperimentConfig:
    suite: str
    modes: tuple[str, ...]
    lengths: tuple[float, ...] = (0.08, 0.14, 0.20, 0.26, 0.32)
    widths: tuple[float, ...] = (0.14, 0.19, 0.24, 0.29, 0.34)
    single_supports: tuple[float, ...] = (0.45, 0.70, 1.00)
    double_supports: tuple[float, ...] = (0.05, 0.18, 0.35)
    samples: int = 300
    seed: int = 42


@dataclass(frozen=True)
class ExperimentRow:
    suite: str
    mode: str
    params: GaitParams
    perturbation: GaitPerturbation
    metrics: TrajectoryMetrics


def build_footsteps(params: GaitParams, perturbation: GaitPerturbation | None = None):
    perturbation = perturbation or GaitPerturbation()
    half_width = params.step_width / 2.0
    footsteps = FootSteps([0.0, -half_width], [0.0, half_width])
    footsteps.add_phase(params.initial_double_support, "none")

    for i in range(params.steps):
        foot = "left" if i % 2 == 0 else "right"
        y = half_width if foot == "left" else -half_width
        footsteps.add_phase(
            params.single_support_duration,
            foot,
            [
                (i + 1) * params.step_length + perturbation.foot_noise_x,
                y + perturbation.foot_noise_y,
            ],
        )
        footsteps.add_phase(params.double_support_duration, "none")

    if params.final_double_support > params.double_support_duration:
        footsteps.add_phase(params.final_double_support - params.double_support_duration, "none")
    return footsteps


def build_trajectories(params: GaitParams,
                       mode="baseline",
                       base_foot_height=0.0,
                       perturbation: GaitPerturbation | None = None):
    mode = normalize_mode(mode)
    perturbation = perturbation or GaitPerturbation()
    params = replace(params, com_height=params.com_height * perturbation.com_height_scale)
    footsteps = build_footsteps(params, perturbation)
    zmp_mode, smooth_double_support, bias = _zmp_options(mode, params)
    zmp_traj = ZmpClass(
        footsteps,
        mode=zmp_mode,
        foot_length=params.foot_length,
        foot_width=params.foot_width,
        safety_margin=params.zmp_safety_margin if zmp_mode != "baseline" else 0.0,
        bias=bias,
        smooth_double_support=smooth_double_support,
    )
    if mode == "preview_control":
        com_traj = PreviewCoMClass(zmp_traj, com_z_nominal=params.com_height, dt=params.com_dt)
    else:
        com_traj = CoMClass(zmp_traj, com_z_nominal=params.com_height, dt=params.com_dt)
    left_ank = LeftLeg(footsteps, base_foot_height)
    right_ank = RightLeg(footsteps, base_foot_height)
    return footsteps, zmp_traj, com_traj, left_ank, right_ank


def normalize_mode(mode):
    if mode in ("baseline", "baseline_fixed"):
        return "baseline_fixed"
    if mode in ("adaptive", "full_adaptive"):
        return "full_adaptive"
    return mode


def _zmp_options(mode, params):
    if mode in ("baseline_fixed", "preview_control", "dcm_heuristic"):
        return "baseline", False, (0.0, 0.0)
    if mode == "support_polygon":
        return "adaptive", False, (0.0, 0.0)
    if mode in ("smooth_ds", "phase_adaptive"):
        return "adaptive", True, (0.0, 0.0)
    if mode in ("zmp_bias", "full_adaptive"):
        return "adaptive", True, (params.zmp_bias_x, params.zmp_bias_y)
    raise ValueError(f"Unknown planner mode: {mode}")


class PreviewCoMClass:
    def __init__(self, zmp_traj: ZmpClass, com_z_nominal=0.9, dt=0.02):
        self.zmp_traj = zmp_traj
        self.dt = dt
        self.t_end = self.zmp_traj.footsteps.timetime[-1]
        self.com_z = com_z_nominal
        self.g = 9.8
        self.L = self.com_z / self.g
        self.N = max(2, int(self.t_end / self.dt))
        self.x_opt = self._solve_axis(0)
        self.y_opt = self._solve_axis(1)
        self.x_traj = self.x_opt[:, 0]
        self.y_traj = self.y_opt[:, 0]

    def _solve_axis(self, direction):
        A = np.array([
            [1.0, self.dt, 0.5 * self.dt**2],
            [0.0, 1.0, self.dt],
            [0.0, 0.0, 1.0],
        ])
        B = np.array([[self.dt**3 / 6.0], [0.5 * self.dt**2], [self.dt]])
        C = np.array([[1.0, 0.0, -self.L]])
        Q = C.T @ C * 100.0 + np.diag([1e-3, 1e-3, 1e-4])
        R = np.array([[1e-2]])
        P = solve_discrete_are(A, B, Q, R)
        K = np.linalg.solve(B.T @ P @ B + R, B.T @ P @ A)

        refs = np.array([self.zmp_traj(k * self.dt)[direction] for k in range(self.N)])
        x = np.array([refs[0], 0.0, 0.0])
        out = []
        for ref in refs:
            target = np.array([ref, 0.0, 0.0])
            u = float((-K @ (x - target))[0])
            x = A @ x + B[:, 0] * u
            out.append(x.copy())
        return np.asarray(out)

    def __call__(self, t):
        idx = int(t / self.dt)
        idx = min(max(idx, 0), len(self.x_traj) - 1)
        return np.array([self.x_traj[idx], self.y_traj[idx], self.com_z])

    def compute_moment(self, t):
        idx = int(t / self.dt)
        idx = min(max(idx, 0), self.N - 1)
        zmp_ref = self.zmp_traj(t)
        moment_y = 10.0 * self.g * (self.x_opt[idx, 0] - zmp_ref[0] - self.L * self.x_opt[idx, 2])
        moment_x = -10.0 * self.g * (self.y_opt[idx, 0] - zmp_ref[1] - self.L * self.y_opt[idx, 2])
        return float(np.linalg.norm([moment_x, moment_y]))


def load_talos_context():
    robot = example_robot_data.load("talos")
    model = robot.model
    data = robot.data
    pin.framesForwardKinematics(model, data, robot.q0)
    left_foot_id = model.getFrameId("leg_left_sole_fix_joint")
    right_foot_id = model.getFrameId("leg_right_sole_fix_joint")
    left_z = data.oMf[left_foot_id].translation[2]
    right_z = data.oMf[right_foot_id].translation[2]
    return RobotContext(robot, model, data, left_foot_id, right_foot_id, (left_z + right_z) / 2.0)


def evaluate_gait(params: GaitParams,
                  mode="baseline",
                  context: RobotContext | None = None,
                  eval_dt: float | None = None,
                  ik_maxiter=40,
                  perturbation: GaitPerturbation | None = None,
                  optimizer_evals=0):
    mode = normalize_mode(mode)
    perturbation = perturbation or GaitPerturbation()
    context = load_talos_context() if context is None else context
    eval_dt = params.eval_dt if eval_dt is None else eval_dt
    footsteps, zmp_traj, com_traj, left_ank, right_ank = build_trajectories(
        params, mode=mode, base_foot_height=context.base_foot_height, perturbation=perturbation
    )
    params = replace(params, com_height=params.com_height * perturbation.com_height_scale, eval_dt=eval_dt)

    animator = Talos(
        context.robot, context.model, context.data,
        context.left_foot_id, context.right_foot_id,
        left_ank, right_ank, com_traj, footsteps,
        params.com_height, context.base_foot_height, eval_dt,
        zmp_traj=zmp_traj,
        ik_maxiter=ik_maxiter,
    )

    q = context.robot.q0.copy()
    t_total = footsteps.timetime[-1]
    ts = np.arange(0.0, t_total, eval_dt)
    ik_errors = []
    com_errors = []
    foot_errors = []
    torso_errors = []
    zmp_margins = []
    com_points = []
    zmp_points = []
    capture_point_margins = []
    moments = []
    start = time.perf_counter()

    for t in ts:
        com = com_traj(t)
        com = np.array([com[0] + perturbation.com_offset_x, com[1] + perturbation.com_offset_y, com[2]])
        left_target = left_ank(t)
        right_target = right_ank(t)
        phase = footsteps.get_phase_type(t)
        support, swing = _legs_for_phase(phase)

        q = animator.IK_CoM_solve(
            support, swing,
            right_target if swing else left_target,
            right_target if support else left_target,
            np.array([com[0], com[1], params.com_height]),
            q,
        )

        pin.framesForwardKinematics(context.model, context.data, q)
        actual_left = context.data.oMf[context.left_foot_id].translation
        actual_right = context.data.oMf[context.right_foot_id].translation
        actual_com = pin.centerOfMass(context.model, context.data, q)

        left_error = np.linalg.norm(actual_left - left_target)
        right_error = np.linalg.norm(actual_right - right_target)
        com_error = np.linalg.norm(actual_com - np.array([com[0], com[1], params.com_height]))
        foot_error = max(left_error, right_error)

        foot_errors.append(foot_error)
        com_errors.append(com_error)
        ik_errors.append(max(foot_error, com_error))
        torso_errors.append(_torso_orientation_error(context))
        zmp_margins.append(zmp_traj.margin(t))
        com_points.append(com[:2])
        zmp_points.append(zmp_traj(t))
        moments.append(com_traj.compute_moment(t))

    runtime = time.perf_counter() - start
    capture_point_margins = _capture_point_margins(com_points, zmp_traj, ts, params)
    metrics = _metrics(
        ik_errors, com_errors, foot_errors, torso_errors, zmp_margins,
        capture_point_margins, com_points, zmp_points, moments, runtime,
        params, strict_capture_point=mode == "dcm_heuristic",
        optimizer_evals=optimizer_evals,
    )
    return PlanResult(params, params, metrics, mode, True, "evaluated",
                      footsteps=footsteps, zmp_traj=zmp_traj, com_traj=com_traj,
                      left_ank=left_ank, right_ank=right_ank)


def plan_adaptive_gait(initial_params: GaitParams,
                       context: RobotContext | None = None,
                       maxiter=8,
                       eval_dt: float | None = None,
                       ik_maxiter=25,
                       planner_mode="full_adaptive",
                       perturbation: GaitPerturbation | None = None):
    planner_mode = normalize_mode(planner_mode)
    perturbation = perturbation or GaitPerturbation()
    context = load_talos_context() if context is None else context
    baseline = evaluate_gait(
        initial_params, mode="baseline_fixed", context=context,
        eval_dt=eval_dt, ik_maxiter=ik_maxiter, perturbation=perturbation,
    )
    cache = {}
    fields, x0, bounds = _optimization_problem(initial_params, planner_mode)

    def params_from_x(x):
        updates = {field: float(value) for field, value in zip(fields, x)}
        return replace(initial_params, **updates)

    def objective(x):
        key = tuple(np.round(x, 4))
        if key not in cache:
            params = params_from_x(x)
            try:
                cache[key] = evaluate_gait(
                    params, mode=planner_mode, context=context,
                    eval_dt=eval_dt, ik_maxiter=ik_maxiter,
                    perturbation=perturbation,
                )
            except Exception as exc:  # ponytail: optimizer only needs a bad score; expose the real error in cache.
                cache[key] = exc
        result = cache[key]
        if isinstance(result, Exception):
            return 1e9

        metrics = result.metrics
        neg_margin = max(0.0, -metrics.min_zmp_margin)
        scale = np.array([_field_scale(field) for field in fields])
        drift = np.sum(((np.asarray(x) - x0) / scale) ** 2) if len(fields) else 0.0
        return (
            50.0 * metrics.max_ik_error
            + 25.0 * metrics.max_com_error
            + 25.0 * metrics.max_foot_error
            + 0.05 * metrics.max_moment_norm
            + 200.0 * neg_margin
            + 100.0 * metrics.max_capture_point_violation
            + 0.15 * drift
        )

    # ponytail: bounded Nelder-Mead caps expensive full-body IK evaluations; use richer MPC if this must be real-time.
    opt = minimize(
        objective, x0,
        method="Nelder-Mead",
        bounds=bounds,
        options={"maxfev": maxiter, "disp": False},
    )
    adapted_params = params_from_x(opt.x)
    adapted = evaluate_gait(
        adapted_params, mode=planner_mode, context=context,
        eval_dt=eval_dt, ik_maxiter=ik_maxiter,
        perturbation=perturbation, optimizer_evals=len(cache),
    )
    adapted.initial_metrics = baseline.metrics
    adapted.initial_params = initial_params
    adapted.success = bool(opt.success or adapted.metrics.feasible)
    adapted.message = opt.message if isinstance(opt.message, str) else str(opt.message)
    return adapted


def _optimization_problem(params, planner_mode):
    choices = {
        "zmp_bias": ("zmp_bias_x", "zmp_bias_y"),
        "phase_adaptive": ("single_support_duration", "double_support_duration"),
        "full_adaptive": (
            "step_length", "step_width",
            "single_support_duration", "double_support_duration",
            "zmp_bias_x", "zmp_bias_y",
        ),
    }
    fields = choices.get(planner_mode, ())
    bounds_by_field = {
        "step_length": (0.05, 0.35),
        "step_width": (max(params.foot_width + 0.02, 0.14), 0.35),
        "single_support_duration": (0.40, 1.20),
        "double_support_duration": (0.05, 0.50),
        "zmp_bias_x": (-0.05, 0.05),
        "zmp_bias_y": (-0.04, 0.04),
    }
    return (
        fields,
        np.array([getattr(params, field) for field in fields], dtype=float),
        [bounds_by_field[field] for field in fields],
    )


def _field_scale(field):
    return {
        "step_length": 0.08,
        "step_width": 0.08,
        "single_support_duration": 0.30,
        "double_support_duration": 0.20,
        "zmp_bias_x": 0.03,
        "zmp_bias_y": 0.03,
    }[field]


def run_sweep(base_params: GaitParams,
              output_dir="results",
              lengths=(0.08, 0.12, 0.16),
              widths=(0.18, 0.22, 0.26),
              adaptive_maxiter=5):
    os.makedirs(output_dir, exist_ok=True)
    context = load_talos_context()
    rows = []

    for length in lengths:
        for width in widths:
            params = replace(base_params, step_length=length, step_width=width)
            baseline = evaluate_gait(params, "baseline_fixed", context=context, ik_maxiter=25)
            adaptive = plan_adaptive_gait(
                params, context=context, maxiter=adaptive_maxiter,
                ik_maxiter=20, planner_mode="full_adaptive",
            )
            rows.append(_row("baseline_fixed", params, baseline.metrics))
            rows.append(_row("full_adaptive", adaptive.adapted_params, adaptive.metrics))

    csv_path = os.path.join(output_dir, "sweep.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    _plot_sweep(rows, os.path.join(output_dir, "feasible_region.png"))
    return rows, csv_path


def _legs_for_phase(phase):
    if phase == "left":
        return 1, 0
    if phase == "right":
        return 0, 1
    return 0, 1


def _torso_orientation_error(context):
    torso_id = context.model.getFrameId("torso_2_link")
    rotation = context.data.oMf[torso_id].rotation
    world_x = np.array([1.0, 0.0, 0.0])
    world_z = np.array([0.0, 0.0, 1.0])
    return float(np.linalg.norm(rotation[:, 0] - world_x) + np.linalg.norm(rotation[:, 2] - world_z))


def _finite_max(values):
    values = np.asarray(values, dtype=float)
    return float(np.max(values)) if len(values) and np.all(np.isfinite(values)) else float("inf")


def _finite_mean(values):
    values = np.asarray(values, dtype=float)
    return float(np.mean(values)) if len(values) and np.all(np.isfinite(values)) else float("inf")


def _capture_point_margins(com_points, zmp_traj, ts, params):
    com_points = np.asarray(com_points, dtype=float)
    if len(com_points) < 2:
        return np.array([])
    velocities = np.vstack([
        np.zeros(2),
        np.diff(com_points, axis=0) / params.eval_dt,
    ])
    omega = np.sqrt(9.8 / params.com_height)
    cps = com_points + velocities / omega
    return np.array([polygon_margin(cp, zmp_traj.polygon(t)) for cp, t in zip(cps, ts)], dtype=float)


def _max_com_jerk(com_points, dt):
    com_points = np.asarray(com_points, dtype=float)
    if len(com_points) < 4:
        return 0.0
    jerk = np.diff(com_points, n=3, axis=0) / (dt ** 3)
    return float(np.max(np.linalg.norm(jerk, axis=1)))


def _max_transition_jump(points):
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return 0.0
    return float(np.max(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def _failure_reason(metrics):
    if not np.isfinite(metrics.max_ik_error):
        return "nan"
    if metrics.max_ik_error > 0.08:
        return "ik_error"
    if metrics.max_com_error > 0.08:
        return "com_error"
    if metrics.max_foot_error > 0.06:
        return "foot_error"
    if metrics.min_zmp_margin < -1e-6:
        return "zmp_margin"
    if metrics.max_capture_point_violation > 0.0:
        return "capture_point"
    return "ok"


def _metrics(ik_errors, com_errors, foot_errors, torso_errors, zmp_margins,
             capture_point_margins, com_points, zmp_points, moments, runtime,
             params, strict_capture_point=False, optimizer_evals=0):
    ik_errors = np.asarray(ik_errors, dtype=float)
    com_errors = np.asarray(com_errors, dtype=float)
    foot_errors = np.asarray(foot_errors, dtype=float)
    torso_errors = np.asarray(torso_errors, dtype=float)
    zmp_margins = np.asarray(zmp_margins, dtype=float)
    capture_point_margins = np.asarray(capture_point_margins, dtype=float)
    moments = np.asarray(moments, dtype=float)
    zmp_violation_count = int(np.sum(zmp_margins < -1e-6)) if len(zmp_margins) else 0
    cp_violation = max(0.0, -float(np.min(capture_point_margins))) if len(capture_point_margins) else 0.0

    feasible = bool(
        np.all(np.isfinite(ik_errors))
        and np.all(np.isfinite(zmp_margins))
        and _finite_max(ik_errors) <= 0.08
        and _finite_max(com_errors) <= 0.08
        and _finite_max(foot_errors) <= 0.06
        and np.all(np.isfinite(zmp_margins))
        and float(np.min(zmp_margins)) >= -1e-6
        and (not strict_capture_point or cp_violation <= 1e-6)
    )

    metrics = TrajectoryMetrics(
        max_ik_error=_finite_max(ik_errors),
        mean_ik_error=_finite_mean(ik_errors),
        max_com_error=_finite_max(com_errors),
        mean_com_error=_finite_mean(com_errors),
        max_foot_error=_finite_max(foot_errors),
        mean_foot_error=_finite_mean(foot_errors),
        max_torso_orientation_error=_finite_max(torso_errors),
        min_zmp_margin=float(np.min(zmp_margins)) if len(zmp_margins) and np.all(np.isfinite(zmp_margins)) else float("-inf"),
        min_normalized_zmp_margin=(
            float(np.min(zmp_margins) / max(params.foot_width / 2.0, 1e-9))
            if len(zmp_margins) and np.all(np.isfinite(zmp_margins)) else float("-inf")
        ),
        max_capture_point_violation=cp_violation,
        max_com_jerk=_max_com_jerk(com_points, params.eval_dt),
        max_support_transition_jump=_max_transition_jump(zmp_points),
        max_moment_norm=_finite_max(moments),
        ik_failure_count=int(np.sum(ik_errors > 0.08)) if len(ik_errors) else 0,
        constraint_violation_count=zmp_violation_count + int(cp_violation > 0.0),
        optimizer_evals=int(optimizer_evals),
        runtime_sec=float(runtime),
        feasible=feasible,
        failure_reason="ok",
    )
    return replace(metrics, failure_reason=_failure_reason(metrics))


def _row(mode, params, metrics):
    row = {"mode": mode, **asdict(params), **metrics.as_dict()}
    return row


def _plot_sweep(rows, path):
    modes = [mode for mode in ("baseline_fixed", "full_adaptive") if any(row["mode"] == mode for row in rows)]
    fig, axes = plt.subplots(1, len(modes), figsize=(5 * len(modes), 4), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, mode in zip(axes, modes):
        subset = [row for row in rows if row["mode"] == mode]
        colors = ["tab:green" if row["feasible"] else "tab:red" for row in subset]
        ax.scatter([row["step_length"] for row in subset], [row["step_width"] for row in subset], c=colors)
        ax.set_title(mode)
        ax.set_xlabel("step length [m]")
        ax.grid(True)
    axes[0].set_ylabel("step width [m]")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_benchmark(suite, output_dir="results/experiment", limit=None, seed=42, samples=None, adaptive_maxiter=8):
    suite = suite.lower()
    os.makedirs(output_dir, exist_ok=True)
    context = load_talos_context()
    config = _benchmark_config(suite, seed=seed, samples=samples)
    cases = _benchmark_cases(config)
    if limit is not None:
        cases = cases[:limit]

    rows = []
    failure_candidates = []
    for case_id, (params, perturbation) in enumerate(cases):
        case_rows = []
        for mode in config.modes:
            result = _evaluate_mode(
                params, mode, context,
                perturbation=perturbation,
                adaptive_maxiter=adaptive_maxiter,
            )
            row = _experiment_row(config.suite, case_id, mode, result.adapted_params, perturbation, result.metrics)
            rows.append(row)
            case_rows.append(row)
        failure_candidates.append(case_rows)

    csv_path = os.path.join(output_dir, f"{suite}.csv")
    _write_rows(csv_path, rows)
    summary_path = os.path.join(output_dir, f"{suite}_summary.csv")
    _write_rows(summary_path, _summary_rows(rows))
    _plot_benchmark(rows, output_dir, suite)
    if suite in ("failure_cases", "wide_grid", "ablation"):
        _plot_failure_cases(failure_candidates, output_dir)
    return rows, csv_path


def _benchmark_config(suite, seed=42, samples=None):
    if suite == "ablation":
        return ExperimentConfig(
            suite=suite,
            modes=("baseline_fixed", "support_polygon", "smooth_ds", "zmp_bias", "phase_adaptive", "full_adaptive"),
            seed=seed,
        )
    if suite in ("wide_grid", "strong_baselines"):
        return ExperimentConfig(
            suite=suite,
            modes=("baseline_fixed", "preview_control", "dcm_heuristic", "full_adaptive"),
            seed=seed,
        )
    if suite == "robustness":
        return ExperimentConfig(
            suite=suite,
            modes=("baseline_fixed", "preview_control", "dcm_heuristic", "full_adaptive"),
            samples=samples or 300,
            seed=seed,
        )
    if suite == "failure_cases":
        return ExperimentConfig(
            suite=suite,
            modes=("baseline_fixed", "preview_control", "dcm_heuristic", "full_adaptive"),
            samples=samples or 80,
            seed=seed,
        )
    raise ValueError(f"Unknown benchmark suite: {suite}")


def _benchmark_cases(config):
    rng = np.random.default_rng(config.seed)
    if config.suite in ("robustness", "failure_cases"):
        cases = []
        for _ in range(config.samples):
            params = GaitParams(
                step_length=float(rng.uniform(0.08, 0.34)),
                step_width=float(rng.uniform(0.14, 0.34)),
                single_support_duration=float(rng.uniform(0.40, 1.20)),
                double_support_duration=float(rng.uniform(0.05, 0.50)),
            )
            perturbation = GaitPerturbation(
                com_offset_x=float(rng.uniform(-0.03, 0.03)),
                com_offset_y=float(rng.uniform(-0.03, 0.03)),
                foot_noise_x=float(rng.uniform(-0.02, 0.02)),
                foot_noise_y=float(rng.uniform(-0.02, 0.02)),
                com_height_scale=float(rng.uniform(0.95, 1.05)),
            )
            cases.append((params, perturbation))
        return cases

    return [
        (
            GaitParams(
                step_length=length,
                step_width=width,
                single_support_duration=single_support,
                double_support_duration=double_support,
            ),
            GaitPerturbation(),
        )
        for length in config.lengths
        for width in config.widths
        for single_support in config.single_supports
        for double_support in config.double_supports
    ]


def _evaluate_mode(params, mode, context, perturbation, adaptive_maxiter):
    if mode in ("zmp_bias", "phase_adaptive", "full_adaptive"):
        return plan_adaptive_gait(
            params, context=context, maxiter=adaptive_maxiter,
            ik_maxiter=20, planner_mode=mode, perturbation=perturbation,
        )
    return evaluate_gait(params, mode=mode, context=context, ik_maxiter=20, perturbation=perturbation)


def _experiment_row(suite, case_id, mode, params, perturbation, metrics):
    return {
        "suite": suite,
        "case_id": case_id,
        "mode": mode,
        **asdict(params),
        **{f"perturb_{k}": v for k, v in asdict(perturbation).items()},
        **metrics.as_dict(),
    }


def _write_rows(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _summary_rows(rows):
    metrics = [
        "max_ik_error", "max_com_error", "max_foot_error",
        "min_zmp_margin", "max_capture_point_violation",
        "max_com_jerk", "max_support_transition_jump",
        "max_moment_norm", "runtime_sec",
    ]
    out = []
    for mode in sorted({row["mode"] for row in rows}):
        subset = [row for row in rows if row["mode"] == mode]
        row = {
            "mode": mode,
            "n": len(subset),
            "success_rate": sum(boolish(r["feasible"]) for r in subset) / max(len(subset), 1),
        }
        for metric in metrics:
            vals = np.array([float(r[metric]) for r in subset], dtype=float)
            row[f"{metric}_mean"] = float(np.mean(vals))
            row[f"{metric}_std"] = float(np.std(vals))
            row[f"{metric}_max"] = float(np.max(vals))
        out.append(row)
    return out


def boolish(value):
    return value is True or value == "True" or value == "true" or value == 1


def _plot_benchmark(rows, output_dir, suite):
    _plot_success_rates(rows, os.path.join(output_dir, f"{suite}_success_rate.png"))
    _plot_metric_box(rows, "max_moment_norm", os.path.join(output_dir, f"{suite}_moment_box.png"))
    _plot_metric_box(rows, "max_ik_error", os.path.join(output_dir, f"{suite}_ik_box.png"))
    if suite in ("wide_grid", "strong_baselines", "ablation"):
        _plot_feasible_heatmap(rows, os.path.join(output_dir, f"{suite}_feasible_heatmap.png"))


def _plot_success_rates(rows, path):
    summary = _summary_rows(rows)
    modes = [row["mode"] for row in summary]
    rates = [row["success_rate"] for row in summary]
    fig, ax = plt.subplots(figsize=(max(7, len(modes) * 1.2), 4))
    ax.bar(modes, rates, color="tab:blue")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("success rate")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_metric_box(rows, metric, path):
    modes = sorted({row["mode"] for row in rows})
    values = [[float(row[metric]) for row in rows if row["mode"] == mode] for mode in modes]
    fig, ax = plt.subplots(figsize=(max(7, len(modes) * 1.2), 4))
    ax.boxplot(values, showfliers=False)
    ax.set_ylabel(metric)
    ax.set_xticks(range(1, len(modes) + 1))
    ax.set_xticklabels(modes)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_feasible_heatmap(rows, path):
    modes = sorted({row["mode"] for row in rows})
    fig, axes = plt.subplots(1, len(modes), figsize=(4 * len(modes), 4), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, mode in zip(axes, modes):
        subset = [row for row in rows if row["mode"] == mode]
        colors = ["tab:green" if boolish(row["feasible"]) else "tab:red" for row in subset]
        ax.scatter([float(row["step_length"]) for row in subset], [float(row["step_width"]) for row in subset], c=colors, s=12)
        ax.set_title(mode)
        ax.set_xlabel("step length [m]")
        ax.grid(True)
    axes[0].set_ylabel("step width [m]")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_failure_cases(case_rows, output_dir):
    selected = []
    for rows in case_rows:
        by_mode = {row["mode"]: row for row in rows}
        baseline = by_mode.get("baseline_fixed")
        adaptive = by_mode.get("full_adaptive")
        if baseline and adaptive and not boolish(baseline["feasible"]) and boolish(adaptive["feasible"]):
            selected.append(("baseline_fail_adaptive_ok", baseline, adaptive))
    for rows in case_rows:
        by_mode = {row["mode"]: row for row in rows}
        baseline = by_mode.get("baseline_fixed")
        adaptive = by_mode.get("full_adaptive")
        if baseline and adaptive and not boolish(baseline["feasible"]) and not boolish(adaptive["feasible"]):
            selected.append(("both_fail", baseline, adaptive))
    selected = selected[:5]
    if not selected:
        return
    path = os.path.join(output_dir, "failure_cases.csv")
    flat = []
    for label, left, right in selected:
        flat.append({"case_type": label, **left})
        flat.append({"case_type": label, **right})
    _write_rows(path, flat)
