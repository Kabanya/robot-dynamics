"""Generate deterministic development figures from frozen results."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import matplotlib


matplotlib.use("Agg")

from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402


FIGURE_FILENAMES = (
    "figure1_pipeline.png",
    "figure2_selective_risk.png",
    "figure3_rollout_budget_efficiency.png",
)

METHOD_ORDER = (
    "zmp_cop_margin",
    "ik_joint_margin",
    "inverse_dynamics_slack",
    "black_box_parameters",
    "uncalibrated_phase_sequence",
    "risk_calibrated_phase_sequence",
)

ROBOTS = ("talos", "icub")
CONDITIONS = ("id", "ood")

_FROZEN_PROTOCOL = {
    "pilot_rollouts": 200,
    "train_base_gaits": 500,
    "tune_base_gaits": 150,
    "calibration_base_gaits": 450,
    "test_base_gaits": 600,
    "ood_base_gaits": 200,
    "perturbations_per_gait": 3,
    "steps": 6,
    "dt": 0.01,
    "candidate_pool_size": 2048,
    "rollout_budgets": (5, 10, 20),
    "screening_repetitions": 30,
}
_METHOD_LABELS = {
    "zmp_cop_margin": "B1 ZMP/CoP margin",
    "ik_joint_margin": "B2 IK + joint margin",
    "inverse_dynamics_slack": "B3 dynamics slack",
    "black_box_parameters": "B4 parameter-only ML",
    "uncalibrated_phase_sequence": "B5 phase sequence",
    "risk_calibrated_phase_sequence": "B6 calibrated",
}
_ROBOT_LABELS = {"talos": "TALOS", "icub": "iCub"}
_CONDITION_LABELS = {"id": "ID", "ood": "OOD"}
_COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00")
_MARKERS = ("o", "s", "^", "D", "P", "X")
_LINESTYLES = ("-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)))
_PENDING_MARKERS = ("pending", "template", "not_results", "not results", "tbd", "todo")
_PNG_METADATA = {"Software": "robot-dynamics deterministic figure generator"}
_RC_PARAMS = {
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.grid": False,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 14,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.transparent": False,
    "lines.linewidth": 1.7,
    "lines.markersize": 6,
    "path.simplify": False,
}


class FigureDataError(ValueError):
    """Raised when frozen result inputs are missing, pending, or inconsistent."""


@dataclass(frozen=True)
class FigureInputs:
    protocol: dict[str, Any]
    selective_metrics: dict[tuple[str, str], dict[str, float | int]]
    screening_metrics: dict[tuple[str, str, str, int], dict[str, float | int]]


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise FigureDataError(f"required result file is missing: {path}")
    try:
        with path.open(encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise FigureDataError(f"cannot read valid JSON from {path}: {exc}") from exc


def _reject_pending(value: Any, location: str) -> None:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if any(marker in normalized for marker in _PENDING_MARKERS):
            raise FigureDataError(
                f"{location} contains pending/template result value {value!r}"
            )
    elif isinstance(value, Mapping):
        for key, child in value.items():
            _reject_pending(child, f"{location}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            _reject_pending(child, f"{location}[{index}]")


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FigureDataError(f"{location} must be a JSON object")
    return value


def _list(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise FigureDataError(f"{location} must be a JSON array")
    return value


def _field(mapping: Mapping[str, Any], name: str, location: str) -> Any:
    if name not in mapping:
        raise FigureDataError(f"{location}.{name} is missing")
    return mapping[name]


def _number(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FigureDataError(f"{location} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise FigureDataError(f"{location} must be a finite number")
    return result


def _integer(value: Any, location: str) -> int:
    result = _number(value, location)
    if not result.is_integer():
        raise FigureDataError(f"{location} must be an integer")
    return int(result)


def _csv_integer(value: Any, location: str) -> int:
    try:
        parsed = int(str(value))
    except (TypeError, ValueError) as exc:
        raise FigureDataError(f"{location} must be an integer") from exc
    return parsed


def _csv_number(value: Any, location: str) -> float:
    try:
        parsed = float(str(value))
    except (TypeError, ValueError) as exc:
        raise FigureDataError(f"{location} must be a finite number") from exc
    if not math.isfinite(parsed):
        raise FigureDataError(f"{location} must be a finite number")
    return parsed


def _csv_boolean(value: Any, location: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise FigureDataError(f"{location} must be true or false")


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-12)


def _binomial_cdf(successes: int, trials: int, probability: float) -> float:
    if probability <= 0.0:
        return 1.0
    if probability >= 1.0:
        return 1.0 if successes >= trials else 0.0
    log_probability = math.log(probability)
    log_complement = math.log1p(-probability)
    logarithms = [
        math.lgamma(trials + 1)
        - math.lgamma(index + 1)
        - math.lgamma(trials - index + 1)
        + index * log_probability
        + (trials - index) * log_complement
        for index in range(successes + 1)
    ]
    scale = max(logarithms)
    return math.exp(scale) * math.fsum(
        math.exp(value - scale) for value in logarithms
    )


def _clopper_pearson_upper(
    failures: int,
    accepted: int,
    alpha: float = 0.05,
) -> float:
    if accepted == 0 or failures == accepted:
        return 1.0
    if failures == 0:
        return 1.0 - alpha ** (1.0 / accepted)
    lower = failures / accepted
    upper = 1.0
    for _ in range(80):
        midpoint = (lower + upper) / 2.0
        if _binomial_cdf(failures, accepted, midpoint) > alpha:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


def _validate_protocol(payload: Any) -> dict[str, Any]:
    _reject_pending(payload, "protocol")
    protocol = dict(_mapping(payload, "protocol"))
    robots = _list(_field(protocol, "robots", "protocol"), "protocol.robots")
    if tuple(robots) != ROBOTS:
        raise FigureDataError(
            "protocol.robots must be exactly ['talos', 'icub']"
        )
    if _field(protocol, "scientific_protocol", "protocol") is not True:
        raise FigureDataError(
            "protocol.scientific_protocol must be true; smoke results are not "
            "manuscript evidence"
        )

    positive_integers = (
        "pilot_rollouts",
        "train_base_gaits",
        "tune_base_gaits",
        "calibration_base_gaits",
        "test_base_gaits",
        "ood_base_gaits",
        "perturbations_per_gait",
        "steps",
        "candidate_pool_size",
        "screening_repetitions",
    )
    for name in positive_integers:
        parsed = _integer(_field(protocol, name, "protocol"), f"protocol.{name}")
        if parsed <= 0:
            raise FigureDataError(f"protocol.{name} must be positive")
        protocol[name] = parsed
    protocol["dt"] = _number(_field(protocol, "dt", "protocol"), "protocol.dt")
    if protocol["dt"] <= 0.0:
        raise FigureDataError("protocol.dt must be positive")

    budget_values = _list(
        _field(protocol, "rollout_budgets", "protocol"),
        "protocol.rollout_budgets",
    )
    budgets = tuple(
        _integer(value, f"protocol.rollout_budgets[{index}]")
        for index, value in enumerate(budget_values)
    )
    if not budgets or any(value <= 0 for value in budgets):
        raise FigureDataError("protocol.rollout_budgets must be positive")
    if tuple(sorted(set(budgets))) != budgets:
        raise FigureDataError(
            "protocol.rollout_budgets must be unique and increasing"
        )
    protocol["rollout_budgets"] = budgets
    for name, expected in _FROZEN_PROTOCOL.items():
        if protocol[name] != expected:
            raise FigureDataError(
                f"protocol.{name}={protocol[name]!r} does not match the frozen "
                f"scientific value {expected!r}"
            )
    return protocol


def _validate_evaluation(
    payload: Any,
    protocol: Mapping[str, Any],
) -> dict[tuple[str, str], dict[str, float | int]]:
    _reject_pending(payload, "evaluation")
    root = _mapping(payload, "evaluation")
    methods = _mapping(_field(root, "methods", "evaluation"), "evaluation.methods")
    test = _mapping(_field(methods, "test", "evaluation.methods"), "evaluation.methods.test")
    output: dict[tuple[str, str], dict[str, float | int]] = {}
    expected_rows = int(protocol["test_base_gaits"])

    for robot in ROBOTS:
        robot_location = f"evaluation.methods.test.{robot}"
        result = _mapping(_field(test, robot, "evaluation.methods.test"), robot_location)
        rows = _integer(_field(result, "rows", robot_location), f"{robot_location}.rows")
        if rows != expected_rows:
            raise FigureDataError(
                f"{robot_location}.rows={rows} does not match "
                f"protocol.test_base_gaits={expected_rows}"
            )
        prespecified = _mapping(
            _field(result, "prespecified", robot_location),
            f"{robot_location}.prespecified",
        )
        if set(prespecified) != set(METHOD_ORDER):
            raise FigureDataError(
                f"{robot_location}.prespecified must contain exactly the six "
                "frozen methods"
            )
        for method in METHOD_ORDER:
            location = f"{robot_location}.prespecified.{method}"
            metrics = _mapping(
                _field(prespecified, method, f"{robot_location}.prespecified"),
                location,
            )
            accepted = _integer(_field(metrics, "accepted", location), f"{location}.accepted")
            failures = _integer(
                _field(metrics, "false_safe_count", location),
                f"{location}.false_safe_count",
            )
            coverage = _number(_field(metrics, "coverage", location), f"{location}.coverage")
            risk = _number(
                _field(metrics, "false_safe_risk", location),
                f"{location}.false_safe_risk",
            )
            upper = _number(
                _field(metrics, "false_safe_upper", location),
                f"{location}.false_safe_upper",
            )
            bound_valid = _field(metrics, "confidence_bound_valid", location)
            if bound_valid is not True:
                raise FigureDataError(
                    f"{location}.confidence_bound_valid must be true for the ID "
                    "manuscript figure"
                )
            if not 0 <= accepted <= rows:
                raise FigureDataError(f"{location}.accepted is outside [0, rows]")
            if not 0 <= failures <= accepted:
                raise FigureDataError(
                    f"{location}.false_safe_count is outside [0, accepted]"
                )
            if not 0.0 <= coverage <= 1.0 or not _close(
                coverage, accepted / rows
            ):
                raise FigureDataError(
                    f"{location}.coverage does not equal accepted / rows"
                )
            expected_risk = failures / accepted if accepted else 0.0
            if not 0.0 <= risk <= 1.0 or not _close(risk, expected_risk):
                raise FigureDataError(
                    f"{location}.false_safe_risk does not equal "
                    "false_safe_count / accepted"
                )
            if not risk <= upper <= 1.0:
                raise FigureDataError(
                    f"{location}.false_safe_upper must be between risk and 1"
                )
            exact_upper = _clopper_pearson_upper(failures, accepted)
            if not math.isclose(
                upper,
                exact_upper,
                rel_tol=1e-8,
                abs_tol=1e-10,
            ):
                raise FigureDataError(
                    f"{location}.false_safe_upper is not the one-sided 95% "
                    "Clopper-Pearson bound"
                )
            output[(robot, method)] = {
                "accepted": accepted,
                "coverage": coverage,
                "false_safe_count": failures,
                "false_safe_risk": risk,
                "false_safe_upper": upper,
            }
    return output


def _read_screening_rows(
    path: Path,
    protocol: Mapping[str, Any],
) -> dict[tuple[str, str, str, int], list[dict[str, Any]]]:
    if not path.is_file():
        raise FigureDataError(f"required result file is missing: {path}")
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
    except OSError as exc:
        raise FigureDataError(f"cannot read screening.csv: {exc}") from exc
    if not rows:
        raise FigureDataError("screening.csv contains no result rows")

    budgets = tuple(int(value) for value in protocol["rollout_budgets"])
    repetitions = int(protocol["screening_repetitions"])
    pool_size = int(protocol["candidate_pool_size"])
    groups: dict[tuple[str, str, str, int], list[dict[str, Any]]] = {}
    paired_pools: dict[tuple[str, str, int], tuple[int, int]] = {}
    required_columns = {
        "method",
        "robot",
        "condition",
        "repetition",
        "budget",
        "success",
        "rollouts",
        "rollout_runtime_seconds",
        "pool_size",
        "pool_seed",
    }
    missing_columns = required_columns - set(rows[0])
    if missing_columns:
        raise FigureDataError(
            "screening.csv is missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    for index, source in enumerate(rows, start=2):
        location = f"screening.csv row {index}"
        _reject_pending(source, location)
        method = str(source["method"])
        robot = str(source["robot"])
        condition = str(source["condition"])
        if method not in METHOD_ORDER:
            raise FigureDataError(f"{location}.method is not a frozen method")
        if robot not in ROBOTS:
            raise FigureDataError(f"{location}.robot is not talos or icub")
        if condition not in CONDITIONS:
            raise FigureDataError(f"{location}.condition is not id or ood")
        repetition = _csv_integer(source["repetition"], f"{location}.repetition")
        budget = _csv_integer(source["budget"], f"{location}.budget")
        success = _csv_boolean(source["success"], f"{location}.success")
        rollouts = _csv_integer(source["rollouts"], f"{location}.rollouts")
        runtime = _csv_number(
            source["rollout_runtime_seconds"],
            f"{location}.rollout_runtime_seconds",
        )
        row_pool_size = _csv_integer(source["pool_size"], f"{location}.pool_size")
        pool_seed = _csv_integer(source["pool_seed"], f"{location}.pool_seed")
        if repetition not in range(repetitions):
            raise FigureDataError(f"{location}.repetition is outside protocol")
        if budget not in budgets:
            raise FigureDataError(f"{location}.budget is outside protocol")
        if not 0 <= rollouts <= budget:
            raise FigureDataError(f"{location}.rollouts is outside [0, budget]")
        if runtime < 0.0:
            raise FigureDataError(
                f"{location}.rollout_runtime_seconds must be nonnegative"
            )
        if row_pool_size != pool_size:
            raise FigureDataError(
                f"{location}.pool_size does not match protocol.candidate_pool_size"
            )
        paired_key = (robot, condition, repetition)
        pool = (pool_seed, row_pool_size)
        if paired_key in paired_pools and paired_pools[paired_key] != pool:
            raise FigureDataError(
                f"{location} does not preserve the paired candidate pool"
            )
        paired_pools[paired_key] = pool
        key = (method, robot, condition, budget)
        groups.setdefault(key, []).append({
            "repetition": repetition,
            "success": success,
            "rollouts": rollouts,
            "runtime": runtime,
        })

    expected_keys = {
        (method, robot, condition, budget)
        for method in METHOD_ORDER
        for robot in ROBOTS
        for condition in CONDITIONS
        for budget in budgets
    }
    if set(groups) != expected_keys:
        missing = len(expected_keys - set(groups))
        extra = len(set(groups) - expected_keys)
        raise FigureDataError(
            "screening.csv does not contain every frozen "
            f"method/robot/condition/budget cell (missing={missing}, extra={extra})"
        )
    expected_repetitions = set(range(repetitions))
    for key, selected in groups.items():
        observed = [int(row["repetition"]) for row in selected]
        if len(observed) != repetitions or set(observed) != expected_repetitions:
            raise FigureDataError(
                f"screening.csv cell {key} has missing or duplicate repetitions"
            )
    return groups


def _validate_screening_summary(
    payload: Any,
    raw_groups: Mapping[tuple[str, str, str, int], list[dict[str, Any]]],
    protocol: Mapping[str, Any],
) -> dict[tuple[str, str, str, int], dict[str, float | int]]:
    _reject_pending(payload, "screening_summary")
    rows = _list(payload, "screening_summary")
    parsed: dict[tuple[str, str, str, int], dict[str, float | int]] = {}
    expected_repetitions = int(protocol["screening_repetitions"])
    expected_pool_size = int(protocol["candidate_pool_size"])

    for index, source_value in enumerate(rows):
        location = f"screening_summary[{index}]"
        source = _mapping(source_value, location)
        method = str(_field(source, "method", location))
        robot = str(_field(source, "robot", location))
        condition = str(_field(source, "condition", location))
        budget = _integer(_field(source, "budget", location), f"{location}.budget")
        key = (method, robot, condition, budget)
        if key not in raw_groups:
            raise FigureDataError(
                f"{location} has no corresponding screening.csv cell"
            )
        if key in parsed:
            raise FigureDataError(f"{location} duplicates summary cell {key}")
        repetitions = _integer(
            _field(source, "repetitions", location),
            f"{location}.repetitions",
        )
        pool_size = _integer(
            _field(source, "pool_size", location),
            f"{location}.pool_size",
        )
        success_rate = _number(
            _field(source, "success_rate", location),
            f"{location}.success_rate",
        )
        mean_rollouts = _number(
            _field(source, "mean_rollouts", location),
            f"{location}.mean_rollouts",
        )
        mean_runtime = _number(
            _field(source, "mean_rollout_runtime_seconds", location),
            f"{location}.mean_rollout_runtime_seconds",
        )
        success_by_repetition = _mapping(
            _field(source, "success_by_repetition", location),
            f"{location}.success_by_repetition",
        )

        raw = sorted(raw_groups[key], key=lambda row: int(row["repetition"]))
        expected_success = {
            str(int(row["repetition"])): bool(row["success"]) for row in raw
        }
        observed_success = {
            str(name): value for name, value in success_by_repetition.items()
        }
        if observed_success != expected_success:
            raise FigureDataError(
                f"{location}.success_by_repetition does not match screening.csv"
            )
        expected_rate = math.fsum(
            1.0 if bool(row["success"]) else 0.0 for row in raw
        ) / len(raw)
        expected_rollouts = math.fsum(float(row["rollouts"]) for row in raw) / len(raw)
        expected_runtime = math.fsum(float(row["runtime"]) for row in raw) / len(raw)
        if repetitions != expected_repetitions or pool_size != expected_pool_size:
            raise FigureDataError(
                f"{location} protocol counts do not match screening.csv"
            )
        if (
            not _close(success_rate, expected_rate)
            or not _close(mean_rollouts, expected_rollouts)
            or not _close(mean_runtime, expected_runtime)
        ):
            raise FigureDataError(
                f"{location} aggregate values do not match screening.csv"
            )
        if not 0.0 <= success_rate <= 1.0:
            raise FigureDataError(f"{location}.success_rate is outside [0, 1]")
        parsed[key] = {
            "success_rate": success_rate,
            "mean_rollouts": mean_rollouts,
            "mean_rollout_runtime_seconds": mean_runtime,
        }

    if set(parsed) != set(raw_groups):
        missing = len(set(raw_groups) - set(parsed))
        raise FigureDataError(
            f"screening_summary.json is incomplete relative to screening.csv "
            f"(missing={missing})"
        )
    return parsed


def _load_inputs(results_directory: Path) -> FigureInputs:
    protocol = _validate_protocol(_load_json(results_directory / "protocol.json"))
    evaluation = _validate_evaluation(
        _load_json(results_directory / "evaluation.json"),
        protocol,
    )
    raw = _read_screening_rows(results_directory / "screening.csv", protocol)
    screening = _validate_screening_summary(
        _load_json(results_directory / "screening_summary.json"),
        raw,
        protocol,
    )
    return FigureInputs(protocol, evaluation, screening)


def _save(figure: plt.Figure, path: Path) -> None:
    figure.savefig(
        path,
        dpi=180,
        format="png",
        metadata=_PNG_METADATA,
        bbox_inches="tight",
        pad_inches=0.08,
    )
    plt.close(figure)


def _pipeline_figure(protocol: Mapping[str, Any], path: Path) -> None:
    stages = (
        (
            "1  Frozen\nsamples",
            "Sobol base gaits / robot\n"
            f"train {protocol['train_base_gaits']} · tune {protocol['tune_base_gaits']}\n"
            f"OOD {protocol['ood_base_gaits']} · "
            f"{protocol['perturbations_per_gait']} perturbations",
        ),
        (
            "2  Whole-body\nreference",
            f"{protocol['steps']}-step sequence\nΔt={protocol['dt']:g} s\n"
            "bounded IK + contacts",
        ),
        (
            "3  Phase\nsignature",
            "kinematics · dynamics\ncontact · touchdown",
        ),
        (
            "4  Six scores",
            "B1–B4 baselines\nB5–B6 sequence",
        ),
        (
            "5  IID\ncalibration",
            f"calibration {protocol['calibration_base_gaits']}\n"
            f"untouched ID test {protocol['test_base_gaits']}\nper robot",
        ),
        (
            "6  Screen +\nverify",
            f"pool {protocol['candidate_pool_size']} / robot\nbudgets "
            + "/".join(str(value) for value in protocol["rollout_budgets"])
            + f"\n{protocol['screening_repetitions']} paired repeats",
        ),
    )
    box_colors = ("#EAF2F8", "#E8F6F3", "#FEF9E7", "#FDEDEC", "#F4ECF7", "#FBE9E7")
    with plt.rc_context(_RC_PARAMS):
        figure, axis = plt.subplots(figsize=(12.2, 3.2))
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.axis("off")
        figure.suptitle(
            "Frozen sequence-screening pipeline",
            x=0.5,
            y=0.98,
            fontweight="bold",
        )
        width = 0.137
        gap = 0.027
        lefts = [0.009 + index * (width + gap) for index in range(len(stages))]
        for index, ((title, detail), color, left) in enumerate(
            zip(stages, box_colors, lefts)
        ):
            patch = FancyBboxPatch(
                (left, 0.29),
                width,
                0.45,
                boxstyle="round,pad=0.012,rounding_size=0.018",
                linewidth=1.2,
                edgecolor="#334155",
                facecolor=color,
            )
            axis.add_patch(patch)
            axis.text(
                left + width / 2,
                0.62,
                title,
                ha="center",
                va="center",
                fontsize=8.5,
                fontweight="bold",
                color="#172033",
                linespacing=1.05,
            )
            axis.text(
                left + width / 2,
                0.43,
                detail,
                ha="center",
                va="center",
                fontsize=7.1,
                linespacing=1.35,
                color="#334155",
            )
            if index < len(stages) - 1:
                axis.annotate(
                    "",
                    xy=(lefts[index + 1] - 0.004, 0.515),
                    xytext=(left + width + 0.004, 0.515),
                    arrowprops={
                        "arrowstyle": "-|>",
                        "color": "#475569",
                        "lw": 1.5,
                        "shrinkA": 0,
                        "shrinkB": 0,
                    },
                )
        axis.text(
            0.5,
            0.14,
            "Every reported final candidate passes the separate full-rollout oracle.",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="#9A3412",
        )
        _save(figure, path)


def _selective_risk_figure(
    metrics: Mapping[tuple[str, str], Mapping[str, float | int]],
    path: Path,
) -> None:
    with plt.rc_context(_RC_PARAMS):
        figure, axes = plt.subplots(
            1,
            2,
            figsize=(10.6, 5.2),
            sharex=True,
            sharey=True,
        )
        figure.subplots_adjust(
            left=0.08,
            right=0.985,
            top=0.82,
            bottom=0.25,
            wspace=0.065,
        )
        figure.suptitle(
            "Held-out ID selective risk at prespecified thresholds",
            fontweight="bold",
        )
        for axis, robot in zip(axes, ROBOTS):
            for index, method in enumerate(METHOD_ORDER):
                point = metrics[(robot, method)]
                coverage = float(point["coverage"])
                risk = float(point["false_safe_risk"])
                upper = float(point["false_safe_upper"])
                axis.errorbar(
                    coverage,
                    risk,
                    yerr=[[0.0], [upper - risk]],
                    fmt=_MARKERS[index],
                    color=_COLORS[index],
                    markeredgecolor="#172033",
                    markeredgewidth=0.45,
                    elinewidth=1.35,
                    capsize=3,
                    label=_METHOD_LABELS[method],
                    zorder=3,
                )
            axis.set_title(_ROBOT_LABELS[robot], fontweight="bold")
            axis.set_xlim(-0.035, 1.035)
            axis.set_ylim(-0.035, 1.035)
            axis.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
            axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
            axis.grid(True, color="#CBD5E1", linewidth=0.65, alpha=0.8)
            axis.set_xlabel("Acceptance coverage")
        axes[0].set_ylabel("Observed false-safe risk")
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.015),
            ncol=3,
            frameon=False,
        )
        figure.text(
            0.5,
            0.16,
            "Markers are point estimates; upper whiskers are one-sided 95% "
            "Clopper–Pearson bounds. Accept-none remains at zero coverage.",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#475569",
        )
        _save(figure, path)


def _rollout_budget_figure(
    metrics: Mapping[tuple[str, str, str, int], Mapping[str, float | int]],
    budgets: Sequence[int],
    path: Path,
) -> None:
    with plt.rc_context(_RC_PARAMS):
        figure, axes = plt.subplots(
            2,
            2,
            figsize=(11.0, 8.0),
            sharex=True,
            sharey=True,
        )
        figure.subplots_adjust(
            left=0.08,
            right=0.985,
            top=0.89,
            bottom=0.19,
            hspace=0.22,
            wspace=0.03,
        )
        figure.suptitle(
            "Rollout-verified success by oracle budget",
            fontweight="bold",
        )
        for row_index, condition in enumerate(CONDITIONS):
            for column_index, robot in enumerate(ROBOTS):
                axis = axes[row_index][column_index]
                for method_index, method in enumerate(METHOD_ORDER):
                    values = [
                        float(
                            metrics[(method, robot, condition, int(budget))][
                                "success_rate"
                            ]
                        )
                        for budget in budgets
                    ]
                    axis.plot(
                        budgets,
                        values,
                        color=_COLORS[method_index],
                        marker=_MARKERS[method_index],
                        linestyle=_LINESTYLES[method_index],
                        markeredgecolor="#172033",
                        markeredgewidth=0.4,
                        label=_METHOD_LABELS[method],
                    )
                axis.set_title(
                    f"{_ROBOT_LABELS[robot]} · {_CONDITION_LABELS[condition]}",
                    fontweight="bold",
                )
                axis.set_xticks(list(budgets))
                axis.set_ylim(-0.035, 1.035)
                axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
                axis.grid(True, color="#CBD5E1", linewidth=0.65, alpha=0.8)
                if column_index == 0:
                    axis.set_ylabel("Verified success rate")
                if row_index == len(CONDITIONS) - 1:
                    axis.set_xlabel("Full-rollout oracle budget")
        handles, labels = axes[0][0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.012),
            ncol=3,
            frameon=False,
        )
        figure.text(
            0.5,
            0.12,
            "Each point aggregates the frozen paired repetitions over the same "
            "candidate pools.",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#475569",
        )
        _save(figure, path)


def _check_output_directory(output: Path) -> None:
    if output.exists() and not output.is_dir():
        raise FigureDataError(f"figure output path is not a directory: {output}")
    if output.is_dir():
        expected = set(FIGURE_FILENAMES)
        entries = list(output.iterdir())
        unexpected = sorted(path.name for path in entries if path.name not in expected)
        if unexpected:
            raise FigureDataError(
                f"{output} contains unexpected files; refusing to alter them: "
                + ", ".join(unexpected)
            )
        non_files = sorted(path.name for path in entries if not path.is_file())
        if non_files:
            raise FigureDataError(
                f"{output} contains non-file figure targets: "
                + ", ".join(non_files)
            )


def generate_figures(
    results_directory: str | os.PathLike[str],
    output_directory: str | os.PathLike[str] | None = None,
) -> tuple[Path, ...]:
    """Validate frozen inputs and atomically emit exactly three PNG figures."""
    results = Path(results_directory)
    inputs = _load_inputs(results)
    output = Path(output_directory) if output_directory is not None else results / "figures"
    _check_output_directory(output)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".figure-staging-",
        dir=output.parent,
    ) as temporary_directory:
        staging = Path(temporary_directory)
        staged_paths = tuple(staging / name for name in FIGURE_FILENAMES)
        _pipeline_figure(inputs.protocol, staged_paths[0])
        _selective_risk_figure(inputs.selective_metrics, staged_paths[1])
        _rollout_budget_figure(
            inputs.screening_metrics,
            inputs.protocol["rollout_budgets"],
            staged_paths[2],
        )
        for path in staged_paths:
            if not path.is_file() or path.stat().st_size == 0:
                raise FigureDataError(f"figure renderer did not produce {path.name}")
        output.mkdir(parents=True, exist_ok=True)
        for path in staged_paths:
            os.replace(path, output / path.name)
    return tuple(output / name for name in FIGURE_FILENAMES)


def cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/experiment"),
        help="directory containing the frozen result files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="figure directory (default: RESULTS_DIR/figures)",
    )
    args = parser.parse_args(argv)
    try:
        paths = generate_figures(args.results_dir, args.output_dir)
    except FigureDataError as exc:
        parser.error(str(exc))
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
