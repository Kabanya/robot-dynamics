"""Frozen sampling and persistence utilities for the experiment protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from time import perf_counter
from typing import Callable, Sequence

import numpy as np
from scipy.stats import beta, qmc
from scipy.special import expit
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .feasibility import (
    GaitSample,
    PhysicsSignature,
    RolloutResult,
    build_whole_body_trajectory,
    compute_physics_signature,
    load_robot_spec,
    rollout,
)
from .surrogate import RiskCalibratedSurrogate


_ROBOTS = ("talos", "icub")
_SPLITS = ("pilot", "train", "tune", "calibration", "test", "ood")
MODEL_PARAMETER_COLUMNS = (
    "step_length",
    "step_width",
    "single_support_duration",
    "double_support_duration",
    "com_height_scale",
    "zmp_bias_x",
    "zmp_bias_y",
    "friction",
    "payload_fraction",
    "timing_error_seconds",
    "impulse",
)
_CONTACT_AUDIT_COLUMNS = (
    "actual_single_support_fraction",
    "actual_double_support_fraction",
    "contact_schedule_match_fraction",
)
_HEURISTIC_METHODS = {
    "zmp_cop_margin",
    "ik_joint_margin",
    "inverse_dynamics_slack",
}
_MATCHED_COVERAGE_BASELINES = (
    "zmp_cop_margin",
    "ik_joint_margin",
    "inverse_dynamics_slack",
    "black_box_parameters",
)
_SCREENING_METHODS = (
    *_MATCHED_COVERAGE_BASELINES,
    "uncalibrated_phase_sequence",
    "risk_calibrated_phase_sequence",
)
_MODEL_FILENAMES = (
    "surrogate.pkl",
    "black_box.pkl",
    "ablation_no_normalization.pkl",
    "ablation_no_phase_separation.pkl",
    "ablation_no_touchdown.pkl",
)
_SCIENTIFIC_SOURCE_FILES = (
    "__init__.py",
    "com.py",
    "factor.py",
    "feasibility.py",
    "footsteps.py",
    "gait.py",
    "legs.py",
    "experiment.py",
    "support.py",
    "surrogate.py",
    "talos.py",
    "zmp.py",
)
_VALIDITY_TEST_FILES = (
    "tests/test_feasibility.py",
    "tests/test_whole_body_trajectory.py",
    "tests/test_physics_signature.py",
    "tests/test_rollout.py",
)


@dataclass(frozen=True)
class ProtocolConfig:
    """Sizes and seeds for the frozen two-robot experimental design."""

    seed: int = 2026
    study_revision: int = 2
    robots: tuple[str, ...] = _ROBOTS
    pilot_rollouts: int = 200
    train_base_gaits: int = 500
    tune_base_gaits: int = 150
    calibration_base_gaits: int = 450
    test_base_gaits: int = 600
    ood_base_gaits: int = 200
    perturbations_per_gait: int = 3
    steps: int = 6
    dt: float = 0.01
    candidate_pool_size: int = 2048
    rollout_budgets: tuple[int, ...] = (5, 10, 20)
    screening_repetitions: int = 30
    scientific_protocol: bool = True
    confirmation_protocol: bool = False
    pilot_revision: int = 0
    step_length_range: tuple[float, float] = (0.10, 0.40)
    step_width_range: tuple[float, float] = (0.85, 1.15)
    single_support_range: tuple[float, float] = (1.4, 2.8)
    double_support_range: tuple[float, float] = (0.2, 0.8)
    com_height_range: tuple[float, float] = (0.90, 1.05)
    zmp_bias_x_range: tuple[float, float] = (-0.25, 0.25)
    zmp_bias_y_range: tuple[float, float] = (-0.25, 0.25)
    id_payload_range: tuple[float, float] = (0.0, 0.10)

    def __post_init__(self):
        object.__setattr__(self, "robots", tuple(self.robots))
        object.__setattr__(self, "rollout_budgets", tuple(self.rollout_budgets))
        if self.study_revision != 2:
            raise ValueError("protocol study revision must be 2")
        if self.pilot_revision not in (0, 1):
            raise ValueError("pilot_revision permits only the initial pilot and one revision")
        domains = {
            "step_length_range": (0.10, 0.40),
            "step_width_range": (0.85, 1.15),
            "single_support_range": (1.4, 2.8),
            "double_support_range": (0.2, 0.8),
            "com_height_range": (0.90, 1.05),
            "zmp_bias_x_range": (-0.25, 0.25),
            "zmp_bias_y_range": (-0.25, 0.25),
        }
        for name, domain in domains.items():
            values = tuple(float(value) for value in getattr(self, name))
            object.__setattr__(self, name, values)
            if (
                len(values) != 2
                or not domain[0] <= values[0] < values[1] <= domain[1]
            ):
                raise ValueError(f"{name} must be an ordered subrange of {domain}")
        payload = tuple(float(value) for value in self.id_payload_range)
        object.__setattr__(self, "id_payload_range", payload)
        if len(payload) != 2 or not 0.0 <= payload[0] < payload[1] <= 0.10:
            raise ValueError("id_payload_range must be an ordered subrange of (0.0, 0.10)")
        if self.pilot_revision == 0 and any(
            getattr(self, name) != domain for name, domain in domains.items()
        ):
            raise ValueError("initial pilot must use the full prespecified ranges")
        if self.scientific_protocol and self.confirmation_protocol:
            raise ValueError(
                "full scientific and confirmation protocols are mutually exclusive"
            )
        if self.scientific_protocol:
            frozen = {
                "robots": _ROBOTS,
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
                "id_payload_range": (0.0, 0.10),
            }
            if any(getattr(self, name) != value for name, value in frozen.items()):
                raise ValueError("scientific protocol sizes and budgets are frozen")
        if self.confirmation_protocol:
            frozen = {
                "robots": _ROBOTS,
                "pilot_rollouts": 200,
                "train_base_gaits": 1,
                "tune_base_gaits": 1,
                "calibration_base_gaits": 1,
                "test_base_gaits": 1,
                "ood_base_gaits": 1,
                "perturbations_per_gait": 1,
                "steps": 6,
                "dt": 0.01,
                "candidate_pool_size": 4,
                "rollout_budgets": (1,),
                "screening_repetitions": 1,
                "pilot_revision": 1,
                "step_length_range": (0.10, 0.16),
                "step_width_range": (1.00, 1.10),
                "single_support_range": (2.70, 2.80),
                "double_support_range": (0.70, 0.80),
                "com_height_range": (0.92, 0.98),
                "zmp_bias_x_range": (-0.05, 0.05),
                "zmp_bias_y_range": (-0.25, -0.10),
                "id_payload_range": (0.0, 0.05),
            }
            if any(getattr(self, name) != value for name, value in frozen.items()):
                raise ValueError("confirmation protocol is frozen")

    @classmethod
    def from_json(cls, path: str) -> "ProtocolConfig":
        with open(path) as stream:
            payload = json.load(stream)
        if payload.get("study_revision") != 2:
            raise ValueError("protocol study revision must be 2")
        return cls(**payload)

    @classmethod
    def smoke(cls, seed: int = 42) -> "ProtocolConfig":
        """Return a tiny, explicitly non-scientific plumbing configuration."""
        return cls(
            seed=seed,
            pilot_rollouts=1,
            train_base_gaits=1,
            tune_base_gaits=1,
            calibration_base_gaits=1,
            test_base_gaits=1,
            ood_base_gaits=1,
            perturbations_per_gait=1,
            steps=1,
            dt=0.02,
            candidate_pool_size=4,
            rollout_budgets=(1,),
            screening_repetitions=1,
            scientific_protocol=False,
        )


@dataclass(frozen=True)
class ExperimentCase:
    robot: str
    split: str
    base_gait_id: str
    perturbation_index: int
    sample: GaitSample


@dataclass(frozen=True)
class RecordArrays:
    X: np.ndarray
    X_parameters: np.ndarray
    labels: np.ndarray
    robots: np.ndarray
    groups: np.ndarray
    feature_names: tuple[str, ...]


@dataclass(frozen=True)
class _ScreeningOutcome:
    success: bool
    failure_reason: str
    runtime_seconds: float
    oracle_wall_seconds: float = 0.0
    actual_oracle_wall_seconds: float = 0.0
    cache_hit: bool = False


def _sobol_points(count: int, dimensions: int, seed: int) -> np.ndarray:
    if count <= 0:
        return np.empty((0, dimensions))
    exponent = int(math.ceil(math.log2(count)))
    return qmc.Sobol(dimensions, scramble=True, seed=seed).random_base2(exponent)[:count]


def _iid_points(count: int, dimensions: int, seed: int) -> np.ndarray:
    """Draw independent uniform task samples for exact-binomial inference."""
    return np.random.default_rng(seed).random((count, dimensions))


def _permuted_rows(points: np.ndarray, seed: int) -> np.ndarray:
    return points[np.random.default_rng(seed).permutation(len(points))]


def _assert_cross_design_correlation_below_limit(
    gait_points: np.ndarray,
    perturbation_points: np.ndarray,
    context: str,
    perturbations: int = 1,
) -> None:
    if len(gait_points) != len(perturbation_points):
        raise ValueError("gait and perturbation designs must have equal rows")
    for slot in range(perturbations):
        gait = gait_points[slot::perturbations]
        disturbance = perturbation_points[slot::perturbations]
        if len(gait) < 64:
            continue
        cross = np.corrcoef(gait.T, disturbance.T)[
            :gait.shape[1], gait.shape[1]:
        ]
        maximum = float(np.max(np.abs(cross)))
        if not np.isfinite(maximum) or maximum >= 0.25:
            raise RuntimeError(
                f"{context} slot {slot} cross-design correlation "
                f"{maximum:.3f} exceeds 0.25"
            )


def _stream_seed(root: int, robot_index: int, split_index: int, stream: int) -> int:
    sequence = np.random.SeedSequence([root, robot_index, split_index, stream])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _scale(values: np.ndarray, low: float, high: float) -> np.ndarray:
    return low + (high - low) * values


def _base_parameters(
    point: np.ndarray, config: ProtocolConfig
) -> tuple[float, ...]:
    return (
        float(_scale(point[0], *config.step_length_range)),
        float(_scale(point[1], *config.step_width_range)),
        float(_scale(point[2], *config.single_support_range)),
        float(_scale(point[3], *config.double_support_range)),
        float(_scale(point[4], *config.com_height_range)),
        float(_scale(point[5], *config.zmp_bias_x_range)),
        float(_scale(point[6], *config.zmp_bias_y_range)),
    )


def _perturbation_parameters(
    point: np.ndarray, ood: bool, config: ProtocolConfig
) -> tuple[float, ...]:
    if not ood:
        return (
            float(_scale(point[0], 0.4, 0.8)),
            float(_scale(point[1], *config.id_payload_range)),
            float(_scale(point[2], -0.02, 0.02)),
            float(_scale(point[3], 0.0, 0.04)),
        )
    friction = (
        _scale(2.0 * point[0], 0.25, 0.40)
        if point[0] < 0.5
        else _scale(2.0 * (point[0] - 0.5), 0.80, 0.90)
    )
    timing = (
        -_scale(2.0 * point[2], 0.02, 0.04)
        if point[2] < 0.5
        else _scale(2.0 * (point[2] - 0.5), 0.02, 0.04)
    )
    return (
        float(friction),
        float(_scale(point[1], 0.10, 0.15)),
        float(timing),
        float(_scale(point[3], 0.04, 0.08)),
    )


def generate_cases(config: ProtocolConfig) -> list[ExperimentCase]:
    """Generate grouped design splits and IID calibration/test sequences."""
    split_sizes = {
        "pilot": config.pilot_rollouts,
        "train": config.train_base_gaits,
        "tune": config.tune_base_gaits,
        "calibration": config.calibration_base_gaits,
        "test": config.test_base_gaits,
        "ood": config.ood_base_gaits,
    }
    cases: list[ExperimentCase] = []
    for robot_index, robot in enumerate(config.robots):
        for split_index, split in enumerate(_SPLITS):
            base_count = split_sizes[split]
            inference_split = split in ("calibration", "test")
            perturbations = (
                1
                if split == "pilot" or inference_split
                else config.perturbations_per_gait
            )
            point_generator = _iid_points if inference_split else _sobol_points
            base = point_generator(
                base_count, 7,
                _stream_seed(config.seed, robot_index, split_index, 0),
            )
            perturb = point_generator(
                base_count * perturbations, 4,
                _stream_seed(config.seed, robot_index, split_index, 1),
            ).reshape(base_count, perturbations, 4)
            if not inference_split:
                perturb = _permuted_rows(
                    perturb,
                    _stream_seed(config.seed, robot_index, split_index, 3),
                )
            if (
                (config.scientific_protocol or config.confirmation_protocol)
                and not inference_split
            ):
                _assert_cross_design_correlation_below_limit(
                    np.repeat(base, perturbations, axis=0),
                    perturb.reshape(-1, 4),
                    f"{robot}/{split}",
                    perturbations=perturbations,
                )
            sample_seeds = np.random.SeedSequence(
                [config.seed, robot_index, split_index, 2]
            ).generate_state(base_count * perturbations, dtype=np.uint32)
            seed_index = 0
            for gait_index, gait_point in enumerate(base):
                gait = _base_parameters(gait_point, config)
                base_id = f"{robot}-{split}-{gait_index:06d}"
                for perturbation_index, perturbation_point in enumerate(perturb[gait_index]):
                    disturbance = _perturbation_parameters(
                        perturbation_point, ood=split == "ood", config=config
                    )
                    cases.append(ExperimentCase(
                        robot=robot,
                        split=split,
                        base_gait_id=base_id,
                        perturbation_index=perturbation_index,
                        sample=GaitSample(
                            *gait,
                            *disturbance,
                            seed=int(sample_seeds[seed_index]),
                            ood=split == "ood",
                        ),
                    ))
                    seed_index += 1
    return cases


def generate_candidate_pool(
    robot: str,
    seed: int,
    count: int = 2048,
    ood: bool = False,
    config: ProtocolConfig | None = None,
) -> list[ExperimentCase]:
    """Generate the common downstream Sobol pool for one robot/condition."""
    if robot not in _ROBOTS or count < 1:
        raise ValueError("robot must be talos/icub and count must be positive")
    if config is None:
        config = ProtocolConfig(seed=seed)
    robot_index = _ROBOTS.index(robot)
    base = _sobol_points(count, 7, _stream_seed(seed, robot_index, 97, 0))
    perturb = _sobol_points(count, 4, _stream_seed(seed, robot_index, 97, 1))
    perturb = _permuted_rows(
        perturb, _stream_seed(seed, robot_index, 97, 3)
    )
    if config.scientific_protocol:
        _assert_cross_design_correlation_below_limit(
            base, perturb, f"{robot}/screening"
        )
    sample_seeds = np.random.SeedSequence([seed, robot_index, 97, 2]).generate_state(
        count, dtype=np.uint32
    )
    split = "screening_ood" if ood else "screening_id"
    return [
        ExperimentCase(
            robot,
            split,
            f"{robot}-{split}-{index:06d}",
            0,
            GaitSample(
                *_base_parameters(base[index], config),
                *_perturbation_parameters(perturb[index], ood, config),
                seed=int(sample_seeds[index]),
                ood=ood,
            ),
        )
        for index in range(count)
    ]


def flatten_record(
    case: ExperimentCase,
    signature: PhysicsSignature,
    rollout: RolloutResult,
) -> dict[str, object]:
    """Flatten one labeled sequence into a stable, CSV-safe record."""
    if len(signature.feature_names) != len(signature.values):
        raise ValueError("physics signature names and values have different lengths")
    if len(signature.raw_values) not in (0, len(signature.feature_names)):
        raise ValueError("raw physics signature has a different length")
    record: dict[str, object] = {
        "split": case.split,
        "robot": case.robot,
        "base_gait_id": case.base_gait_id,
        "perturbation_index": case.perturbation_index,
        **{name: value for name, value in asdict(case.sample).items() if name != "ood"},
        "ood": case.sample.ood,
        "label": int(not rollout.success),
        "failure_reason": rollout.failure_reason,
        "failure_index": int(rollout.failure_index),
        "peak_torque_joint": rollout.peak_torque_joint,
        "peak_torque_ratio": rollout.peak_torque_ratio,
        "runtime_seconds": float(rollout.runtime_seconds),
    }
    processed = (
        len(rollout.time)
        if rollout.failure_index < 0
        else min(len(rollout.time), rollout.failure_index + 1)
    )
    if (
        processed
        and rollout.active_contacts.shape == (len(rollout.time), 2)
        and rollout.scheduled_contacts.shape == (len(rollout.time), 2)
    ):
        active = rollout.active_contacts[:processed]
        scheduled = rollout.scheduled_contacts[:processed]
        counts = np.sum(active, axis=1)
        record.update({
            "actual_single_support_fraction": float(np.mean(counts == 1)),
            "actual_double_support_fraction": float(np.mean(counts == 2)),
            "contact_schedule_match_fraction": float(np.mean(
                np.all(active == scheduled, axis=1)
            )),
        })
    else:
        record.update({
            "actual_single_support_fraction": float("nan"),
            "actual_double_support_fraction": float("nan"),
            "contact_schedule_match_fraction": float("nan"),
        })
    record.update({
        f"feature.{name}": float(value)
        for name, value in zip(signature.feature_names, signature.values)
    })
    record.update({
        f"raw_feature.{name}": float(value)
        for name, value in zip(signature.feature_names, signature.raw_values)
    })
    return record


def write_dataset(
    records: Sequence[dict[str, object]],
    config: ProtocolConfig,
    output_directory: str,
    experiment_fingerprint: str | None = None,
) -> tuple[str, str]:
    """Write one flat CSV and its reproducibility manifest atomically enough for reruns."""
    if not records:
        raise ValueError("cannot persist an empty dataset")
    os.makedirs(output_directory, exist_ok=True)
    feature_columns = [name for name in records[0] if name.startswith("feature.")]
    raw_feature_columns = [
        name for name in records[0] if name.startswith("raw_feature.")
    ]
    fieldnames = list(records[0])
    expected = set(fieldnames)
    if any(set(record) != expected for record in records):
        raise ValueError("all records must have the same flat schema")
    csv_path = os.path.join(output_directory, "dataset.csv")
    with open(csv_path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    if experiment_fingerprint is None:
        experiment_fingerprint = _experiment_fingerprint(
            config, output_directory
        )
    manifest = {
        **asdict(config),
        "schema_version": 1,
        "grouped_split": True,
        "sampling_design": {
            "train_tune_ood": "grouped_independently_permuted_scrambled_sobol",
            "calibration_test": "iid_singleton_sequences",
        },
        "feature_columns": feature_columns,
        "raw_feature_columns": raw_feature_columns,
        "record_count": len(records),
        "experiment_fingerprint": experiment_fingerprint,
        "dataset_sha256": _file_sha256(csv_path),
        "results_status": (
            "campaign_pending_analysis"
            if config.scientific_protocol
            else "independent_confirmation_pending_analysis"
            if config.confirmation_protocol
            else "unvalidated_smoke"
        ),
    }
    manifest_path = os.path.join(output_directory, "manifest.json")
    with open(manifest_path, "w") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    protocol_path = os.path.join(output_directory, "protocol.json")
    with open(protocol_path, "w") as stream:
        json.dump(asdict(config), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return csv_path, manifest_path


def _distribution_identity(distribution) -> tuple[str, str]:
    name = distribution.metadata["Name"].casefold().replace("_", "-")
    return name, distribution.version


def _distribution_files_sha256(distribution) -> str:
    if not distribution.files:
        raise RuntimeError(
            f"cannot lock non-Conda distribution without a file inventory: "
            f"{distribution.metadata['Name']}"
        )
    digest = hashlib.sha256()
    for relative in sorted(
        distribution.files, key=lambda path: str(path).lower()
    ):
        path = distribution.locate_file(relative)
        if not os.path.isfile(path):
            raise RuntimeError(
                f"cannot lock missing distribution file: {path}"
            )
        digest.update(str(relative).replace("\\", "/").encode())
        digest.update(b"\0")
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _pip_distribution_records(
    distributions, conda_python_distributions: set[tuple[str, str]]
) -> dict[str, dict[str, str]]:
    records = {}
    for distribution in distributions:
        if _distribution_identity(distribution) in conda_python_distributions:
            continue
        records[distribution.metadata["Name"]] = {
            "version": distribution.version,
            "files_sha256": _distribution_files_sha256(distribution),
        }
    return dict(sorted(records.items(), key=lambda item: item[0].lower()))


def _environment_payload() -> dict[str, object]:
    distributions = [
        distribution
        for distribution in importlib.metadata.distributions()
        if distribution.metadata["Name"]
    ]
    packages = sorted({
        distribution.metadata["Name"]: distribution.version
        for distribution in distributions
    }.items(), key=lambda item: item[0].lower())
    conda_packages = []
    conda_python_metadata = set()
    conda_meta = os.path.join(sys.prefix, "conda-meta")
    if os.path.isdir(conda_meta):
        for filename in sorted(os.listdir(conda_meta)):
            if not filename.endswith(".json"):
                continue
            with open(os.path.join(conda_meta, filename)) as stream:
                package = json.load(stream)
            conda_packages.append({
                name: package.get(name)
                for name in ("name", "version", "build", "url", "sha256")
            })
            for path in package.get("files", ()):
                parts = path.replace("\\", "/").split("/")
                for index, part in enumerate(parts):
                    if part.endswith((".dist-info", ".egg-info")):
                        conda_python_metadata.add("/".join(parts[:index + 1]))
                        break
    conda_python_distributions = {
        _distribution_identity(importlib.metadata.Distribution.at(
            os.path.join(sys.prefix, path)
        ))
        for path in conda_python_metadata
    }
    pip_packages = _pip_distribution_records(
        distributions, conda_python_distributions
    )
    robot_models = {}
    for robot in _ROBOTS:
        spec = load_robot_spec(robot)
        urdf = spec.robot.urdf
        with open(urdf, "rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        collision_meshes = []
        for mesh_path in sorted({
            geometry.meshPath
            for geometry in spec.robot.collision_model.geometryObjects
            if geometry.meshPath and os.path.isfile(geometry.meshPath)
        }):
            collision_meshes.append({
                "file": os.path.basename(mesh_path),
                "sha256": _file_sha256(mesh_path),
            })
        robot_models[robot] = {
            "urdf": os.path.basename(urdf),
            "urdf_sha256": digest,
            "collision_meshes": collision_meshes,
        }
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "packages": dict(packages),
        "conda_packages": conda_packages,
        "pip_packages": pip_packages,
        "robot_models": robot_models,
    }


def _experiment_fingerprint(
    config: ProtocolConfig,
    output_directory: str,
    *,
    extra_paths: Sequence[str] = (),
    require_environment_lock: bool = False,
) -> str:
    environment = _environment_payload()
    lock_path = os.path.join(output_directory, "environment.lock")
    if require_environment_lock:
        if not os.path.exists(lock_path):
            raise RuntimeError("run experiment lock before the scientific campaign")
        with open(lock_path) as stream:
            if json.load(stream) != environment:
                raise RuntimeError("current environment differs from environment.lock")

    digest = hashlib.sha256(json.dumps(
        {"config": asdict(config), "environment": environment},
        sort_keys=True,
    ).encode())
    source_directory = os.path.dirname(__file__)
    for filename in _SCIENTIFIC_SOURCE_FILES:
        path = os.path.join(source_directory, filename)
        if not os.path.exists(path):
            raise RuntimeError(f"scientific source is missing: {path}")
        digest.update(filename.encode())
        with open(path, "rb") as stream:
            digest.update(stream.read())
    for path in extra_paths:
        if not os.path.exists(path):
            raise RuntimeError(f"fingerprinted input is missing: {path}")
        digest.update(os.path.basename(path).encode())
        with open(path, "rb") as stream:
            digest.update(stream.read())
    return digest.hexdigest()


def _case_identity(case: ExperimentCase) -> tuple[object, ...]:
    return (
        case.robot,
        case.split,
        case.base_gait_id,
        case.perturbation_index,
    )


def _case_checkpoint_path(
    directory: str, fingerprint: str, case: ExperimentCase
) -> str:
    key = json.dumps(_case_identity(case), separators=(",", ":")).encode()
    return os.path.join(directory, fingerprint, f"{hashlib.sha256(key).hexdigest()}.json")


def _read_case_checkpoint(
    path: str, fingerprint: str, case: ExperimentCase
) -> dict[str, object] | None:
    if not os.path.exists(path):
        return None
    with open(path) as stream:
        payload = json.load(stream)
    if (
        payload.get("fingerprint") != fingerprint
        or tuple(payload.get("case", ())) != _case_identity(case)
        or not isinstance(payload.get("record"), dict)
    ):
        raise RuntimeError(f"case checkpoint does not match this run: {path}")
    record = payload["record"]
    if (
        str(record.get("robot")) != case.robot
        or str(record.get("split")) != case.split
        or str(record.get("base_gait_id")) != case.base_gait_id
        or int(record.get("perturbation_index", -1)) != case.perturbation_index
        or int(record.get("label", -1)) not in (0, 1)
    ):
        raise RuntimeError(f"case checkpoint has invalid output identity: {path}")
    _assert_record_matches_case(record, case)
    return record


def _write_case_checkpoint(
    path: str,
    fingerprint: str,
    case: ExperimentCase,
    record: dict[str, object],
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp"
    _write_json(temporary, {
        "fingerprint": fingerprint,
        "case": list(_case_identity(case)),
        "record": record,
    })
    os.replace(temporary, path)


def run_cases(
    config: ProtocolConfig,
    output_directory: str,
    splits: Sequence[str] | None = None,
    runner=None,
    workers: int = 1,
) -> tuple[list[dict[str, object]], str, str]:
    """Label selected cases with deterministic per-case checkpointing."""
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise ValueError("workers must be a positive integer")
    selected_splits = set(_SPLITS if splits is None else splits)
    unknown = selected_splits - set(_SPLITS)
    if unknown:
        raise ValueError(f"unknown splits: {sorted(unknown)}")
    default_runner = runner is None
    validated_protocol = (
        config.scientific_protocol or config.confirmation_protocol
    )
    if validated_protocol and not default_runner:
        raise RuntimeError("validated case generation forbids injected runners")
    if runner is None:
        worker_state = threading.local()

        def runner(case, protocol):
            if not hasattr(worker_state, "specs"):
                worker_state.specs = {}
            if case.robot not in worker_state.specs:
                worker_state.specs[case.robot] = load_robot_spec(case.robot)
            spec = worker_state.specs[case.robot]
            trajectory = build_whole_body_trajectory(
                spec, case.sample, steps=protocol.steps, dt=protocol.dt
            )
            signature = compute_physics_signature(spec, trajectory, case.sample)
            return flatten_record(case, signature, rollout(spec, trajectory, case.sample))

    cases = [
        case for case in generate_cases(config) if case.split in selected_splits
    ]
    fingerprint = _experiment_fingerprint(
        config,
        output_directory,
        require_environment_lock=validated_protocol and default_runner,
    )
    checkpoint_directory = os.path.join(output_directory, "case_records")
    records_by_case = {}
    pending = []
    for case in cases:
        path = _case_checkpoint_path(checkpoint_directory, fingerprint, case)
        record = _read_case_checkpoint(path, fingerprint, case)
        if record is None:
            pending.append(case)
        else:
            records_by_case[_case_identity(case)] = record

    def compute(case):
        record = runner(case, config)
        _write_case_checkpoint(
            _case_checkpoint_path(checkpoint_directory, fingerprint, case),
            fingerprint,
            case,
            record,
        )
        return record

    if workers == 1:
        generated = map(compute, pending)
        for case, record in zip(pending, generated):
            records_by_case[_case_identity(case)] = record
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for case, record in zip(pending, executor.map(compute, pending)):
                records_by_case[_case_identity(case)] = record
    records = [records_by_case[_case_identity(case)] for case in cases]
    csv_path, manifest_path = write_dataset(
        records, config, output_directory, fingerprint
    )
    return records, csv_path, manifest_path


def summarize_pilot(
    records: Sequence[dict[str, object]], config: ProtocolConfig
) -> dict[str, object]:
    """Audit the prespecified 25--75% success-rate pilot freeze rule."""
    per_robot = {}
    all_balanced = True
    for robot in config.robots:
        selected = [
            record for record in records
            if record["split"] == "pilot" and record["robot"] == robot
        ]
        if not selected:
            raise ValueError(f"pilot has no rows for {robot}")
        successes = sum(int(record["label"]) == 0 for record in selected)
        success_fraction = successes / len(selected)
        balanced = 0.25 <= success_fraction <= 0.75
        all_balanced &= balanced
        per_robot[robot] = {
            "rollouts": len(selected),
            "successes": successes,
            "success_fraction": success_fraction,
            "balanced_25_75": balanced,
        }
    if all_balanced:
        status = "ready_for_dataset"
    elif config.pilot_revision == 0:
        status = "range_adjustment_required"
    else:
        status = "validity_blocked"
    return {
        "status": status,
        "protocol_frozen": all_balanced,
        "pilot_revision": config.pilot_revision,
        "protocol": asdict(config),
        "robots": per_robot,
    }


def assert_pilot_frozen(
    summary: dict[str, object],
    config: ProtocolConfig,
    experiment_fingerprint: str | None = None,
) -> None:
    """Refuse a scientific campaign that has not passed its pilot gate."""
    expected = json.dumps(asdict(config), sort_keys=True)
    observed = json.dumps(summary.get("protocol"), sort_keys=True)
    if expected != observed:
        raise RuntimeError("pilot and dataset protocol configurations differ")
    if not summary.get("protocol_frozen"):
        raise RuntimeError(
            f"pilot is not frozen: {summary.get('status', 'unknown status')}"
        )
    if (
        experiment_fingerprint is not None
        and summary.get("experiment_fingerprint") != experiment_fingerprint
    ):
        raise RuntimeError("pilot and campaign experiment fingerprints differ")


def _file_sha256(path: str) -> str:
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _verify_dataset_provenance(
    dataset_path: str, config: ProtocolConfig
) -> dict[str, str]:
    directory = os.path.dirname(os.path.abspath(dataset_path))
    manifest_path = os.path.join(directory, "manifest.json")
    with open(manifest_path) as stream:
        manifest = json.load(stream)
    expected_fingerprint = _experiment_fingerprint(
        config,
        directory,
        require_environment_lock=config.scientific_protocol,
    )
    dataset_sha256 = _file_sha256(dataset_path)
    if manifest.get("experiment_fingerprint") != expected_fingerprint:
        raise RuntimeError("dataset was produced by a different experiment")
    if manifest.get("dataset_sha256") != dataset_sha256:
        raise RuntimeError("dataset.csv differs from its manifest")
    return {
        "experiment_fingerprint": expected_fingerprint,
        "dataset_sha256": dataset_sha256,
    }


def assert_initial_pilot_evidence(
    output_directory: str, revised_config: ProtocolConfig
) -> None:
    """Require an archived, imbalanced full-range pilot before revision one."""
    if revised_config.pilot_revision != 1:
        raise ValueError("initial-pilot evidence is required only for revision one")
    paths = {
        "records": os.path.join(output_directory, "pilot_records_revision0.csv"),
        "manifest": os.path.join(output_directory, "pilot_manifest_revision0.json"),
        "protocol": os.path.join(output_directory, "protocol_revision0.json"),
        "summary": os.path.join(output_directory, "pilot_summary_revision0.json"),
    }
    if not all(os.path.exists(path) for path in paths.values()):
        raise RuntimeError("revision one requires preserved revision-zero pilot evidence")
    initial = ProtocolConfig.from_json(paths["protocol"])
    initial_values, revised_values = asdict(initial), asdict(revised_config)
    unchanged = (
        name for name in initial_values
        if name != "pilot_revision" and not name.endswith("_range")
    )
    if initial.pilot_revision != 0 or any(
        initial_values[name] != revised_values[name] for name in unchanged
    ):
        raise RuntimeError("revised pilot differs beyond the permitted gait ranges")

    with open(paths["summary"]) as stream:
        summary = json.load(stream)
    if (
        summary.get("status") != "range_adjustment_required"
        or summary.get("protocol_frozen")
        or summary.get("dataset_sha256") != _file_sha256(paths["records"])
        or summary.get("manifest_sha256") != _file_sha256(paths["manifest"])
    ):
        raise RuntimeError("revision-zero pilot evidence is incomplete or inconsistent")
    records = read_dataset(paths["records"])
    with open(paths["manifest"]) as stream:
        manifest = json.load(stream)
    for name, expected in initial_values.items():
        if json.dumps(manifest.get(name), sort_keys=True) != json.dumps(
            expected, sort_keys=True
        ):
            raise RuntimeError(f"revision-zero manifest mismatch: {name}")
    expected_cases = {
        (
            case.robot,
            case.split,
            case.base_gait_id,
            case.perturbation_index,
        ): case
        for case in generate_cases(initial)
        if case.split == "pilot"
    }
    observed = set()
    for record in records:
        key = (
            str(record["robot"]),
            str(record["split"]),
            str(record["base_gait_id"]),
            int(record["perturbation_index"]),
        )
        if key in observed or key not in expected_cases:
            raise RuntimeError("revision-zero pilot cases are duplicated or unexpected")
        observed.add(key)
        _assert_record_matches_case(record, expected_cases[key])
        if int(record["label"]) not in (0, 1):
            raise RuntimeError("revision-zero pilot labels must be binary")
    if observed != set(expected_cases):
        raise RuntimeError("revision-zero pilot is missing cases")
    archived_summary = dict(summary)
    archived_summary.pop("dataset_sha256", None)
    archived_summary.pop("manifest_sha256", None)
    archived_summary.pop("experiment_fingerprint", None)
    if json.dumps(archived_summary, sort_keys=True) != json.dumps(
        summarize_pilot(records, initial), sort_keys=True
    ):
        raise RuntimeError("revision-zero pilot summary does not match its records")


def _assert_pilot_provenance(
    output_directory: str,
    config: ProtocolConfig,
    experiment_fingerprint: str,
) -> None:
    revision = config.pilot_revision
    paths = {
        "records": os.path.join(
            output_directory, f"pilot_records_revision{revision}.csv"
        ),
        "manifest": os.path.join(
            output_directory, f"pilot_manifest_revision{revision}.json"
        ),
        "protocol": os.path.join(
            output_directory, f"protocol_revision{revision}.json"
        ),
        "summary": os.path.join(
            output_directory, f"pilot_summary_revision{revision}.json"
        ),
    }
    active_summary_path = os.path.join(output_directory, "pilot_summary.json")
    with open(active_summary_path) as stream:
        active_summary = json.load(stream)
    with open(paths["summary"]) as stream:
        archived_summary = json.load(stream)
    if active_summary != archived_summary:
        raise RuntimeError("active and archived pilot summaries differ")
    assert_pilot_frozen(active_summary, config, experiment_fingerprint)
    if (
        active_summary.get("dataset_sha256") != _file_sha256(paths["records"])
        or active_summary.get("manifest_sha256") != _file_sha256(paths["manifest"])
    ):
        raise RuntimeError("pilot records or manifest differ from their hashes")
    with open(paths["manifest"]) as stream:
        manifest = json.load(stream)
    if manifest.get("experiment_fingerprint") != experiment_fingerprint:
        raise RuntimeError("pilot manifest refers to a different experiment")
    archived_config = ProtocolConfig.from_json(paths["protocol"])
    if asdict(archived_config) != asdict(config):
        raise RuntimeError("archived pilot protocol differs from campaign protocol")
    if revision == 1:
        assert_initial_pilot_evidence(output_directory, config)


def read_dataset(path: str) -> list[dict[str, str]]:
    with open(path, newline="") as stream:
        return list(csv.DictReader(stream))


def _assert_record_matches_case(
    record: dict[str, object], case: ExperimentCase
) -> None:
    for name, expected in asdict(case.sample).items():
        if name == "ood":
            continue
        observed = int(record[name]) if name == "seed" else float(record[name])
        if observed != expected:
            raise ValueError(f"record input differs from generated protocol: {name}")


def validate_campaign_dataset(
    records: Sequence[dict[str, object]],
    config: ProtocolConfig,
    manifest: dict[str, object],
) -> None:
    """Reject campaign CSVs that do not match the frozen mixed sampling protocol."""
    if not records:
        raise ValueError("campaign dataset is empty")
    for name, expected in asdict(config).items():
        if json.dumps(manifest.get(name), sort_keys=True) != json.dumps(
            expected, sort_keys=True
        ):
            raise ValueError(f"manifest protocol mismatch: {name}")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("grouped_split") is not True
        or manifest.get("sampling_design") != {
            "train_tune_ood": "grouped_independently_permuted_scrambled_sobol",
            "calibration_test": "iid_singleton_sequences",
        }
        or int(manifest.get("record_count", -1)) != len(records)
    ):
        raise ValueError("manifest schema or row count is invalid")

    columns = list(records[0])
    if any(list(record) != columns for record in records):
        raise ValueError("campaign rows do not share one ordered schema")
    features = [name for name in columns if name.startswith("feature.")]
    raw_features = [name for name in columns if name.startswith("raw_feature.")]
    if (
        not features
        or len(features) != len(raw_features)
        or manifest.get("feature_columns") != features
        or manifest.get("raw_feature_columns") != raw_features
    ):
        raise ValueError("campaign feature schema differs from the manifest")

    expected_cases = {
        (
            case.robot,
            case.split,
            case.base_gait_id,
            case.perturbation_index,
        ): case
        for case in generate_cases(config)
        if case.split != "pilot"
    }
    observed_keys = set()
    numeric_columns = (
        *MODEL_PARAMETER_COLUMNS,
        "runtime_seconds",
        *_CONTACT_AUDIT_COLUMNS,
        *features,
        *raw_features,
    )
    for record in records:
        key = (
            str(record["robot"]),
            str(record["split"]),
            str(record["base_gait_id"]),
            int(record["perturbation_index"]),
        )
        if key in observed_keys or key not in expected_cases:
            raise ValueError("duplicate or unexpected campaign case")
        observed_keys.add(key)
        _assert_record_matches_case(record, expected_cases[key])
        if int(record["label"]) not in (0, 1):
            raise ValueError("labels must be binary")
        if any(not math.isfinite(float(record[name])) for name in numeric_columns):
            raise ValueError("campaign contains non-finite numeric data")
        audit = [float(record[name]) for name in _CONTACT_AUDIT_COLUMNS]
        if any(not 0.0 <= value <= 1.0 for value in audit) or sum(audit[:2]) > 1.0 + 1e-12:
            raise ValueError("campaign contains invalid contact-schedule audit data")
        ood = str(record["ood"]).lower()
        if ood not in ("true", "false", "1", "0") or (
            ood in ("true", "1")
        ) != (key[1] == "ood"):
            raise ValueError("OOD flag and split disagree")

    if observed_keys != set(expected_cases):
        raise ValueError("campaign is missing frozen protocol cases")


def read_campaign_dataset(
    path: str, config: ProtocolConfig
) -> list[dict[str, str]]:
    """Load a dataset only with its colocated pilot and manifest evidence."""
    directory = os.path.dirname(os.path.abspath(path))
    provenance = _verify_dataset_provenance(path, config)
    with open(os.path.join(directory, "pilot_summary.json")) as stream:
        assert_pilot_frozen(
            json.load(stream),
            config,
            provenance["experiment_fingerprint"],
        )
    with open(os.path.join(directory, "manifest.json")) as stream:
        manifest = json.load(stream)
    records = read_dataset(path)
    validate_campaign_dataset(records, config, manifest)
    return records


def records_to_arrays(
    records: Sequence[dict[str, object]], split: str | None = None
) -> RecordArrays:
    """Convert flat records to pooled arrays without encoding robot identity."""
    selected = [record for record in records if split is None or record["split"] == split]
    if not selected:
        raise ValueError(f"no records for split {split!r}")
    signature_columns = tuple(
        name for name in selected[0] if name.startswith("feature.")
    )
    feature_names = (*MODEL_PARAMETER_COLUMNS, *signature_columns)
    X_parameters = np.asarray([
        [float(record[name]) for name in MODEL_PARAMETER_COLUMNS]
        for record in selected
    ])
    signatures = np.asarray([
        [float(record[name]) for name in signature_columns]
        for record in selected
    ])
    X = np.column_stack((X_parameters, signatures))
    return RecordArrays(
        X=X,
        X_parameters=X_parameters,
        labels=np.asarray([int(record["label"]) for record in selected]),
        robots=np.asarray([str(record["robot"]) for record in selected]),
        groups=np.asarray([str(record["base_gait_id"]) for record in selected]),
        feature_names=tuple(feature_names),
    )


def ablation_arrays(
    records: Sequence[dict[str, object]], split: str, ablation: str
) -> RecordArrays:
    """Build the three feature-removal matrices fixed before evaluation."""
    selected = [record for record in records if record["split"] == split]
    if not selected:
        raise ValueError(f"no records for split {split!r}")
    parameters = np.asarray([
        [float(record[name]) for name in MODEL_PARAMETER_COLUMNS]
        for record in selected
    ])
    normalized_columns = [
        name for name in selected[0] if name.startswith("feature.")
    ]

    if ablation == "no_normalization":
        columns = [
            name for name in selected[0] if name.startswith("raw_feature.")
        ]
        if len(columns) != len(normalized_columns):
            raise ValueError("raw features are required for normalization ablation")
        signatures = np.asarray([
            [float(record[name]) for name in columns] for record in selected
        ])
        names = tuple(columns)
    elif ablation == "no_touchdown":
        columns = [
            name for name in normalized_columns if ".touchdown." not in name
        ]
        signatures = np.asarray([
            [float(record[name]) for name in columns] for record in selected
        ])
        names = tuple(columns)
    elif ablation == "no_phase_separation":
        groups: dict[tuple[str, str], list[str]] = {}
        for name in normalized_columns:
            _, _, channel, statistic = name.split(".", 3)
            groups.setdefault((channel, statistic), []).append(name)
        blocks, names_list = [], []
        for (channel, statistic), columns in groups.items():
            values = np.asarray([
                [float(record[name]) for name in columns] for record in selected
            ])
            blocks.append(np.sort(values, axis=1))
            names_list.extend(
                f"phase_agnostic.{channel}.{statistic}.phase_rank{rank}"
                for rank in range(len(columns))
            )
        signatures = np.column_stack(blocks)
        names = tuple(names_list)
    else:
        raise ValueError(f"unknown feature ablation {ablation!r}")

    return RecordArrays(
        X=np.column_stack((parameters, signatures)),
        X_parameters=parameters,
        labels=np.asarray([int(record["label"]) for record in selected]),
        robots=np.asarray([str(record["robot"]) for record in selected]),
        groups=np.asarray([str(record["base_gait_id"]) for record in selected]),
        feature_names=(*MODEL_PARAMETER_COLUMNS, *names),
    )


def _fit_calibrated(
    train: RecordArrays,
    tune: RecordArrays,
    calibration: RecordArrays,
    seed: int,
) -> RiskCalibratedSurrogate:
    model = RiskCalibratedSurrogate(seed).fit(
        train.X,
        train.labels,
        train.robots,
        tune_data=(tune.X, tune.labels, tune.robots),
    )
    model.calibrate_threshold(
        calibration.X, calibration.labels, calibration.robots
    )
    return model


def train_surrogates(
    records: Sequence[dict[str, object]],
    output_directory: str,
    seed: int = 0,
    config: ProtocolConfig | None = None,
):
    """Fit the parameter-only baseline and calibrated phase-sequence model."""
    train = records_to_arrays(records, "train")
    tune = records_to_arrays(records, "tune")
    calibration = records_to_arrays(records, "calibration")
    surrogate = _fit_calibrated(train, tune, calibration, seed)
    black_box = RiskCalibratedSurrogate(seed).fit(
        train.X_parameters,
        train.labels,
        train.robots,
        tune_data=(tune.X_parameters, tune.labels, tune.robots),
    )
    os.makedirs(output_directory, exist_ok=True)
    surrogate.save(os.path.join(output_directory, "surrogate.pkl"))
    black_box.save(os.path.join(output_directory, "black_box.pkl"))
    ablations = {}
    for name in (
        "no_normalization", "no_phase_separation", "no_touchdown"
    ):
        model = _fit_calibrated(
            ablation_arrays(records, "train", name),
            ablation_arrays(records, "tune", name),
            ablation_arrays(records, "calibration", name),
            seed,
        )
        model.save(os.path.join(output_directory, f"ablation_{name}.pkl"))
        ablations[name] = {
            "best_params": model.best_params_,
            "calibration": model.calibration_,
        }
    ablations["no_risk_calibration"] = {
        "model": "surrogate.pkl",
        "threshold": 0.5,
    }
    model_sha256 = {
        filename: _file_sha256(os.path.join(output_directory, filename))
        for filename in _MODEL_FILENAMES
    }
    experiment_fingerprint = None
    dataset_sha256 = None
    if config is not None:
        experiment_fingerprint = _experiment_fingerprint(
            config,
            output_directory,
            require_environment_lock=config.scientific_protocol,
        )
        dataset_path = os.path.join(output_directory, "dataset.csv")
        if os.path.exists(dataset_path):
            provenance = _verify_dataset_provenance(dataset_path, config)
            experiment_fingerprint = provenance["experiment_fingerprint"]
            dataset_sha256 = provenance["dataset_sha256"]
        elif config.scientific_protocol:
            raise RuntimeError("scientific training requires the campaign dataset")
    training_fingerprint = hashlib.sha256(json.dumps({
        "experiment_fingerprint": experiment_fingerprint,
        "dataset_sha256": dataset_sha256,
        "model_sha256": model_sha256,
    }, sort_keys=True).encode()).hexdigest()
    metadata = {
        "seed": seed,
        "protocol": asdict(config) if config is not None else None,
        "experiment_fingerprint": experiment_fingerprint,
        "dataset_sha256": dataset_sha256,
        "model_sha256": model_sha256,
        "training_fingerprint": training_fingerprint,
        "split_rows": {
            "train": len(train.labels),
            "tune": len(tune.labels),
            "calibration": len(calibration.labels),
        },
        "feature_names": list(train.feature_names),
        "surrogate_best_params": surrogate.best_params_,
        "black_box_best_params": black_box.best_params_,
        "calibration": surrogate.calibration_,
        "ablations": ablations,
    }
    metadata_path = os.path.join(output_directory, "training.json")
    with open(metadata_path, "w") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return surrogate, black_box, metadata_path


def assert_training_protocol(
    output_directory: str, config: ProtocolConfig
) -> dict[str, object]:
    return assert_training_provenance(output_directory, config)


def assert_training_provenance(
    output_directory: str, config: ProtocolConfig
) -> dict[str, object]:
    with open(os.path.join(output_directory, "training.json")) as stream:
        metadata = json.load(stream)
    if json.dumps(metadata.get("protocol"), sort_keys=True) != json.dumps(
        asdict(config), sort_keys=True
    ):
        raise RuntimeError("trained models and requested protocol differ")
    expected_experiment = _experiment_fingerprint(
        config,
        output_directory,
        require_environment_lock=config.scientific_protocol,
    )
    if metadata.get("experiment_fingerprint") != expected_experiment:
        raise RuntimeError("training and current experiment fingerprints differ")
    dataset_path = os.path.join(output_directory, "dataset.csv")
    if os.path.exists(dataset_path):
        dataset = _verify_dataset_provenance(dataset_path, config)
        if metadata.get("dataset_sha256") != dataset["dataset_sha256"]:
            raise RuntimeError("training refers to a different dataset")
    elif config.scientific_protocol or metadata.get("dataset_sha256") is not None:
        raise RuntimeError("training dataset is missing")
    model_sha256 = {
        filename: _file_sha256(os.path.join(output_directory, filename))
        for filename in _MODEL_FILENAMES
    }
    if metadata.get("model_sha256") != model_sha256:
        raise RuntimeError("a trained model differs from training.json")
    expected_training = hashlib.sha256(json.dumps({
        "experiment_fingerprint": expected_experiment,
        "dataset_sha256": metadata.get("dataset_sha256"),
        "model_sha256": model_sha256,
    }, sort_keys=True).encode()).hexdigest()
    if metadata.get("training_fingerprint") != expected_training:
        raise RuntimeError("training fingerprint is invalid")
    return {
        "experiment_fingerprint": expected_experiment,
        "dataset_sha256": metadata.get("dataset_sha256"),
        "model_sha256": model_sha256,
        "training_fingerprint": expected_training,
    }


def _assert_validity_provenance(
    output_directory: str, config: ProtocolConfig
) -> dict[str, object]:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(output_directory, "validity.json")
    with open(path) as stream:
        validity = json.load(stream)
    experiment_fingerprint = _experiment_fingerprint(
        config,
        output_directory,
        require_environment_lock=(
            config.scientific_protocol or config.confirmation_protocol
        ),
    )
    test_sha256 = {
        test_file: _file_sha256(os.path.join(root, test_file))
        for test_file in _VALIDITY_TEST_FILES
    }
    expected = hashlib.sha256(json.dumps({
        "experiment_fingerprint": experiment_fingerprint,
        "test_sha256": test_sha256,
    }, sort_keys=True).encode()).hexdigest()
    if (
        validity.get("experiment_fingerprint") != experiment_fingerprint
        or validity.get("test_sha256") != test_sha256
        or validity.get("validity_fingerprint") != expected
    ):
        raise RuntimeError("validity evidence is stale or inconsistent")
    return validity


def _assert_evaluation_provenance(
    output_directory: str, training_fingerprint: str
) -> dict[str, object]:
    with open(os.path.join(output_directory, "evaluation.json")) as stream:
        evaluation = json.load(stream)
    if (
        evaluation.get("provenance", {}).get("training_fingerprint")
        != training_fingerprint
    ):
        raise RuntimeError("evaluation refers to different trained models")
    return evaluation


def _physics_score(arrays: RecordArrays, tokens: tuple[str, ...]) -> np.ndarray:
    indices = [
        index for index, name in enumerate(arrays.feature_names)
        if index >= len(MODEL_PARAMETER_COLUMNS) and any(token in name for token in tokens)
    ]
    if not indices:
        return np.full(len(arrays.labels), 0.5)
    violation = np.max(arrays.X[:, indices], axis=1)
    return expit(4.0 * np.clip(violation, -100.0, 100.0))


def score_methods(
    records: Sequence[dict[str, object]],
    surrogate: RiskCalibratedSurrogate,
    black_box: RiskCalibratedSurrogate,
    split: str,
) -> dict[str, np.ndarray]:
    arrays = records_to_arrays(records, split)
    return _score_arrays(arrays, surrogate, black_box)


def _score_arrays(
    arrays: RecordArrays,
    surrogate: RiskCalibratedSurrogate,
    black_box: RiskCalibratedSurrogate,
) -> dict[str, np.ndarray]:
    proposed = surrogate.predict_failure_score(arrays.X)
    return {
        "zmp_cop_margin": _physics_score(arrays, (".zmp.", ".cop.")),
        "ik_joint_margin": _physics_score(
            arrays, (".ik_residual.", ".joint_position.")
        ),
        "inverse_dynamics_slack": _physics_score(
            arrays, (".dynamics_slack.",)
        ),
        "black_box_parameters": black_box.predict_failure_score(arrays.X_parameters),
        "uncalibrated_phase_sequence": proposed,
        "risk_calibrated_phase_sequence": proposed,
    }


def _exact_acceptance_mask(scores: np.ndarray, accepted: int) -> np.ndarray:
    """Select exactly the lowest-scoring rows with stable tie breaking."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("scores must be a finite one-dimensional vector")
    if not isinstance(accepted, (int, np.integer)):
        raise TypeError("accepted must be an integer")
    if not 0 <= accepted <= len(scores):
        raise ValueError("accepted must be between zero and the row count")
    mask = np.zeros(len(scores), dtype=bool)
    mask[np.argsort(scores, kind="stable")[:accepted]] = True
    return mask


def _acceptance_mask(
    scores: np.ndarray,
    selector: float | np.ndarray | None,
) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if selector is None:
        return np.zeros(len(scores), dtype=bool)
    if np.isscalar(selector):
        return scores <= float(selector)
    mask = np.asarray(selector, dtype=bool)
    if mask.shape != scores.shape:
        raise ValueError("acceptance mask must match score rows")
    return mask


def paired_group_bootstrap(
    labels: np.ndarray,
    proposed_scores: np.ndarray,
    baseline_scores: dict[str, np.ndarray],
    proposed_threshold: float | np.ndarray | None,
    groups: np.ndarray,
    *,
    repetitions: int = 2000,
    seed: int = 0,
) -> dict[str, object]:
    """Bootstrap the reselected best-baseline risk at matched coverage."""
    labels = np.asarray(labels, dtype=int)
    proposed_scores = np.asarray(proposed_scores, dtype=float)
    groups = np.asarray(groups, dtype=str)
    baselines = {
        name: np.asarray(scores, dtype=float)
        for name, scores in baseline_scores.items()
    }
    if (
        not baselines
        or labels.shape != proposed_scores.shape
        or labels.shape != groups.shape
        or any(scores.shape != labels.shape for scores in baselines.values())
    ):
        raise ValueError("paired bootstrap inputs must have matching rows")
    if repetitions < 1:
        raise ValueError("bootstrap repetitions must be positive")

    proposed_accepts = _acceptance_mask(proposed_scores, proposed_threshold)
    def compare(indices):
        selected_labels = labels[indices]
        selected_proposed = proposed_accepts[indices]
        accepted = int(np.sum(selected_proposed))
        if not accepted:
            return None
        proposed_risk = float(np.mean(selected_labels[selected_proposed]))
        risks = {
            name: float(np.mean(selected_labels[
                _exact_acceptance_mask(scores[indices], accepted)
            ]))
            for name, scores in baselines.items()
        }
        best_name = min(risks, key=lambda name: (risks[name], name))
        return best_name, proposed_risk, risks[best_name]

    observed_comparison = compare(np.arange(len(labels)))
    if observed_comparison is None:
        best_baseline = None
        proposed_risk = baseline_risk = observed = None
    else:
        best_baseline, proposed_risk, baseline_risk = observed_comparison
        observed = baseline_risk - proposed_risk
    unique_groups = np.unique(groups)
    members = {
        group: np.flatnonzero(groups == group) for group in unique_groups
    }
    rng = np.random.default_rng(seed)
    differences = []
    if observed is not None:
        for _ in range(repetitions):
            sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
            indices = np.concatenate([members[group] for group in sampled])
            replicate = compare(indices)
            if replicate is not None:
                differences.append(replicate[2] - replicate[1])
    interval = (
        [float(value) for value in np.percentile(differences, (2.5, 97.5))]
        if differences else [None, None]
    )
    return {
        "groups": len(unique_groups),
        "baseline": best_baseline,
        "baseline_candidates": len(baselines),
        "reselected_each_replicate": True,
        "repetitions": repetitions,
        "valid_repetitions": len(differences),
        "proposed_risk": proposed_risk,
        "baseline_risk": baseline_risk,
        "risk_difference": observed,
        "relative_risk_reduction": (
            observed / baseline_risk
            if observed is not None and baseline_risk and baseline_risk > 0.0
            else None
        ),
        "ci95": interval,
    }


def evaluate_records(
    records: Sequence[dict[str, object]],
    surrogate: RiskCalibratedSurrogate,
    black_box: RiskCalibratedSurrogate,
    split: str,
    robot: str | None = None,
) -> dict[str, object]:
    selected = [
        record for record in records
        if record["split"] == split and (robot is None or record["robot"] == robot)
    ]
    arrays = records_to_arrays(selected)
    scores = score_methods(selected, surrogate, black_box, split)
    runtimes = np.asarray([float(record["runtime_seconds"]) for record in selected])
    reasons = np.asarray([str(record["failure_reason"]) for record in selected])
    natural = {name: 0.5 for name in scores}
    natural["risk_calibrated_phase_sequence"] = surrogate.threshold_
    proposed_acceptance = np.asarray(surrogate.accept(arrays.X), dtype=bool)
    calibrated_accepts = int(np.sum(proposed_acceptance))
    matched: dict[str, float | np.ndarray | None] = {
        name: (
            proposed_acceptance
            if name == "risk_calibrated_phase_sequence"
            else (
                0.5
                if name == "uncalibrated_phase_sequence"
                else _exact_acceptance_mask(values, calibrated_accepts)
            )
        )
        for name, values in scores.items()
    }
    prespecified = evaluate_score_methods(
        arrays.labels,
        scores,
        natural,
        runtimes,
        reasons,
        alpha=0.05 if split == "test" else None,
    )
    matched_metrics = evaluate_score_methods(
        arrays.labels,
        scores,
        matched,
        runtimes,
        reasons,
        alpha=None,
    )
    if split == "test":
        for name in (
            "uncalibrated_phase_sequence",
            "risk_calibrated_phase_sequence",
        ):
            matched_metrics[name].update(calibration_metrics(
                arrays.labels, scores[name], matched[name], alpha=0.05
            ))
    comparison = paired_group_bootstrap(
        arrays.labels,
        scores["risk_calibrated_phase_sequence"],
        {
            name: scores[name]
            for name in _MATCHED_COVERAGE_BASELINES
        },
        proposed_acceptance,
        arrays.groups,
        seed=0,
    )
    return {
        "split": split,
        "robot": robot or "pooled",
        "rows": len(selected),
        "prespecified": prespecified,
        "matched_coverage": matched_metrics,
        "selection_rules": {
            "prespecified": natural,
            "matched_coverage": {
                name: (
                    {"kind": "calibrated_threshold", "count": calibrated_accepts}
                    if name == "risk_calibrated_phase_sequence"
                    else (
                        {"kind": "native_threshold", "threshold": 0.5}
                        if name == "uncalibrated_phase_sequence"
                        else {"kind": "exact_top_k", "count": calibrated_accepts}
                    )
                )
                for name in scores
            },
        },
        "comparison_to_best_baseline": comparison,
    }


def load_ablation_models(output_directory: str) -> dict[str, RiskCalibratedSurrogate]:
    return {
        name: RiskCalibratedSurrogate.load(os.path.join(
            output_directory, f"ablation_{name}.pkl"
        ))
        for name in (
            "no_normalization", "no_phase_separation", "no_touchdown"
        )
    }


def evaluate_ablations(
    records: Sequence[dict[str, object]],
    surrogate: RiskCalibratedSurrogate,
    ablation_models: dict[str, RiskCalibratedSurrogate],
    split: str,
    robot: str | None = None,
) -> dict[str, object]:
    selected = [
        record for record in records
        if record["split"] == split and (robot is None or record["robot"] == robot)
    ]
    runtimes = np.asarray([
        float(record["runtime_seconds"]) for record in selected
    ])
    reasons = np.asarray([str(record["failure_reason"]) for record in selected])
    results = {}
    for name, model in ablation_models.items():
        arrays = ablation_arrays(selected, split, name)
        results[name] = evaluate_score_methods(
            arrays.labels,
            {name: model.predict_failure_score(arrays.X)},
            {name: model.threshold_},
            runtimes,
            reasons,
        )[name]
    arrays = records_to_arrays(selected)
    results["no_risk_calibration"] = evaluate_score_methods(
        arrays.labels,
        {"no_risk_calibration": surrogate.predict_failure_score(arrays.X)},
        {"no_risk_calibration": 0.5},
        runtimes,
        reasons,
    )["no_risk_calibration"]
    return results


def _write_json(path: str, payload: object) -> str:
    with open(path, "w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return path


def summarize_screening(
    rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Aggregate paired downstream repetitions without mixing protocol cells."""
    cells: dict[tuple[object, ...], list[dict[str, object]]] = {}
    paired_pools: dict[tuple[object, ...], tuple[int, int]] = {}
    keys = ("method", "robot", "condition", "budget")
    for row in rows:
        if "pool_seed" not in row or "pool_size" not in row:
            raise ValueError("screening rows must identify their candidate pool")
        paired_key = (
            row["robot"],
            row["condition"],
            int(row["repetition"]),
        )
        pool = (int(row["pool_seed"]), int(row["pool_size"]))
        if paired_key in paired_pools and paired_pools[paired_key] != pool:
            raise ValueError(
                f"mismatched paired candidate pool in cell {paired_key}"
            )
        paired_pools[paired_key] = pool
        key = tuple(row[name] for name in keys)
        cells.setdefault(key, []).append(row)
    summary = []
    for key in sorted(cells):
        selected = cells[key]
        repetitions = {int(row["repetition"]) for row in selected}
        pool_sizes = {int(row["pool_size"]) for row in selected}
        if len(repetitions) != len(selected):
            raise ValueError(f"duplicate screening repetition in cell {key}")
        if len(pool_sizes) != 1:
            raise ValueError(f"inconsistent candidate pool size in cell {key}")
        summary.append({
            **dict(zip(keys, key)),
            "repetitions": len(repetitions),
            "pool_size": pool_sizes.pop(),
            "success_by_repetition": {
                str(int(row["repetition"])): bool(row["success"])
                for row in sorted(selected, key=lambda item: int(item["repetition"]))
            },
            "success_rate": float(np.mean([
                bool(row["success"]) for row in selected
            ])),
            "mean_rollouts": float(np.mean([
                int(row["rollouts"]) for row in selected
            ])),
            "rollouts_by_repetition": {
                str(int(row["repetition"])): int(row["rollouts"])
                for row in sorted(selected, key=lambda item: int(item["repetition"]))
            },
            "mean_rollout_runtime_seconds": float(np.mean([
                float(row["rollout_runtime_seconds"]) for row in selected
            ])),
            "mean_oracle_wall_seconds": float(np.mean([
                float(row.get("oracle_wall_seconds", 0.0)) for row in selected
            ])),
            "mean_actual_oracle_wall_seconds": float(np.mean([
                float(row.get(
                    "actual_oracle_wall_seconds",
                    row.get("oracle_wall_seconds", 0.0),
                ))
                for row in selected
            ])),
            "mean_feature_runtime_seconds": float(np.mean([
                float(row.get("feature_runtime_seconds", 0.0))
                for row in selected
            ])),
            "mean_candidate_trajectory_ik_seconds": float(np.mean([
                float(row.get("candidate_trajectory_ik_seconds", 0.0))
                for row in selected
            ])),
            "mean_signature_runtime_seconds": float(np.mean([
                float(row.get("signature_runtime_seconds", 0.0))
                for row in selected
            ])),
            "mean_inference_runtime_seconds": float(np.mean([
                float(row.get("inference_runtime_seconds", 0.0))
                for row in selected
            ])),
        })
    return summary


def _paired_screening_difference(
    proposed: dict[str, object],
    baselines: Sequence[dict[str, object]],
    *,
    repetitions: int = 2000,
) -> dict[str, object] | None:
    proposed_by_repetition = proposed.get("success_by_repetition")
    baseline_maps = {
        str(row["method"]): row.get("success_by_repetition")
        for row in baselines
    }
    if (
        not isinstance(proposed_by_repetition, dict)
        or not baseline_maps
        or any(not isinstance(values, dict) for values in baseline_maps.values())
    ):
        return None
    keys = sorted(proposed_by_repetition, key=int)
    if any(set(values) != set(keys) for values in baseline_maps.values()):
        raise ValueError("paired screening summaries have different repetitions")
    proposed_values = np.asarray(
        [bool(proposed_by_repetition[key]) for key in keys], dtype=float
    )
    baseline_values = {
        name: np.asarray([bool(values[key]) for key in keys], dtype=float)
        for name, values in baseline_maps.items()
    }
    best = max(
        baseline_values,
        key=lambda name: (float(np.mean(baseline_values[name])), name),
    )
    observed = float(np.mean(proposed_values - baseline_values[best]))
    rng = np.random.default_rng(0)
    differences = []
    for _ in range(repetitions):
        indices = rng.integers(0, len(keys), len(keys))
        replicate_best = max(
            baseline_values,
            key=lambda name: (
                float(np.mean(baseline_values[name][indices])),
                name,
            ),
        )
        differences.append(float(np.mean(
            proposed_values[indices] - baseline_values[replicate_best][indices]
        )))
    return {
        "baseline": best,
        "success_rate_difference": observed,
        "paired_bootstrap_repetitions": repetitions,
        "baseline_reselected_each_replicate": len(baseline_values) > 1,
        "ci95": [
            float(value) for value in np.percentile(differences, (2.5, 97.5))
        ],
    }


def _paired_rollout_reduction(
    proposed: dict[str, object],
    baseline: dict[str, object],
) -> dict[str, float] | None:
    proposed_map = proposed.get("rollouts_by_repetition")
    baseline_map = baseline.get("rollouts_by_repetition")
    if not isinstance(proposed_map, dict) or not isinstance(baseline_map, dict):
        return None
    if set(proposed_map) != set(baseline_map):
        raise ValueError("paired screening summaries have different repetitions")
    keys = sorted(proposed_map, key=int)
    proposed_calls = np.asarray([float(proposed_map[key]) for key in keys])
    baseline_calls = np.asarray([float(baseline_map[key]) for key in keys])
    if (
        not len(keys)
        or not np.isfinite(proposed_calls).all()
        or not np.isfinite(baseline_calls).all()
        or np.any(proposed_calls < 0.0)
        or np.any(baseline_calls < 0.0)
    ):
        raise ValueError("paired rollout counts must be finite and nonnegative")
    proposed_mean = float(np.mean(proposed_calls))
    baseline_mean = float(np.mean(baseline_calls))
    if baseline_mean == 0.0:
        return None
    return {
        "proposed_mean_rollouts": proposed_mean,
        "baseline_mean_rollouts": baseline_mean,
        "rollout_reduction": 1.0 - proposed_mean / baseline_mean,
    }


def readiness_gate(
    evaluation: dict[str, object],
    screening_summary: Sequence[dict[str, object]],
    *,
    validity_passed: bool,
) -> dict[str, object]:
    """Apply the frozen reporting gates without turning failure into a claim."""
    robot_details = {}
    all_statistical = True
    expected_cells = {
        (method, robot, condition, budget)
        for method in _SCREENING_METHODS
        for robot in _ROBOTS
        for condition in ("id", "ood")
        for budget in (5, 10, 20)
    }
    indexed_cells = {
        (
            str(row["method"]),
            str(row["robot"]),
            str(row["condition"]),
            int(row["budget"]),
        ): row
        for row in screening_summary
    }
    screening_protocol_complete = (
        set(indexed_cells) == expected_cells
        and all(
            int(row.get("repetitions", 0)) == 30
            and int(row.get("pool_size", 0)) == 2048
            for row in indexed_cells.values()
        )
    )
    all_downstream = screening_protocol_complete
    test_results = evaluation.get("methods", {}).get("test", {})
    for robot in _ROBOTS:
        result = test_results.get(robot)
        if result is None:
            robot_details[robot] = {"passed": False, "reason": "missing_test_result"}
            all_statistical = all_downstream = False
            continue
        proposed = result["prespecified"]["risk_calibrated_phase_sequence"]
        comparison = result["comparison_to_best_baseline"]
        interval = comparison.get("ci95", [None, None])
        statistical = {
            "false_safe_upper_at_most_0_05":
                proposed["false_safe_upper"] <= 0.05,
            "coverage_at_least_0_20": proposed["coverage"] >= 0.20,
            "relative_risk_reduction_at_least_0_30":
                comparison.get("relative_risk_reduction") is not None
                and comparison["relative_risk_reduction"] >= 0.30,
            "bootstrap_lower_above_zero":
                interval[0] is not None and interval[0] > 0.0,
        }
        all_statistical &= all(statistical.values())

        budget_20_cells = [
            row for row in screening_summary
            if row["robot"] == robot
            and row["condition"] == "id"
            and int(row["budget"]) == 20
        ]
        proposed_cells = [
            row for row in budget_20_cells
            if row["method"] == "risk_calibrated_phase_sequence"
        ]
        baseline_cells = [
            row for row in budget_20_cells
            if row["method"] != "risk_calibrated_phase_sequence"
        ]
        reduced_budget_cells = [
            row for row in screening_summary
            if row["robot"] == robot
            and row["condition"] == "id"
            and row["method"] == "risk_calibrated_phase_sequence"
            and int(row["budget"]) <= 10
        ]
        if proposed_cells and baseline_cells:
            proposed_cell = proposed_cells[0]
            success_comparison = _paired_screening_difference(
                proposed_cell, baseline_cells
            )
            if success_comparison is None:
                downstream = {
                    "passed": False,
                    "reason": "missing_paired_downstream_outcomes",
                }
                all_downstream = False
                robot_details[robot] = {
                    "statistical": statistical,
                    "downstream": downstream,
                    "passed": False,
                }
                continue
            best_baseline = next(
                row for row in baseline_cells
                if row["method"] == success_comparison["baseline"]
            )
            efficiency = []
            for row in reduced_budget_cells:
                comparison = _paired_screening_difference(
                    row, [best_baseline]
                )
                rollout_comparison = _paired_rollout_reduction(
                    row, best_baseline
                )
                if (
                    comparison is not None
                    and rollout_comparison is not None
                    and float(row["success_rate"]) > 0.0
                    and comparison["success_rate_difference"] >= 0.0
                    and comparison["ci95"][0] >= 0.0
                    and rollout_comparison["rollout_reduction"] >= 0.50
                ):
                    efficiency.append((
                        row, comparison, rollout_comparison
                    ))
            efficient = (
                min(efficiency, key=lambda item: int(item[0]["budget"]))
                if efficiency else None
            )
            success_gain_passed = (
                success_comparison["success_rate_difference"] >= 0.15
                and success_comparison["ci95"][0] > 0.0
            )
            downstream_passed = (
                screening_protocol_complete
                and (success_gain_passed or efficient is not None)
            )
            downstream = {
                "baseline": best_baseline["method"],
                "baseline_budget": 20,
                "baseline_success_rate": float(best_baseline["success_rate"]),
                "success_rate_gain": success_comparison[
                    "success_rate_difference"
                ],
                "success_rate_gain_ci95": success_comparison["ci95"],
                "matched_success_budget": (
                    int(efficient[0]["budget"]) if efficient else None
                ),
                "matched_success_difference_ci95": (
                    efficient[1]["ci95"] if efficient else None
                ),
                "proposed_mean_rollouts": (
                    efficient[2]["proposed_mean_rollouts"]
                    if efficient else None
                ),
                "baseline_mean_rollouts": (
                    efficient[2]["baseline_mean_rollouts"]
                    if efficient else None
                ),
                "rollout_reduction": (
                    efficient[2]["rollout_reduction"] if efficient else None
                ),
                "screening_protocol_complete": screening_protocol_complete,
                "passed": downstream_passed,
            }
        else:
            downstream = {"passed": False, "reason": "missing_budget_20_cell"}
        all_downstream &= downstream["passed"]
        robot_details[robot] = {
            "statistical": statistical,
            "downstream": downstream,
            "passed": all(statistical.values()) and downstream["passed"],
        }

    ready = bool(validity_passed and all_statistical and all_downstream)
    return {
        "ready": ready,
        "validity_passed": bool(validity_passed),
        "screening_protocol_complete": screening_protocol_complete,
        "claim_mode": "full_prespecified_claim" if ready else "benchmark_or_negative_result",
        "robots": robot_details,
    }


_PLAIN_TEST_RUNNER = """
import inspect
import runpy
import sys

namespace = runpy.run_path(sys.argv[1], run_name="__validity__")
tests = [
    value
    for name, value in namespace.items()
    if name.startswith("test_")
    and inspect.isfunction(value)
    and value.__module__ == "__validity__"
    and not inspect.signature(value).parameters
]
for test in tests:
    test()
print(f"{len(tests)} tests passed")
"""


def _run_plain_test_file(
    test_file: str, root: str, environment: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", _PLAIN_TEST_RUNNER, test_file],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def run_validity_tests(
    output_directory: str, config: ProtocolConfig
) -> dict[str, object]:
    """Run every physical-oracle regression defined in the validity files."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_files = _VALIDITY_TEST_FILES
    environment = os.environ.copy()
    environment.update({
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    })
    checks = []
    for test_file in test_files:
        completed = _run_plain_test_file(test_file, root, environment)
        checks.append({
            "test": test_file,
            "returncode": completed.returncode,
            "last_output_line": (
                (completed.stdout or completed.stderr).strip().splitlines()[-1]
                if (completed.stdout or completed.stderr).strip() else ""
            ),
        })
    experiment_fingerprint = _experiment_fingerprint(
        config,
        output_directory,
        require_environment_lock=config.scientific_protocol,
    )
    test_sha256 = {
        test_file: _file_sha256(os.path.join(root, test_file))
        for test_file in test_files
    }
    validity_fingerprint = hashlib.sha256(json.dumps({
        "experiment_fingerprint": experiment_fingerprint,
        "test_sha256": test_sha256,
    }, sort_keys=True).encode()).hexdigest()
    report = {
        "passed": all(check["returncode"] == 0 for check in checks),
        "checks": checks,
        "experiment_fingerprint": experiment_fingerprint,
        "test_sha256": test_sha256,
        "validity_fingerprint": validity_fingerprint,
    }
    os.makedirs(output_directory, exist_ok=True)
    _write_json(os.path.join(output_directory, "validity.json"), report)
    return report


def write_environment_lock(output_directory: str) -> str:
    """Capture exact installed Python package and platform versions."""
    os.makedirs(output_directory, exist_ok=True)
    payload = _environment_payload()
    lock_path = _write_json(
        os.path.join(output_directory, "environment.lock"),
        payload,
    )
    explicit_path = os.path.join(output_directory, "conda-explicit.txt")
    with open(explicit_path, "w") as stream:
        stream.write(
            "# Create with: conda create --name reproduction "
            "--file conda-explicit.txt\n"
            "# Then install: conda run --name reproduction python -m pip "
            "install -r pip-requirements.txt\n"
            "# platform and builds are fixed by the package URLs\n"
            "@EXPLICIT\n"
        )
        for package in payload["conda_packages"]:
            if package.get("url"):
                stream.write(f"{package['url']}\n")
    pip_path = os.path.join(output_directory, "pip-requirements.txt")
    with open(pip_path, "w") as stream:
        stream.write("# Version-pinned packages installed outside Conda\n")
        for name, record in payload["pip_packages"].items():
            stream.write(f"{name}=={record['version']}\n")
    return lock_path


def verify_campaign_provenance(
    output_directory: str, config: ProtocolConfig
) -> dict[str, object]:
    """Reject any campaign assembled from stale or mixed-stage outputs."""
    try:
        training = assert_training_provenance(output_directory, config)
        if config.scientific_protocol:
            _assert_pilot_provenance(
                output_directory,
                config,
                str(training["experiment_fingerprint"]),
            )
        evaluation = _assert_evaluation_provenance(
            output_directory, str(training["training_fingerprint"])
        )
        validity = _assert_validity_provenance(output_directory, config)
        with open(os.path.join(
            output_directory, "screening_summary.json"
        )) as stream:
            screening_summary = json.load(stream)
        readiness_path = os.path.join(output_directory, "readiness.json")
        with open(readiness_path) as stream:
            readiness = json.load(stream)
        if not isinstance(readiness, dict) or not isinstance(
            screening_summary, list
        ):
            raise TypeError("readiness and screening summary must be objects")
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise RuntimeError("campaign provenance evidence is unreadable") from error
    provenance = readiness.get("provenance", {})
    workers = int(provenance.get("screening_workers", 0))
    if workers < 1:
        raise RuntimeError("readiness does not identify screening workers")
    screening_fingerprint = _screening_fingerprint(
        config, output_directory, workers
    )
    expected = {
        "experiment_fingerprint": training["experiment_fingerprint"],
        "training_fingerprint": training["training_fingerprint"],
        "screening_fingerprint": screening_fingerprint,
        "evaluation_sha256": _file_sha256(os.path.join(
            output_directory, "evaluation.json"
        )),
        "validity_sha256": _file_sha256(os.path.join(
            output_directory, "validity.json"
        )),
        "screening_csv_sha256": _file_sha256(os.path.join(
            output_directory, "screening.csv"
        )),
        "screening_calls_sha256": _file_sha256(os.path.join(
            output_directory, "screening_calls.csv"
        )),
        "screening_summary_sha256": _file_sha256(os.path.join(
            output_directory, "screening_summary.json"
        )),
        "screening_workers": workers,
    }
    if provenance != expected:
        raise RuntimeError("readiness provenance chain is stale or inconsistent")
    persisted_gate = {
        name: value for name, value in readiness.items()
        if name != "provenance"
    }
    recomputed_gate = readiness_gate(
        evaluation,
        screening_summary,
        validity_passed=bool(validity.get("passed")),
    )
    if persisted_gate != recomputed_gate:
        raise RuntimeError("readiness gate is stale or inconsistent")
    return readiness


def cli(argv=None, runner=None) -> int:
    """Headless entrypoint for the frozen pilot/dataset/train/evaluate workflow."""
    parser = argparse.ArgumentParser(prog="python -m src.experiment")
    parser.add_argument(
        "action",
        choices=(
            "lock", "validate", "pilot", "dataset", "train", "evaluate",
            "screen", "smoke",
        ),
    )
    parser.add_argument("--output-dir", default="results/experiment")
    parser.add_argument("--dataset-csv")
    parser.add_argument("--protocol-json")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)
    dataset_path = args.dataset_csv or os.path.join(args.output_dir, "dataset.csv")
    config = (
        ProtocolConfig.from_json(args.protocol_json)
        if args.protocol_json
        else ProtocolConfig() if args.seed is None else ProtocolConfig(seed=args.seed)
    )
    if args.protocol_json and args.seed is not None:
        config = replace(config, seed=args.seed)

    if args.action == "lock":
        print(f"experiment lock: {write_environment_lock(args.output_dir)}")
        return 0

    if args.action == "validate":
        report = run_validity_tests(args.output_dir, config)
        print(f"experiment validate: passed={report['passed']}")
        return 0 if report["passed"] else 1

    if args.action == "pilot":
        revision = config.pilot_revision
        archive_paths = {
            "records": os.path.join(
                args.output_dir, f"pilot_records_revision{revision}.csv"
            ),
            "manifest": os.path.join(
                args.output_dir, f"pilot_manifest_revision{revision}.json"
            ),
            "protocol": os.path.join(
                args.output_dir, f"protocol_revision{revision}.json"
            ),
            "summary": os.path.join(
                args.output_dir, f"pilot_summary_revision{revision}.json"
            ),
        }
        if any(os.path.exists(path) for path in archive_paths.values()):
            raise RuntimeError(f"pilot revision {revision} is already archived")
        if revision == 1:
            assert_initial_pilot_evidence(args.output_dir, config)
        records, csv_path, manifest_path = run_cases(
            config,
            args.output_dir,
            splits=("pilot",),
            runner=runner,
            workers=args.workers,
        )
        summary = summarize_pilot(records, config)
        with open(manifest_path) as stream:
            summary["experiment_fingerprint"] = json.load(stream)[
                "experiment_fingerprint"
            ]
        summary["dataset_sha256"] = _file_sha256(csv_path)
        summary["manifest_sha256"] = _file_sha256(manifest_path)
        summary_path = _write_json(
            os.path.join(args.output_dir, "pilot_summary.json"), summary
        )
        for source, destination in (
            (csv_path, archive_paths["records"]),
            (manifest_path, archive_paths["manifest"]),
            (os.path.join(args.output_dir, "protocol.json"), archive_paths["protocol"]),
            (summary_path, archive_paths["summary"]),
        ):
            shutil.copyfile(source, destination)
        print(
            f"experiment pilot: {len(records)} rows -> {csv_path}; {manifest_path}; "
            f"{summary_path}; status={summary['status']}"
        )
        return 0

    if args.action == "dataset":
        pilot_path = os.path.join(args.output_dir, "pilot_summary.json")
        if not os.path.exists(pilot_path):
            raise RuntimeError("run and freeze the pilot before generating the dataset")
        with open(pilot_path) as stream:
            assert_pilot_frozen(
                json.load(stream),
                config,
                _experiment_fingerprint(
                    config,
                    args.output_dir,
                    require_environment_lock=config.scientific_protocol,
                ),
            )
        records, csv_path, manifest_path = run_cases(
            config,
            args.output_dir,
            splits=_SPLITS[1:],
            runner=runner,
            workers=args.workers,
        )
        print(f"experiment dataset: {len(records)} rows -> {csv_path}; {manifest_path}")
        return 0

    if args.action == "train":
        train_surrogates(
            read_campaign_dataset(dataset_path, config),
            args.output_dir,
            config.seed,
            config,
        )
        print(f"experiment train: models -> {args.output_dir}")
        return 0

    if args.action == "evaluate":
        records = read_campaign_dataset(dataset_path, config)
        training = assert_training_provenance(args.output_dir, config)
        surrogate = RiskCalibratedSurrogate.load(os.path.join(args.output_dir, "surrogate.pkl"))
        black_box = RiskCalibratedSurrogate.load(os.path.join(args.output_dir, "black_box.pkl"))
        ablation_models = load_ablation_models(args.output_dir)
        payload = {
            "provenance": {
                "training_fingerprint": training["training_fingerprint"],
            },
            "methods": {
                split: {
                    name: evaluate_records(
                        records, surrogate, black_box, split, robot
                    )
                    for name, robot in (
                        ("pooled", None), ("talos", "talos"), ("icub", "icub")
                    )
                }
                for split in ("test", "ood")
            },
            "ablations": {
                split: {
                    name: evaluate_ablations(
                        records, surrogate, ablation_models, split, robot
                    )
                    for name, robot in (
                        ("pooled", None), ("talos", "talos"), ("icub", "icub")
                    )
                }
                for split in ("test", "ood")
            }
        }
        path = _write_json(os.path.join(args.output_dir, "evaluation.json"), payload)
        print(f"experiment evaluate: {path}")
        return 0

    if args.action == "screen":
        training = assert_training_provenance(args.output_dir, config)
        evaluation = _assert_evaluation_provenance(
            args.output_dir, str(training["training_fingerprint"])
        )
        validity = _assert_validity_provenance(args.output_dir, config)
        surrogate = RiskCalibratedSurrogate.load(
            os.path.join(args.output_dir, "surrogate.pkl")
        )
        black_box = RiskCalibratedSurrogate.load(
            os.path.join(args.output_dir, "black_box.pkl")
        )
        rows, csv_path, summary_path = run_screening_campaign(
            config,
            surrogate,
            black_box,
            args.output_dir,
            workers=args.workers,
        )
        gate = readiness_gate(
            evaluation,
            summarize_screening(rows),
            validity_passed=bool(validity.get("passed")),
        )
        gate["provenance"] = {
            "experiment_fingerprint": training["experiment_fingerprint"],
            "training_fingerprint": training["training_fingerprint"],
            "screening_fingerprint": _screening_fingerprint(
                config, args.output_dir, args.workers
            ),
            "evaluation_sha256": _file_sha256(os.path.join(
                args.output_dir, "evaluation.json"
            )),
            "validity_sha256": _file_sha256(os.path.join(
                args.output_dir, "validity.json"
            )),
            "screening_csv_sha256": _file_sha256(csv_path),
            "screening_calls_sha256": _file_sha256(os.path.join(
                args.output_dir, "screening_calls.csv"
            )),
            "screening_summary_sha256": _file_sha256(summary_path),
            "screening_workers": args.workers,
        }
        gate_path = _write_json(
            os.path.join(args.output_dir, "readiness.json"), gate
        )
        verify_campaign_provenance(args.output_dir, config)
        print(f"experiment screen: {csv_path}; {summary_path}; {gate_path}")
        return 0

    config = ProtocolConfig.smoke(config.seed)
    records, _, _ = run_cases(
        config, args.output_dir, runner=runner, workers=args.workers
    )
    surrogate, black_box, _ = train_surrogates(
        records, args.output_dir, config.seed
    )
    ablation_models = load_ablation_models(args.output_dir)
    evaluation = {
        "methods": {
            split: evaluate_records(records, surrogate, black_box, split)
            for split in ("test", "ood")
        },
        "ablations": {
            split: evaluate_ablations(
                records, surrogate, ablation_models, split
            )
            for split in ("test", "ood")
        },
    }
    _write_json(os.path.join(args.output_dir, "evaluation.json"), evaluation)

    test_cases = [case for case in generate_cases(config) if case.split == "test"]
    test_arrays = records_to_arrays(records, "test")
    order = np.argsort(surrogate.predict_failure_score(test_arrays.X), kind="stable")
    specs = {}

    def final_oracle(case):
        if case.robot not in specs:
            specs[case.robot] = load_robot_spec(case.robot)
        spec = specs[case.robot]
        trajectory = build_whole_body_trajectory(
            spec, case.sample, steps=config.steps, dt=config.dt
        )
        return rollout(spec, trajectory, case.sample)

    screening = verify_ranked_candidates(
        surrogate,
        test_arrays.X[order],
        [test_cases[index] for index in order],
        config.rollout_budgets[0],
        final_oracle,
    )
    _write_json(os.path.join(args.output_dir, "smoke.json"), screening)
    print(f"experiment smoke: {len(records)} rows; verified success={screening['success']}")
    return 0


def _clopper_pearson_upper(failures: int, accepted: int, alpha: float) -> float:
    if accepted == 0 or failures == accepted:
        return 1.0
    return float(beta.ppf(1.0 - alpha, failures + 1, accepted - failures))


def calibration_metrics(
    labels: np.ndarray,
    failure_scores: np.ndarray,
    threshold: float | np.ndarray | None,
    alpha: float | None = 0.05,
) -> dict[str, float | int | bool | None]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(failure_scores, dtype=float)
    if labels.shape != scores.shape:
        raise ValueError("labels and scores must have matching rows")
    accepted_mask = _acceptance_mask(scores, threshold)
    accepted = int(np.sum(accepted_mask))
    failures = int(np.sum(labels[accepted_mask]))
    return {
        "accepted": accepted,
        "coverage": accepted / len(labels) if len(labels) else 0.0,
        "false_safe_count": failures,
        "false_safe_risk": failures / accepted if accepted else 0.0,
        "false_safe_upper": (
            _clopper_pearson_upper(failures, accepted, alpha)
            if alpha is not None else None
        ),
        "confidence_bound_valid": alpha is not None,
    }


def classification_metrics(
    labels: np.ndarray, failure_scores: np.ndarray, bins: int = 10
) -> dict[str, float | None]:
    """Return probability-quality metrics for failure as the positive class."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(failure_scores, dtype=float)
    if labels.ndim != 1 or scores.shape != labels.shape or not len(labels):
        raise ValueError("labels and scores must be non-empty matching vectors")
    if not np.isin(labels, (0, 1)).all() or not np.isfinite(scores).all():
        raise ValueError("labels must be binary and scores finite")
    scores = np.clip(scores, 0.0, 1.0)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (scores >= lower) & (
            scores <= upper if index == bins - 1 else scores < upper
        )
        if np.any(mask):
            ece += float(np.mean(mask)) * abs(
                float(np.mean(scores[mask])) - float(np.mean(labels[mask]))
            )
    return {
        "pr_auc": float(average_precision_score(labels, scores)),
        "roc_auc": (
            float(roc_auc_score(labels, scores))
            if len(np.unique(labels)) == 2 else None
        ),
        "brier": float(brier_score_loss(labels, scores)),
        "ece": ece,
    }


def evaluate_score_methods(
    labels: np.ndarray,
    method_scores: dict[str, np.ndarray],
    thresholds: dict[str, float | np.ndarray | None],
    runtimes: np.ndarray,
    failure_reasons: np.ndarray,
    alpha: float | None = 0.05,
) -> dict[str, dict[str, object]]:
    """Evaluate selective risk and predictive quality for named score methods."""
    labels = np.asarray(labels, dtype=int)
    runtimes = np.asarray(runtimes, dtype=float)
    reasons = np.asarray(failure_reasons, dtype=str)
    if runtimes.shape != labels.shape or reasons.shape != labels.shape:
        raise ValueError("runtime and failure reason rows must match labels")
    breakdown = {
        reason: int(np.sum(reasons == reason))
        for reason in sorted(set(reasons) - {""})
    }
    result = {}
    for name, scores in method_scores.items():
        if name not in thresholds:
            raise ValueError(f"missing threshold for {name}")
        scores = np.asarray(scores, dtype=float)
        predictive = classification_metrics(labels, scores)
        if name in _HEURISTIC_METHODS:
            # Margin-to-score maps preserve ranking and the zero-margin
            # threshold, but are not fitted probabilities.
            predictive["brier"] = None
            predictive["ece"] = None
        result[name] = {
            **predictive,
            **calibration_metrics(labels, scores, thresholds[name], alpha),
            "rollout_runtime_seconds": float(np.sum(runtimes)),
            "failure_breakdown": breakdown,
        }
    return result


def verify_ranked_candidates(
    surrogate,
    features: np.ndarray,
    candidates: Sequence[object],
    budget: int,
    oracle: Callable[[object], RolloutResult],
) -> dict[str, object]:
    """Spend rollout budget only on accepted candidates and never infer success."""
    if budget < 1:
        raise ValueError("budget must be positive")
    accepted = np.asarray(surrogate.accept(features), dtype=bool)
    selected = np.flatnonzero(accepted)[:budget]
    if not len(selected):
        return {
            "success": False,
            "verified_by_rollout": False,
            "rollouts": 0,
            "failure_reason": "no_accepted_candidate",
        }
    last_reason = ""
    for rollout_count, index in enumerate(selected, start=1):
        result = oracle(candidates[int(index)])
        if result.success:
            return {
                "success": True,
                "verified_by_rollout": True,
                "rollouts": rollout_count,
                "candidate_index": int(index),
                "failure_reason": "",
            }
        last_reason = result.failure_reason
    return {
        "success": False,
        "verified_by_rollout": True,
        "rollouts": len(selected),
        "failure_reason": last_reason,
    }


def screening_experiment(
    method_scores: dict[str, np.ndarray],
    candidates: Sequence[object],
    budgets: Sequence[int],
    oracle: Callable[[object], RolloutResult],
    accepted_masks: dict[str, np.ndarray] | None = None,
    call_log: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    """Run paired budget comparisons over one shared candidate pool."""
    rows = []
    for method, scores in method_scores.items():
        scores = np.asarray(scores, dtype=float)
        if scores.shape != (len(candidates),) or not np.isfinite(scores).all():
            raise ValueError(f"invalid scores for {method}")
        accepted = (
            np.ones(len(candidates), dtype=bool)
            if accepted_masks is None or method not in accepted_masks
            else np.asarray(accepted_masks[method], dtype=bool)
        )
        if accepted.shape != (len(candidates),):
            raise ValueError(f"invalid acceptance mask for {method}")
        order = [index for index in np.argsort(scores, kind="stable") if accepted[index]]
        for budget in budgets:
            if budget < 1:
                raise ValueError("budgets must be positive")
            success = False
            runtime = 0.0
            oracle_wall = 0.0
            actual_oracle_wall = 0.0
            used = 0
            reason = "no_accepted_candidate" if not order else ""
            for rank, index in enumerate(order[:budget], start=1):
                result = oracle(candidates[int(index)])
                counterfactual_wall = float(getattr(
                    result, "oracle_wall_seconds", result.runtime_seconds
                ))
                actual_wall = float(getattr(
                    result, "actual_oracle_wall_seconds", counterfactual_wall
                ))
                used += 1
                runtime += result.runtime_seconds
                oracle_wall += counterfactual_wall
                actual_oracle_wall += actual_wall
                reason = result.failure_reason
                if call_log is not None:
                    call_log.append({
                        "method": method,
                        "budget": int(budget),
                        "rank": rank,
                        "candidate_index": int(index),
                        "candidate_id": str(getattr(
                            candidates[int(index)], "base_gait_id", index
                        )),
                        "score": float(scores[int(index)]),
                        "success": bool(result.success),
                        "failure_reason": str(result.failure_reason),
                        "rollout_runtime_seconds": float(result.runtime_seconds),
                        "oracle_wall_seconds": counterfactual_wall,
                        "actual_oracle_wall_seconds": actual_wall,
                        "cache_hit": bool(getattr(result, "cache_hit", False)),
                    })
                if result.success:
                    success, reason = True, ""
                    break
            rows.append({
                "method": method,
                "budget": int(budget),
                "success": success,
                "rollouts": used,
                "rollout_runtime_seconds": runtime,
                "oracle_wall_seconds": oracle_wall,
                "actual_oracle_wall_seconds": actual_oracle_wall,
                "failure_reason": reason,
            })
    return rows


def _screening_fingerprint(
    config: ProtocolConfig, output_directory: str, workers: int = 1
) -> str:
    training_path = os.path.join(output_directory, "training.json")
    if not os.path.exists(training_path):
        if config.scientific_protocol:
            raise RuntimeError("scientific screening requires trained models")
        return _experiment_fingerprint(config, output_directory)
    training = assert_training_provenance(output_directory, config)
    return hashlib.sha256(json.dumps({
        "training_fingerprint": training["training_fingerprint"],
        "workers": workers,
    }, sort_keys=True).encode()).hexdigest()


def _read_screening_cell(
    path: str,
    fingerprint: str,
    robot: str,
    condition: str,
    repetition: int,
    pool_seed: int,
    config: ProtocolConfig,
) -> tuple[list[dict[str, object]], list[dict[str, object]]] | None:
    if not os.path.exists(path):
        return None
    with open(path) as stream:
        payload = json.load(stream)
    expected = {
        "fingerprint": fingerprint,
        "robot": robot,
        "condition": condition,
        "repetition": repetition,
        "pool_seed": pool_seed,
        "pool_size": config.candidate_pool_size,
    }
    if any(payload.get(name) != value for name, value in expected.items()):
        raise RuntimeError(f"screening checkpoint does not match this run: {path}")
    rows = payload.get("rows")
    calls = payload.get("calls")
    expected_count = len(_SCREENING_METHODS) * len(config.rollout_budgets)
    if (
        not isinstance(rows, list)
        or len(rows) != expected_count
        or {str(row.get("method")) for row in rows} != set(_SCREENING_METHODS)
        or {int(row.get("budget", -1)) for row in rows}
        != set(config.rollout_budgets)
        or not isinstance(calls, list)
    ):
        raise RuntimeError(f"screening checkpoint is incomplete: {path}")
    return rows, calls


def _write_screening_cell(
    path: str,
    fingerprint: str,
    robot: str,
    condition: str,
    repetition: int,
    pool_seed: int,
    pool_size: int,
    rows: Sequence[dict[str, object]],
    calls: Sequence[dict[str, object]],
) -> None:
    temporary = f"{path}.tmp"
    _write_json(temporary, {
        "fingerprint": fingerprint,
        "robot": robot,
        "condition": condition,
        "repetition": repetition,
        "pool_seed": pool_seed,
        "pool_size": pool_size,
        "rows": list(rows),
        "calls": list(calls),
    })
    os.replace(temporary, path)


def run_screening_campaign(
    config: ProtocolConfig,
    surrogate: RiskCalibratedSurrogate,
    black_box: RiskCalibratedSurrogate,
    output_directory: str,
    *,
    feature_runner: Callable[[ExperimentCase, ProtocolConfig], dict[str, object]]
    | None = None,
    oracle: Callable[[ExperimentCase], RolloutResult] | None = None,
    workers: int = 1,
) -> tuple[list[dict[str, object]], str, str]:
    """Run all paired robot/condition/repetition downstream screening cells."""
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if config.scientific_protocol and (
        feature_runner is not None or oracle is not None
    ):
        raise RuntimeError("scientific screening forbids injected callbacks")
    os.makedirs(output_directory, exist_ok=True)
    cell_directory = os.path.join(output_directory, "screening_cells")
    os.makedirs(cell_directory, exist_ok=True)
    fingerprint = _screening_fingerprint(config, output_directory, workers)
    worker_state = threading.local()

    def get_spec(robot):
        if not hasattr(worker_state, "specs"):
            worker_state.specs = {}
        if robot not in worker_state.specs:
            worker_state.specs[robot] = load_robot_spec(robot)
        return worker_state.specs[robot]

    if feature_runner is None:
        def feature_runner(case, protocol):
            spec = get_spec(case.robot)
            started = perf_counter()
            trajectory = build_whole_body_trajectory(
                spec, case.sample, steps=protocol.steps, dt=protocol.dt
            )
            trajectory_done = perf_counter()
            signature = compute_physics_signature(spec, trajectory, case.sample)
            signature_done = perf_counter()
            placeholder = RolloutResult(
                np.empty(0),
                np.empty((0, spec.model.nq)),
                np.empty((0, spec.model.nv)),
                np.empty((0, 2, 6)),
                True,
                "",
            )
            record = flatten_record(case, signature, placeholder)
            record["_candidate_trajectory_ik_seconds"] = (
                trajectory_done - started
            )
            record["_signature_seconds"] = signature_done - trajectory_done
            return record

    if oracle is None:
        def oracle(case):
            spec = get_spec(case.robot)
            trajectory = build_whole_body_trajectory(
                spec, case.sample, steps=config.steps, dt=config.dt
            )
            return rollout(spec, trajectory, case.sample)

    cells = []
    pending = []
    completed = {}
    for repetition in range(config.screening_repetitions):
        for condition_index, (condition, ood) in enumerate(
            (("id", False), ("ood", True))
        ):
            pool_seed = int(np.random.SeedSequence([
                config.seed, repetition, condition_index, 991
            ]).generate_state(1, dtype=np.uint32)[0])
            for robot in config.robots:
                cell_path = os.path.join(
                    cell_directory,
                    f"{repetition:03d}_{condition}_{robot}.json",
                )
                cell = (
                    robot,
                    condition,
                    ood,
                    repetition,
                    pool_seed,
                    cell_path,
                )
                cells.append(cell)
                checkpoint = _read_screening_cell(
                    cell_path,
                    fingerprint,
                    robot,
                    condition,
                    repetition,
                    pool_seed,
                    config,
                )
                if checkpoint is not None:
                    completed[cell_path] = checkpoint
                    continue
                pending.append(cell)

    def compute_cell(cell):
        robot, condition, ood, repetition, pool_seed, _ = cell
        candidates = generate_candidate_pool(
            robot,
            pool_seed,
            config.candidate_pool_size,
            ood,
            config,
        )
        feature_started = perf_counter()
        records = [feature_runner(case, config) for case in candidates]
        feature_seconds = perf_counter() - feature_started
        trajectory_ik_seconds = sum(
            float(record.pop("_candidate_trajectory_ik_seconds", 0.0))
            for record in records
        )
        signature_seconds = sum(
            float(record.pop("_signature_seconds", 0.0))
            for record in records
        )
        inference_started = perf_counter()
        arrays = records_to_arrays(records)
        scores = _score_arrays(arrays, surrogate, black_box)
        accepted = surrogate.accept(arrays.X)
        inference_seconds = perf_counter() - inference_started
        cache = {}

        def cached_oracle(case):
            if case.base_gait_id not in cache:
                started = perf_counter()
                result = oracle(case)
                wall_seconds = perf_counter() - started
                cache[case.base_gait_id] = _ScreeningOutcome(
                    bool(result.success),
                    str(result.failure_reason),
                    float(result.runtime_seconds),
                    wall_seconds,
                    wall_seconds,
                )
                return cache[case.base_gait_id]
            return replace(
                cache[case.base_gait_id],
                actual_oracle_wall_seconds=0.0,
                cache_hit=True,
            )

        call_rows = []
        cell_rows = screening_experiment(
            scores,
            candidates,
            config.rollout_budgets,
            cached_oracle,
            accepted_masks={
                "risk_calibrated_phase_sequence": accepted
            },
            call_log=call_rows,
        )
        for row in cell_rows:
            row.update({
                "robot": robot,
                "condition": condition,
                "repetition": repetition,
                "pool_seed": pool_seed,
                "pool_size": len(candidates),
                "feature_runtime_seconds": feature_seconds,
                "candidate_trajectory_ik_seconds": trajectory_ik_seconds,
                "signature_runtime_seconds": signature_seconds,
                "inference_runtime_seconds": inference_seconds,
            })
        for call in call_rows:
            call.update({
                "robot": robot,
                "condition": condition,
                "repetition": repetition,
                "pool_seed": pool_seed,
                "feature_runtime_seconds": feature_seconds,
                "candidate_trajectory_ik_seconds": trajectory_ik_seconds,
                "signature_runtime_seconds": signature_seconds,
                "inference_runtime_seconds": inference_seconds,
            })
        return cell_rows, call_rows

    def persist(cell, payload):
        robot, condition, _, repetition, pool_seed, cell_path = cell
        cell_rows, call_rows = payload
        completed[cell_path] = payload
        _write_screening_cell(
            cell_path,
            fingerprint,
            robot,
            condition,
            repetition,
            pool_seed,
            config.candidate_pool_size,
            cell_rows,
            call_rows,
        )

    if workers == 1:
        for cell in pending:
            persist(cell, compute_cell(cell))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(compute_cell, cell): cell for cell in pending
            }
            for future in as_completed(futures):
                persist(futures[future], future.result())

    rows = [
        row
        for cell in cells
        for row in completed[cell[-1]][0]
    ]
    calls = [
        call
        for cell in cells
        for call in completed[cell[-1]][1]
    ]

    csv_path = os.path.join(output_directory, "screening.csv")
    with open(csv_path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    calls_path = os.path.join(output_directory, "screening_calls.csv")
    call_fields = [
        "method", "budget", "rank", "candidate_index", "candidate_id", "score",
        "success", "failure_reason", "rollout_runtime_seconds",
        "oracle_wall_seconds", "actual_oracle_wall_seconds", "cache_hit",
        "robot", "condition",
        "repetition", "pool_seed", "feature_runtime_seconds",
        "candidate_trajectory_ik_seconds", "signature_runtime_seconds",
        "inference_runtime_seconds",
    ]
    with open(calls_path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=call_fields)
        writer.writeheader()
        writer.writerows(calls)
    summary_path = _write_json(
        os.path.join(output_directory, "screening_summary.json"),
        summarize_screening(rows),
    )
    return rows, csv_path, summary_path


if __name__ == "__main__":
    raise SystemExit(cli())
