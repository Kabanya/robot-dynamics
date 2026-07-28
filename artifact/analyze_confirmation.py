"""Analyze the frozen ranking confirmation without refitting on its labels."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from src.experiment import (
    _physics_score,
    ablation_arrays,
    read_dataset,
    records_to_arrays,
)


METHODS = {
    "B1_zmp_cop": (".zmp.", ".cop."),
    "B2_ik_joint": (".ik_residual.", ".joint_position."),
    "B3_dynamics_slack": (".dynamics_slack.",),
}
BUDGETS = (5, 10, 20)
ROBOTS = ("talos", "icub")
FIGURES = (
    "figure2_domain_gate.pdf",
    "figure3_heldout_ranking.pdf",
)


def _fit(features: np.ndarray, labels: np.ndarray):
    return HistGradientBoostingClassifier(
        max_leaf_nodes=7,
        learning_rate=0.08,
        max_iter=150,
        l2_regularization=1.0,
        random_state=2026,
    ).fit(features, labels)


def _metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "pr_auc": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
    }


def _with_robot_id(features: np.ndarray, robots: np.ndarray) -> np.ndarray:
    return np.column_stack((features, (robots == "icub").astype(float)))


def _macro_pr_auc(
    labels: np.ndarray, robots: np.ndarray, scores: np.ndarray
) -> float:
    return float(np.mean([
        average_precision_score(labels[robots == robot], scores[robots == robot])
        for robot in ROBOTS
    ]))


def _bootstrap_primary(
    labels: np.ndarray,
    robots: np.ndarray,
    confirmations: np.ndarray,
    proposed: np.ndarray,
    comparator: np.ndarray,
    seed: int,
    repetitions: int,
) -> dict[str, object]:
    if repetitions < 1:
        raise ValueError("bootstrap repetitions must be positive")
    rng = np.random.default_rng(seed)
    confirmation_ids = tuple(np.unique(confirmations))
    members = {
        (confirmation, robot): np.flatnonzero(
            (confirmations == confirmation) & (robots == robot)
        )
        for confirmation in confirmation_ids
        for robot in ROBOTS
    }
    differences = np.empty(repetitions)
    for repetition in range(repetitions):
        indices = np.concatenate([
            rng.choice(indices, size=len(indices), replace=True)
            for indices in members.values()
        ])
        differences[repetition] = _macro_pr_auc(
            labels[indices], robots[indices], proposed[indices]
        ) - _macro_pr_auc(
            labels[indices], robots[indices], comparator[indices]
        )
    observed = _macro_pr_auc(labels, robots, proposed) - _macro_pr_auc(
        labels, robots, comparator
    )
    by_confirmation = {
        str(confirmation): (
            _macro_pr_auc(
                labels[confirmations == confirmation],
                robots[confirmations == confirmation],
                proposed[confirmations == confirmation],
            )
            - _macro_pr_auc(
                labels[confirmations == confirmation],
                robots[confirmations == confirmation],
                comparator[confirmations == confirmation],
            )
        )
        for confirmation in confirmation_ids
    }
    interval = np.percentile(differences, (2.5, 97.5))
    return {
        "metric": "macro_within_robot_pr_auc_difference",
        "proposed": "B5_whole_body",
        "comparator": "B4_parameters_robot",
        "delta_pr_auc": float(observed),
        "delta_by_confirmation": by_confirmation,
        "ci95": [float(interval[0]), float(interval[1])],
        "bootstrap_seed": int(seed),
        "bootstrap_repetitions": int(repetitions),
        "bootstrap_interpretation": "conditional_descriptive",
        "passed": bool(
            interval[0] > 0.0
            and all(delta > 0.0 for delta in by_confirmation.values())
        ),
    }


def _success_at_budget(
    labels: np.ndarray,
    robots: np.ndarray,
    scores: dict[str, np.ndarray],
    seed: int,
    repetitions: int,
) -> dict[str, object]:
    result = {}
    for robot_index, robot in enumerate(ROBOTS):
        members = np.flatnonzero(robots == robot)
        rng = np.random.default_rng([seed, robot_index, 7919])
        successes = {
            method: {budget: [] for budget in BUDGETS}
            for method in ("B4_parameters_robot", "B5_whole_body")
        }
        first_success_ranks = {
            method: [] for method in ("B4_parameters_robot", "B5_whole_body")
        }
        for _ in range(repetitions):
            pool = rng.choice(members, size=50, replace=False)
            for method in successes:
                order = pool[np.argsort(scores[method][pool], kind="stable")]
                ordered_labels = labels[order]
                successful = ordered_labels == 0
                ranks = np.flatnonzero(successful)
                first_success_ranks[method].append(
                    int(ranks[0] + 1) if len(ranks) else 51
                )
                for budget in BUDGETS:
                    successes[method][budget].append(
                        bool(np.any(successful[:budget]))
                    )
        robot_result = {}
        for method in successes:
            robot_result[method] = {
                "success_rate": {
                    str(budget): float(np.mean(values))
                    for budget, values in successes[method].items()
                },
                "mean_rank_to_first_success": float(np.mean(
                    first_success_ranks[method]
                )),
            }
        robot_result["B5_minus_B4_success_rate"] = {
            str(budget): float(
                np.mean(successes["B5_whole_body"][budget])
                - np.mean(successes["B4_parameters_robot"][budget])
            )
            for budget in BUDGETS
        }
        result[robot] = robot_result
    return result


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _save_figure(figure, path: Path) -> None:
    figure.savefig(
        path,
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None, "Creator": "robot-dynamics"},
    )
    figure.savefig(
        path.with_suffix(".png"),
        dpi=180,
        bbox_inches="tight",
        metadata={"Software": "robot-dynamics"},
    )
    plt.close(figure)


def _label_failure_bars(axis, bars, robot):
    annotations = axis.bar_label(
        bars,
        labels=[
            (
                f"{'TALOS' if robot == 'talos' else 'iCub'} "
                f"{int(bar.get_width())}"
                if index == 0
                else str(int(bar.get_width())) if bar.get_width() else ""
            )
            for index, bar in enumerate(bars)
        ],
        padding=2,
        fontsize=7,
    )
    if robot == "icub" and annotations:
        x, y = annotations[0].get_position()
        annotations[0].set_position((x, y - 3))
    return annotations


def generate_confirmation_figures(
    result: dict[str, object], output_dir
) -> list[Path]:
    """Generate the two manuscript figures from one frozen analysis JSON."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 8,
        "figure.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    methods = (
        "B1_zmp_cop",
        "B2_ik_joint",
        "B3_dynamics_slack",
        "B4_parameters_robot",
        "B5_whole_body",
    )
    labels = ("B1", "B2", "B3", "B4", "B5")
    values = [
        result["heldout_ranking"]["pooled"][method]["pr_auc"]
        for method in methods
    ]
    figure, axis = plt.subplots(figsize=(3.45, 2.25))
    axis.bar(labels, values, color=("#999999",) * 3 + ("#0072B2", "#D55E00"))
    axis.set(ylim=(0, 1), ylabel="Failure PR-AUC")
    delta = result["primary"]["delta_pr_auc"]
    lower, upper = result["primary"]["ci95"]
    axis.set_title(
        f"Macro B5 - B4 = {delta:+.3f}  "
        f"(conditional 95% CI {lower:+.3f}, {upper:+.3f})"
    )
    axis.grid(axis="y", alpha=0.2)
    _save_figure(figure, output / FIGURES[0])

    blue = "#0072B2"
    orange = "#D55E00"
    figure, axes = plt.subplots(2, 2, figsize=(7.0, 4.0))
    figure.subplots_adjust(
        left=0.09, right=0.98, bottom=0.10, top=0.94, wspace=0.30, hspace=0.42
    )

    axis = axes[0, 0]
    x = np.arange(len(ROBOTS))
    for offset, method, color, hatch in (
        (-0.18, "B4_parameters_robot", blue, "//"),
        (0.18, "B5_whole_body", orange, None),
    ):
        bars = axis.bar(
            x + offset,
            [
                result["heldout_ranking"][robot][method]["pr_auc"]
                for robot in ROBOTS
            ],
            width=0.36,
            label=method.split("_", 1)[0],
            color=color,
            edgecolor="#333333",
            linewidth=0.5,
            hatch=hatch,
        )
        axis.bar_label(bars, fmt="%.3f", padding=2, fontsize=7)
    axis.set(
        xticks=x,
        xticklabels=("TALOS", "iCub"),
        ylabel="Failure PR-AUC",
        ylim=(0, 0.8),
        title="(a) Per-robot failure ranking",
    )
    axis.legend(frameon=False, fontsize=7, loc="upper left")

    axis = axes[0, 1]
    failure_composition = result["failure_composition"]
    reasons = sorted(
        set().union(*(failure_composition[robot] for robot in ROBOTS)),
        key=lambda reason: (
            -sum(failure_composition[robot].get(reason, 0) for robot in ROBOTS),
            reason,
        ),
    )
    y = np.arange(len(reasons))
    for offset, robot, color, hatch in (
        (-0.18, "talos", blue, "//"),
        (0.18, "icub", orange, None),
    ):
        bars = axis.barh(
            y + offset,
            [failure_composition[robot].get(reason, 0) for reason in reasons],
            height=0.36,
            label="TALOS" if robot == "talos" else "iCub",
            color=color,
            edgecolor="#333333",
            linewidth=0.5,
            hatch=hatch,
        )
        _label_failure_bars(axis, bars, robot)
    axis.set(
        yticks=y,
        yticklabels=[
            reason.replace("_", " ").replace("dynamics", "dyn.")
            for reason in reasons
        ],
        xlabel="First failures (count)",
        title="(b) Failure composition",
    )
    axis.invert_yaxis()

    axis = axes[1, 0]
    metrics = ("pr_auc", "roc_auc")
    x = np.arange(len(metrics))
    pooled_ablation = result["representation_ablation"]["pooled"]
    for offset, representation, color, marker, label in (
        (-0.14, "phase_resolved", blue, "o", "Phase-resolved"),
        (0.0, "phase_agnostic", orange, "s", "Phase-agnostic"),
        (0.14, "no_touchdown", "#777777", "^", "No touchdown"),
    ):
        values = [pooled_ablation[representation][metric] for metric in metrics]
        axis.scatter(
            x + offset,
            values,
            label=label,
            marker=marker,
            s=34,
            facecolor=color,
            edgecolor=color,
            linewidth=1.2,
            zorder=2,
        )
    axis.set(
        xticks=x,
        xticklabels=("PR-AUC", "ROC-AUC"),
        ylabel="Score",
        xlim=(-0.45, 1.45),
        ylim=(0, 1),
        title="(c) Pooled representation ablation",
    )
    axis.legend(frameon=False, fontsize=7, loc="lower left")

    axis = axes[1, 1]
    x = np.arange(len(ROBOTS))
    for offset, method, color, hatch in (
        (-0.18, "B4_parameters_robot", blue, "//"),
        (0.18, "B5_whole_body", orange, None),
    ):
        bars = axis.bar(
            x + offset,
            [
                result["success_at_budget"][robot][method][
                    "mean_rank_to_first_success"
                ]
                for robot in ROBOTS
            ],
            width=0.36,
            label=method.split("_", 1)[0],
            color=color,
            edgecolor="#333333",
            linewidth=0.5,
            hatch=hatch,
        )
        axis.bar_label(bars, fmt="%.3f", padding=2, fontsize=7)
    axis.set(
        xticks=x,
        xticklabels=("TALOS", "iCub"),
        ylabel="Mean rank to first feasible gait",
        ylim=(0, 1.6),
        title="(d) Rollout-budget diagnostic",
    )
    axis.legend(frameon=False, fontsize=7, loc="upper left")

    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
        axis.title.set_fontsize(8)
        axis.xaxis.label.set_fontsize(8)
        axis.yaxis.label.set_fontsize(8)
        axis.tick_params(labelsize=7)
        axis.spines[["top", "right"]].set_visible(False)
    _save_figure(figure, output / FIGURES[1])
    return [output / name for name in FIGURES]


def analyze(
    development_csvs,
    confirmation_csvs,
    bootstrap_seed=62026,
    repetitions=10000,
) -> dict[str, object]:
    development_paths = [Path(path) for path in development_csvs]
    confirmation_paths = (
        [Path(confirmation_csvs)]
        if isinstance(confirmation_csvs, (str, Path))
        else [Path(path) for path in confirmation_csvs]
    )
    if len(development_paths) != 2:
        raise ValueError("analysis requires exactly two development CSVs")
    if not confirmation_paths:
        raise ValueError("analysis requires at least one confirmation CSV")
    development_parts = [read_dataset(str(path)) for path in development_paths]
    if any(len(records) != 100 for records in development_parts):
        raise ValueError("each development CSV must contain exactly 100 rows")
    development_records = [
        record for records in development_parts for record in records
    ]
    confirmation_parts = [
        read_dataset(str(path)) for path in confirmation_paths
    ]
    confirmation_names = []
    for index, (path, records) in enumerate(
        zip(confirmation_paths, confirmation_parts)
    ):
        counts = Counter(str(record["robot"]) for record in records)
        if counts != Counter({"talos": 200, "icub": 200}):
            raise ValueError("analysis requires 200 confirmation rows per robot")
        protocol_path = path.with_name("protocol.json")
        if protocol_path.is_file():
            with protocol_path.open() as stream:
                confirmation_names.append(f"seed{json.load(stream)['seed']}")
        else:
            confirmation_names.append(f"set{index + 1}")
    if len(set(confirmation_names)) != len(confirmation_names):
        raise ValueError("confirmation identifiers must be unique")
    confirmation_records = [
        record for records in confirmation_parts for record in records
    ]
    confirmations = np.concatenate([
        np.full(len(records), name, dtype=object)
        for name, records in zip(confirmation_names, confirmation_parts)
    ])
    confirmation_counts = Counter(
        str(record["robot"]) for record in confirmation_records
    )
    development_counts = Counter(
        str(record["robot"]) for record in development_records
    )
    if development_counts != Counter({"talos": 100, "icub": 100}):
        raise ValueError("analysis requires 100 development rows per robot")
    development_seeds = {
        int(record["seed"]) for record in development_records
    }
    confirmation_seeds = {
        int(record["seed"]) for record in confirmation_records
    }
    if development_seeds & confirmation_seeds:
        raise ValueError("development and confirmation share rollout seeds")
    if sum(len({
        int(record["seed"]) for record in records
    }) for records in confirmation_parts) != len(confirmation_seeds):
        raise ValueError("confirmation scrambles share rollout seeds")

    development = records_to_arrays(development_records, "pilot")
    confirmation = records_to_arrays(confirmation_records, "pilot")
    phase_agnostic_development = ablation_arrays(
        development_records, "pilot", "no_phase_separation"
    )
    phase_agnostic_confirmation = ablation_arrays(
        confirmation_records, "pilot", "no_phase_separation"
    )
    no_touchdown_development = ablation_arrays(
        development_records, "pilot", "no_touchdown"
    )
    no_touchdown_confirmation = ablation_arrays(
        confirmation_records, "pilot", "no_touchdown"
    )
    for robot in ROBOTS:
        labels = confirmation.labels[confirmation.robots == robot]
        if set(labels) != {0, 1}:
            raise ValueError(f"confirmation requires both labels for {robot}")

    parameter_model = _fit(
        _with_robot_id(development.X_parameters, development.robots),
        development.labels,
    )
    whole_body_model = _fit(
        _with_robot_id(
            phase_agnostic_development.X,
            phase_agnostic_development.robots,
        ),
        phase_agnostic_development.labels,
    )
    phase_resolved_model = _fit(
        _with_robot_id(development.X, development.robots),
        development.labels,
    )
    no_touchdown_model = _fit(
        _with_robot_id(no_touchdown_development.X, no_touchdown_development.robots),
        no_touchdown_development.labels,
    )
    scores = {
        method: _physics_score(confirmation, tokens)
        for method, tokens in METHODS.items()
    }
    scores.update({
        "B4_parameters_robot": parameter_model.predict_proba(
            _with_robot_id(confirmation.X_parameters, confirmation.robots)
        )[:, 1],
        "B5_whole_body": whole_body_model.predict_proba(
            _with_robot_id(
                phase_agnostic_confirmation.X,
                phase_agnostic_confirmation.robots,
            )
        )[:, 1],
    })
    phase_resolved_scores = phase_resolved_model.predict_proba(
        _with_robot_id(confirmation.X, confirmation.robots)
    )[:, 1]
    no_touchdown_scores = no_touchdown_model.predict_proba(
        _with_robot_id(no_touchdown_confirmation.X, no_touchdown_confirmation.robots)
    )[:, 1]

    ranking = {}
    for scope, mask in {
        "pooled": np.ones(len(confirmation.labels), dtype=bool),
        "talos": confirmation.robots == "talos",
        "icub": confirmation.robots == "icub",
    }.items():
        ranking[scope] = {
            method: _metrics(
                confirmation.labels[mask], method_scores[mask]
            )
            for method, method_scores in scores.items()
        }
    representation_ablation = {
        scope: {
            "phase_agnostic": _metrics(
                confirmation.labels[mask], scores["B5_whole_body"][mask]
            ),
            "phase_resolved": _metrics(
                confirmation.labels[mask], phase_resolved_scores[mask]
            ),
            "no_touchdown": _metrics(
                confirmation.labels[mask], no_touchdown_scores[mask]
            ),
        }
        for scope, mask in {
            "pooled": np.ones(len(confirmation.labels), dtype=bool),
            "talos": confirmation.robots == "talos",
            "icub": confirmation.robots == "icub",
        }.items()
    }
    failure_composition = {
        robot: dict(sorted(Counter(
            str(record["failure_reason"])
            for record in confirmation_records
            if str(record["robot"]) == robot
            and int(record["label"]) == 1
            and str(record["failure_reason"])
        ).items()))
        for robot in ROBOTS
    }
    return {
        "status": "independent_ranking_confirmation",
        "rows": {
            "development": len(development_records),
            "confirmation": len(confirmation_records),
            "confirmation_by_robot": dict(sorted(confirmation_counts.items())),
            "confirmation_by_scramble": {
                name: len(records)
                for name, records in zip(confirmation_names, confirmation_parts)
            },
        },
        "data_sha256": {
            "development": [_sha256(path) for path in development_paths],
            "confirmation": [_sha256(path) for path in confirmation_paths],
        },
        "class_counts": {
            robot: {
                "success": int(np.sum(
                    confirmation.labels[confirmation.robots == robot] == 0
                )),
                "failure": int(np.sum(
                    confirmation.labels[confirmation.robots == robot] == 1
                )),
            }
            for robot in ROBOTS
        },
        "model": {
            "max_leaf_nodes": 7,
            "learning_rate": 0.08,
            "max_iter": 150,
            "l2_regularization": 1.0,
            "random_state": 2026,
            "robot_id_feature": {
                "B4_parameters_robot": True,
                "B5_whole_body": True,
            },
            "proposed_representation": "normalized_phase_agnostic_whole_body",
        },
        "primary": _bootstrap_primary(
            confirmation.labels,
            confirmation.robots,
            confirmations,
            scores["B5_whole_body"],
            scores["B4_parameters_robot"],
            int(bootstrap_seed),
            int(repetitions),
        ),
        "heldout_ranking": ranking,
        "representation_ablation": representation_ablation,
        "failure_composition": failure_composition,
        "success_at_budget": _success_at_budget(
            confirmation.labels,
            confirmation.robots,
            scores,
            int(bootstrap_seed),
            int(repetitions),
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--development-csv",
        action="append",
        default=None,
    )
    parser.add_argument("--confirmation-dir", action="append")
    parser.add_argument(
        "--output-dir",
        default="results/confirmation",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=62026)
    parser.add_argument("--repetitions", type=int, default=10000)
    parser.add_argument("--figures-dir", default="submission/ijhr/figures")
    args = parser.parse_args(argv)
    development = args.development_csv or [
        "results/development-seed12026/dataset.csv",
        "results/domain-gate-seed22026/dataset.csv",
    ]
    confirmation_directories = [
        Path(path) for path in (
            args.confirmation_dir or [
                "results/confirmation-seed42026",
                "results/confirmation-seed52026",
            ]
        )
    ]
    result = analyze(
        development,
        [directory / "dataset.csv" for directory in confirmation_directories],
        args.bootstrap_seed,
        args.repetitions,
    )
    output_directory = Path(args.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    output = output_directory / "confirmation_analysis.json"
    with output.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    figures = generate_confirmation_figures(result, args.figures_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"confirmation analysis: {output}")
    print("figures:", *(str(path) for path in figures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
