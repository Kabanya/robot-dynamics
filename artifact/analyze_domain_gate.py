"""Analyze the frozen engineering data and independent shared-domain gate."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times"],
    "text.usetex": True,
})

from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import beta, fisher_exact  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from src.experiment import (  # noqa: E402
    _physics_score,
    ablation_arrays,
    read_dataset,
    records_to_arrays,
)


METHODS = {
    "B1": ("ZMP/CoP", (".zmp.", ".cop.")),
    "B2": ("IK/joint", (".ik_residual.", ".joint_position.")),
    "B3": ("dynamics slack", (".dynamics_slack.",)),
}
COLORS = {
    "blue": "#2F6B9A",
    "gold": "#D49A28",
    "orange": "#D26A3A",
    "olive": "#6F7C3E",
    "pink": "#B65A7A",
    "ink": "#243447",
    "grid": "#D8DEE7",
}


def _interval(successes: int, trials: int) -> tuple[float, float]:
    lower = 0.0 if successes == 0 else float(beta.ppf(
        0.025, successes, trials - successes + 1
    ))
    upper = 1.0 if successes == trials else float(beta.ppf(
        0.975, successes + 1, trials - successes
    ))
    return lower, upper


def _fit(features: np.ndarray, labels: np.ndarray) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_leaf_nodes=7,
        learning_rate=0.08,
        max_iter=150,
        l2_regularization=1.0,
        random_state=2026,
    ).fit(features, labels)


def _learned_scores(train, test) -> dict[str, np.ndarray]:
    parameters = _fit(train.X_parameters, train.labels)
    sequence = _fit(train.X, train.labels)
    return {
        "B4": parameters.predict_proba(test.X_parameters)[:, 1],
        "B5": sequence.predict_proba(test.X)[:, 1],
    }


def _ranking_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | None]:
    if len(np.unique(labels)) < 2:
        return {"pr_auc": None, "roc_auc": None}
    return {
        "pr_auc": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
    }


def analyze(training_csv: Path, test_csv: Path, gate_summary: Path) -> dict[str, object]:
    training_records = read_dataset(str(training_csv))
    test_records = read_dataset(str(test_csv))
    if len(training_records) != 100 or len(test_records) != 100:
        raise ValueError("analysis requires exactly 100 training and 100 gate records")
    if {record["seed"] for record in training_records} & {
        record["seed"] for record in test_records
    }:
        raise ValueError("training and gate samples share rollout seeds")

    train = records_to_arrays(training_records, "pilot")
    test = records_to_arrays(test_records, "pilot")
    scores = {
        method: _physics_score(test, tokens)
        for method, (_, tokens) in METHODS.items()
    }
    scores.update(_learned_scores(train, test))

    ranking = {}
    for scope, mask in {
        "pooled": np.ones(len(test.labels), dtype=bool),
        "talos": test.robots == "talos",
        "icub": test.robots == "icub",
    }.items():
        ranking[scope] = {
            method: _ranking_metrics(test.labels[mask], values[mask])
            for method, values in scores.items()
        }
    risk_at_20_percent = {}
    for robot in ("talos", "icub"):
        indices = np.flatnonzero(test.robots == robot)
        risk_at_20_percent[robot] = {}
        for method, values in scores.items():
            accepted = indices[np.argsort(values[indices], kind="stable")[:10]]
            failures = int(np.sum(test.labels[accepted]))
            risk_at_20_percent[robot][method] = {
                "accepted": 10,
                "failures": failures,
                "false_safe_risk": failures / 10,
            }

    ablations = {}
    for name in ("no_normalization", "no_phase_separation", "no_touchdown"):
        ablation_train = ablation_arrays(training_records, "pilot", name)
        ablation_test = ablation_arrays(test_records, "pilot", name)
        values = _fit(
            ablation_train.X, ablation_train.labels
        ).predict_proba(ablation_test.X)[:, 1]
        ablations[name] = {
            scope: _ranking_metrics(ablation_test.labels[mask], values[mask])
            for scope, mask in {
                "pooled": np.ones(len(ablation_test.labels), dtype=bool),
                "talos": ablation_test.robots == "talos",
                "icub": ablation_test.robots == "icub",
            }.items()
        }

    with gate_summary.open(encoding="utf-8") as stream:
        gate = json.load(stream)
    success = {
        robot: int(gate["robots"][robot]["successes"])
        for robot in ("talos", "icub")
    }
    odds_ratio, p_value = fisher_exact([
        [success["talos"], 50 - success["talos"]],
        [success["icub"], 50 - success["icub"]],
    ])
    failures = {
        robot: dict(sorted(Counter(
            record["failure_reason"] or "success"
            for record in test_records if record["robot"] == robot
        ).items()))
        for robot in ("talos", "icub")
    }
    return {
        "status": "independent_domain_gate_with_exploratory_frozen_training",
        "not_formal_risk_calibration": True,
        "training_rows": len(training_records),
        "test_rows": len(test_records),
        "gate": gate,
        "success_rate_ci95": {
            robot: {
                "successes": successes,
                "trials": 50,
                "rate": successes / 50,
                "exact_ci95": list(_interval(successes, 50)),
            }
            for robot, successes in success.items()
        },
        "talos_icub_fisher": {
            "odds_ratio": float(odds_ratio),
            "two_sided_p": float(p_value),
        },
        "failure_breakdown": failures,
        "heldout_ranking": ranking,
        "risk_at_20_percent_coverage": risk_at_20_percent,
        "ablations": ablations,
        "rollout_runtime_seconds": {
            robot: {
                "mean": float(np.mean([
                    float(record["runtime_seconds"])
                    for record in test_records if record["robot"] == robot
                ])),
                "median": float(np.median([
                    float(record["runtime_seconds"])
                    for record in test_records if record["robot"] == robot
                ])),
            }
            for robot in ("talos", "icub")
        },
        "method_labels": {
            **{key: value[0] for key, value in METHODS.items()},
            "B4": "parameter-only ML",
            "B5": "phase-sequence ML",
        },
    }


def _save(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(figure)


def _pipeline(output: Path) -> None:
    stages = (
        ("Engineering set", "100 frozen rollouts\ntraining only"),
        ("Whole-body\nsignature", r"SSP $\cdot$ touchdown $\cdot$ DSP" "\nnormalized margins"),
        ("Five rankings", "B1--B3 physics\nB4--B5 learned"),
        ("Independent\ngate", "100 new rollouts\n50 per robot"),
        ("Decision", "per-robot balance\nheld-out ranking"),
    )
    figure, axis = plt.subplots(figsize=(10.4, 2.5))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    for index, (title, detail) in enumerate(stages):
        left = 0.01 + index * 0.198
        axis.add_patch(FancyBboxPatch(
            (left, 0.28), 0.17, 0.48,
            boxstyle="round,pad=0.012",
            facecolor="#EDF3F8" if index % 2 == 0 else "#FAF3E5",
            edgecolor=COLORS["ink"],
        ))
        axis.text(left + 0.085, 0.61, title, ha="center", va="center",
                  weight="bold", color=COLORS["ink"], fontsize=9)
        axis.text(left + 0.085, 0.43, detail, ha="center", va="center",
                  color=COLORS["ink"], fontsize=8)
        if index < len(stages) - 1:
            axis.annotate("", xy=(left + 0.195, 0.52), xytext=(left + 0.175, 0.52),
                          arrowprops={"arrowstyle": "-|>", "color": COLORS["ink"]})
    axis.set_title("Independent evaluation of robot-normalized feasibility screening",
                   weight="bold", color=COLORS["ink"])
    _save(figure, output / "figure1_pipeline")


def _outcomes(result: dict[str, object], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    robots = ("talos", "icub")
    labels = ("TALOS", "iCub")
    rates = [result["success_rate_ci95"][robot]["rate"] for robot in robots]
    intervals = [result["success_rate_ci95"][robot]["exact_ci95"] for robot in robots]
    axes[0].axhspan(0.25, 0.75, color="#EDF3F8", zorder=0)
    axes[0].errorbar(
        labels,
        rates,
        yerr=[
            [rate - interval[0] for rate, interval in zip(rates, intervals)],
            [interval[1] - rate for rate, interval in zip(rates, intervals)],
        ],
        fmt="o",
        color=COLORS["blue"],
        markeredgecolor=COLORS["ink"],
        capsize=5,
    )
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Successful rollout fraction")
    axes[0].set_title("Shared-domain feasibility", weight="bold")
    axes[0].grid(axis="y", color=COLORS["grid"])
    for index, rate in enumerate(rates):
        axes[0].text(index, rate + 0.06, f"{int(rate * 50)}/50",
                     ha="center", color=COLORS["ink"])

    categories = ("success", "normal_force", "friction", "impact_dynamics", "other")
    palette = (
        COLORS["blue"], COLORS["gold"], COLORS["orange"],
        COLORS["olive"], COLORS["pink"],
    )
    left = np.zeros(2)
    for category, color in zip(categories, palette):
        values = []
        for robot in robots:
            breakdown = result["failure_breakdown"][robot]
            if category == "other":
                values.append(sum(
                    count for name, count in breakdown.items()
                    if name not in categories[:-1]
                ))
            else:
                values.append(breakdown.get(category, 0))
        axes[1].barh(labels, values, left=left, label=category.replace("_", " "),
                     color=color, edgecolor="white")
        left += values
    axes[1].set_xlim(0, 50)
    axes[1].set_title("First outcome", weight="bold")
    axes[1].legend(
        frameon=False,
        fontsize=7,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
    )
    axes[1].grid(axis="x", color=COLORS["grid"])
    figure.suptitle("Independent 50-per-robot domain gate", weight="bold")
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    _save(figure, output / "figure2_domain_gate")


def _ranking(result: dict[str, object], output: Path) -> None:
    methods = ("B1", "B2", "B3", "B4", "B5")
    figure, axes = plt.subplots(1, 3, figsize=(11.2, 3.8), sharey=True)
    x = np.arange(len(methods))
    for axis, scope, title in zip(
        axes, ("pooled", "talos", "icub"), ("Pooled", "TALOS", "iCub")
    ):
        pr = [result["heldout_ranking"][scope][method]["pr_auc"] for method in methods]
        roc = [result["heldout_ranking"][scope][method]["roc_auc"] for method in methods]
        axis.bar(x - 0.18, [np.nan if value is None else value for value in pr],
                 0.36, label="PR-AUC", color=COLORS["blue"])
        axis.bar(x + 0.18, [np.nan if value is None else value for value in roc],
                 0.36, label="ROC-AUC", color=COLORS["gold"])
        axis.set_xticks(x, methods)
        axis.set_ylim(0, 1)
        axis.set_title(title, weight="bold")
        axis.grid(axis="y", color=COLORS["grid"])
    axes[0].set_ylabel("Held-out score")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
    )
    figure.suptitle(
        "Ranking on the independent payload-capped domain",
        weight="bold",
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.86))
    _save(figure, output / "figure3_heldout_ranking")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-csv",
        default="results/development-seed12026/dataset.csv",
    )
    parser.add_argument(
        "--test-dir",
        default="results/domain-gate-seed22026",
    )
    parser.add_argument("--figure-dir", default="submission/ijhr/figures")
    args = parser.parse_args(argv)
    test_directory = Path(args.test_dir)
    output = Path(args.figure_dir)
    output.mkdir(parents=True, exist_ok=True)
    result = analyze(
        Path(args.training_csv),
        test_directory / "dataset.csv",
        test_directory / "domain_gate_summary.json",
    )
    with (test_directory / "analysis.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    _pipeline(output)
    _outcomes(result, output)
    _ranking(result, output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
