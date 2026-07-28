import csv
import json
from pathlib import Path
import sys
import tempfile


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment import MODEL_PARAMETER_COLUMNS

try:
    from artifact.analyze_confirmation import (
        analyze,
        generate_confirmation_figures,
    )
except ImportError:
    analyze = None
    generate_confirmation_figures = None

try:
    from artifact.analyze_confirmation import _label_failure_bars
except ImportError:
    _label_failure_bars = None


FEATURE_COLUMNS = (
    "feature.single_support.zmp.min",
    "feature.single_support.ik_residual.min",
    "feature.single_support.joint_position.min",
    "feature.single_support.dynamics_slack.min",
)


def _write_rows(path: Path, rows: int, seed_start: int, source: int) -> None:
    fieldnames = [
        "split",
        "robot",
        "base_gait_id",
        "perturbation_index",
        *MODEL_PARAMETER_COLUMNS,
        "seed",
        "label",
        "failure_reason",
        "runtime_seconds",
        *FEATURE_COLUMNS,
        *(name.replace("feature.", "raw_feature.", 1) for name in FEATURE_COLUMNS),
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        per_robot = rows // 2
        for index in range(rows):
            robot = "talos" if index < per_robot else "icub"
            local = index % per_robot
            label = int((local + source) % 4 in (0, 1))
            parameter_signal = 0.01 * ((local + source) % 7)
            raw_signal = 2.0 * label + 0.01 * (local % 5)
            row = {
                "split": "pilot",
                "robot": robot,
                "base_gait_id": f"{robot}-{source}-{local:04d}",
                "perturbation_index": 0,
                "seed": seed_start + index,
                "label": label,
                "failure_reason": "friction" if label else "",
                "runtime_seconds": 1.0,
            }
            row.update({
                name: parameter_signal + column * 0.001
                for column, name in enumerate(MODEL_PARAMETER_COLUMNS)
            })
            row.update({
                name: raw_signal + 0.02 * ((local + column) % 9)
                for column, name in enumerate(FEATURE_COLUMNS)
            })
            row.update({
                name.replace("feature.", "raw_feature.", 1):
                raw_signal + column * 0.001
                for column, name in enumerate(FEATURE_COLUMNS)
            })
            writer.writerow(row)


def _fixture(root: Path):
    development_one = root / "development_one.csv"
    development_two = root / "development_two.csv"
    confirmation = root / "confirmation.csv"
    _write_rows(development_one, 100, 10_000, 0)
    _write_rows(development_two, 100, 20_000, 1)
    _write_rows(confirmation, 400, 30_000, 2)
    return [development_one, development_two], confirmation


def _repeated_fixture(root: Path):
    development, first = _fixture(root)
    second = root / "confirmation_second.csv"
    _write_rows(second, 400, 40_000, 3)
    return development, [first, second]


def test_confirmation_analysis_is_deterministic_and_has_primary_gate():
    assert analyze is not None, "analyze_confirmation.analyze is missing"
    with tempfile.TemporaryDirectory() as temporary:
        development, confirmation = _fixture(Path(temporary))
        first = analyze(
            development, confirmation, bootstrap_seed=32026, repetitions=200
        )
        second = analyze(
            development, confirmation, bootstrap_seed=32026, repetitions=200
        )
        assert json.dumps(first, sort_keys=True) == json.dumps(
            second, sort_keys=True
        )
        assert first["rows"] == {
            "development": 200,
            "confirmation": 400,
            "confirmation_by_robot": {"icub": 200, "talos": 200},
            "confirmation_by_scramble": {"set1": 400},
        }
        primary = first["primary"]
        assert primary["metric"] == "macro_within_robot_pr_auc_difference"
        assert primary["proposed"] == "B5_whole_body"
        assert primary["comparator"] == "B4_parameters_robot"
        assert primary["delta_pr_auc"] > 0.0
        assert len(primary["ci95"]) == 2
        assert primary["passed"] is (
            primary["ci95"][0] > 0.0
            and all(
                value > 0.0
                for value in primary["delta_by_confirmation"].values()
            )
        )
        assert set(first["success_at_budget"]) == {"talos", "icub"}
        assert first["failure_composition"] == {
            "talos": {"friction": 100},
            "icub": {"friction": 100},
        }


def test_repeated_confirmation_is_stratified_by_scramble_and_robot():
    with tempfile.TemporaryDirectory() as temporary:
        development, confirmation = _repeated_fixture(Path(temporary))
        result = analyze(
            development, confirmation, bootstrap_seed=62026, repetitions=100
        )
        assert result["rows"] == {
            "development": 200,
            "confirmation": 800,
            "confirmation_by_robot": {"icub": 400, "talos": 400},
            "confirmation_by_scramble": {"set1": 400, "set2": 400},
        }
        assert set(result["primary"]["delta_by_confirmation"]) == {
            "set1",
            "set2",
        }
        assert result["model"]["robot_id_feature"] == {
            "B4_parameters_robot": True,
            "B5_whole_body": True,
        }
        assert set(result["representation_ablation"]["pooled"]) == {
            "phase_agnostic",
            "phase_resolved",
            "no_touchdown",
        }


def test_confirmation_figures_are_complete_and_deterministic():
    assert generate_confirmation_figures is not None
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        development, confirmation = _fixture(root)
        result = analyze(
            development, confirmation, bootstrap_seed=32026, repetitions=20
        )
        first = generate_confirmation_figures(result, root / "first")
        second = generate_confirmation_figures(result, root / "second")
        assert [path.name for path in first] == [
            "figure2_domain_gate.pdf",
            "figure3_heldout_ranking.pdf",
        ]
        assert all(path.stat().st_size > 1_000 for path in first)
        assert [path.read_bytes() for path in first] == [
            path.read_bytes() for path in second
        ]


def test_icub_friction_label_is_lowered():
    assert _label_failure_bars is not None
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots()
    try:
        bars = axis.barh([0.18], [41])
        labels = _label_failure_bars(axis, bars, "icub")
        assert labels[0].get_text() == "iCub 41"
        assert labels[0].get_position() == (2, -3)
    finally:
        plt.close(figure)


def test_confirmation_analysis_rejects_seed_overlap():
    with tempfile.TemporaryDirectory() as temporary:
        development, confirmation = _fixture(Path(temporary))
        rows = list(csv.DictReader(confirmation.open(newline="")))
        rows[0]["seed"] = "10000"
        with confirmation.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        try:
            analyze(development, confirmation, repetitions=10)
        except ValueError as error:
            assert "share rollout seeds" in str(error)
        else:
            raise AssertionError("analysis accepted overlapping rollout seeds")


def test_confirmation_analysis_requires_200_rows_per_robot():
    with tempfile.TemporaryDirectory() as temporary:
        development, confirmation = _fixture(Path(temporary))
        rows = list(csv.DictReader(confirmation.open(newline="")))[:-1]
        with confirmation.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        try:
            analyze(development, confirmation, repetitions=10)
        except ValueError as error:
            assert "200 confirmation rows per robot" in str(error)
        else:
            raise AssertionError("analysis accepted an incomplete confirmation")


if __name__ == "__main__":
    test_confirmation_analysis_is_deterministic_and_has_primary_gate()
    test_repeated_confirmation_is_stratified_by_scramble_and_robot()
    test_confirmation_figures_are_complete_and_deterministic()
    test_icub_friction_label_is_lowered()
    test_confirmation_analysis_rejects_seed_overlap()
    test_confirmation_analysis_requires_200_rows_per_robot()
