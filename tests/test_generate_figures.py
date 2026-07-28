import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from artifact.generate_figures import (  # noqa: E402
    FIGURE_FILENAMES,
    FigureDataError,
    generate_figures,
)


METHODS = (
    "zmp_cop_margin",
    "ik_joint_margin",
    "inverse_dynamics_slack",
    "black_box_parameters",
    "uncalibrated_phase_sequence",
    "risk_calibrated_phase_sequence",
)


def _write_valid_fixture(root: Path) -> None:
    root.mkdir()
    protocol = {
        "seed": 2026,
        "robots": ["talos", "icub"],
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
        "rollout_budgets": [5, 10, 20],
        "screening_repetitions": 30,
        "scientific_protocol": True,
    }
    (root / "protocol.json").write_text(json.dumps(protocol))

    evaluation = {"methods": {"test": {}}}
    for robot in ("talos", "icub"):
        metrics = {}
        for method_index, method in enumerate(METHODS):
            accepted = (method_index + 1) * 60
            failures = 0
            risk = failures / accepted
            metrics[method] = {
                "accepted": accepted,
                "coverage": accepted / protocol["test_base_gaits"],
                "false_safe_count": failures,
                "false_safe_risk": risk,
                "false_safe_upper": 1.0 - 0.05 ** (1.0 / accepted),
                "confidence_bound_valid": True,
            }
        evaluation["methods"]["test"][robot] = {
            "split": "test",
            "robot": robot,
            "rows": protocol["test_base_gaits"],
            "prespecified": metrics,
        }
    (root / "evaluation.json").write_text(json.dumps(evaluation))

    raw_rows = []
    for robot_index, robot in enumerate(("talos", "icub")):
        for condition_index, condition in enumerate(("id", "ood")):
            for repetition in range(protocol["screening_repetitions"]):
                pool_seed = 1000 + 100 * condition_index + repetition
                for method_index, method in enumerate(METHODS):
                    for budget in protocol["rollout_budgets"]:
                        raw_rows.append({
                            "method": method,
                            "robot": robot,
                            "condition": condition,
                            "repetition": repetition,
                            "budget": budget,
                            "success": (
                                method_index + robot_index + condition_index
                                + repetition + budget
                            ) % 3 == 0,
                            "rollouts": min(
                                budget, 1 + (method_index + repetition) % budget
                            ),
                            "rollout_runtime_seconds": (
                                0.1 * (method_index + 1) + 0.01 * repetition
                            ),
                            "failure_reason": "",
                            "pool_size": protocol["candidate_pool_size"],
                            "pool_seed": pool_seed,
                        })
    with (root / "screening.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(raw_rows[0]))
        writer.writeheader()
        writer.writerows(raw_rows)

    grouped = {}
    for row in raw_rows:
        key = (
            row["method"], row["robot"], row["condition"], row["budget"]
        )
        grouped.setdefault(key, []).append(row)
    summary = []
    for key in sorted(grouped):
        rows = grouped[key]
        summary.append({
            "method": key[0],
            "robot": key[1],
            "condition": key[2],
            "budget": key[3],
            "repetitions": len(rows),
            "pool_size": protocol["candidate_pool_size"],
            "success_by_repetition": {
                str(row["repetition"]): row["success"]
                for row in sorted(rows, key=lambda item: item["repetition"])
            },
            "success_rate": sum(row["success"] for row in rows) / len(rows),
            "mean_rollouts": sum(row["rollouts"] for row in rows) / len(rows),
            "mean_rollout_runtime_seconds": (
                sum(row["rollout_runtime_seconds"] for row in rows) / len(rows)
            ),
        })
    (root / "screening_summary.json").write_text(json.dumps(summary))


class GenerateFiguresTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_generate_figures_is_complete_png_and_byte_deterministic(self):
        results = self.root / "results"
        _write_valid_fixture(results)

        first = generate_figures(results, self.root / "first")
        second = generate_figures(results, self.root / "second")

        assert tuple(path.name for path in first) == FIGURE_FILENAMES
        assert tuple(path.name for path in second) == FIGURE_FILENAMES
        assert sorted(path.name for path in (self.root / "first").iterdir()) == list(
            FIGURE_FILENAMES
        )
        for first_path, second_path in zip(first, second):
            first_bytes = first_path.read_bytes()
            second_bytes = second_path.read_bytes()
            assert first_bytes.startswith(b"\x89PNG\r\n\x1a\n")
            assert len(first_bytes) > 5_000
            assert hashlib.sha256(first_bytes).digest() == hashlib.sha256(
                second_bytes
            ).digest()

    def test_generate_figures_refuses_pending_without_partial_output(self):
        results = self.root / "results"
        _write_valid_fixture(results)
        evaluation_path = results / "evaluation.json"
        evaluation = json.loads(evaluation_path.read_text())
        evaluation["methods"]["test"]["talos"]["prespecified"][
            "risk_calibrated_phase_sequence"
        ]["false_safe_risk"] = "pending"
        evaluation_path.write_text(json.dumps(evaluation))
        output = self.root / "figures"

        with self.assertRaisesRegex(FigureDataError, "pending"):
            generate_figures(results, output)

        assert not output.exists()

    def test_generate_figures_refuses_stale_screening_summary(self):
        results = self.root / "results"
        _write_valid_fixture(results)
        summary_path = results / "screening_summary.json"
        summary = json.loads(summary_path.read_text())
        summary[0]["success_rate"] = 0.123456
        summary_path.write_text(json.dumps(summary))
        output = self.root / "figures"

        with self.assertRaisesRegex(FigureDataError, "screening.csv"):
            generate_figures(results, output)

        assert not output.exists()

    def test_generate_figures_refuses_non_binomial_confidence_bound(self):
        results = self.root / "results"
        _write_valid_fixture(results)
        evaluation_path = results / "evaluation.json"
        evaluation = json.loads(evaluation_path.read_text())
        evaluation["methods"]["test"]["icub"]["prespecified"][
            "risk_calibrated_phase_sequence"
        ]["false_safe_upper"] = 0.1
        evaluation_path.write_text(json.dumps(evaluation))
        output = self.root / "figures"

        with self.assertRaisesRegex(FigureDataError, "Clopper-Pearson"):
            generate_figures(results, output)

        assert not output.exists()

    def test_generate_figures_refuses_smoke_scale_protocol(self):
        results = self.root / "results"
        _write_valid_fixture(results)
        protocol_path = results / "protocol.json"
        protocol = json.loads(protocol_path.read_text())
        protocol["candidate_pool_size"] = 8
        protocol_path.write_text(json.dumps(protocol))
        output = self.root / "figures"

        with self.assertRaisesRegex(FigureDataError, "frozen scientific value"):
            generate_figures(results, output)

        assert not output.exists()


if __name__ == "__main__":
    unittest.main()
