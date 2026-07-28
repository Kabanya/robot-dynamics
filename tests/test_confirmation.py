"""Checks for the independent confirmation runner."""

from pathlib import Path
import sys
import tempfile


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from artifact.run_confirmation import (
        build_config,
        main,
        summarize,
    )
except ImportError:
    build_config = main = summarize = None

try:
    from artifact.generate_rollout_video import (
        _frame_indices,
        select_cases,
    )
except ImportError:
    _frame_indices = select_cases = None


def _records(talos_successes=120, icub_successes=80):
    return [
        *(
            {
                "robot": "talos",
                "label": int(index >= talos_successes),
                "failure_reason": "" if index < talos_successes else "friction",
            }
            for index in range(200)
        ),
        *(
            {
                "robot": "icub",
                "label": int(index >= icub_successes),
                "failure_reason": "" if index < icub_successes else "normal_force",
            }
            for index in range(200)
        ),
    ]


def test_confirmation_protocol_is_frozen():
    assert build_config is not None, "run_confirmation.build_config is missing"
    config = build_config(seed=42026)
    assert config.seed == 42026
    assert config.confirmation_protocol is True
    assert config.scientific_protocol is False
    assert config.pilot_rollouts == 200
    assert config.steps == 6
    assert config.dt == 0.01
    assert config.step_length_range == (0.10, 0.16)
    assert config.step_width_range == (1.00, 1.10)
    assert config.single_support_range == (2.70, 2.80)
    assert config.double_support_range == (0.70, 0.80)
    assert config.com_height_range == (0.92, 0.98)
    assert config.zmp_bias_x_range == (-0.05, 0.05)
    assert config.zmp_bias_y_range == (-0.25, -0.10)
    assert config.id_payload_range == (0.0, 0.05)


def test_confirmation_summary_requires_200_rows_per_robot():
    summary = summarize(_records(), seed=42026)
    assert summary["seed"] == 42026
    assert summary["confirmation_not_tunable"] is True
    assert summary["robots"]["talos"] == {
        "rollouts": 200,
        "successes": 120,
        "failures": 80,
        "outcomes": {"friction": 80, "success": 120},
    }
    assert summary["robots"]["icub"]["successes"] == 80
    assert summary["robots"]["icub"]["failures"] == 120

    try:
        summarize(_records()[:-1], seed=42026)
    except ValueError as error:
        assert "exactly 200 icub records" in str(error)
    else:
        raise AssertionError("summary accepted an incomplete confirmation set")


def test_confirmation_cli_refuses_to_overwrite():
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "existing"
        output.mkdir()
        try:
            main(["--output-dir", str(output)])
        except SystemExit as error:
            assert "refusing to overwrite" in str(error)
        else:
            raise AssertionError("confirmation CLI overwrote an existing directory")


def test_video_cases_are_successful_longest_steps():
    assert select_cases is not None
    rows = [
        {
            "robot": robot,
            "label": label,
            "step_length": step,
            "failure_index": failure_index,
            "seed": seed,
        }
        for robot in ("talos", "icub")
        for label, step, failure_index, seed in (
            (1, "0.16", 20, 1),
            (1, "0.12", 80, 4),
            (0, "0.11", -1, 2),
            (0, "0.15", -1, 3),
        )
    ]
    assert {
        robot: {
            outcome: int(row["seed"])
            for outcome, row in outcomes.items()
        }
        for robot, outcomes in select_cases(rows).items()
    } == {
        "talos": {"success": 3, "failure": 4},
        "icub": {"success": 3, "failure": 4},
    }


def test_video_frames_include_the_full_available_rollout():
    assert _frame_indices(35, stride=10, count=8) == [0, 10, 20, 30, 30, 30, 30, 30]
    assert _frame_indices(95, stride=10, count=8) == list(range(0, 95, 10))


if __name__ == "__main__":
    test_confirmation_protocol_is_frozen()
    test_confirmation_summary_requires_200_rows_per_robot()
    test_confirmation_cli_refuses_to_overwrite()
    test_video_cases_are_successful_longest_steps()
    test_video_frames_include_the_full_available_rollout()
