"""Checks for the shared-domain gate runner."""

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from artifact.run_domain_gate import build_config, summarize


def test_domain_gate_is_fixed_and_requires_each_robot_to_be_balanced():
    config = build_config()
    assert config.seed == 22026
    assert config.pilot_rollouts == 50
    assert config.id_payload_range == (0.0, 0.05)
    assert config.steps == 6
    assert config.dt == 0.01

    records = [
        *({"robot": "talos", "label": int(index >= 25),
           "failure_reason": "friction" if index >= 25 else ""}
          for index in range(50)),
        *({"robot": "icub", "label": int(index >= 12),
           "failure_reason": "friction" if index >= 12 else ""}
          for index in range(50)),
    ]
    summary = summarize(records)
    assert summary["robots"]["talos"]["balanced_25_75"] is True
    assert summary["robots"]["icub"]["balanced_25_75"] is False
    assert summary["shared_domain_supported"] is False
    assert summary["robots"]["talos"]["outcomes"] == {
        "friction": 25,
        "success": 25,
    }


if __name__ == "__main__":
    test_domain_gate_is_fixed_and_requires_each_robot_to_be_balanced()
