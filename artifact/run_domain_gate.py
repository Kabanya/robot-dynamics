"""Run the one-shot shared-domain gate used after the engineering prescreen."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path

from src.experiment import ProtocolConfig, _write_json, run_cases


ROBOTS = ("talos", "icub")


def build_config() -> ProtocolConfig:
    return replace(
        ProtocolConfig.smoke(seed=22026),
        pilot_rollouts=50,
        steps=6,
        dt=0.01,
        pilot_revision=1,
        step_length_range=(0.10, 0.16),
        step_width_range=(1.00, 1.10),
        single_support_range=(2.70, 2.80),
        double_support_range=(0.70, 0.80),
        com_height_range=(0.92, 0.98),
        zmp_bias_x_range=(-0.05, 0.05),
        zmp_bias_y_range=(-0.25, -0.10),
        id_payload_range=(0.0, 0.05),
    )


def summarize(records) -> dict[str, object]:
    robots = {}
    for robot in ROBOTS:
        selected = [record for record in records if record["robot"] == robot]
        if len(selected) != 50:
            raise ValueError(f"domain gate requires exactly 50 {robot} records")
        successes = sum(int(record["label"]) == 0 for record in selected)
        outcomes = Counter(
            str(record["failure_reason"]) or "success" for record in selected
        )
        robots[robot] = {
            "rollouts": 50,
            "successes": successes,
            "success_fraction": successes / 50,
            "balanced_25_75": 13 <= successes <= 37,
            "outcomes": dict(sorted(outcomes.items())),
        }
    return {
        "study": "independent_shared_domain_gate",
        "official_surrogate_campaign": False,
        "no_range_adjustment_or_rerun": True,
        "decision_rule": "pass only when each robot has 13-37 successes out of 50",
        "shared_domain_supported": all(
            result["balanced_25_75"] for result in robots.values()
        ),
        "robots": robots,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="results/domain-gate-seed22026",
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    output = Path(args.output_dir)
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")

    records, _, manifest_path = run_cases(
        build_config(),
        str(output),
        splits=("pilot",),
        workers=args.workers,
    )
    summary = summarize(records)
    with open(manifest_path, encoding="utf-8") as stream:
        manifest = json.load(stream)
    summary["experiment_fingerprint"] = manifest["experiment_fingerprint"]
    path = _write_json(str(output / "domain_gate_summary.json"), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"domain gate summary: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
