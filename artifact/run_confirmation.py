"""Run the frozen 400-rollout confirmation for the ranking claim."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path

from src.experiment import (
    ProtocolConfig,
    _write_json,
    run_cases,
    run_validity_tests,
    write_environment_lock,
)


ROBOTS = ("talos", "icub")


def build_config(seed: int = 42026) -> ProtocolConfig:
    return replace(
        ProtocolConfig.smoke(seed=seed),
        confirmation_protocol=True,
        pilot_rollouts=200,
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


def summarize(records, seed: int) -> dict[str, object]:
    robots = {}
    for robot in ROBOTS:
        selected = [record for record in records if record["robot"] == robot]
        if len(selected) != 200:
            raise ValueError(
                f"confirmation requires exactly 200 {robot} records"
            )
        successes = sum(int(record["label"]) == 0 for record in selected)
        outcomes = Counter(
            str(record["failure_reason"]) or "success" for record in selected
        )
        robots[robot] = {
            "rollouts": 200,
            "successes": successes,
            "failures": 200 - successes,
            "outcomes": dict(sorted(outcomes.items())),
        }
    return {
        "study": "independent_whole_body_confirmation",
        "seed": int(seed),
        "confirmation_not_tunable": True,
        "robots": robots,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42026)
    parser.add_argument("--output-dir")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    output = Path(
        args.output_dir
        or f"results/confirmation-seed{args.seed}"
    )
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    output.mkdir(parents=True)

    config = build_config(args.seed)
    write_environment_lock(str(output))
    validity = run_validity_tests(str(output), config)
    if not validity["passed"]:
        raise SystemExit("physical validity gate failed")
    records, _, manifest_path = run_cases(
        config,
        str(output),
        splits=("pilot",),
        workers=args.workers,
    )
    summary = summarize(records, args.seed)
    with open(manifest_path, encoding="utf-8") as stream:
        manifest = json.load(stream)
    summary["experiment_fingerprint"] = manifest["experiment_fingerprint"]
    path = _write_json(str(output / "confirmation_summary.json"), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"confirmation summary: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
