import gc
import os
import sys
import tempfile
import weakref
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src import experiment
from src.feasibility import PhysicsSignature, RolloutResult, WholeBodyTrajectory


class _ScoreModel:
    threshold_ = 0.5

    def predict_failure_score(self, X):
        return np.linspace(0.1, 0.9, len(X))

    def accept(self, X):
        return np.ones(len(X), dtype=bool)


def test_screening_releases_trajectories_and_full_rollout_results():
    trajectory_refs = []
    result_refs = []
    peak_trajectories = 0
    peak_results = 0

    def build_trajectory(*_args, **_kwargs):
        nonlocal peak_trajectories
        gc.collect()
        trajectory = WholeBodyTrajectory(
            np.array([0.0]),
            np.zeros((1, 1)),
            np.zeros((1, 1)),
            np.zeros((1, 1)),
            np.zeros((1, 4, 4)),
            np.zeros((1, 4, 4)),
            np.zeros((1, 3)),
            ("double_support",),
            0.01,
        )
        trajectory_refs.append(weakref.ref(trajectory))
        peak_trajectories = max(
            peak_trajectories,
            sum(reference() is not None for reference in trajectory_refs),
        )
        return trajectory

    def final_rollout(*_args):
        nonlocal peak_results
        gc.collect()
        result = RolloutResult(
            np.arange(64, dtype=float),
            np.zeros((64, 8)),
            np.zeros((64, 8)),
            np.zeros((64, 2, 6)),
            False,
            "tracking",
            runtime=0.2,
        )
        result_refs.append(weakref.ref(result))
        peak_results = max(
            peak_results,
            sum(reference() is not None for reference in result_refs),
        )
        return result

    config = replace(
        experiment.ProtocolConfig.smoke(seed=17),
        robots=("talos",),
        candidate_pool_size=3,
        rollout_budgets=(2,),
    )
    spec = SimpleNamespace(model=SimpleNamespace(nq=1, nv=1))
    signature = PhysicsSignature(
        ("single_support.torque.min",),
        np.array([0.0]),
        np.array([0.0]),
        ("ok",),
    )

    with (
        patch.object(experiment, "load_robot_spec", return_value=spec),
        patch.object(
            experiment,
            "build_whole_body_trajectory",
            new=build_trajectory,
        ),
        patch.object(
            experiment,
            "compute_physics_signature",
            new=lambda *_args: signature,
        ),
        patch.object(experiment, "rollout", new=final_rollout),
        patch.object(
            experiment,
            "_environment_payload",
            return_value={"test": "screening-memory"},
        ),
        tempfile.TemporaryDirectory() as directory,
    ):
        experiment.run_screening_campaign(
            config,
            _ScoreModel(),
            _ScoreModel(),
            directory,
        )

    assert peak_trajectories == 1
    assert peak_results == 1


if __name__ == "__main__":
    test_screening_releases_trajectories_and_full_rollout_results()
    print("screening memory test passed")
