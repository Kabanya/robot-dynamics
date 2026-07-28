import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from dataclasses import asdict, replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import src.experiment as experiment
from src.feasibility import PhysicsSignature, RolloutResult
from src.experiment import (
    MODEL_PARAMETER_COLUMNS,
    ExperimentCase,
    ProtocolConfig,
    _exact_acceptance_mask,
    ablation_arrays,
    assert_pilot_frozen,
    assert_training_provenance,
    calibration_metrics,
    classification_metrics,
    cli,
    evaluate_records,
    evaluate_score_methods,
    flatten_record,
    generate_candidate_pool,
    generate_cases,
    paired_group_bootstrap,
    records_to_arrays,
    readiness_gate,
    run_cases,
    run_screening_campaign,
    screening_experiment,
    score_methods,
    summarize_pilot,
    summarize_screening,
    train_surrogates,
    validate_campaign_dataset,
    verify_campaign_provenance,
    verify_ranked_candidates,
    write_dataset,
    write_environment_lock,
)


def test_scrambled_sobol_protocol_is_deterministic_and_grouped():
    config = ProtocolConfig.smoke(seed=2718)
    first = generate_cases(config)
    second = generate_cases(config)
    assert first == second
    assert first != generate_cases(replace(config, seed=2719))

    expected = {
        "pilot": 2,
        "train": 2,
        "tune": 2,
        "calibration": 2,
        "test": 2,
        "ood": 2,
    }
    assert {
        split: sum(case.split == split for case in first)
        for split in expected
    } == expected

    split_by_group = {}
    for case in first:
        key = (case.robot, case.base_gait_id)
        split_by_group.setdefault(key, set()).add(case.split)
    assert all(len(splits) == 1 for splits in split_by_group.values())
    assert all(case.sample.ood == (case.split == "ood") for case in first)


def _maximum_cross_design_correlation(cases):
    gait_fields = (
        "step_length", "step_width", "single_support_duration",
        "double_support_duration", "com_height_scale", "zmp_bias_x",
        "zmp_bias_y",
    )
    perturbation_fields = (
        "friction", "payload_fraction", "timing_error_seconds", "impulse",
    )
    gait = np.asarray([
        [getattr(case.sample, field) for field in gait_fields]
        for case in cases
    ])
    perturbation = np.asarray([
        [getattr(case.sample, field) for field in perturbation_fields]
        for case in cases
    ])
    correlation = np.corrcoef(gait.T, perturbation.T)
    return float(np.max(np.abs(correlation[:len(gait_fields), len(gait_fields):])))


def test_grouped_sobol_splits_do_not_index_couple_gait_and_perturbations():
    cases = generate_cases(ProtocolConfig(seed=2026))
    for robot in ("talos", "icub"):
        for split in ("pilot", "train", "tune", "ood"):
            slots = {
                case.perturbation_index for case in cases
                if case.robot == robot and case.split == split
            }
            for slot in slots:
                selected = [
                    case for case in cases
                    if case.robot == robot and case.split == split
                    and case.perturbation_index == slot
                ]
                maximum = _maximum_cross_design_correlation(selected)
                assert maximum < 0.25, (robot, split, slot, maximum)


def test_candidate_pool_does_not_index_couple_gait_and_perturbations():
    config = ProtocolConfig(seed=2026)
    for robot in ("talos", "icub"):
        for ood in (False, True):
            candidates = generate_candidate_pool(
                robot, seed=2026, count=2048, ood=ood, config=config
            )
            assert _maximum_cross_design_correlation(candidates) < 0.10, (
                robot, ood, _maximum_cross_design_correlation(candidates)
            )


def test_cross_design_gate_checks_each_grouped_perturbation_slot():
    gait = np.linspace(0.0, 1.0, 100)[:, None]
    repeated_gait = np.repeat(gait, 3, axis=0)
    perturbations = np.column_stack((
        gait[:, 0], 1.0 - gait[:, 0], np.full(len(gait), 0.5)
    )).reshape(-1, 1)
    try:
        experiment._assert_cross_design_correlation_below_limit(
            repeated_gait, perturbations, "synthetic", perturbations=3
        )
    except RuntimeError as error:
        assert "slot" in str(error)
    except TypeError as error:
        raise AssertionError("cross-design gate does not inspect grouped slots") from error
    else:
        raise AssertionError("slot-wise index coupling was accepted")


def test_scientific_case_design_rejects_index_coupled_streams():
    original = experiment._permuted_rows
    experiment._permuted_rows = lambda points, _seed: points
    try:
        try:
            generate_cases(ProtocolConfig(seed=2026))
        except RuntimeError as error:
            assert "cross-design correlation" in str(error)
        else:
            raise AssertionError("index-coupled scientific cases were accepted")
    finally:
        experiment._permuted_rows = original


def test_scientific_candidate_pool_rejects_index_coupled_streams():
    original = experiment._permuted_rows
    experiment._permuted_rows = lambda points, _seed: points
    try:
        try:
            generate_candidate_pool(
                "talos", seed=2026, count=2048,
                config=ProtocolConfig(seed=2026),
            )
        except RuntimeError as error:
            assert "cross-design correlation" in str(error)
        else:
            raise AssertionError("index-coupled scientific pool was accepted")
    finally:
        experiment._permuted_rows = original


def test_default_protocol_has_exact_counts_and_iid_inference_units():
    config = ProtocolConfig()
    assert config.seed == 2026
    assert config.study_revision == 2
    assert config.calibration_base_gaits == 450
    assert config.test_base_gaits == 600
    cases = generate_cases(config)
    counts = {
        split: sum(case.split == split for case in cases)
        for split in ("pilot", "train", "tune", "calibration", "test", "ood")
    }
    assert counts == {
        "pilot": 400,
        "train": 3000,
        "tune": 900,
        "calibration": 900,
        "test": 1200,
        "ood": 1200,
    }
    grouped = {}
    for case in cases:
        grouped.setdefault((case.robot, case.base_gait_id), []).append(case)
    assert all(
        len(group) == (
            1 if group[0].split in ("pilot", "calibration", "test") else 3
        )
        for group in grouped.values()
    )
    assert all(
        case.perturbation_index == 0
        for case in cases
        if case.split in ("calibration", "test")
    )
    try:
        replace(config, candidate_pool_size=8)
    except ValueError:
        pass
    else:
        raise AssertionError("scientific screening pool size was mutable")


def test_protocol_json_supports_one_auditable_pilot_range_revision():
    revised = replace(
        ProtocolConfig.smoke(seed=8),
        pilot_revision=1,
        step_length_range=(0.20, 0.30),
        step_width_range=(0.90, 1.10),
    )
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "protocol.json")
        with open(path, "w") as stream:
            json.dump(asdict(revised), stream)
        restored = ProtocolConfig.from_json(path)
        legacy = asdict(revised)
        del legacy["study_revision"]
        with open(path, "w") as stream:
            json.dump(legacy, stream)
        try:
            ProtocolConfig.from_json(path)
        except ValueError as error:
            assert "study revision" in str(error)
        else:
            raise AssertionError("legacy protocol was accepted as revision 2")
    assert restored == revised
    for case in generate_cases(restored):
        assert 0.20 <= case.sample.step_length <= 0.30
        assert 0.90 <= case.sample.step_width <= 1.10
    try:
        replace(ProtocolConfig(seed=44), step_length_range=(0.20, 0.30))
    except ValueError:
        pass
    else:
        raise AssertionError("initial pilot accepted a revised gait range")


def test_non_scientific_protocol_can_narrow_id_payload_without_changing_ood():
    config = replace(
        ProtocolConfig.smoke(seed=22026),
        pilot_rollouts=50,
        id_payload_range=(0.0, 0.05),
    )
    cases = generate_cases(config)
    id_payloads = [
        case.sample.payload_fraction for case in cases if not case.sample.ood
    ]
    ood_payloads = [
        case.sample.payload_fraction for case in cases if case.sample.ood
    ]

    assert max(id_payloads) <= 0.05
    assert max(id_payloads) > 0.04
    assert min(ood_payloads) >= 0.10


def _record(case, label=0):
    signature = PhysicsSignature(
        ("single_support.torque.max", "touchdown.cop.p05"),
        np.array([0.25, -0.5]),
        np.array([1e-9]),
        ("optimal",),
        np.array([25.0, -0.05]),
    )
    rollout = RolloutResult(
        np.array([0.0]),
        np.zeros((1, 2)),
        np.zeros((1, 2)),
        np.zeros((1, 2, 6)),
        success=not label,
        failure_reason="" if not label else "torque",
        active_contacts=np.array([[True, False]]),
        scheduled_contacts=np.array([[True, True]]),
        failure_index=0 if label else -1,
        peak_torque_joint="l_knee" if label else "",
        peak_torque_ratio=1.25 if label else None,
        runtime=0.125,
    )
    return flatten_record(case, signature, rollout)


def test_manifest_and_flat_csv_capture_required_schema():
    config = ProtocolConfig.smoke(seed=11)
    records = [_record(case, index % 2) for index, case in enumerate(generate_cases(config))]
    with tempfile.TemporaryDirectory() as directory:
        csv_path, manifest_path = write_dataset(records, config, directory)
        with open(csv_path, newline="") as stream:
            rows = list(csv.DictReader(stream))
        with open(manifest_path) as stream:
            manifest = json.load(stream)
        with open(os.path.join(directory, "protocol.json")) as stream:
            protocol = json.load(stream)

    required = {
        "split", "robot", "base_gait_id", "perturbation_index", "seed",
        "step_length", "step_width", "single_support_duration",
        "double_support_duration", "com_height_scale", "zmp_bias_x",
        "zmp_bias_y", "friction", "payload_fraction",
        "timing_error_seconds", "impulse", "label", "failure_reason",
        "failure_index", "peak_torque_joint", "peak_torque_ratio",
        "runtime_seconds", "actual_single_support_fraction",
        "actual_double_support_fraction", "contact_schedule_match_fraction",
        "feature.single_support.torque.max",
        "feature.touchdown.cop.p05",
        "raw_feature.single_support.torque.max",
        "raw_feature.touchdown.cop.p05",
    }
    assert required <= set(rows[0])
    assert len(rows) == len(records)
    assert rows[0]["actual_single_support_fraction"] == "1.0"
    assert rows[0]["actual_double_support_fraction"] == "0.0"
    assert rows[0]["contact_schedule_match_fraction"] == "0.0"
    assert rows[1]["failure_index"] == "0"
    assert rows[1]["peak_torque_joint"] == "l_knee"
    assert rows[1]["peak_torque_ratio"] == "1.25"
    assert manifest["seed"] == config.seed
    assert manifest["grouped_split"] is True
    assert manifest["sampling_design"]["train_tune_ood"] == (
        "grouped_independently_permuted_scrambled_sobol"
    )
    assert manifest["feature_columns"] == [
        "feature.single_support.torque.max", "feature.touchdown.cop.p05"
    ]
    assert manifest["scientific_protocol"] is False
    assert manifest["results_status"] == "unvalidated_smoke"
    assert len(manifest["experiment_fingerprint"]) == 64
    assert len(manifest["dataset_sha256"]) == 64
    assert protocol["seed"] == config.seed


def test_environment_lock_records_robot_model_hashes():
    with tempfile.TemporaryDirectory() as directory:
        path = write_environment_lock(directory)
        with open(path) as stream:
            lock = json.load(stream)
        explicit_exists = os.path.exists(os.path.join(
            directory, "conda-explicit.txt"
        ))
    assert set(lock["robot_models"]) == {"talos", "icub"}
    assert all(
        len(model["urdf_sha256"]) == 64
        for model in lock["robot_models"].values()
    )
    assert "conda_packages" in lock
    assert explicit_exists
    assert all(
        model["collision_meshes"]
        and all(len(mesh["sha256"]) == 64 for mesh in model["collision_meshes"])
        for model in lock["robot_models"].values()
    )


def test_environment_lock_writes_exact_pip_requirements():
    with tempfile.TemporaryDirectory() as directory:
        path = write_environment_lock(directory)
        with open(path) as stream:
            lock = json.load(stream)
        with open(os.path.join(directory, "pip-requirements.txt")) as stream:
            requirements = [
                line.strip()
                for line in stream
                if line.strip() and not line.startswith("#")
            ]
    assert requirements == [
        f"{name}=={record['version']}"
        for name, record in sorted(
            lock["pip_packages"].items(), key=lambda item: item[0].lower()
        )
    ]
    assert all(
        len(record["files_sha256"]) == 64
        for record in lock["pip_packages"].values()
    )


def test_pip_lock_excludes_conda_owned_distribution_identity():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)

        class Distribution:
            def __init__(self, name, version):
                self.metadata = {"Name": name}
                self.version = version
                self.files = (
                    Path(f"{name}-{version}.dist-info/METADATA"),
                    Path(f"{name}.py"),
                )

            def locate_file(self, path):
                return root / path

        owned = Distribution("owned", "1.0")
        external = Distribution("external", "2.0")
        for distribution in (owned, external):
            for path in distribution.files:
                target = distribution.locate_file(path)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(f"{distribution.metadata['Name']}:{path}")

        records = experiment._pip_distribution_records(
            (owned, external), {("owned", "1.0")}
        )
        first_digest = records["external"]["files_sha256"]
        external.locate_file(Path("external.py")).write_text("changed")
        changed = experiment._pip_distribution_records(
            (owned, external), {("owned", "1.0")}
        )

    assert set(records) == {"external"}
    assert records["external"]["version"] == "2.0"
    assert len(first_digest) == 64
    assert changed["external"]["files_sha256"] != first_digest


def test_pip_lock_rejects_external_distribution_without_file_inventory():
    class Distribution:
        metadata = {"Name": "external"}
        version = "1.0"
        files = None

    try:
        experiment._pip_distribution_records((Distribution(),), set())
    except RuntimeError as error:
        assert "without a file inventory" in str(error)
    else:
        raise AssertionError("uninventoried external distribution was accepted")


def test_plain_test_runner_executes_every_defined_test_function():
    with tempfile.TemporaryDirectory() as directory:
        marker = os.path.join(directory, "calls.txt")
        test_file = os.path.join(directory, "test_sample.py")
        with open(test_file, "w") as stream:
            stream.write(
                "from pathlib import Path\n"
                f"marker = Path({marker!r})\n"
                "def test_first():\n"
                "    marker.write_text(marker.read_text() + 'first\\n' "
                "if marker.exists() else 'first\\n')\n"
                "def test_second():\n"
                "    marker.write_text(marker.read_text() + 'second\\n')\n"
                "def test_requires_fixture(value):\n"
                "    marker.write_text(marker.read_text() + str(value))\n"
                "if __name__ == '__main__':\n"
                "    test_first()\n"
            )
        completed = experiment._run_plain_test_file(
            test_file, directory, os.environ.copy()
        )
        with open(marker) as stream:
            calls = stream.read().splitlines()
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "2 tests passed"
    assert calls == ["first", "second"]


def test_campaign_dataset_integrity_rejects_group_leakage():
    config = ProtocolConfig.smoke(seed=12)
    records = [
        _record(case)
        for case in generate_cases(config)
        if case.split != "pilot"
    ]
    with tempfile.TemporaryDirectory() as directory:
        _, manifest_path = write_dataset(records, config, directory)
        with open(manifest_path) as stream:
            manifest = json.load(stream)
    validate_campaign_dataset(records, config, manifest)
    records[0]["base_gait_id"] = records[-1]["base_gait_id"]
    try:
        validate_campaign_dataset(records, config, manifest)
    except ValueError:
        pass
    else:
        raise AssertionError("cross-split base-gait leakage was accepted")


def test_case_runner_filters_splits_and_persists_once():
    config = ProtocolConfig.smoke(seed=99)
    calls = []

    def fake_runner(case, _config):
        calls.append((case.robot, case.split, case.base_gait_id))
        return _record(case, label=case.robot == "icub")

    with tempfile.TemporaryDirectory() as directory:
        records, csv_path, manifest_path = run_cases(
            config,
            directory,
            splits=("train", "test"),
            runner=fake_runner,
            workers=2,
        )
        assert os.path.exists(csv_path) and os.path.exists(manifest_path)
        resumed, _, _ = run_cases(
            config,
            directory,
            splits=("train", "test"),
            runner=lambda *_: (_ for _ in ()).throw(
                AssertionError("completed case must resume")
            ),
            workers=2,
        )
        checkpoints = [
            name
            for root, _, files in os.walk(os.path.join(directory, "case_records"))
            for name in files
        ]
    assert len(records) == len(calls) == 4
    assert resumed == records
    assert len(checkpoints) == len(records)
    assert {record["split"] for record in records} == {"train", "test"}


def test_screening_fingerprint_binds_transitive_source_files():
    config = ProtocolConfig.smoke(seed=7)
    original_file = experiment.__file__
    with tempfile.TemporaryDirectory() as directory:
        source = os.path.join(directory, "src")
        output = os.path.join(directory, "output")
        os.makedirs(source)
        os.makedirs(output)
        for name in experiment._SCIENTIFIC_SOURCE_FILES:
            with open(os.path.join(source, name), "w") as stream:
                stream.write(name)
        gait = os.path.join(source, "gait.py")
        with open(gait, "w") as stream:
            stream.write("first")
        experiment.__file__ = os.path.join(source, "experiment.py")
        try:
            first = experiment._screening_fingerprint(config, output)
            with open(gait, "w") as stream:
                stream.write("second")
            second = experiment._screening_fingerprint(config, output)
        finally:
            experiment.__file__ = original_file
    assert first != second


def test_pilot_balance_is_a_hard_gate_and_allows_at_most_one_range_revision():
    config = ProtocolConfig.smoke(seed=99)
    pilot_cases = [
        case for case in generate_cases(config) if case.split == "pilot"
    ]
    balanced_records = []
    for case in pilot_cases:
        for index, label in enumerate((0, 0, 1, 1)):
            balanced_records.append(_record(
                ExperimentCase(
                    case.robot,
                    case.split,
                    f"{case.base_gait_id}-{index}",
                    case.perturbation_index,
                    case.sample,
                ),
                label,
            ))
    summary = summarize_pilot(balanced_records, config)
    assert summary["protocol_frozen"] is True
    assert summary["status"] == "ready_for_dataset"
    assert_pilot_frozen(summary, config)

    unbalanced = summarize_pilot(
        [_record(case, 1) for case in pilot_cases], config
    )
    assert unbalanced["protocol_frozen"] is False
    assert unbalanced["status"] == "range_adjustment_required"
    try:
        assert_pilot_frozen(unbalanced, config)
    except RuntimeError:
        pass
    else:
        raise AssertionError("dataset gate accepted an unbalanced pilot")

    revised = summarize_pilot(
        [_record(case, 1) for case in pilot_cases],
        replace(config, pilot_revision=1),
    )
    assert revised["status"] == "validity_blocked"


def test_revised_pilot_requires_and_preserves_initial_evidence():
    initial = ProtocolConfig(seed=73, scientific_protocol=False)
    revised = replace(
        initial,
        pilot_revision=1,
        step_length_range=(0.20, 0.30),
    )
    with tempfile.TemporaryDirectory() as directory:
        revised_path = os.path.join(directory, "revised.json")
        with open(revised_path, "w") as stream:
            json.dump(asdict(revised), stream)
        try:
            cli([
                "pilot", "--protocol-json", revised_path,
                "--output-dir", directory,
            ], runner=lambda case, _: _record(case, 1))
        except RuntimeError:
            pass
        else:
            raise AssertionError("revised pilot ran without initial evidence")

    with tempfile.TemporaryDirectory() as directory:
        initial_path = os.path.join(directory, "initial.json")
        with open(initial_path, "w") as stream:
            json.dump(asdict(initial), stream)
        cli([
            "pilot", "--protocol-json", initial_path,
            "--output-dir", directory,
        ], runner=lambda case, _: _record(case, 1))
        revised_path = os.path.join(directory, "revised.json")
        with open(revised_path, "w") as stream:
            json.dump(asdict(revised), stream)
        cli([
            "pilot", "--protocol-json", revised_path,
            "--output-dir", directory,
        ], runner=lambda case, _: _record(
            case, int(case.base_gait_id.rsplit("-", 1)[-1]) % 2
        ))
        with open(os.path.join(directory, "pilot_summary_revision0.json")) as stream:
            first = json.load(stream)
        with open(os.path.join(directory, "pilot_summary_revision1.json")) as stream:
            second = json.load(stream)
    assert first["status"] == "range_adjustment_required"
    assert second["status"] == "ready_for_dataset"


def test_selective_metrics_use_failure_among_accepted_and_exact_upper_bound():
    metrics = calibration_metrics(
        labels=np.array([0, 1, 0, 1]),
        failure_scores=np.array([0.1, 0.2, 0.8, 0.9]),
        threshold=0.25,
        alpha=0.05,
    )
    assert metrics["accepted"] == 2
    assert metrics["coverage"] == 0.5
    assert metrics["false_safe_risk"] == 0.5
    assert metrics["false_safe_upper"] >= metrics["false_safe_risk"]
    assert _exact_acceptance_mask(np.array([0.1, 0.1, 0.1, 0.2]), 2).tolist() == [
        True, True, False, False
    ]


def test_paired_bootstrap_resamples_base_gaits_and_is_deterministic():
    labels = np.array([0, 1, 0, 1, 0, 1])
    groups = np.array(["a", "a", "b", "b", "c", "c"])
    proposed = np.array([0.1, 0.9] * 3)
    baseline = np.array([0.9, 0.1] * 3)
    first = paired_group_bootstrap(
        labels,
        proposed,
        {"baseline": baseline},
        proposed_threshold=0.5,
        groups=groups,
        repetitions=200,
        seed=7,
    )
    second = paired_group_bootstrap(
        labels,
        proposed,
        {"baseline": baseline},
        proposed_threshold=0.5,
        groups=groups,
        repetitions=200,
        seed=7,
    )
    assert first == second
    assert first["groups"] == 3
    assert first["baseline"] == "baseline"
    assert first["reselected_each_replicate"] is True
    assert first["risk_difference"] == 1.0
    assert first["ci95"] == [1.0, 1.0]


def test_model_matrix_contains_parameters_and_signature_but_no_robot_identity():
    cases = generate_cases(ProtocolConfig.smoke(seed=13))
    records = [_record(case, index % 2) for index, case in enumerate(cases)]
    arrays = records_to_arrays(records, "train")
    assert arrays.X.shape == (2, len(MODEL_PARAMETER_COLUMNS) + 2)
    assert arrays.X_parameters.shape == (2, len(MODEL_PARAMETER_COLUMNS))
    assert "robot" not in arrays.feature_names
    assert arrays.robots.tolist() == ["talos", "icub"]
    assert arrays.labels.tolist() == [1, 1]


def test_ablation_matrices_remove_only_the_prespecified_information():
    cases = generate_cases(ProtocolConfig.smoke(seed=13))
    records = [_record(case, index % 2) for index, case in enumerate(cases)]
    no_touchdown = ablation_arrays(records, "train", "no_touchdown")
    assert no_touchdown.X.shape[1] == len(MODEL_PARAMETER_COLUMNS) + 1
    assert not any("touchdown" in name for name in no_touchdown.feature_names)

    no_normalization = ablation_arrays(records, "train", "no_normalization")
    np.testing.assert_allclose(no_normalization.X[:, -2:], [[25.0, -0.05]] * 2)

    phased = []
    for record in records:
        expanded = dict(record)
        expanded["feature.double_support.torque.max"] = 0.75
        expanded["raw_feature.double_support.torque.max"] = 75.0
        phased.append(expanded)
    no_phase = ablation_arrays(phased, "train", "no_phase_separation")
    assert no_phase.X.shape[1] == len(MODEL_PARAMETER_COLUMNS) + 3
    assert all("phase_rank" in name for name in no_phase.feature_names[
        len(MODEL_PARAMETER_COLUMNS):
    ])
    np.testing.assert_allclose(no_phase.X[0, -3:], [0.25, 0.75, -0.5])


def test_evaluation_exposes_all_six_methods_and_calibration_metrics():
    labels = np.array([0, 1, 0, 1])
    perfect = np.array([0.0, 1.0, 0.0, 1.0])
    metrics = classification_metrics(labels, perfect, bins=2)
    assert metrics["pr_auc"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["brier"] == 0.0
    assert metrics["ece"] == 0.0

    methods = {
        "zmp_cop_margin": perfect,
        "ik_joint_margin": perfect,
        "inverse_dynamics_slack": perfect,
        "black_box_parameters": perfect,
        "uncalibrated_phase_sequence": perfect,
        "risk_calibrated_phase_sequence": perfect,
    }
    evaluated = evaluate_score_methods(
        labels, methods,
        thresholds={name: 0.5 for name in methods},
        runtimes=np.array([0.1, 0.2, 0.3, 0.4]),
        failure_reasons=np.array(["", "torque", "", "tracking"]),
    )
    assert set(evaluated) == set(methods)
    assert evaluated["zmp_cop_margin"]["brier"] is None
    assert evaluated["zmp_cop_margin"]["ece"] is None
    assert evaluated["risk_calibrated_phase_sequence"]["brier"] == 0.0
    assert evaluated["risk_calibrated_phase_sequence"]["ece"] == 0.0
    assert evaluated["risk_calibrated_phase_sequence"]["accepted"] == 2
    assert evaluated["risk_calibrated_phase_sequence"]["false_safe_risk"] == 0.0
    assert evaluated["risk_calibrated_phase_sequence"]["rollout_runtime_seconds"] == 1.0
    assert evaluated["risk_calibrated_phase_sequence"]["failure_breakdown"] == {
        "torque": 1, "tracking": 1
    }


def test_primary_matched_comparison_excludes_calibration_ablation():
    class ScoreModel:
        threshold_ = 0.5

        def predict_failure_score(self, X):
            return np.linspace(0.1, 0.9, len(X))

        def accept(self, X):
            return self.predict_failure_score(X) <= self.threshold_

    records = [
        _record(case, case.robot == "icub")
        for case in generate_cases(ProtocolConfig.smoke(seed=4))
    ]
    result = evaluate_records(
        records, ScoreModel(), ScoreModel(), split="test"
    )
    assert result["comparison_to_best_baseline"]["baseline"] in {
        "zmp_cop_margin",
        "ik_joint_margin",
        "inverse_dynamics_slack",
        "black_box_parameters",
    }
    assert result["comparison_to_best_baseline"]["baseline_candidates"] == 4
    assert result["comparison_to_best_baseline"][
        "reselected_each_replicate"
    ] is True
    assert result["selection_rules"]["matched_coverage"][
        "uncalibrated_phase_sequence"
    ]["kind"] == "native_threshold"
    assert result["matched_coverage"]["zmp_cop_margin"][
        "false_safe_upper"
    ] is None
    assert result["matched_coverage"]["zmp_cop_margin"][
        "confidence_bound_valid"
    ] is False
    assert result["matched_coverage"]["risk_calibrated_phase_sequence"][
        "false_safe_upper"
    ] is not None
    assert result["matched_coverage"]["risk_calibrated_phase_sequence"][
        "confidence_bound_valid"
    ] is True
    ood = evaluate_records(
        records, ScoreModel(), ScoreModel(), split="ood"
    )
    assert ood["prespecified"]["risk_calibrated_phase_sequence"][
        "false_safe_upper"
    ] is None
    assert ood["prespecified"]["risk_calibrated_phase_sequence"][
        "confidence_bound_valid"
    ] is False


def test_training_uses_frozen_train_tune_calibration_splits_and_saves_models():
    config = ProtocolConfig.smoke(seed=5)
    cases = generate_cases(config)
    records = [_record(case, label=case.robot == "icub") for case in cases]
    with tempfile.TemporaryDirectory() as directory:
        write_dataset(records, config, directory)
        surrogate, black_box, metadata_path = train_surrogates(
            records, directory, seed=5, config=config
        )
        assert os.path.exists(os.path.join(directory, "surrogate.pkl"))
        assert os.path.exists(os.path.join(directory, "black_box.pkl"))
        for name in (
            "no_normalization", "no_phase_separation", "no_touchdown"
        ):
            assert os.path.exists(os.path.join(
                directory, f"ablation_{name}.pkl"
            ))
        with open(metadata_path) as stream:
            metadata = json.load(stream)
        assert_training_provenance(directory, config)
        model_path = os.path.join(directory, "surrogate.pkl")
        with open(model_path, "rb") as stream:
            original_model = stream.read()
        with open(model_path, "ab") as stream:
            stream.write(b"tampered")
        try:
            assert_training_provenance(directory, config)
        except RuntimeError:
            pass
        else:
            raise AssertionError("modified model passed the training hash chain")
        with open(model_path, "wb") as stream:
            stream.write(original_model)
        provenance = assert_training_provenance(directory, config)
        robot_result = {
            "prespecified": {
                "risk_calibrated_phase_sequence": {
                    "false_safe_upper": 0.04,
                    "coverage": 0.25,
                }
            },
            "comparison_to_best_baseline": {
                "relative_risk_reduction": 0.40,
                "ci95": [0.01, 0.08],
            },
        }
        evaluation = {
            "provenance": {
                "training_fingerprint": provenance["training_fingerprint"]
            },
            "methods": {
                "test": {"talos": robot_result, "icub": robot_result}
            },
        }
        experiment._write_json(
            os.path.join(directory, "evaluation.json"),
            evaluation,
        )
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        test_sha256 = {
            test_file: experiment._file_sha256(os.path.join(root, test_file))
            for test_file in experiment._VALIDITY_TEST_FILES
        }
        validity_fingerprint = hashlib.sha256(json.dumps({
            "experiment_fingerprint": provenance["experiment_fingerprint"],
            "test_sha256": test_sha256,
        }, sort_keys=True).encode()).hexdigest()
        experiment._write_json(
            os.path.join(directory, "validity.json"),
            {
                "passed": True,
                "experiment_fingerprint": provenance["experiment_fingerprint"],
                "test_sha256": test_sha256,
                "validity_fingerprint": validity_fingerprint,
            },
        )
        screening_rows = [
            {
                "method": method,
                "robot": robot,
                "condition": condition,
                "repetition": repetition,
                "budget": budget,
                "success": True,
                "rollouts": (
                    2 if method == "risk_calibrated_phase_sequence" else 4
                ),
                "rollout_runtime_seconds": (
                    0.2 if method == "risk_calibrated_phase_sequence" else 0.4
                ),
                "failure_reason": "",
                "pool_size": 2048,
                "pool_seed": 1000 + repetition,
            }
            for robot in ("talos", "icub")
            for condition in ("id", "ood")
            for repetition in range(30)
            for budget in (5, 10, 20)
            for method in (
                "zmp_cop_margin",
                "ik_joint_margin",
                "inverse_dynamics_slack",
                "black_box_parameters",
                "uncalibrated_phase_sequence",
                "risk_calibrated_phase_sequence",
            )
        ]
        with open(
            os.path.join(directory, "screening.csv"), "w", newline=""
        ) as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(screening_rows[0])
            )
            writer.writeheader()
            writer.writerows(screening_rows)
        with open(
            os.path.join(directory, "screening_calls.csv"), "w"
        ) as stream:
            stream.write("method,success\nproposed,True\n")
        screening_summary = summarize_screening(screening_rows)
        experiment._write_json(
            os.path.join(directory, "screening_summary.json"),
            screening_summary,
        )
        screening_fingerprint = experiment._screening_fingerprint(
            config, directory, 1
        )
        terminal_files = {
            "evaluation_sha256": "evaluation.json",
            "validity_sha256": "validity.json",
            "screening_csv_sha256": "screening.csv",
            "screening_calls_sha256": "screening_calls.csv",
            "screening_summary_sha256": "screening_summary.json",
        }
        readiness = readiness_gate(
            evaluation, screening_summary, validity_passed=True
        )
        readiness["provenance"] = {
            "experiment_fingerprint": provenance[
                "experiment_fingerprint"
            ],
            "training_fingerprint": provenance["training_fingerprint"],
            "screening_fingerprint": screening_fingerprint,
            **{
                field: experiment._file_sha256(os.path.join(
                    directory, filename
                ))
                for field, filename in terminal_files.items()
            },
            "screening_workers": 1,
        }
        readiness_path = os.path.join(directory, "readiness.json")
        experiment._write_json(readiness_path, readiness)
        verify_campaign_provenance(directory, config)
        tampered_readiness = []
        for field, value in (
            ("ready", False),
            ("claim_mode", "benchmark_or_negative_result"),
        ):
            payload = json.loads(json.dumps(readiness))
            payload[field] = value
            tampered_readiness.append(payload)
        payload = json.loads(json.dumps(readiness))
        payload["robots"]["talos"]["passed"] = False
        tampered_readiness.append(payload)
        payload = json.loads(json.dumps(readiness))
        payload["forged_gate_field"] = True
        tampered_readiness.append(payload)
        payload = json.loads(json.dumps(readiness))
        payload.pop("claim_mode")
        tampered_readiness.append(payload)
        for payload in tampered_readiness:
            assert payload["provenance"] == readiness["provenance"]
            experiment._write_json(readiness_path, payload)
            try:
                verify_campaign_provenance(directory, config)
            except RuntimeError:
                pass
            else:
                raise AssertionError(
                    "tampered readiness gate passed provenance verification"
                )
        experiment._write_json(readiness_path, readiness)
        for filename in terminal_files.values():
            path = os.path.join(directory, filename)
            with open(path, "rb") as stream:
                original = stream.read()
            with open(path, "ab") as stream:
                stream.write(b"tampered")
            try:
                verify_campaign_provenance(directory, config)
            except RuntimeError:
                pass
            else:
                raise AssertionError(
                    f"modified terminal evidence passed: {filename}"
                )
            with open(path, "wb") as stream:
                stream.write(original)
    assert surrogate.n_features_in_ == len(MODEL_PARAMETER_COLUMNS) + 2
    assert black_box.n_features_in_ == len(MODEL_PARAMETER_COLUMNS)
    assert metadata["split_rows"] == {"train": 2, "tune": 2, "calibration": 2}
    assert metadata["protocol"]["seed"] == 5
    assert set(metadata["ablations"]) == {
        "no_normalization", "no_phase_separation", "no_touchdown",
        "no_risk_calibration",
    }
    assert set(score_methods(records, surrogate, black_box, "test")) == {
        "zmp_cop_margin",
        "ik_joint_margin",
        "inverse_dynamics_slack",
        "black_box_parameters",
        "uncalibrated_phase_sequence",
        "risk_calibrated_phase_sequence",
    }


class _AcceptFirst:
    def accept(self, X):
        return np.arange(len(X)) == 0


def test_downstream_success_is_never_written_without_oracle_verification():
    X = np.array([[0.1], [0.2]])
    candidates = ["first", "second"]
    failed_calls = []

    def failed_oracle(candidate):
        failed_calls.append(candidate)
        return RolloutResult(
            np.array([0.0]), np.zeros((1, 1)), np.zeros((1, 1)),
            np.zeros((1, 2, 6)), False, "tracking",
        )

    failed = verify_ranked_candidates(
        _AcceptFirst(), X, candidates, budget=1, oracle=failed_oracle
    )
    assert failed_calls == ["first"]
    assert failed["success"] is False
    assert failed["verified_by_rollout"] is True
    assert failed["failure_reason"] == "tracking"

    passed = verify_ranked_candidates(
        _AcceptFirst(), X, candidates, budget=1,
        oracle=lambda candidate: RolloutResult(
            np.array([0.0]), np.zeros((1, 1)), np.zeros((1, 1)),
            np.zeros((1, 2, 6)), True, "",
        ),
    )
    assert passed["success"] is True
    assert passed["verified_by_rollout"] is True


def test_screening_uses_same_deterministic_pool_and_requested_budgets():
    first = generate_candidate_pool("talos", seed=31, count=8, ood=False)
    second = generate_candidate_pool("talos", seed=31, count=8, ood=False)
    assert first == second and len(first) == 8
    assert len(generate_candidate_pool("icub", seed=31, count=2048, ood=True)) == 2048

    labels = {case.base_gait_id: index == 2 for index, case in enumerate(first)}

    def oracle(case):
        return RolloutResult(
            np.array([0.0]), np.zeros((1, 1)), np.zeros((1, 1)),
            np.zeros((1, 2, 6)), labels[case.base_gait_id],
            "" if labels[case.base_gait_id] else "tracking", runtime=0.2,
        )

    rows = screening_experiment(
        {"a": np.arange(8), "b": np.array([7, 6, 0, 5, 4, 3, 2, 1])},
        first,
        budgets=(1, 3),
        oracle=oracle,
    )
    assert len(rows) == 4
    assert next(row for row in rows if row["method"] == "a" and row["budget"] == 3)["success"]
    assert not next(row for row in rows if row["method"] == "a" and row["budget"] == 1)["success"]


def test_screening_summary_preserves_paired_robot_condition_and_budget_cells():
    rows = [
        {
            "method": method,
            "robot": "talos",
            "condition": "id",
            "repetition": repetition,
            "budget": 5,
            "success": method == "proposed" or repetition == 0,
            "rollouts": 2 + repetition,
            "rollout_runtime_seconds": 0.4 + repetition,
            "failure_reason": "",
            "pool_size": 2048,
            "pool_seed": 700 + repetition,
        }
        for method in ("baseline", "proposed")
        for repetition in range(2)
    ]
    summary = summarize_screening(rows)
    proposed = next(cell for cell in summary if cell["method"] == "proposed")
    baseline = next(cell for cell in summary if cell["method"] == "baseline")
    assert proposed["repetitions"] == baseline["repetitions"] == 2
    assert proposed["success_rate"] == 1.0
    assert baseline["success_rate"] == 0.5
    assert proposed["mean_rollouts"] == baseline["mean_rollouts"] == 2.5
    assert proposed.get("rollouts_by_repetition") == {"0": 2, "1": 3}
    assert proposed["pool_size"] == baseline["pool_size"] == 2048

    rows[-1]["pool_seed"] += 1
    try:
        summarize_screening(rows)
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched paired pool seeds must be rejected")


def test_screening_campaign_runs_all_paired_cells_and_persists_raw_and_summary():
    class ScoreModel:
        threshold_ = 0.5

        def predict_failure_score(self, X):
            return np.linspace(0.1, 0.9, len(X))

        def accept(self, X):
            return self.predict_failure_score(X) <= self.threshold_

    config = replace(
        ProtocolConfig.smoke(seed=123),
        screening_repetitions=2,
        rollout_budgets=(1, 2),
    )

    def feature_runner(case, _config):
        return _record(case)

    def oracle(case):
        success = int(case.base_gait_id.rsplit("-", 1)[-1]) % 3 == 0
        return RolloutResult(
            np.array([0.0]), np.zeros((1, 1)), np.zeros((1, 1)),
            np.zeros((1, 2, 6)), success,
            "" if success else "tracking", runtime=0.2,
        )

    with tempfile.TemporaryDirectory() as directory:
        rows, csv_path, summary_path = run_screening_campaign(
            config,
            ScoreModel(),
            ScoreModel(),
            directory,
            feature_runner=feature_runner,
            oracle=oracle,
            workers=2,
        )
        checkpoints = os.listdir(os.path.join(directory, "screening_cells"))
        assert len(checkpoints) == 2 * 2 * 2

        def must_not_run(*_args, **_kwargs):
            raise AssertionError("completed screening cells must resume")

        resumed, _, _ = run_screening_campaign(
            config,
            ScoreModel(),
            ScoreModel(),
            directory,
            feature_runner=must_not_run,
            oracle=must_not_run,
            workers=2,
        )
        with open(csv_path, newline="") as stream:
            persisted = list(csv.DictReader(stream))
        with open(summary_path) as stream:
            summary = json.load(stream)
        with open(os.path.join(directory, "screening_calls.csv"), newline="") as stream:
            calls = list(csv.DictReader(stream))
    assert resumed == rows
    assert len(rows) == len(persisted) == 2 * 2 * 2 * 6 * 2
    assert len(summary) == 2 * 2 * 6 * 2
    assert {row["condition"] for row in rows} == {"id", "ood"}
    assert {row["robot"] for row in rows} == {"talos", "icub"}
    assert all(row["pool_size"] == config.candidate_pool_size for row in rows)
    assert calls
    assert all(call["candidate_id"] and call["rank"] for call in calls)
    assert "actual_oracle_wall_seconds" in calls[0]
    cache_hits = [call for call in calls if call["cache_hit"] == "True"]
    assert cache_hits
    assert all(float(call["oracle_wall_seconds"]) > 0.0 for call in cache_hits)
    assert all(
        float(call["actual_oracle_wall_seconds"]) == 0.0
        for call in cache_hits
    )
    assert all(
        cell["mean_actual_oracle_wall_seconds"]
        <= cell["mean_oracle_wall_seconds"]
        for cell in summary
    )


def test_readiness_gate_requires_validity_risk_coverage_bootstrap_and_downstream():
    robot_result = {
        "prespecified": {
            "risk_calibrated_phase_sequence": {
                "false_safe_upper": 0.04,
                "coverage": 0.25,
            }
        },
        "comparison_to_best_baseline": {
            "relative_risk_reduction": 0.40,
            "ci95": [0.01, 0.08],
        },
    }
    evaluation = {
        "methods": {
            "test": {"talos": robot_result, "icub": robot_result}
        }
    }
    screening = [
        {
            "method": method,
            "robot": robot,
            "condition": condition,
            "budget": budget,
            "success_rate": (
                1.0
                if method == "risk_calibrated_phase_sequence"
                else 0.50
            ),
            "success_by_repetition": {
                str(index): (
                    method == "risk_calibrated_phase_sequence" or index < 15
                )
                for index in range(30)
            },
            "mean_rollouts": float(budget),
            "repetitions": 30,
            "pool_size": 2048,
        }
        for robot in ("talos", "icub")
        for condition in ("id", "ood")
        for budget in (5, 10, 20)
        for method in (
            "zmp_cop_margin",
            "ik_joint_margin",
            "inverse_dynamics_slack",
            "black_box_parameters",
            "uncalibrated_phase_sequence",
            "risk_calibrated_phase_sequence",
        )
    ]
    passed = readiness_gate(evaluation, screening, validity_passed=True)
    assert passed["ready"] is True
    assert not readiness_gate(
        evaluation, screening, validity_passed=False
    )["ready"]
    screening[0]["repetitions"] = 29
    assert not readiness_gate(
        evaluation, screening, validity_passed=True
    )["ready"]


def test_readiness_rejects_accept_none_as_rollout_reduction():
    robot_result = {
        "prespecified": {
            "risk_calibrated_phase_sequence": {
                "false_safe_upper": 0.04,
                "coverage": 0.25,
            }
        },
        "comparison_to_best_baseline": {
            "relative_risk_reduction": 0.40,
            "ci95": [0.01, 0.08],
        },
    }
    evaluation = {
        "methods": {"test": {"talos": robot_result, "icub": robot_result}}
    }
    screening = [
        {
            "method": method,
            "robot": robot,
            "condition": condition,
            "budget": budget,
            "success_rate": (
                0.0 if method == "risk_calibrated_phase_sequence" else 1.0
            ),
            "success_by_repetition": {
                str(index): method != "risk_calibrated_phase_sequence"
                for index in range(30)
            },
            "mean_rollouts": (
                0.0 if method == "risk_calibrated_phase_sequence" else float(budget)
            ),
            "repetitions": 30,
            "pool_size": 2048,
        }
        for robot in ("talos", "icub")
        for condition in ("id", "ood")
        for budget in (5, 10, 20)
        for method in (
            "zmp_cop_margin",
            "ik_joint_margin",
            "inverse_dynamics_slack",
            "black_box_parameters",
            "uncalibrated_phase_sequence",
            "risk_calibrated_phase_sequence",
        )
    ]
    result = readiness_gate(evaluation, screening, validity_passed=True)
    assert result["ready"] is False
    assert not result["robots"]["talos"]["downstream"]["passed"]
    for row in screening:
        row["success_rate"] = 0.0
    assert not readiness_gate(
        evaluation, screening, validity_passed=True
    )["ready"]


def test_readiness_uses_paired_actual_calls_for_rollout_reduction():
    robot_result = {
        "prespecified": {
            "risk_calibrated_phase_sequence": {
                "false_safe_upper": 0.04,
                "coverage": 0.25,
            }
        },
        "comparison_to_best_baseline": {
            "relative_risk_reduction": 0.40,
            "ci95": [0.01, 0.08],
        },
    }
    evaluation = {
        "methods": {"test": {"talos": robot_result, "icub": robot_result}}
    }
    screening = [
        {
            "method": method,
            "robot": robot,
            "condition": condition,
            "budget": budget,
            "success_rate": 1.0,
            "success_by_repetition": {
                str(index): True for index in range(30)
            },
            "mean_rollouts": (
                float(budget)
                if method == "risk_calibrated_phase_sequence" else 1.0
            ),
            "rollouts_by_repetition": {
                str(index): (
                    budget
                    if method == "risk_calibrated_phase_sequence" else 1
                )
                for index in range(30)
            },
            "repetitions": 30,
            "pool_size": 2048,
        }
        for robot in ("talos", "icub")
        for condition in ("id", "ood")
        for budget in (5, 10, 20)
        for method in (
            "zmp_cop_margin",
            "ik_joint_margin",
            "inverse_dynamics_slack",
            "black_box_parameters",
            "uncalibrated_phase_sequence",
            "risk_calibrated_phase_sequence",
        )
    ]
    assert not readiness_gate(
        evaluation, screening, validity_passed=True
    )["ready"]

    for row in screening:
        if (
            row["method"] == "risk_calibrated_phase_sequence"
            and row["budget"] == 5
        ):
            row["mean_rollouts"] = 2.0
            row["rollouts_by_repetition"] = {
                str(index): 2 for index in range(30)
            }
        elif (
            row["method"] != "risk_calibrated_phase_sequence"
            and row["budget"] == 20
        ):
            row["mean_rollouts"] = 4.0
            row["rollouts_by_repetition"] = {
                str(index): 4 for index in range(30)
            }
    result = readiness_gate(evaluation, screening, validity_passed=True)
    assert result["ready"] is True
    assert result["robots"]["talos"]["downstream"][
        "rollout_reduction"
    ] == 0.5


def test_smoke_cli_writes_outputs_but_not_unverified_success():
    def fake_runner(case, _config):
        return _record(case, label=case.robot == "icub")

    with tempfile.TemporaryDirectory() as directory:
        assert cli(["smoke", "--output-dir", directory, "--seed", "19"], runner=fake_runner) == 0
        for name in (
            "dataset.csv", "manifest.json", "surrogate.pkl", "black_box.pkl",
            "training.json", "evaluation.json", "smoke.json",
        ):
            assert os.path.exists(os.path.join(directory, name)), name
        with open(os.path.join(directory, "smoke.json")) as stream:
            smoke = json.load(stream)
    assert smoke["success"] is False
    assert smoke["verified_by_rollout"] is False
    assert smoke["failure_reason"] == "no_accepted_candidate"


if __name__ == "__main__":
    test_scrambled_sobol_protocol_is_deterministic_and_grouped()
    test_grouped_sobol_splits_do_not_index_couple_gait_and_perturbations()
    test_candidate_pool_does_not_index_couple_gait_and_perturbations()
    test_cross_design_gate_checks_each_grouped_perturbation_slot()
    test_scientific_case_design_rejects_index_coupled_streams()
    test_scientific_candidate_pool_rejects_index_coupled_streams()
    test_default_protocol_has_exact_counts_and_iid_inference_units()
    test_protocol_json_supports_one_auditable_pilot_range_revision()
    test_manifest_and_flat_csv_capture_required_schema()
    test_environment_lock_records_robot_model_hashes()
    test_environment_lock_writes_exact_pip_requirements()
    test_pip_lock_excludes_conda_owned_distribution_identity()
    test_pip_lock_rejects_external_distribution_without_file_inventory()
    test_plain_test_runner_executes_every_defined_test_function()
    test_campaign_dataset_integrity_rejects_group_leakage()
    test_case_runner_filters_splits_and_persists_once()
    test_screening_fingerprint_binds_transitive_source_files()
    test_pilot_balance_is_a_hard_gate_and_allows_at_most_one_range_revision()
    test_revised_pilot_requires_and_preserves_initial_evidence()
    test_selective_metrics_use_failure_among_accepted_and_exact_upper_bound()
    test_paired_bootstrap_resamples_base_gaits_and_is_deterministic()
    test_model_matrix_contains_parameters_and_signature_but_no_robot_identity()
    test_ablation_matrices_remove_only_the_prespecified_information()
    test_evaluation_exposes_all_six_methods_and_calibration_metrics()
    test_primary_matched_comparison_excludes_calibration_ablation()
    test_training_uses_frozen_train_tune_calibration_splits_and_saves_models()
    test_downstream_success_is_never_written_without_oracle_verification()
    test_screening_uses_same_deterministic_pool_and_requested_budgets()
    test_screening_summary_preserves_paired_robot_condition_and_budget_cells()
    test_screening_campaign_runs_all_paired_cells_and_persists_raw_and_summary()
    test_readiness_gate_requires_validity_risk_coverage_bootstrap_and_downstream()
    test_readiness_rejects_accept_none_as_rollout_reduction()
    test_readiness_uses_paired_actual_calls_for_rollout_reduction()
    test_smoke_cli_writes_outputs_but_not_unverified_success()
    print("experiment tests passed")
