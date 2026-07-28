import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from artifact.package_artifact import (
    MAX_BYTES,
    _contains_identifying_path,
    _verify_confirmation_results,
    _verify_figures,
    build_package,
)
from artifact.generate_figures import generate_figures
from tests.test_generate_figures import _write_valid_fixture
from tests.test_confirmation_analysis import _fixture
from artifact.analyze_confirmation import (
    analyze,
    generate_confirmation_figures,
)


def _required_inputs(root, readme="requirements"):
    results = root / "results"
    figures, video = results / "figures", results / "video"
    figures.mkdir(parents=True)
    video.mkdir()
    (root / "ReadMe.txt").write_text(readme)
    (root / "Summary.txt").write_text("contents")
    (results / "validity.json").write_text(json.dumps({"passed": True}))
    (results / "pilot_summary.json").write_text(
        json.dumps({"protocol_frozen": True})
    )
    for index in range(3):
        (figures / f"figure{index}.png").write_bytes(bytes((index,)))
    (video / "rollout.mp4").write_bytes(b"video")
    return results


def _required_manifest(extra=""):
    return (
        "ReadMe.txt\tReadMe.txt\n"
        "Summary.txt\tSummary.txt\n"
        "results/experiment/figures/\t{results}/figures/*\n"
        "results/experiment/video/\t{results}/video/*\n"
        + extra
    )


def test_artifact_package_is_relative_verified_and_deterministic():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        results = _required_inputs(root)
        manifest = root / "manifest.txt"
        manifest.write_text(_required_manifest())
        first, second = root / "first.zip", root / "second.zip"
        skip_provenance = lambda _: None
        build_package(
            root, results, manifest, first,
            provenance_verifier=skip_provenance,
            figure_verifier=lambda *_: None,
        )
        build_package(
            root, results, manifest, second,
            provenance_verifier=skip_provenance,
            figure_verifier=lambda *_: None,
        )
        assert hashlib.sha256(first.read_bytes()).digest() == hashlib.sha256(
            second.read_bytes()
        ).digest()
        with zipfile.ZipFile(first) as archive:
            assert archive.namelist() == sorted(archive.namelist())
            assert all(not name.startswith("/") for name in archive.namelist())


def test_confirmation_package_verifies_counts_and_hashes():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        results = root / "confirmation"
        results.mkdir()
        development = [root / "dev1.csv", root / "dev2.csv"]
        for index, path in enumerate(development):
            path.write_text(f"development {index}\n")
        confirmations = [
            root / "confirmation-seed42026",
            root / "confirmation-seed52026",
        ]
        digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        for seed, confirmation in zip((42026, 52026), confirmations):
            confirmation.mkdir()
            (confirmation / "dataset.csv").write_text(f"confirmation {seed}\n")
            (confirmation / "protocol.json").write_text(json.dumps({"seed": seed}))
            (confirmation / "validity.json").write_text(json.dumps({"passed": True}))
            (confirmation / "confirmation_summary.json").write_text(json.dumps({
                "confirmation_not_tunable": True,
                "robots": {
                    "talos": {"rollouts": 200},
                    "icub": {"rollouts": 200},
                },
            }))
        (results / "confirmation_analysis.json").write_text(json.dumps({
            "rows": {
                "confirmation": 800,
                "confirmation_by_robot": {"talos": 400, "icub": 400},
                "confirmation_by_scramble": {
                    "seed42026": 400,
                    "seed52026": 400,
                },
            },
            "data_sha256": {
                "confirmation": [
                    digest(path / "dataset.csv") for path in confirmations
                ],
                "development": [digest(path) for path in development],
            },
            "primary": {
                "bootstrap_repetitions": 10000,
                "passed": False,
                "ci95": [-0.01, 0.10],
                "delta_by_confirmation": {
                    "seed42026": 0.01,
                    "seed52026": 0.02,
                },
            },
        }))
        _verify_confirmation_results(results, development, confirmations)
        dataset = confirmations[0] / "dataset.csv"
        dataset.write_text("changed\n")
        try:
            _verify_confirmation_results(results, development, confirmations)
        except RuntimeError as error:
            assert "hash" in str(error)
        else:
            raise AssertionError("changed confirmation CSV passed verification")


def test_artifact_package_rejects_parent_archive_path():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        results = _required_inputs(root)
        source = root / "escape.txt"
        source.write_text("payload")
        manifest = root / "manifest.txt"
        manifest.write_text(_required_manifest("../escape.txt\tescape.txt\n"))

        try:
            build_package(
                root, results, manifest, root / "artifact.zip",
                provenance_verifier=lambda _: None,
                figure_verifier=lambda *_: None,
            )
        except RuntimeError as error:
            assert "archive path" in str(error)
        else:
            raise AssertionError("parent archive path was accepted")


def test_artifact_package_rejects_source_symlink_outside_roots():
    with (
        tempfile.TemporaryDirectory() as directory,
        tempfile.TemporaryDirectory() as outside_directory,
    ):
        root = Path(directory)
        results = _required_inputs(root)
        outside = Path(outside_directory) / "secret.txt"
        outside.write_text("payload")
        link = root / "secret.txt"
        link.symlink_to(outside)
        manifest = root / "manifest.txt"
        manifest.write_text(_required_manifest("secret.txt\tsecret.txt\n"))

        try:
            build_package(
                root, results, manifest, root / "artifact.zip",
                provenance_verifier=lambda _: None,
                figure_verifier=lambda *_: None,
            )
        except RuntimeError as error:
            assert "source" in str(error)
        else:
            raise AssertionError("source symlink was accepted")


def test_artifact_package_preserves_existing_output_on_late_failure():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        results = _required_inputs(root, "/" + "Users/alice/private")
        manifest = root / "manifest.txt"
        manifest.write_text(_required_manifest())
        output = root / "artifact.zip"
        output.write_bytes(b"known-good")

        try:
            build_package(
                root, results, manifest, output,
                provenance_verifier=lambda _: None,
                figure_verifier=lambda *_: None,
            )
        except RuntimeError as error:
            assert "identifying absolute path" in str(error)
        else:
            raise AssertionError("identifying path was accepted")
        assert output.read_bytes() == b"known-good"


def test_identity_path_scanner_does_not_match_its_packaged_sources():
    root = Path(__file__).resolve().parents[1]
    for path in (
        root / "artifact/package_artifact.py",
        Path(__file__),
    ):
        assert not _contains_identifying_path(path.read_bytes()), path
    assert _contains_identifying_path(b"/" + b"Users/alice/private")
    assert _contains_identifying_path(b"/" + b"home/alice/private")
    assert _contains_identifying_path(b"C:" + b"\\Users\\alice\\private")


def test_figure_verifier_rejects_stale_packaged_png():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        results = root / "results"
        _write_valid_fixture(results)
        paths = generate_figures(results)
        entries = {
            f"results/experiment/figures/{path.name}": path for path in paths
        }
        _verify_figures(results, entries)
        paths[0].write_bytes(b"stale")

        try:
            _verify_figures(results, entries)
        except RuntimeError as error:
            assert "regenerated" in str(error)
        else:
            raise AssertionError("stale figure passed provenance verification")


def test_figure_verifier_accepts_only_regenerated_confirmation_pdfs():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        development, confirmation = _fixture(root)
        result = analyze(development, confirmation, repetitions=20)
        results = root / "results"
        results.mkdir()
        (results / "confirmation_analysis.json").write_text(json.dumps(result))
        paths = generate_confirmation_figures(result, root / "figures")
        entries = {f"figures/{path.name}": path for path in paths}
        _verify_figures(results, entries)
        paths[-1].write_bytes(b"stale")
        try:
            _verify_figures(results, entries)
        except RuntimeError as error:
            assert "regenerated" in str(error)
        else:
            raise AssertionError("stale confirmation figure passed verification")


def test_release_manifest_preserves_runnable_repository_layout():
    assert MAX_BYTES == 50_000_000
    root = Path(__file__).resolve().parents[1]
    lines = (
        root / "artifact/package_manifest.txt"
    ).read_text().splitlines()
    entries = {}
    for line in lines:
        if line and not line.startswith("#"):
            target, source = line.split("\t", 1)
            entries[source] = target.removeprefix("?")

    for script in (
        "analyze_confirmation.py",
        "generate_rollout_video.py",
        "package_artifact.py",
        "run_confirmation.py",
    ):
        source = f"artifact/{script}"
        assert entries[source] == source
    assert entries["{confirmation42026}/dataset.csv"] == (
        "data/confirmation_seed42026/dataset.csv"
    )
    assert entries["{confirmation52026}/dataset.csv"] == (
        "data/confirmation_seed52026/dataset.csv"
    )
    assert entries["{results}/confirmation_analysis.json"] == (
        "confirmation_analysis.json"
    )
    assert entries["{confirmation42026}/video/rollout_examples.mp4"] == (
        "video/rollout_examples.mp4"
    )
    assert sum(target.startswith("video/") for target in entries.values()) == 1
    assert not any(
        token in source.lower()
        for source in entries
        for token in (".urdf", ".dae", ".stl", ".obj")
    )


def test_frozen_confirmation_commands_are_documented():
    root = Path(__file__).resolve().parents[1]
    commands = (root / "artifact/README.md").read_text()
    assert "run_confirmation" in commands
    assert "--workers 4" in commands
    assert "analyze_confirmation" in commands


if __name__ == "__main__":
    test_artifact_package_is_relative_verified_and_deterministic()
    test_confirmation_package_verifies_counts_and_hashes()
    test_artifact_package_rejects_parent_archive_path()
    test_artifact_package_rejects_source_symlink_outside_roots()
    test_artifact_package_preserves_existing_output_on_late_failure()
    test_identity_path_scanner_does_not_match_its_packaged_sources()
    test_figure_verifier_rejects_stale_packaged_png()
    test_figure_verifier_accepts_only_regenerated_confirmation_pdfs()
    test_release_manifest_preserves_runnable_repository_layout()
    test_frozen_confirmation_commands_are_documented()
    print("artifact package test passed")
