"""Build the reproducibility artifact ZIP from an explicit allowlist."""

import argparse
import glob
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
import zipfile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiment import ProtocolConfig, verify_campaign_provenance
from artifact.generate_figures import (
    FIGURE_FILENAMES,
    generate_figures,
)
from artifact.analyze_confirmation import (
    FIGURES as CONFIRMATION_FIGURES,
    generate_confirmation_figures,
)


MAX_BYTES = 50_000_000
TEMPLATE_MARKER = b"TEMPLATE - DO NOT SUBMIT"
DEVELOPMENT_PATHS = (
    REPO_ROOT
    / "results/development-seed12026"
    / "dataset.csv",
    REPO_ROOT
    / "results/domain-gate-seed22026"
    / "dataset.csv",
)
CONFIRMATION_SEEDS = (42026, 52026)


def _confirmation_dirs(results_dir):
    return [Path(f"{results_dir}-seed{seed}") for seed in CONFIRMATION_SEEDS]


def _validate_archive_path(name):
    path = PurePosixPath(name)
    if (
        not name
        or "\x00" in name
        or "\\" in name
        or path.is_absolute()
        or ".." in path.parts
        or re.match(r"^[A-Za-z]:", name)
    ):
        raise RuntimeError(f"unsafe archive path: {name!r}")


def _contains_identifying_path(data):
    unix_home = rb"(?:/" + rb"Users/|/" + rb"home/)" + rb"[^/\s]+/"
    windows_home = rb"[A-Za-z]:\\" + rb"Users\\" + rb"[^\\\s]+\\"
    return bool(
        str(Path.home()).encode() in data
        or re.search(unix_home, data)
        or re.search(windows_home, data)
    )


def _entries(repo_root, results_dir, manifest_path):
    repo_root, results_dir = Path(repo_root).resolve(), Path(results_dir).resolve()
    entries = {}
    for line in Path(manifest_path).read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        archive_name, source_pattern = line.split("\t", 1)
        optional = archive_name.startswith("?")
        archive_name = archive_name.removeprefix("?")
        _validate_archive_path(archive_name)
        source_pattern = source_pattern.replace("{results}", str(results_dir))
        for seed, directory in zip(
            CONFIRMATION_SEEDS, _confirmation_dirs(results_dir)
        ):
            source_pattern = source_pattern.replace(
                f"{{confirmation{seed}}}", str(directory)
            )
        if not os.path.isabs(source_pattern):
            source_pattern = str(repo_root / source_pattern)
        matches = sorted(glob.glob(source_pattern))
        if not matches and optional:
            continue
        if not matches:
            raise RuntimeError(f"required artifact input is missing: {source_pattern}")
        for source in matches:
            source = Path(source)
            resolved = source.resolve()
            if source.is_symlink() or not resolved.is_file() or not any(
                resolved.is_relative_to(root) for root in (repo_root, results_dir)
            ):
                raise RuntimeError(f"unsafe artifact source: {source}")
            target = (
                archive_name + source.name
                if archive_name.endswith("/") else archive_name
            )
            _validate_archive_path(target)
            if target in entries:
                raise RuntimeError(f"duplicate archive path: {target}")
            entries[target] = resolved
    return entries


def _verify_results(results_dir):
    if (Path(results_dir) / "confirmation_analysis.json").is_file():
        _verify_confirmation_results(
            results_dir, DEVELOPMENT_PATHS, _confirmation_dirs(results_dir)
        )
        return
    config = ProtocolConfig.from_json(str(Path(results_dir) / "protocol.json"))
    verify_campaign_provenance(str(results_dir), config)


def _verify_confirmation_results(results_dir, development_paths, confirmation_dirs):
    results_dir = Path(results_dir)
    development_paths = [Path(path) for path in development_paths]
    with open(results_dir / "confirmation_analysis.json") as stream:
        analysis = json.load(stream)
    confirmation_dirs = [Path(path) for path in confirmation_dirs]
    counts = analysis.get("rows", {})
    primary = analysis.get("primary", {})
    deltas = primary.get("delta_by_confirmation", {})
    if (
        len(confirmation_dirs) != 2
        or counts.get("confirmation") != 800
        or counts.get("confirmation_by_robot") != {"icub": 400, "talos": 400}
        or counts.get("confirmation_by_scramble") != {
            "seed42026": 400,
            "seed52026": 400,
        }
        or primary.get("bootstrap_repetitions") != 10000
        or primary.get("passed") is not (
            primary.get("ci95", [0])[0] > 0
            and all(deltas.get(f"seed{seed}", 0) > 0 for seed in CONFIRMATION_SEEDS)
        )
    ):
        raise RuntimeError("confirmation evidence is incomplete or inconsistent")
    for seed, directory in zip(CONFIRMATION_SEEDS, confirmation_dirs):
        with open(directory / "protocol.json") as stream:
            protocol = json.load(stream)
        with open(directory / "confirmation_summary.json") as stream:
            summary = json.load(stream)
        with open(directory / "validity.json") as stream:
            validity = json.load(stream)
        robots = summary.get("robots", {})
        if (
            protocol.get("seed") != seed
            or not summary.get("confirmation_not_tunable")
            or any(
                robots.get(robot, {}).get("rollouts") != 200
                for robot in ("talos", "icub")
            )
            or not validity.get("passed")
        ):
            raise RuntimeError("confirmation evidence is incomplete or inconsistent")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    hashes = analysis.get("data_sha256", {})
    if (
        hashes.get("confirmation") != [
            digest(path / "dataset.csv") for path in confirmation_dirs
        ]
        or hashes.get("development") != [
            digest(path) for path in development_paths
        ]
    ):
        raise RuntimeError("confirmation or development data hash differs")


def _verify_figures(results_dir, entries):
    packaged = {
        PurePosixPath(name).name: source
        for name, source in entries.items()
        if PurePosixPath(name).parent.name == "figures"
    }
    confirmation = Path(results_dir) / "confirmation_analysis.json"
    if confirmation.is_file():
        if set(packaged) != set(CONFIRMATION_FIGURES):
            raise RuntimeError(
                "artifact confirmation figure filenames do not match"
            )
        with confirmation.open() as stream:
            result = json.load(stream)
        with tempfile.TemporaryDirectory(
            prefix=".confirmation-figure-verification-"
        ) as directory:
            regenerated = generate_confirmation_figures(result, directory)
            if any(
                path.read_bytes() != packaged[path.name].read_bytes()
                for path in regenerated
            ):
                raise RuntimeError(
                    "packaged figures differ from regenerated results"
                )
        return
    if set(packaged) != set(FIGURE_FILENAMES):
        raise RuntimeError("artifact figure filenames do not match the frozen set")
    with tempfile.TemporaryDirectory(prefix=".figure-verification-") as directory:
        regenerated = generate_figures(
            results_dir, Path(directory) / "figures"
        )
        if any(
            path.read_bytes() != packaged[path.name].read_bytes()
            for path in regenerated
        ):
            raise RuntimeError("packaged figures differ from regenerated results")


def build_package(
    repo_root,
    results_dir,
    manifest_path,
    output_path,
    provenance_verifier=_verify_results,
    figure_verifier=_verify_figures,
):
    repo_root, results_dir = Path(repo_root), Path(results_dir)
    provenance_verifier(results_dir)
    entries = _entries(repo_root, results_dir, manifest_path)
    expected_figures = (
        len(CONFIRMATION_FIGURES)
        if (results_dir / "confirmation_analysis.json").is_file()
        else len(FIGURE_FILENAMES)
    )
    if (
        sum(PurePosixPath(name).parent.name == "figures" for name in entries)
        != expected_figures
    ):
        raise RuntimeError(
            f"artifact requires exactly {expected_figures} generated figures"
        )
    if sum(PurePosixPath(name).parent.name == "video" for name in entries) != 1:
        raise RuntimeError("artifact requires exactly one video")
    for name in ("ReadMe.txt", "Summary.txt"):
        if TEMPLATE_MARKER in entries[name].read_bytes():
            raise RuntimeError(f"{name} is still a pre-campaign template")
    if not (results_dir / "confirmation_analysis.json").is_file():
        with open(results_dir / "validity.json") as stream:
            if not json.load(stream).get("passed"):
                raise RuntimeError("physical validity gate did not pass")
        with open(results_dir / "pilot_summary.json") as stream:
            if not json.load(stream).get("protocol_frozen"):
                raise RuntimeError("pilot protocol is not frozen")
    figure_verifier(results_dir, entries)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        with zipfile.ZipFile(
            temporary_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for name, source in sorted(entries.items()):
                data = source.read_bytes()
                if _contains_identifying_path(data):
                    raise RuntimeError(
                        f"identifying absolute path found in {source}"
                    )
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                archive.writestr(info, data, compresslevel=9)
        if temporary_path.stat().st_size > MAX_BYTES:
            raise RuntimeError("artifact exceeds the 50 MB release limit")
        with zipfile.ZipFile(temporary_path) as archive:
            if archive.testzip() is not None or archive.namelist() != sorted(entries):
                raise RuntimeError("artifact ZIP verification failed")
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return output_path


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        default="results/confirmation",
    )
    parser.add_argument(
        "--output",
        default=(
            "results/confirmation/"
            "reproducibility_artifact.zip"
        ),
    )
    args = parser.parse_args(argv)
    repo_root = REPO_ROOT
    manifest = Path(__file__).with_name("package_manifest.txt")
    path = build_package(repo_root, args.results_dir, manifest, args.output)
    print(path)


if __name__ == "__main__":
    main()
