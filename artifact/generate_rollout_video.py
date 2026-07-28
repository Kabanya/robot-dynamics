"""Render one browser-free TALOS+iCub MP4 from archived confirmation cases."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile

import matplotlib
import numpy as np
import pinocchio as pin

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402
from scipy.spatial import ConvexHull  # noqa: E402

from src.feasibility import (
    GaitSample,
    build_whole_body_trajectory,
    load_robot_spec,
    rollout,
)


ROBOTS = ("talos", "icub")
PARAMETERS = (
    "step_length",
    "step_width",
    "single_support_duration",
    "double_support_duration",
    "com_height_scale",
    "zmp_bias_x",
    "zmp_bias_y",
    "friction",
    "payload_fraction",
    "timing_error_seconds",
    "impulse",
)


def select_cases(records) -> dict[str, dict[str, object]]:
    selected = {}
    for robot in ROBOTS:
        successes = [
            row for row in records
            if row["robot"] == robot and int(row["label"]) == 0
        ]
        failures = [
            row for row in records
            if row["robot"] == robot and int(row["label"]) == 1
        ]
        if not successes or not failures:
            raise ValueError(f"{robot} video requires success and failure")
        selected[robot] = {
            "success": max(
                successes,
                key=lambda row: (float(row["step_length"]), -int(row["seed"])),
            ),
            "failure": max(
                failures,
                key=lambda row: (
                    int(row["failure_index"]),
                    float(row["step_length"]),
                    -int(row["seed"]),
                ),
            ),
        }
    return selected


def _sample(row) -> GaitSample:
    return GaitSample(
        **{name: float(row[name]) for name in PARAMETERS},
        seed=int(row["seed"]),
        ood=str(row.get("ood", "false")).lower() in {"true", "1"},
    )


def _prepare_icub_visuals(robot) -> None:
    rotation = pin.SE3(
        np.array(((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0))),
        np.zeros(3),
    )
    for visual in robot.visual_model.geometryObjects:
        if Path(visual.meshPath).suffix.lower() == ".dae":
            visual.meshPath = ""
            visual.placement = visual.placement * rotation


def _frame_indices(stop: int, stride: int = 10, count: int = 80) -> list[int]:
    indices = list(range(0, stop, stride))
    if not indices:
        raise ValueError("video case has no rollout states")
    return (indices + [indices[-1]] * count)[:max(count, len(indices))]


def _local_meshes(robot, vertices_per_body: int = 80):
    meshes = []
    for geometry_id, visual in enumerate(robot.visual_model.geometryObjects):
        geometry = visual.geometry
        if not hasattr(geometry, "vertices"):
            continue
        vertices = np.asarray(geometry.vertices(), dtype=float)
        count = min(len(vertices), vertices_per_body)
        if count < 4:
            continue
        sampled = vertices[np.linspace(0, len(vertices) - 1, count, dtype=int)]
        triangles = ConvexHull(sampled, qhull_options="QJ").simplices
        scale = np.asarray(visual.meshScale, dtype=float)
        meshes.append((geometry_id, sampled[triangles] * scale))
    return meshes


def _joint_points(robot, q) -> np.ndarray:
    pin.forwardKinematics(robot.model, robot.data, q)
    return np.array(
        [robot.data.oMi[joint].translation for joint in range(1, robot.model.njoints)]
    )


def _world_faces(robot, q, meshes):
    pin.updateGeometryPlacements(
        robot.model,
        robot.data,
        robot.visual_model,
        robot.visual_data,
        q,
    )
    faces = []
    for geometry_id, local_faces in meshes:
        placement = robot.visual_data.oMg[geometry_id]
        faces.append(local_faces @ placement.rotation.T + placement.translation)
    return np.concatenate(faces)


def _draw_skeleton(axis, robot, q, color, width, alpha=1.0, zorder=None) -> None:
    points = _joint_points(robot, q)
    for joint in range(1, robot.model.njoints):
        parent = int(robot.model.parents[joint])
        if parent == 0:
            continue
        segment = np.vstack((points[parent - 1], points[joint - 1]))
        axis.plot(
            *segment.T,
            color=color,
            linewidth=width,
            alpha=alpha,
            zorder=zorder,
        )


def _record_clip(
    robot_name,
    outcome,
    row,
    directory: Path,
    ffmpeg: str,
) -> Path:
    spec = load_robot_spec(robot_name)
    sample = _sample(row)
    trajectory = build_whole_body_trajectory(spec, sample, steps=6, dt=0.01)
    result = rollout(spec, trajectory, sample)
    if bool(result.success) != (outcome == "success"):
        raise RuntimeError(f"archived {robot_name} {outcome} no longer reproduces")

    robot = spec.robot
    if robot_name == "icub":
        _prepare_icub_visuals(robot)
    meshes = _local_meshes(robot)
    frame_dir = directory / f"{robot_name}_{outcome}_frames"
    frame_dir.mkdir()
    stop = len(result.q) if result.success else result.failure_index + 1
    indices = _frame_indices(stop)
    extent_points = np.concatenate(
        [
            _joint_points(robot, result.q[index])
            for index in sorted(set(indices))
        ]
    )
    center = (extent_points.min(axis=0) + extent_points.max(axis=0)) / 2
    radius = max(np.ptp(extent_points, axis=0).max() * 0.6, 1.15 * spec.leg_length)
    status_color = "#17864b" if result.success else "#c43b3b"
    label = (
        f"{robot_name.upper()} — SUCCESS"
        if result.success else
        f"{robot_name.upper()} — FAILURE: {result.failure_reason}"
    )

    figure = plt.figure(figsize=(7.2, 7.2), dpi=100)
    axis = figure.add_subplot(111, projection="3d")
    for number, index in enumerate(indices):
        axis.cla()
        reference_index = min(index, len(trajectory.q) - 1)
        faces = _world_faces(robot, result.q[index], meshes)
        axis.add_collection3d(
            Poly3DCollection(
                faces,
                facecolors=status_color,
                edgecolors=status_color,
                linewidths=0.08,
                alpha=0.82,
            )
        )
        _draw_skeleton(
            axis,
            robot,
            trajectory.q[reference_index],
            color="#555555",
            width=3.4,
            zorder=100,
        )
        _draw_skeleton(axis, robot, result.q[index], color=status_color, width=1.3)
        grid = np.linspace(center[0] - radius, center[0] + radius, 7)
        for value in grid:
            axis.plot(
                [value, value],
                [center[1] - radius, center[1] + radius],
                [0, 0],
                color="#dddddd",
                linewidth=0.5,
            )
            axis.plot(
                [center[0] - radius, center[0] + radius],
                [value - center[0] + center[1]] * 2,
                [0, 0],
                color="#dddddd",
                linewidth=0.5,
            )
        axis.set(
            xlim=(center[0] - radius, center[0] + radius),
            ylim=(center[1] - radius, center[1] + radius),
            zlim=(max(-0.05, center[2] - radius), center[2] + radius),
        )
        axis.set_box_aspect((1, 1, 1))
        axis.view_init(elev=16, azim=-62)
        axis.set_axis_off()
        axis.set_title(
            f"{label}\nactual configuration (color) · planned reference (gray)",
            color=status_color,
            fontsize=12,
            fontweight="bold",
            pad=2,
        )
        figure.subplots_adjust(left=0, right=1, bottom=0, top=0.93)
        figure.savefig(frame_dir / f"{number:05d}.png", dpi=100)
    plt.close(figure)

    clip = directory / f"{robot_name}_{outcome}.mp4"
    subprocess.run(
        [
            ffmpeg, "-loglevel", "error", "-y", "-framerate", "10",
            "-i", str(frame_dir / "%05d.png"), "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(clip),
        ],
        check=True,
    )
    return clip


def generate_video(dataset, output) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required")
    dataset, output = Path(dataset), Path(output)
    with dataset.open(newline="", encoding="utf-8") as stream:
        selected = select_cases(list(csv.DictReader(stream)))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".rollout-video-", dir=output.parent) as name:
        temporary = Path(name)
        clips = [
            _record_clip(
                robot,
                outcome,
                selected[robot][outcome],
                temporary,
                ffmpeg,
            )
            for robot in ROBOTS
            for outcome in ("success", "failure")
        ]
        combined = temporary / output.name
        subprocess.run(
            [
                ffmpeg, "-loglevel", "error", "-y",
                "-i", str(clips[0]), "-i", str(clips[1]),
                "-i", str(clips[2]), "-i", str(clips[3]),
                "-filter_complex",
                "[0:v][1:v][2:v][3:v]concat=n=4:v=1:a=0[v]",
                "-map", "[v]", "-c:v", "libx264",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(combined),
            ],
            check=True,
        )
        os.replace(combined, output)
    manifest = {
        "video": output.name,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "cases": {
            robot: {
                outcome: {
                    "base_gait_id": selected[robot][outcome]["base_gait_id"],
                    "seed": int(selected[robot][outcome]["seed"]),
                }
                for outcome in ("success", "failure")
            }
            for robot in ROBOTS
        },
    }
    output.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def generate_video_from_artifact(artifact, output) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with (
        zipfile.ZipFile(artifact) as archive,
        tempfile.TemporaryDirectory(
            prefix=".rollout-video-data-",
            dir=output.parent,
        ) as directory,
    ):
        dataset = Path(directory) / "dataset.csv"
        dataset.write_bytes(
            archive.read("data/confirmation_seed42026/dataset.csv")
        )
        return generate_video(dataset, output)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--confirmation-dir",
        default="results/confirmation-seed42026",
    )
    args = parser.parse_args(argv)
    directory = Path(args.confirmation_dir)
    output = directory / "video" / "rollout_examples.mp4"
    print(generate_video(directory / "dataset.csv", output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
