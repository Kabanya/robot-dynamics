from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import zipfile

import matplotlib
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import artifact.generate_rollout_video as video


def test_artifact_wrapper_extracts_frozen_confirmation_dataset():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        archive_path = root / "artifact.zip"
        output_path = root / "video" / "rollout.mp4"
        expected = b"robot,label\nicub,0\n"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr(
                "data/confirmation_seed42026/dataset.csv",
                expected,
            )

        def verify_dataset(dataset, output):
            assert Path(dataset).read_bytes() == expected
            assert Path(output) == output_path
            return output_path

        with patch.object(video, "generate_video", side_effect=verify_dataset):
            result = video.generate_video_from_artifact(
                archive_path,
                output_path,
            )
        assert result == output_path


def test_reference_skeleton_is_visible_over_actual_surface():
    points = np.array(((-0.6, 0.0, 0.0), (0.6, 0.0, 0.0)))
    robot = SimpleNamespace(
        model=SimpleNamespace(njoints=3, parents=np.array((0, 0, 1)))
    )
    figure = plt.figure(figsize=(3, 3), dpi=100)
    axis = figure.add_subplot(111, projection="3d")
    axis.add_collection3d(
        Poly3DCollection(
            [[(-0.8, -0.1, -0.3), (0.8, -0.1, -0.3),
              (0.8, -0.1, 0.3), (-0.8, -0.1, 0.3)]],
            facecolors="#17864b",
        )
    )
    with patch.object(video, "_joint_points", return_value=points):
        video._draw_skeleton(
            axis, robot, None, color="#555555", width=4, zorder=100
        )
    axis.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1))
    axis.set_axis_off()
    figure.canvas.draw()
    pixels = np.asarray(figure.canvas.buffer_rgba())[..., :3]
    target = np.array((85, 85, 85))
    visible = np.max(np.abs(pixels.astype(int) - target), axis=2) < 20
    plt.close(figure)
    assert visible.sum() > 100


if __name__ == "__main__":
    test_artifact_wrapper_extracts_frozen_confirmation_dataset()
    test_reference_skeleton_is_visible_over_actual_surface()
