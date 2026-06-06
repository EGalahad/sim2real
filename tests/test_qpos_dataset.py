from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sim2real.teleop.qpos_dataset import write_single_motion_qpos_dataset


class DummyRobotCfg:
    name = "toy"

    def __init__(self, mjcf_path: Path) -> None:
        self.mjcf_path = mjcf_path

    @property
    def qpos_size(self) -> int:
        return 8

    def resolve_mjcf_path(self) -> Path:
        return self.mjcf_path


class WriteSingleMotionQposDatasetTest(unittest.TestCase):
    def test_writes_any4hdmi_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mjcf_path = root / "toy.xml"
            mjcf_path.write_text(
                """
<mujoco>
  <worldbody>
    <body name="base">
      <freejoint name="floating_base_joint"/>
      <geom type="sphere" size="0.05" mass="1"/>
      <body name="link">
        <joint name="hinge" type="hinge" axis="0 0 1"/>
        <geom type="sphere" size="0.05" mass="1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
""".strip(),
                encoding="utf-8",
            )

            output_dir = root / "dataset"
            qpos = np.zeros((3, 8), dtype=np.float32)
            qpos[:, 3] = 1.0

            motion_path, manifest_path = write_single_motion_qpos_dataset(
                output_dir,
                robot_cfg=DummyRobotCfg(mjcf_path),
                qpos=qpos,
                fps=30.0,
                dataset_name="dataset",
                source={"test": "qpos_dataset"},
            )

            self.assertEqual(motion_path, output_dir / "motions" / "motion.npz")
            self.assertEqual(manifest_path, output_dir / "manifest.json")
            np.testing.assert_allclose(np.load(motion_path)["qpos"], qpos)
            self.assertFalse((output_dir / "motion.npz").exists())

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["format_version"], 2)
            self.assertEqual(manifest["dataset_name"], "dataset")
            self.assertEqual(manifest["motions_subdir"], "motions")
            self.assertEqual(manifest["qpos_dim"], 8)
            self.assertEqual(
                manifest["qpos_names"],
                [
                    "root_tx",
                    "root_ty",
                    "root_tz",
                    "root_qw",
                    "root_qx",
                    "root_qy",
                    "root_qz",
                    "hinge",
                ],
            )
            self.assertEqual(manifest["num_motions"], 1)
            self.assertAlmostEqual(manifest["timestep"], 1.0 / 30.0)
            self.assertAlmostEqual(manifest["total_hours"], 3.0 / 30.0 / 3600.0)


if __name__ == "__main__":
    unittest.main()
