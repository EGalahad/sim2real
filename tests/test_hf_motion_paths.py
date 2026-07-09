from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sim2real.rl_policy.utils.motion import _any4hdmi_manifest_override_view
from sim2real.sim_env.integrated_sim2sim import _expand_local_path_arg


class HfMotionPathTest(unittest.TestCase):
    def test_integrated_args_preserve_hf_motion_uri(self) -> None:
        uri = "hf://elijahgalahad/any4hdmi-g1-amass-hard/motions/CMU/05/05_06_stageii.npz"

        self.assertEqual(_expand_local_path_arg(uri), uri)
        self.assertEqual(_expand_local_path_arg("~/motion.npz"), str(Path("~/motion.npz").expanduser()))

    def test_manifest_override_resolves_hf_input_before_path_ops(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            dataset_root = temp / "dataset"
            motion_path = dataset_root / "motions" / "CMU" / "05" / "05_06_stageii.npz"
            motion_path.parent.mkdir(parents=True)
            motion_path.write_bytes(b"npz")
            (dataset_root / "manifest.json").write_text(
                json.dumps({"motions_subdir": "motions", "mjcf": "old.xml"}) + "\n",
                encoding="utf-8",
            )
            mjcf_path = temp / "robot.xml"
            mjcf_path.write_text("<mujoco/>\n", encoding="utf-8")
            uri = "hf://elijahgalahad/any4hdmi-g1-amass-hard/motions/CMU/05/05_06_stageii.npz"

            with (
                mock.patch(
                    "sim2real.rl_policy.utils.motion.resolve_input_paths",
                    return_value=[motion_path],
                ) as resolve_input_paths,
                mock.patch(
                    "sim2real.rl_policy.utils.motion.resolve_mjcf_path",
                    return_value=mjcf_path,
                ),
            ):
                view_path = Path(
                    _any4hdmi_manifest_override_view(
                        root_path=uri,
                        base_dir=temp,
                        mjcf_path="hf://elijahgalahad/g1_xmls@main/g1-mode_13_15.xml",
                    )
                )

            resolve_input_paths.assert_called_once_with(temp, uri)
            self.assertEqual(view_path.name, "05_06_stageii.npz")
            self.assertTrue((view_path.parents[3] / "manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
