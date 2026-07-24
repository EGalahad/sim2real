from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import yaml

from sim2real.rl_policy.observations.sonic import (
    sonic_joint_pos_multi_future_wrist_for_smpl,
    sonic_smpl_joints_multi_future_local,
    sonic_smpl_official_encoder_input,
    sonic_smpl_root_ori_b_multi_future,
)


class SonicDictPolicyContractTest(unittest.TestCase):
    def test_yaml_groups_match_onnx_inputs(self) -> None:
        checkpoint_root = (
            Path(__file__).resolve().parents[1] / "checkpoints" / "sonic" / "release"
        )
        expected = {
            "g1": {"g1_input": [640], "proprioception": [930]},
            "smpl": {"smpl_input": [840], "proprioception": [930]},
        }

        for mode, expected_inputs in expected.items():
            with self.subTest(mode=mode):
                config_path = checkpoint_root / mode / "policy.yaml"
                model_path = checkpoint_root / mode / "policy.onnx"
                with config_path.open() as config_file:
                    config = yaml.safe_load(config_file)

                model = onnx.load(model_path)
                onnx.checker.check_model(model)
                actual_inputs = {
                    value.name: [
                        dim.dim_value if dim.dim_value else dim.dim_param
                        for dim in value.type.tensor_type.shape.dim
                    ]
                    for value in model.graph.input
                }
                outputs = {
                    value.name: [
                        dim.dim_value if dim.dim_value else dim.dim_param
                        for dim in value.type.tensor_type.shape.dim
                    ]
                    for value in model.graph.output
                }

                self.assertEqual(actual_inputs, expected_inputs)
                self.assertEqual(list(config["observation"]), list(expected_inputs))
                self.assertNotIn("obs_dict", actual_inputs)
                self.assertEqual(outputs, {"action": [29], "token": [64]})

    def test_split_smpl_observations_match_legacy_layout(self) -> None:
        rng = np.random.default_rng(20260718)
        joint_names = list(sonic_smpl_official_encoder_input.WRIST_JOINT_NAMES)
        joint_names.extend(f"joint_{index}" for index in range(23))
        motion_data = SimpleNamespace(
            smpl_joint_pos_root=rng.standard_normal((1, 10, 24, 3)).astype(np.float32),
            smpl_root_quat_w=np.broadcast_to(
                np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                (1, 10, 4),
            ).copy(),
            joint_pos=rng.standard_normal((1, 10, 29)).astype(np.float32),
        )
        state_processor = SimpleNamespace(
            root_quat_w=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )
        env = SimpleNamespace(
            state_processor=state_processor,
            motion_data=motion_data,
            motion_joint_names=joint_names,
        )
        future_steps = list(range(10))

        legacy = sonic_smpl_official_encoder_input(env=env)
        split = [
            sonic_smpl_joints_multi_future_local(env=env, future_steps=future_steps),
            sonic_smpl_root_ori_b_multi_future(env=env, future_steps=future_steps),
            sonic_joint_pos_multi_future_wrist_for_smpl(
                env=env, future_steps=future_steps
            ),
        ]
        legacy.reset()
        for observation in split:
            observation.reset()

        legacy_input = legacy.compute()
        split_input = np.concatenate([observation.compute() for observation in split])
        legacy_selected_input = np.concatenate(
            [
                legacy_input[922:1642],
                legacy_input[1642:1702],
                legacy_input[1702:1762],
            ]
        )
        np.testing.assert_array_equal(split_input, legacy_selected_input)


if __name__ == "__main__":
    unittest.main()
