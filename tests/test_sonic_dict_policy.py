from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import yaml

from sim2real.rl_policy.observations.sonic import (
    sonic_joint_pos_multi_future_wrist_for_smpl,
    sonic_motion_anchor_ori_heading_mf_nonflat,
    sonic_smpl_joints_multi_future_local,
    sonic_smpl_official_encoder_input,
    sonic_smpl_root_ori_b_multi_future,
)
from sim2real.rl_policy.utils.motion import MotionData
from sim2real.utils.math import (
    matrix_from_quat,
    projected_yaw_quat,
    quat_conjugate,
    quat_from_yaw,
    quat_mul,
)


class SonicDictPolicyContractTest(unittest.TestCase):
    def test_v11_g1_orientation_uses_robot_heading_not_full_pose(self) -> None:
        joint_names = [f"joint_{index}" for index in range(29)]
        ref_quat = quat_mul(
            quat_from_yaw(np.asarray([0.8], dtype=np.float32)),
            np.asarray([[np.cos(0.2), 0.0, np.sin(0.2), 0.0]], dtype=np.float32),
        )[0]
        robot_quat = quat_mul(
            quat_from_yaw(np.asarray([0.8], dtype=np.float32)),
            np.asarray([[np.cos(0.3), np.sin(0.3), 0.0, 0.0]], dtype=np.float32),
        )[0]
        motion_data = MotionData(
            joint_pos=np.zeros((1, 2, 29), dtype=np.float32),
            joint_vel=np.zeros((1, 2, 29), dtype=np.float32),
            body_pos_w=np.zeros((1, 2, 1, 3), dtype=np.float32),
            body_lin_vel_w=np.zeros((1, 2, 1, 3), dtype=np.float32),
            body_quat_w=np.broadcast_to(ref_quat, (1, 2, 1, 4)).copy(),
            body_ang_vel_w=np.zeros((1, 2, 1, 3), dtype=np.float32),
        )
        state_processor = SimpleNamespace(root_quat_w=robot_quat, motion_data=motion_data)
        env = SimpleNamespace(
            state_processor=state_processor,
            motion_data=motion_data,
            motion_future_steps=np.asarray([0, 1]),
            motion_joint_names=joint_names,
            motion_body_names=["pelvis"],
            policy_joint_names=joint_names,
            body_names_simulation=["pelvis"],
        )
        observation = sonic_motion_anchor_ori_heading_mf_nonflat(
            env=env,
            future_steps=[0, 1],
            joint_names=joint_names,
            root_body_name="pelvis",
        )

        observation.update({})

        expected_quat = quat_mul(
            quat_conjugate(projected_yaw_quat(robot_quat.reshape(1, 4))),
            ref_quat.reshape(1, 4),
        )
        expected = np.tile(matrix_from_quat(expected_quat)[..., :, :2].reshape(-1), 2)
        np.testing.assert_allclose(observation.compute(), expected, atol=1e-6)

    def test_yaml_groups_match_onnx_inputs(self) -> None:
        checkpoint_root = Path(__file__).resolve().parents[1] / "checkpoints" / "sonic"
        expected = {
            "release/g1": {"g1_input": [640], "proprioception": [930]},
            "release/smpl": {"smpl_input": [840], "proprioception": [930]},
            "v1_1/g1": {"g1_input": [640], "proprioception": [930]},
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
