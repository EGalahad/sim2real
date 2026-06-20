from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from sim2real.config.robots.base import PICO_RECV_TIME_NS_KEY, PUBLISH_T_NS_KEY, SMPLX_T_NS_KEY
from sim2real.rl_policy.observations.sonic import sonic_smpl_official_encoder_input
from sim2real.rl_policy.utils.motion import MotionData
from sim2real.rl_policy.utils.motion_buffer import RealtimeMotionBuffer, RealtimeSmplMotionBuffer
from sim2real.teleop.smpl_stream import build_smpl_frame_from_xrobot_raw


class DummyRobotCfg:
    joint_names = ("left_joint", "right_joint")
    body_names = ("pelvis", "torso")


def _payload(**timestamps: int) -> dict[str, object]:
    return {
        **timestamps,
        "joint_pos": [0.1, -0.1],
        "body_pos_w": [[0.0, 0.0, 0.5], [0.0, 0.0, 0.8]],
        "body_quat_w": [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
    }


class RealtimeMotionBufferTest(unittest.TestCase):
    def test_uses_pico_receive_time_over_smplx_time(self) -> None:
        buffer = RealtimeMotionBuffer(DummyRobotCfg(), future_steps=[0])
        recv_time_ns = 1_000_000_000_000
        pico_recv_time_ns = recv_time_ns - 5_000_000
        future_smplx_time_ns = recv_time_ns + 258_000_000_000

        buffer._RealtimeMotionBuffer__append_payload(
            _payload(
                **{
                    PICO_RECV_TIME_NS_KEY: pico_recv_time_ns,
                    SMPLX_T_NS_KEY: future_smplx_time_ns,
                }
            ),
            recv_time_ns=recv_time_ns,
        )

        self.assertEqual(buffer.latest_timestamp_ns, pico_recv_time_ns)

    def test_falls_back_to_publish_time_for_legacy_payloads(self) -> None:
        buffer = RealtimeMotionBuffer(DummyRobotCfg(), future_steps=[0])
        recv_time_ns = 1_000_000_000_000
        publish_time_ns = recv_time_ns - 20_000_000

        buffer._RealtimeMotionBuffer__append_payload(
            _payload(**{PUBLISH_T_NS_KEY: publish_time_ns}),
            recv_time_ns=recv_time_ns,
        )

        self.assertEqual(buffer.latest_timestamp_ns, publish_time_ns)

    def test_estimates_reference_velocities_from_neighbor_frames(self) -> None:
        buffer = RealtimeMotionBuffer(DummyRobotCfg(), future_steps=[0])
        t0_ns = 1_000_000_000
        t1_ns = 1_100_000_000
        yaw_delta_rad = 0.2
        body_quat_right = [
            float(np.cos(yaw_delta_rad / 2.0)),
            0.0,
            0.0,
            float(np.sin(yaw_delta_rad / 2.0)),
        ]

        buffer._RealtimeMotionBuffer__append_payload(
            {
                PUBLISH_T_NS_KEY: t0_ns,
                "joint_pos": [0.0, 0.0],
                "body_pos_w": [[0.0, 0.0, 0.5], [0.0, 0.0, 0.8]],
                "body_quat_w": [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            },
            recv_time_ns=t0_ns,
        )
        buffer._RealtimeMotionBuffer__append_payload(
            {
                PUBLISH_T_NS_KEY: t1_ns,
                "joint_pos": [0.2, -0.4],
                "body_pos_w": [[0.2, 0.0, 0.5], [0.0, 0.3, 0.8]],
                "body_quat_w": [body_quat_right, [1.0, 0.0, 0.0, 0.0]],
            },
            recv_time_ns=t1_ns,
        )

        joint_pos = np.zeros((1, 2), dtype=np.float32)
        joint_vel = np.zeros_like(joint_pos)
        body_pos_w = np.zeros((1, 2, 3), dtype=np.float32)
        body_lin_vel_w = np.zeros_like(body_pos_w)
        body_quat_w = np.zeros((1, 2, 4), dtype=np.float32)
        body_ang_vel_w = np.zeros_like(body_pos_w)

        buffer._fill_sample_frames_locked(
            np.asarray([t0_ns + 50_000_000], dtype=np.int64),
            joint_pos,
            joint_vel,
            body_pos_w,
            body_lin_vel_w,
            body_quat_w,
            body_ang_vel_w,
        )

        np.testing.assert_allclose(joint_pos[0], [0.1, -0.2], atol=1e-6)
        np.testing.assert_allclose(joint_vel[0], [2.0, -4.0], atol=1e-6)
        np.testing.assert_allclose(
            body_lin_vel_w[0],
            [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            body_ang_vel_w[0],
            [[0.0, 0.0, 2.0], [0.0, 0.0, 0.0]],
            atol=1e-6,
        )

    def test_smpl_buffer_keeps_robot_cfg_joint_order(self) -> None:
        buffer = RealtimeSmplMotionBuffer(DummyRobotCfg(), future_steps=[0])

        buffer._RealtimeSmplMotionBuffer__append_payload(
            {
                PUBLISH_T_NS_KEY: 1_000_000_000,
                "smpl_body_pose_aa": np.zeros((1, 21, 3), dtype=np.float32),
                "smpl_joint_pos_root": np.zeros((1, 24, 3), dtype=np.float32),
                "smpl_root_quat_w": np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                "joint_pos": np.asarray([[0.25, -0.5]], dtype=np.float32),
            },
            recv_time_ns=1_000_000_000,
        )

        self.assertEqual(buffer.joint_names, list(DummyRobotCfg.joint_names))
        with buffer._lock:
            np.testing.assert_allclose(buffer._joint_pos_frames[0], [0.25, -0.5])

    def test_sonic_smpl_encoder_extracts_wrist_values_by_name(self) -> None:
        wrist_names = sonic_smpl_official_encoder_input.WRIST_JOINT_NAMES
        motion_joint_names = ("left_hip_pitch_joint", *wrist_names, "right_knee_joint")
        num_steps = 10
        joint_pos = np.full((1, num_steps, len(motion_joint_names)), -10.0, dtype=np.float32)
        wrist_indices = [motion_joint_names.index(name) for name in wrist_names]
        expected = np.arange(num_steps * len(wrist_indices), dtype=np.float32).reshape(
            num_steps,
            len(wrist_indices),
        )
        joint_pos[0][:, wrist_indices] = expected

        state_processor = SimpleNamespace(
            motion_joint_names=list(motion_joint_names),
            root_quat_w=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            motion_data=MotionData(
                smpl_joint_pos_root=np.zeros((1, num_steps, 24, 3), dtype=np.float32),
                smpl_root_quat_w=np.broadcast_to(
                    np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                    (1, num_steps, 4),
                ).copy(),
                joint_pos=joint_pos,
            ),
        )
        obs = sonic_smpl_official_encoder_input(
            env=SimpleNamespace(state_processor=state_processor)
        )

        output = obs.compute()
        wrist_slice = output[
            obs.WRIST_OFFSET : obs.WRIST_OFFSET + num_steps * len(wrist_indices)
        ]
        np.testing.assert_allclose(wrist_slice, expected.reshape(-1))

    def test_smpl_raw_frame_requires_official_joint_info(self) -> None:
        body_poses = np.zeros((24, 7), dtype=np.float32)
        body_poses[:, 6] = 1.0

        with self.assertRaises(FileNotFoundError):
            build_smpl_frame_from_xrobot_raw(
                body_poses,
                [f"joint_{idx}" for idx in range(24)],
                human_joints_info_path="/tmp/definitely_missing_smpl_human_joints_info.pkl",
            )


if __name__ == "__main__":
    unittest.main()
