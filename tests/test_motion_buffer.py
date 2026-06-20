from __future__ import annotations

import unittest

import numpy as np

from sim2real.config.robots.base import PICO_RECV_TIME_NS_KEY, PUBLISH_T_NS_KEY, SMPLX_T_NS_KEY
from sim2real.rl_policy.utils.motion_buffer import RealtimeMotionBuffer


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


if __name__ == "__main__":
    unittest.main()
