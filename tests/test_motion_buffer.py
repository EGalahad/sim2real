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


if __name__ == "__main__":
    unittest.main()
