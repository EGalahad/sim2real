from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from sim2real.config.robots.base import (
    JOINT_NAMES_KEY,
    JOINT_POS_KEY,
    MOTION_FIRST_FRAME_KEY,
    PICO_RECV_TIME_NS_KEY,
    PUBLISH_T_NS_KEY,
    SMPLX_T_NS_KEY,
)
from sim2real.rl_policy.observations.sonic import sonic_smpl_official_encoder_input
from sim2real.rl_policy.utils.motion import MotionData
from sim2real.rl_policy.utils.motion_buffer import RealtimeMotionBuffer, RealtimeSmplMotionBuffer
from sim2real.rl_policy.utils.state_processor import StateProcessor
from sim2real.teleop.npz_pub import NpzMotionPublisher, PlaybackState
from sim2real.teleop.smpl_stream import build_smpl_frame_from_xrobot_raw


class DummyRobotCfg:
    joint_names = ("left_joint", "right_joint")
    body_names = ("pelvis", "torso")


class FakeMotionDataset:
    joint_names = ("left_joint", "right_joint", "extra_joint")
    body_names = ("pelvis", "torso")

    def get_slice(
        self,
        motion_ids: np.ndarray,
        starts: np.ndarray,
        steps: np.ndarray,
    ) -> SimpleNamespace:
        del motion_ids, steps
        frame = int(np.asarray(starts).reshape(-1)[0])
        joint_pos = np.asarray(
            [[[frame + 0.1, frame + 0.2, frame + 99.0]]],
            dtype=np.float32,
        )
        joint_vel = np.asarray([[[1.0, 2.0, 99.0]]], dtype=np.float32)
        body_pos_w = np.asarray(
            [[[[frame, 0.0, 0.5], [frame, 0.2, 0.8]]]],
            dtype=np.float32,
        )
        body_quat_w = np.asarray(
            [[[_yaw_quat(0.0), _yaw_quat(0.0)]]],
            dtype=np.float32,
        )
        body_lin_vel_w = np.zeros_like(body_pos_w)
        body_ang_vel_w = np.zeros_like(body_pos_w)
        return SimpleNamespace(
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            body_pos_w=body_pos_w,
            body_lin_vel_w=body_lin_vel_w,
            body_quat_w=body_quat_w,
            body_ang_vel_w=body_ang_vel_w,
        )


class FakeKeyboard:
    def __init__(self, keys: list[str]) -> None:
        self.keys = list(keys)

    def pop_keys(self) -> list[str]:
        keys = list(self.keys)
        self.keys.clear()
        return keys


def _fake_npz_publisher(
    *,
    frame: int = 0,
    motion_length: int = 3,
    state: PlaybackState | None = None,
    paused: bool | None = None,
    loop: bool = False,
    hold_last: bool = True,
    pub_vel: bool = False,
    segment_first_frame: bool = False,
) -> NpzMotionPublisher:
    publisher = object.__new__(NpzMotionPublisher)
    publisher.args = SimpleNamespace(
        motion_path="fake_motion.npz",
        loop=loop,
        hold_last=hold_last,
        pub_vel=pub_vel,
    )
    publisher.robot_cfg = DummyRobotCfg()
    publisher.motion_dataset = FakeMotionDataset()
    publisher.motion_ids = np.asarray([0], dtype=np.int64)
    publisher.frame = int(frame)
    publisher.motion_length = int(motion_length)
    publisher.seq = 0
    if state is None:
        state = PlaybackState.MOTION_PAUSED if paused else PlaybackState.MOTION_PLAYING
    publisher.state = state
    publisher._segment_first_frame = bool(segment_first_frame)
    publisher._stop_after_terminal_payload = False
    publisher.motion_joint_indices = [0, 1]
    publisher.motion_body_indices = [0, 1]
    publisher.root_body_index = 0
    publisher.latest_root_pos_w = np.zeros(3, dtype=np.float32)
    publisher.latest_root_quat_w = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    publisher.aligned_default_joint_pos = np.zeros((2,), dtype=np.float32)
    publisher.aligned_default_body_pos_w = np.asarray(
        [[0.0, 0.0, 0.5], [0.0, 0.0, 0.8]],
        dtype=np.float32,
    )
    publisher.aligned_default_body_quat_w = np.asarray(
        [_yaw_quat(0.0), _yaw_quat(0.0)],
        dtype=np.float32,
    )
    publisher.aligned_default_qpos = np.zeros((9,), dtype=np.float32)
    publisher._capture_aligned_default_pose = lambda: None
    publisher.keyboard = None
    return publisher


def _payload(**timestamps: int) -> dict[str, object]:
    return {
        **timestamps,
        "joint_pos": [0.1, -0.1],
        "body_pos_w": [[0.0, 0.0, 0.5], [0.0, 0.0, 0.8]],
        "body_quat_w": [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
    }


def _yaw_quat(yaw: float) -> list[float]:
    return [float(np.cos(yaw / 2.0)), 0.0, 0.0, float(np.sin(yaw / 2.0))]


def _yaw_from_quat(quat: np.ndarray) -> float:
    q = np.asarray(quat, dtype=np.float32).reshape(4)
    return float(np.arctan2(2.0 * (q[0] * q[3] + q[1] * q[2]), 1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3])))


def _rotate_xy(vector_xy: np.ndarray, yaw: float) -> np.ndarray:
    c = np.cos(yaw)
    s = np.sin(yaw)
    x, y = np.asarray(vector_xy, dtype=np.float32).reshape(2)
    return np.asarray([c * x - s * y, s * x + c * y], dtype=np.float32)


class NpzMotionPublisherTest(unittest.TestCase):
    def test_hold_last_emits_terminal_frame_as_active_payload(self) -> None:
        publisher = _fake_npz_publisher(frame=1, motion_length=3, hold_last=True)

        penultimate_payload = publisher.sample_payload()
        self.assertEqual(penultimate_payload["frame"], 1)
        self.assertFalse(penultimate_payload["paused"])
        self.assertFalse(publisher.paused)
        self.assertEqual(publisher.frame, 2)

        terminal_payload = publisher.sample_payload()
        self.assertEqual(terminal_payload["frame"], 2)
        self.assertFalse(terminal_payload["paused"])
        self.assertTrue(publisher.paused)
        self.assertTrue(publisher.motion_finished)

    def test_first_frame_flag_stays_marked_until_motion_advances(self) -> None:
        publisher = _fake_npz_publisher(
            frame=0,
            motion_length=3,
            state=PlaybackState.MOTION_PAUSED,
            segment_first_frame=True,
        )

        paused_payload = publisher.sample_payload()
        self.assertEqual(paused_payload["frame"], 0)
        self.assertTrue(paused_payload["paused"])
        self.assertTrue(paused_payload[MOTION_FIRST_FRAME_KEY])
        self.assertTrue(publisher._segment_first_frame)

        publisher.state = PlaybackState.MOTION_PLAYING
        active_payload = publisher.sample_payload()
        self.assertEqual(active_payload["frame"], 0)
        self.assertFalse(active_payload["paused"])
        self.assertTrue(active_payload[MOTION_FIRST_FRAME_KEY])
        self.assertFalse(publisher._segment_first_frame)
        self.assertEqual(publisher.frame, 1)

    def test_loop_marks_returned_frame_zero_as_first_frame(self) -> None:
        publisher = _fake_npz_publisher(
            frame=1,
            motion_length=2,
            loop=True,
            hold_last=False,
        )

        last_payload = publisher.sample_payload()
        self.assertEqual(last_payload["frame"], 1)
        self.assertFalse(last_payload[MOTION_FIRST_FRAME_KEY])
        self.assertTrue(publisher._segment_first_frame)
        self.assertEqual(publisher.frame, 0)

        first_payload = publisher.sample_payload()
        self.assertEqual(first_payload["frame"], 0)
        self.assertTrue(first_payload[MOTION_FIRST_FRAME_KEY])
        self.assertFalse(publisher._segment_first_frame)

    def test_keyboard_state_transitions(self) -> None:
        publisher = _fake_npz_publisher(
            frame=2,
            state=PlaybackState.DEFAULT,
            segment_first_frame=True,
        )

        default_payload = publisher.sample_payload()
        self.assertEqual(default_payload["source"], "default")
        self.assertTrue(default_payload["paused"])
        self.assertTrue(default_payload[MOTION_FIRST_FRAME_KEY])
        self.assertEqual(publisher.frame, 2)

        publisher.keyboard = FakeKeyboard(["]"])
        motion_paused_payload = publisher.sample_payload()
        self.assertEqual(publisher.state, PlaybackState.MOTION_PAUSED)
        self.assertEqual(motion_paused_payload["frame"], 0)
        self.assertTrue(motion_paused_payload["paused"])
        self.assertTrue(motion_paused_payload[MOTION_FIRST_FRAME_KEY])

        publisher.keyboard = FakeKeyboard(["space"])
        motion_playing_payload = publisher.sample_payload()
        self.assertEqual(motion_playing_payload["frame"], 0)
        self.assertFalse(motion_playing_payload["paused"])
        self.assertTrue(motion_playing_payload[MOTION_FIRST_FRAME_KEY])
        self.assertEqual(publisher.state, PlaybackState.MOTION_PLAYING)
        self.assertEqual(publisher.frame, 1)

        publisher.keyboard = FakeKeyboard(["space"])
        motion_repaused_payload = publisher.sample_payload()
        self.assertEqual(motion_repaused_payload["frame"], 1)
        self.assertTrue(motion_repaused_payload["paused"])
        self.assertFalse(motion_repaused_payload[MOTION_FIRST_FRAME_KEY])
        self.assertEqual(publisher.frame, 1)

        publisher.keyboard = FakeKeyboard(["x"])
        default_again_payload = publisher.sample_payload()
        self.assertEqual(default_again_payload["source"], "default")
        self.assertTrue(default_again_payload["paused"])
        self.assertTrue(default_again_payload[MOTION_FIRST_FRAME_KEY])
        self.assertEqual(publisher.frame, 1)

        publisher.keyboard = FakeKeyboard(["]"])
        reset_payload = publisher.sample_payload()
        self.assertEqual(reset_payload["frame"], 0)
        self.assertTrue(reset_payload["paused"])
        self.assertTrue(reset_payload[MOTION_FIRST_FRAME_KEY])

    def test_publishes_robot_joint_subset_for_zmq_buffer(self) -> None:
        publisher = _fake_npz_publisher(pub_vel=True)

        payload = publisher.sample_payload()

        self.assertEqual(payload[JOINT_NAMES_KEY], list(DummyRobotCfg.joint_names))
        np.testing.assert_allclose(payload[JOINT_POS_KEY], [0.1, 0.2], atol=1e-6)
        np.testing.assert_allclose(payload["joint_vel"], [1.0, 2.0], atol=1e-6)


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

    def test_holds_latest_frame_with_zero_velocity_when_stream_stops(self) -> None:
        buffer = RealtimeMotionBuffer(DummyRobotCfg(), future_steps=[0])
        t0_ns = 1_000_000_000
        t1_ns = 1_100_000_000
        latest_body_quat = [
            float(np.cos(0.1)),
            0.0,
            0.0,
            float(np.sin(0.1)),
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
                "body_quat_w": [latest_body_quat, [1.0, 0.0, 0.0, 0.0]],
            },
            recv_time_ns=t1_ns,
        )

        joint_pos = np.zeros((2, 2), dtype=np.float32)
        joint_vel = np.empty_like(joint_pos)
        body_pos_w = np.zeros((2, 2, 3), dtype=np.float32)
        body_lin_vel_w = np.empty_like(body_pos_w)
        body_quat_w = np.zeros((2, 2, 4), dtype=np.float32)
        body_ang_vel_w = np.empty_like(body_pos_w)

        buffer._fill_sample_frames_locked(
            np.asarray([t1_ns + 1, t1_ns + 2_000_000_000], dtype=np.int64),
            joint_pos,
            joint_vel,
            body_pos_w,
            body_lin_vel_w,
            body_quat_w,
            body_ang_vel_w,
        )

        np.testing.assert_allclose(joint_pos, [[0.2, -0.4], [0.2, -0.4]], atol=1e-6)
        np.testing.assert_allclose(joint_vel, 0.0, atol=1e-6)
        np.testing.assert_allclose(
            body_pos_w,
            [
                [[0.2, 0.0, 0.5], [0.0, 0.3, 0.8]],
                [[0.2, 0.0, 0.5], [0.0, 0.3, 0.8]],
            ],
            atol=1e-6,
        )
        np.testing.assert_allclose(body_lin_vel_w, 0.0, atol=1e-6)
        expected_latest_quat = np.broadcast_to(
            np.asarray(latest_body_quat, dtype=np.float32),
            (2, 4),
        )
        np.testing.assert_allclose(body_quat_w[:, 0], expected_latest_quat, atol=1e-6)
        np.testing.assert_allclose(body_ang_vel_w, 0.0, atol=1e-6)

    def test_explicit_default_first_frame_replaces_cached_motion_frame(self) -> None:
        buffer = RealtimeMotionBuffer(DummyRobotCfg(), future_steps=[0])
        motion_time_ns = 1_000_000_000
        default_time_ns = 2_000_000_000

        buffer._RealtimeMotionBuffer__append_payload(
            {
                PUBLISH_T_NS_KEY: motion_time_ns,
                "source": "npz",
                "paused": False,
                "joint_pos": [0.2, -0.4],
                "body_pos_w": [[0.2, 0.0, 0.5], [0.0, 0.3, 0.8]],
                "body_quat_w": [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            },
            recv_time_ns=motion_time_ns,
        )
        buffer.cleanup(motion_time_ns + 1)
        self.assertIsNone(buffer.latest_timestamp_ns)

        buffer._RealtimeMotionBuffer__append_payload(
            {
                PUBLISH_T_NS_KEY: default_time_ns,
                "source": "default",
                "paused": True,
                MOTION_FIRST_FRAME_KEY: True,
                "joint_pos": [0.0, 0.0],
                "body_pos_w": [[0.0, 0.0, 0.5], [0.0, 0.0, 0.8]],
                "body_quat_w": [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            },
            recv_time_ns=default_time_ns,
        )

        self.assertEqual(buffer.latest_timestamp_ns, default_time_ns)
        joint_pos = np.zeros((1, 2), dtype=np.float32)
        joint_vel = np.empty_like(joint_pos)
        body_pos_w = np.zeros((1, 2, 3), dtype=np.float32)
        body_lin_vel_w = np.empty_like(body_pos_w)
        body_quat_w = np.zeros((1, 2, 4), dtype=np.float32)
        body_ang_vel_w = np.empty_like(body_pos_w)

        buffer._fill_sample_frames_locked(
            np.asarray([default_time_ns], dtype=np.int64),
            joint_pos,
            joint_vel,
            body_pos_w,
            body_lin_vel_w,
            body_quat_w,
            body_ang_vel_w,
        )

        np.testing.assert_allclose(joint_pos[0], [0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(joint_vel, 0.0, atol=1e-6)
        np.testing.assert_allclose(body_pos_w[0, 0], [0.2, 0.0, 0.5], atol=1e-6)
        np.testing.assert_allclose(
            body_pos_w[0, 1],
            [0.2, 0.0, 0.8],
            atol=1e-6,
        )

    def test_first_frame_segment_is_xy_yaw_continuous(self) -> None:
        buffer = RealtimeMotionBuffer(DummyRobotCfg(), future_steps=[0])
        t0_ns = 1_000_000_000
        t1_ns = 1_100_000_000
        t2_ns = 1_200_000_000
        t3_ns = 1_300_000_000
        prev_yaw = 0.2
        raw_segment_yaw = np.pi / 2.0

        buffer._RealtimeMotionBuffer__append_payload(
            {
                PUBLISH_T_NS_KEY: t0_ns,
                "joint_pos": [0.0, 0.0],
                "body_pos_w": [[0.0, 0.0, 0.5], [0.0, 1.0, 0.8]],
                "body_quat_w": [_yaw_quat(0.0), _yaw_quat(0.0)],
            },
            recv_time_ns=t0_ns,
        )
        buffer._RealtimeMotionBuffer__append_payload(
            {
                PUBLISH_T_NS_KEY: t1_ns,
                "joint_pos": [0.1, -0.1],
                "body_pos_w": [[1.0, 0.0, 0.5], [1.0, 1.0, 0.8]],
                "body_quat_w": [_yaw_quat(prev_yaw), _yaw_quat(prev_yaw)],
            },
            recv_time_ns=t1_ns,
        )
        buffer._RealtimeMotionBuffer__append_payload(
            {
                PUBLISH_T_NS_KEY: t2_ns,
                MOTION_FIRST_FRAME_KEY: True,
                "joint_pos": [0.2, -0.2],
                "body_pos_w": [[10.0, 10.0, 0.5], [10.0, 11.0, 0.8]],
                "body_quat_w": [_yaw_quat(raw_segment_yaw), _yaw_quat(raw_segment_yaw)],
            },
            recv_time_ns=t2_ns,
        )
        buffer._RealtimeMotionBuffer__append_payload(
            {
                PUBLISH_T_NS_KEY: t3_ns,
                "joint_pos": [0.3, -0.3],
                "body_pos_w": [[10.0, 11.0, 0.5], [10.0, 12.0, 0.8]],
                "body_quat_w": [_yaw_quat(raw_segment_yaw), _yaw_quat(raw_segment_yaw)],
            },
            recv_time_ns=t3_ns,
        )

        with buffer._lock:
            segment_first_root_pos = buffer._body_pos_w_frames[2][0].copy()
            segment_first_root_quat = buffer._body_quat_w_frames[2][0].copy()
            segment_next_root_pos = buffer._body_pos_w_frames[3][0].copy()

        np.testing.assert_allclose(segment_first_root_pos[:2], [1.0, 0.0], atol=1e-6)
        self.assertAlmostEqual(_yaw_from_quat(segment_first_root_quat), prev_yaw, places=6)

        yaw_delta = prev_yaw - raw_segment_yaw
        expected_next_xy = np.asarray([1.0, 0.0], dtype=np.float32) + _rotate_xy(
            np.asarray([0.0, 1.0], dtype=np.float32),
            yaw_delta,
        )
        np.testing.assert_allclose(segment_next_root_pos[:2], expected_next_xy, atol=1e-6)

    def test_first_frame_signal_stays_internal_to_motion_buffer(self) -> None:
        buffer = RealtimeMotionBuffer(DummyRobotCfg(), future_steps=[0])
        motion_time_ns = 1_000_000_000

        buffer._RealtimeMotionBuffer__append_payload(
            {
                **_payload(**{PUBLISH_T_NS_KEY: motion_time_ns}),
                "source": "npz",
                "paused": False,
            },
            recv_time_ns=motion_time_ns,
        )

        buffer._RealtimeMotionBuffer__append_payload(
            {
                **_payload(**{PUBLISH_T_NS_KEY: motion_time_ns + 1_000_000}),
                "source": "default",
                "paused": True,
                MOTION_FIRST_FRAME_KEY: True,
            },
            recv_time_ns=motion_time_ns + 1_000_000,
        )

        buffer.get_obs()
        self.assertFalse(hasattr(buffer, "consume_first_frame"))

    def test_cleanup_to_empty_returns_cached_last_frame(self) -> None:
        buffer = RealtimeMotionBuffer(DummyRobotCfg(), future_steps=[0])
        t0_ns = 1_000_000_000
        buffer._RealtimeMotionBuffer__append_payload(
            {
                PUBLISH_T_NS_KEY: t0_ns,
                "joint_pos": [0.2, -0.4],
                "body_pos_w": [[0.2, 0.0, 0.5], [0.0, 0.3, 0.8]],
                "body_quat_w": [_yaw_quat(0.2), [1.0, 0.0, 0.0, 0.0]],
            },
            recv_time_ns=t0_ns,
        )
        buffer.cleanup(t0_ns + 1)

        joint_pos = np.zeros((1, 2), dtype=np.float32)
        joint_vel = np.empty_like(joint_pos)
        body_pos_w = np.zeros((1, 2, 3), dtype=np.float32)
        body_lin_vel_w = np.empty_like(body_pos_w)
        body_quat_w = np.zeros((1, 2, 4), dtype=np.float32)
        body_ang_vel_w = np.empty_like(body_pos_w)

        buffer._fill_sample_frames_locked(
            np.asarray([t0_ns + 2], dtype=np.int64),
            joint_pos,
            joint_vel,
            body_pos_w,
            body_lin_vel_w,
            body_quat_w,
            body_ang_vel_w,
        )

        np.testing.assert_allclose(joint_pos[0], [0.2, -0.4], atol=1e-6)
        np.testing.assert_allclose(joint_vel, 0.0, atol=1e-6)
        np.testing.assert_allclose(body_pos_w[0, 0], [0.2, 0.0, 0.5], atol=1e-6)
        np.testing.assert_allclose(body_lin_vel_w, 0.0, atol=1e-6)
        np.testing.assert_allclose(body_ang_vel_w, 0.0, atol=1e-6)

    def test_partial_future_window_interpolates_then_clamps_to_latest(self) -> None:
        buffer = RealtimeMotionBuffer(DummyRobotCfg(), future_steps=[0])
        t0_ns = 1_000_000_000
        t1_ns = 1_100_000_000
        buffer._RealtimeMotionBuffer__append_payload(
            {
                PUBLISH_T_NS_KEY: t0_ns,
                "joint_pos": [0.0, 0.0],
                "body_pos_w": [[0.0, 0.0, 0.5], [0.0, 0.0, 0.8]],
                "body_quat_w": [_yaw_quat(0.0), [1.0, 0.0, 0.0, 0.0]],
            },
            recv_time_ns=t0_ns,
        )
        buffer._RealtimeMotionBuffer__append_payload(
            {
                PUBLISH_T_NS_KEY: t1_ns,
                "joint_pos": [0.2, -0.4],
                "body_pos_w": [[0.2, 0.0, 0.5], [0.0, 0.3, 0.8]],
                "body_quat_w": [_yaw_quat(0.2), [1.0, 0.0, 0.0, 0.0]],
            },
            recv_time_ns=t1_ns,
        )

        joint_pos = np.zeros((2, 2), dtype=np.float32)
        joint_vel = np.empty_like(joint_pos)
        body_pos_w = np.zeros((2, 2, 3), dtype=np.float32)
        body_lin_vel_w = np.empty_like(body_pos_w)
        body_quat_w = np.zeros((2, 2, 4), dtype=np.float32)
        body_ang_vel_w = np.empty_like(body_pos_w)

        buffer._fill_sample_frames_locked(
            np.asarray([t0_ns + 50_000_000, t1_ns + 1_000_000_000], dtype=np.int64),
            joint_pos,
            joint_vel,
            body_pos_w,
            body_lin_vel_w,
            body_quat_w,
            body_ang_vel_w,
        )

        np.testing.assert_allclose(joint_pos[0], [0.1, -0.2], atol=1e-6)
        np.testing.assert_allclose(joint_vel[0], [2.0, -4.0], atol=1e-6)
        np.testing.assert_allclose(joint_pos[1], [0.2, -0.4], atol=1e-6)
        np.testing.assert_allclose(joint_vel[1], 0.0, atol=1e-6)
        np.testing.assert_allclose(body_pos_w[1, 0], [0.2, 0.0, 0.5], atol=1e-6)

    def test_first_frame_does_not_propagate_to_state_processor_api(self) -> None:
        self.assertFalse(hasattr(StateProcessor, "consume_motion_first_frame"))

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
