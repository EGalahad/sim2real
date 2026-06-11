#!/usr/bin/env python3
"""
Live PICO/XRobot -> G1 ZMQ publisher.

This script reads XR body data from XRobotStreamer, retargets to Unitree G1 with
GMR, forward-kinematics the resulting MuJoCo qpos, and publishes a canonical
motion payload over ZMQ.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import Literal, Optional

import mujoco
import numpy as np
import tyro
import zmq
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import XRobotStreamer
from loop_rate_limiters import RateLimiter

from sim2real.config.robots import get_robot_cfg
from sim2real.config.robots.base import (
    BODY_POS_W_KEY,
    BODY_QUAT_W_KEY,
    JOINT_POS_KEY,
    PICO_RECV_TIME_NS_KEY,
    PUBLISH_T_NS_KEY,
    SEQ_KEY,
    SMPLX_T_NS_KEY,
    XROBOT_BODY_NAMES_KEY,
    XROBOT_BODY_POS_W_KEY,
    XROBOT_BODY_QUAT_W_KEY,
)
from sim2real.utils.common import PORTS, PicoControllerStateMessage
from sim2real.utils.math import quat_conjugate, quat_mul, quat_rotate_inverse_numpy, quat_rotate_numpy, yaw_quat
from sim2real.utils.profiling import ScopedTimer


BODY_POSE_TIMER_NAME = "pico_retarget_pub.body_pose_dict_from_streamer"
RETARGET_TIMER_NAME = "pico_retarget_pub.gmr_retarget"
MODE_HINT_INTERVAL_S = 5.0
POLL_TIMER_NAME = "pico_retarget_pub.poll_pause_toggle"
PAUSE_CAPTURE_TIMER_NAME = "pico_retarget_pub.capture_paused_qpos"
LIVE_TRANSFORM_TIMER_NAME = "pico_retarget_pub.transform_live_body_pose_dict"
SKIP_MAP_TIMER_NAME = "pico_retarget_pub.skip_map_to_robot_bodies"
MIN_HEIGHT_TIMER_NAME = "pico_retarget_pub.apply_min_link_height"
BUILD_PAYLOAD_TIMER_NAME = "pico_retarget_pub.build_payload"
SAMPLE_TIMER_NAME = "pico_retarget_pub.sample_and_retarget"
SEND_TIMER_NAME = "pico_retarget_pub.send_payload"


@dataclass(frozen=True)
class XRobotBodyFrame:
    names: tuple[str, ...]
    body_pos_w: np.ndarray
    body_quat_w: np.ndarray


def _pos_no_z_from_pos(pos: np.ndarray) -> np.ndarray:
    pos_arr = np.asarray(pos, dtype=np.float32).reshape(-1)
    pos_no_z = np.zeros(3, dtype=np.float32)
    pos_no_z[:2] = pos_arr[:2]
    return pos_no_z


def _pelvis_pose_no_z_yaw_from_body_pose_dict(
    body_pose_dict: dict[str, list[np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    pelvis_pose = body_pose_dict["Pelvis"]
    pelvis_pos_no_z = _pos_no_z_from_pos(pelvis_pose[0])
    pelvis_quat = np.asarray(pelvis_pose[1], dtype=np.float32).reshape(-1)
    pelvis_quat_yaw = yaw_quat(pelvis_quat).reshape(4)

    return pelvis_pos_no_z, pelvis_quat_yaw


def _transform_body_pose_dict(
    body_pose_dict: dict[str, list[np.ndarray]],
    *,
    src_pelvis_pos_no_z: np.ndarray,
    src_pelvis_quat_yaw: np.ndarray,
    dst_pelvis_pos_no_z: np.ndarray,
    dst_pelvis_quat_yaw: np.ndarray,
) -> dict[str, list[np.ndarray]]:
    relative_quat_yaw = quat_mul(
        dst_pelvis_quat_yaw,
        quat_conjugate(src_pelvis_quat_yaw),
    ).astype(np.float32, copy=False)

    transformed_body_pose_dict: dict[str, list[np.ndarray]] = {}
    for body_name, pose in body_pose_dict.items():
        body_pos = np.asarray(pose[0], dtype=np.float32).reshape(-1)
        body_quat = np.asarray(pose[1], dtype=np.float32).reshape(-1)

        body_pos_out = body_pos.copy()
        body_pos_local = quat_rotate_inverse_numpy(
            src_pelvis_quat_yaw[None, :],
            (body_pos[:3] - src_pelvis_pos_no_z)[None, :],
        )[0]
        body_pos_transformed = quat_rotate_numpy(
            dst_pelvis_quat_yaw[None, :],
            body_pos_local[None, :],
        )[0] + dst_pelvis_pos_no_z
        body_pos_out[:3] = body_pos_transformed[:3]

        body_quat_out = body_quat.copy()
        body_quat_out = quat_mul(relative_quat_yaw, body_quat).astype(
            np.float32,
            copy=False,
        )

        transformed_body_pose_dict[body_name] = [body_pos_out, body_quat_out]

    return transformed_body_pose_dict

def _xrobot_body_frame_from_pose_dict(
    body_pose_dict: dict[str, list[np.ndarray]],
    body_names: Optional[tuple[str, ...]] = None,
) -> XRobotBodyFrame:
    ordered_names = body_names if body_names is not None else tuple(str(name) for name in body_pose_dict.keys())
    body_pos_w = np.zeros((len(ordered_names), 3), dtype=np.float32)
    body_quat_w = np.zeros((len(ordered_names), 4), dtype=np.float32)
    for body_idx, body_name in enumerate(ordered_names):
        pos, quat = body_pose_dict[body_name]
        body_pos_w[body_idx] = np.asarray(pos, dtype=np.float32)
        body_quat_w[body_idx] = np.asarray(quat, dtype=np.float32)
    return XRobotBodyFrame(
        names=ordered_names,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
    )


def _xrobot_payload_dict(xrobot_frame: Optional[XRobotBodyFrame]) -> dict[str, object]:
    if xrobot_frame is None:
        return {}
    return {
        XROBOT_BODY_NAMES_KEY: list(xrobot_frame.names),
        XROBOT_BODY_POS_W_KEY: np.asarray(xrobot_frame.body_pos_w, dtype=np.float32).tolist(),
        XROBOT_BODY_QUAT_W_KEY: np.asarray(xrobot_frame.body_quat_w, dtype=np.float32).tolist(),
    }


def _body_pose_dict_from_streamer(
    streamer: XRobotStreamer,
) -> tuple[dict[str, list[np.ndarray]], int, int]:
    with ScopedTimer(BODY_POSE_TIMER_NAME):
        body_poses, _body_velocities, _body_accelerations, _imu_timestamps, body_timestamp = (
            streamer.get_raw_body_data()
        )
        pico_recv_time_ns = int(time.time_ns())
        if body_poses is None:
            raise RuntimeError("No XR body data available")

        body_pose_dict: dict[str, list[np.ndarray]] = {}
        for i, body_name in enumerate(streamer.body_joint_names):
            pose = np.asarray(body_poses[i], dtype=np.float32).reshape(-1)
            pos = pose[:3].astype(np.float32, copy=False)
            quat = np.asarray([pose[6], pose[3], pose[4], pose[5]], dtype=np.float32)
            body_pose_dict[body_name] = [pos, quat]

        # Keep the same coordinate transform that the streamer uses for its live path.
        body_pose_dict = streamer.coordinate_transform_unity_data(body_pose_dict).copy()
        return body_pose_dict, int(body_timestamp), pico_recv_time_ns


def _controller_button_pressed(controller_data: object, controller_name: str, key_name: str) -> bool:
    if not isinstance(controller_data, dict):
        return False
    controller = controller_data.get(controller_name)
    if not isinstance(controller, dict):
        return False
    return bool(controller.get(key_name, False))


def _pico_controller_state_from_data(controller_data: object) -> PicoControllerStateMessage:
    timestamp_ns = 0
    if isinstance(controller_data, dict):
        timestamp_ns = int(controller_data.get("timestamp", 0) or time.time_ns())

    return PicoControllerStateMessage(
        timestamp_ns=timestamp_ns,
        A=_controller_button_pressed(controller_data, "RightController", "key_one"),
        B=_controller_button_pressed(controller_data, "RightController", "key_two"),
        X=_controller_button_pressed(controller_data, "LeftController", "key_one"),
        Y=_controller_button_pressed(controller_data, "LeftController", "key_two"),
    )


def _controller_x_pressed(controller_data: object) -> bool:
    return _pico_controller_state_from_data(controller_data).X


class LiveRetargetPublisher:
    def __init__(self, args: "PublisherArgs"):
        self.args = args
        self.robot_cfg = get_robot_cfg(args.robot)
        self.publish_hz = float(args.publish_hz)
        if self.publish_hz <= 0:
            raise ValueError("publish_hz must be > 0")

        self.rate = RateLimiter(frequency=self.publish_hz, warn=True)
        self.streamer = XRobotStreamer()
        self.retarget = GMR(
            src_human="xrobot",
            tgt_robot="unitree_g1",
            actual_human_height=float(args.actual_human_height),
            verbose=bool(args.verbose),
        )

        self.mjcf_path = self.robot_cfg.resolve_mjcf_path()
        self.joint_qpos_indices = self._resolve_joint_qpos_indices()
        self.body_ids = self._resolve_body_ids()
        self.root_body_index = tuple(self.robot_cfg.body_names).index(self.retarget.robot_root_name)
        self.human_to_robot_body_name = self._resolve_human_to_robot_body_name()

        expected_qpos_size = self.robot_cfg.qpos_size
        if self.retarget.configuration.model.nq != expected_qpos_size:
            print(
                "[publish] warning: G1 MJCF qpos size mismatch "
                f"(model.nq={self.retarget.configuration.model.nq}, expected={expected_qpos_size})"
            )

        self.latest_qpos = np.asarray(self.robot_cfg.default_qpos, dtype=np.float32).copy()
        self.paused_qpos = self.latest_qpos.copy()
        (
            self.paused_joint_pos,
            self.paused_body_pos_w,
            self.paused_body_quat_w,
        ) = self._pose_arrays_from_qpos(self.paused_qpos)

        self.tracked_pelvis_pos_no_z = np.zeros(3, dtype=np.float32)
        self.tracked_pelvis_quat_yaw = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.live_start_tracked_pelvis_pos_no_z = np.zeros(3, dtype=np.float32)
        self.live_start_tracked_pelvis_quat_yaw = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.live_start_pico_pelvis_pos_no_z = np.zeros(3, dtype=np.float32)
        self.live_start_pico_pelvis_quat_yaw = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        self.last_stream_wait_log_monotonic = 0.0
        self.min_link_height_offset = 0.0
        self.min_link_height_offset_frame_count = 0
        self.paused = True
        self._needs_live_pelvis_init = False
        self._x_button_was_pressed = False
        self._latest_controller_t_ns = 0
        self._last_mode_hint_monotonic = 0.0

        self._controller_sock = zmq.Context.instance().socket(zmq.PUB)
        self._controller_sock.setsockopt(zmq.LINGER, 0)
        self._controller_sock.setsockopt(zmq.SNDHWM, int(args.controller_hwm))
        self._controller_sock.setsockopt(zmq.CONFLATE, 1)
        self._controller_sock.bind(args.controller_bind)

    def _resolve_joint_qpos_indices(self) -> list[int]:
        model = self.retarget.configuration.model
        joint_qpos_indices: list[int] = []
        for joint_name in self.robot_cfg.joint_names:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise ValueError(f"Failed to resolve joint name in MJCF: {joint_name}")
            joint_qpos_indices.append(int(model.jnt_qposadr[joint_id]))
        return joint_qpos_indices

    def _resolve_body_ids(self) -> list[int]:
        model = self.retarget.configuration.model
        body_ids: list[int] = []
        for body_name in self.robot_cfg.body_names:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0:
                raise ValueError(f"Failed to resolve body name in MJCF: {body_name}")
            body_ids.append(int(body_id))
        return body_ids

    def _resolve_human_to_robot_body_name(self) -> dict[str, str]:
        human_to_robot_body_name: dict[str, str] = {}
        for robot_body_name, entry in self.retarget.ik_match_table1.items():
            human_body_name = str(entry[0])
            human_to_robot_body_name.setdefault(human_body_name, str(robot_body_name))
        return human_to_robot_body_name

    def _pose_arrays_from_qpos(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        data = mujoco.MjData(self.retarget.configuration.model)
        data.qpos[:] = np.asarray(qpos, dtype=np.float32).reshape(-1)
        mujoco.mj_forward(self.retarget.configuration.model, data)
        joint_pos = np.asarray(data.qpos[self.joint_qpos_indices], dtype=np.float32)
        body_pos_w = np.asarray(data.xpos[self.body_ids], dtype=np.float32)
        body_quat_w = np.asarray(data.xquat[self.body_ids], dtype=np.float32)
        return joint_pos, body_pos_w, body_quat_w

    def _apply_min_link_height_offset(
        self,
        body_pose_dict: dict[str, list[np.ndarray]],
        *,
        raw_min_body_z: float,
    ) -> dict[str, list[np.ndarray]]:
        strategy = self.args.min_link_height_align_strategy
        if strategy == "none":
            return body_pose_dict

        frame_offset = float(self.args.min_link_height) - float(raw_min_body_z)

        if strategy == "per_frame":
            z_offset = frame_offset
        else:
            if strategy != "bootstrap":
                raise ValueError(f"Unsupported min_link_height_align_strategy: {strategy}")

            bootstrap_frames = int(self.args.min_link_height_bootstrap_frames)
            if self.min_link_height_offset_frame_count < bootstrap_frames:
                self.min_link_height_offset = frame_offset
                self.min_link_height_offset_frame_count += 1
                if self.min_link_height_offset_frame_count == bootstrap_frames:
                    print(
                        "[Info] fixed min-link-height offset calibrated: "
                        f"{self.min_link_height_offset:.6f} m"
                    )
            z_offset = self.min_link_height_offset

        body_pose_dict_adj: dict[str, list[np.ndarray]] = {}
        for body_name, pose in body_pose_dict.items():
            body_pos = np.asarray(pose[0], dtype=np.float32).copy()
            body_pos[2] += np.float32(z_offset)
            body_pose_dict_adj[body_name] = [body_pos, np.asarray(pose[1], dtype=np.float32).copy()]
        return body_pose_dict_adj

    def _capture_paused_qpos(self) -> None:
        with ScopedTimer(PAUSE_CAPTURE_TIMER_NAME):
            paused_qpos = np.asarray(self.robot_cfg.default_qpos, dtype=np.float32).copy()
            paused_qpos[:2] = np.asarray(self.latest_qpos[:2], dtype=np.float32)
            paused_qpos[3:7] = yaw_quat(np.asarray(self.latest_qpos[3:7], dtype=np.float32)).astype(
                np.float32,
                copy=False,
            )
            self.paused_qpos = paused_qpos
            (
                self.paused_joint_pos,
                self.paused_body_pos_w,
                self.paused_body_quat_w,
            ) = self._pose_arrays_from_qpos(self.paused_qpos)

    def _transform_live_body_pose_dict(
        self,
        body_pose_dict: dict[str, list[np.ndarray]],
    ) -> dict[str, list[np.ndarray]]:
        with ScopedTimer(LIVE_TRANSFORM_TIMER_NAME):
            pelvis_pose_no_z_yaw = _pelvis_pose_no_z_yaw_from_body_pose_dict(body_pose_dict)
            current_pico_pelvis_pos_no_z, current_pico_pelvis_quat_yaw = pelvis_pose_no_z_yaw

            if self._needs_live_pelvis_init:
                self.live_start_tracked_pelvis_pos_no_z = self.tracked_pelvis_pos_no_z.copy()
                self.live_start_tracked_pelvis_quat_yaw = self.tracked_pelvis_quat_yaw.copy()
                self.live_start_pico_pelvis_pos_no_z = current_pico_pelvis_pos_no_z.copy()
                self.live_start_pico_pelvis_quat_yaw = current_pico_pelvis_quat_yaw.copy()
                self._needs_live_pelvis_init = False

            transformed_body_pose_dict = _transform_body_pose_dict(
                body_pose_dict,
                src_pelvis_pos_no_z=self.live_start_pico_pelvis_pos_no_z,
                src_pelvis_quat_yaw=self.live_start_pico_pelvis_quat_yaw,
                dst_pelvis_pos_no_z=self.live_start_tracked_pelvis_pos_no_z,
                dst_pelvis_quat_yaw=self.live_start_tracked_pelvis_quat_yaw,
            )

            tracked_pelvis_pose_no_z_yaw = _pelvis_pose_no_z_yaw_from_body_pose_dict(transformed_body_pose_dict)
            self.tracked_pelvis_pos_no_z = tracked_pelvis_pose_no_z_yaw[0].copy()
            self.tracked_pelvis_quat_yaw = tracked_pelvis_pose_no_z_yaw[1].copy()
            return transformed_body_pose_dict

    def _robot_body_pose_dict_from_human_pose_dict(
        self,
        body_pose_dict: dict[str, list[np.ndarray]],
    ) -> dict[str, list[np.ndarray]]:
        robot_body_pose_dict: dict[str, list[np.ndarray]] = {}
        for human_body_name, pose in body_pose_dict.items():
            robot_body_name = self.human_to_robot_body_name.get(human_body_name)
            if robot_body_name is None:
                continue
            robot_body_pose_dict[robot_body_name] = [
                np.asarray(pose[0], dtype=np.float32),
                np.asarray(pose[1], dtype=np.float32),
            ]
        return robot_body_pose_dict

    def _canonical_body_arrays_from_pose_dict(
        self,
        body_pose_dict: dict[str, list[np.ndarray]],
    ) -> tuple[np.ndarray, np.ndarray]:
        body_pos_w = np.full((len(self.robot_cfg.body_names), 3), np.nan, dtype=np.float32)
        body_quat_w = np.full((len(self.robot_cfg.body_names), 4), np.nan, dtype=np.float32)
        for body_idx, body_name in enumerate(self.robot_cfg.body_names):
            pose = body_pose_dict.get(body_name)
            if pose is None:
                continue
            body_pos_w[body_idx] = np.asarray(pose[0], dtype=np.float32)
            body_quat_w[body_idx] = np.asarray(pose[1], dtype=np.float32)
        return body_pos_w, body_quat_w

    def _skip_retarget_qpos_from_scaled_human_data(
        self,
        body_pose_dict: dict[str, list[np.ndarray]],
    ) -> np.ndarray:
        qpos = np.asarray(self.robot_cfg.default_qpos, dtype=np.float32).copy()
        pelvis_pos_no_z, pelvis_quat_yaw = _pelvis_pose_no_z_yaw_from_body_pose_dict(body_pose_dict)
        qpos[0:2] = pelvis_pos_no_z[:2].copy()
        qpos[3:7] = pelvis_quat_yaw.copy()
        return qpos

    def _poll_pause_toggle(self) -> bool:
        with ScopedTimer(POLL_TIMER_NAME):
            try:
                controller_data = self.streamer.get_controller_data()
            except Exception as exc:
                now = time.monotonic()
                if now - self.last_stream_wait_log_monotonic > 2.0:
                    print(f"[Info] Waiting for PICO controller data... ({exc})")
                    self.last_stream_wait_log_monotonic = now
                return False

            controller_state = _pico_controller_state_from_data(controller_data)
            self._latest_controller_t_ns = int(controller_state.timestamp_ns)
            self._publish_controller_state(controller_state)

            return controller_state.X

    def _publish_controller_state(self, controller_state: PicoControllerStateMessage) -> None:
        try:
            self._controller_sock.send(controller_state.to_bytes(), flags=zmq.NOBLOCK)
        except zmq.Again:
            pass

    def _build_payload(
        self,
        *,
        source_smplx_t_ns: int,
        pico_recv_time_ns: int,
        body_pos_w: np.ndarray,
        body_quat_w: np.ndarray,
        joint_pos: np.ndarray,
        qpos: Optional[np.ndarray] = None,
        xrobot_frame: Optional[XRobotBodyFrame] = None,
    ) -> dict[str, object]:
        with ScopedTimer(BUILD_PAYLOAD_TIMER_NAME):
            publish_t_ns = int(time.time_ns())

            payload = {
                PUBLISH_T_NS_KEY: publish_t_ns,
                SMPLX_T_NS_KEY: int(source_smplx_t_ns),
                PICO_RECV_TIME_NS_KEY: int(pico_recv_time_ns),
                "paused": self.paused,
                JOINT_POS_KEY: np.asarray(joint_pos, dtype=np.float32).tolist(),
                BODY_POS_W_KEY: np.asarray(body_pos_w, dtype=np.float32).tolist(),
                BODY_QUAT_W_KEY: np.asarray(body_quat_w, dtype=np.float32).tolist(),
            }
            if qpos is not None:
                qpos_arr = np.asarray(qpos, dtype=np.float32).reshape(-1)
                self.latest_qpos = qpos_arr.copy()
                payload["qpos"] = qpos_arr.tolist()
            payload.update(_xrobot_payload_dict(xrobot_frame))
            return payload

    def _timer_avg_ms(self, timer_name: str) -> float:
        timer = ScopedTimer._instances.get(timer_name)
        if timer is None or timer.count <= 0:
            return 0.0
        return float(timer.time / timer.count * 1000.0)

    def _timer_last_ms(self, timer_name: str) -> float:
        timer = ScopedTimer._instances.get(timer_name)
        if timer is None or not hasattr(timer, "last_time"):
            return 0.0
        return float(timer.last_time * 1000.0)

    def _reset_late_frame_timers(self) -> None:
        for timer_name in (
            POLL_TIMER_NAME,
            BODY_POSE_TIMER_NAME,
            LIVE_TRANSFORM_TIMER_NAME,
            RETARGET_TIMER_NAME,
            SKIP_MAP_TIMER_NAME,
            MIN_HEIGHT_TIMER_NAME,
            BUILD_PAYLOAD_TIMER_NAME,
            SEND_TIMER_NAME,
            PAUSE_CAPTURE_TIMER_NAME,
            SAMPLE_TIMER_NAME,
        ):
            timer = ScopedTimer._instances.get(timer_name)
            if timer is not None:
                timer.last_time = 0.0

    def _maybe_print_late_frame_breakdown(self, loop_elapsed_s: float) -> None:
        frame_budget_s = 1.0 / self.publish_hz
        if loop_elapsed_s <= frame_budget_s:
            return

        print(
            "[Warning] late frame breakdown: "
            f"loop={loop_elapsed_s * 1000.0:.3f} ms, "
            f"budget={frame_budget_s * 1000.0:.3f} ms, "
            f"poll={self._timer_last_ms(POLL_TIMER_NAME):.3f} ms, "
            f"body_pose={self._timer_last_ms(BODY_POSE_TIMER_NAME):.3f} ms, "
            f"transform={self._timer_last_ms(LIVE_TRANSFORM_TIMER_NAME):.3f} ms, "
            f"retarget={self._timer_last_ms(RETARGET_TIMER_NAME):.3f} ms, "
            f"skip_map={self._timer_last_ms(SKIP_MAP_TIMER_NAME):.3f} ms, "
            f"min_height={self._timer_last_ms(MIN_HEIGHT_TIMER_NAME):.3f} ms, "
            f"build_payload={self._timer_last_ms(BUILD_PAYLOAD_TIMER_NAME):.3f} ms, "
            f"send={self._timer_last_ms(SEND_TIMER_NAME):.3f} ms"
        )

    def _maybe_print_mode_hint(self) -> None:
        now = time.monotonic()
        if now - self._last_mode_hint_monotonic < MODE_HINT_INTERVAL_S:
            return
        self._last_mode_hint_monotonic = now

        if self.paused:
            print("[Info] mode=pause, press x to resume")
            return

        body_pose_avg_ms = self._timer_avg_ms(BODY_POSE_TIMER_NAME)
        retarget_avg_ms = self._timer_avg_ms(RETARGET_TIMER_NAME)
        retarget_label = "self.retarget.update_targets" if self.args.skip_retarget else "self.retarget.retarget"
        print(
            "[Info] mode=live, press x to pause, retargeting avg stats: "
            f"_body_pose_dict_from_streamer={body_pose_avg_ms:.3f} ms, "
            f"{retarget_label}={retarget_avg_ms:.3f} ms"
        )

    def sample_and_retarget(self) -> Optional[dict[str, object]]:
        with ScopedTimer(SAMPLE_TIMER_NAME):
            x_pressed = self._poll_pause_toggle()
            if x_pressed and not self._x_button_was_pressed:
                self.paused = not self.paused

                if self.paused:
                    self._capture_paused_qpos()
                else:
                    self._needs_live_pelvis_init = True
                print(f"[Info] paused toggled to {self.paused} via PICO X button")
            self._x_button_was_pressed = x_pressed

            self._maybe_print_mode_hint()
            if self.paused:
                return self._build_payload(
                    source_smplx_t_ns=self._latest_controller_t_ns,
                    pico_recv_time_ns=time.time_ns(),
                    body_pos_w=self.paused_body_pos_w,
                    body_quat_w=self.paused_body_quat_w,
                    joint_pos=self.paused_joint_pos,
                    qpos=self.paused_qpos,
                    xrobot_frame=None,
                )

            try:
                smplx_data, source_smplx_t_ns, pico_recv_time_ns = _body_pose_dict_from_streamer(self.streamer)
                if source_smplx_t_ns > 0:
                    delay_ms = (pico_recv_time_ns - int(source_smplx_t_ns)) / 1e6
                    print(
                        "[pico body timestamp] "
                        f"source_smplx_t_ns={int(source_smplx_t_ns)} "
                        f"pico_recv_time_ns={pico_recv_time_ns} "
                        f"delay_ms={delay_ms:.3f}",
                        flush=True,
                    )
                else:
                    print(
                        "[pico body timestamp] "
                        f"source_smplx_t_ns={int(source_smplx_t_ns)} "
                        f"pico_recv_time_ns={pico_recv_time_ns} "
                        "delay_ms=unavailable",
                        flush=True,
                    )
            except RuntimeError:
                now = time.monotonic()
                if now - self.last_stream_wait_log_monotonic > 2.0:
                    print("[Info] Waiting for XR body data from PICO...")
                    self.last_stream_wait_log_monotonic = now
                return None

            raw_min_body_z = float(min(np.asarray(pose[0], dtype=np.float32)[2] for pose in smplx_data.values()))
            smplx_data = self._transform_live_body_pose_dict(smplx_data)
            with ScopedTimer(MIN_HEIGHT_TIMER_NAME):
                smplx_data = self._apply_min_link_height_offset(
                    smplx_data,
                    raw_min_body_z=raw_min_body_z,
                )

            with ScopedTimer(RETARGET_TIMER_NAME):
                if self.args.skip_retarget:
                    self.retarget.update_targets(smplx_data, offset_to_ground=False)
                else:
                    self.retarget.retarget(smplx_data, offset_to_ground=False)

            processed_human_data = getattr(self.retarget, "scaled_human_data", None)
            processed_human_data["Pelvis"][0][2] += 0.05
            xrobot_frame = None
            if isinstance(processed_human_data, dict) and processed_human_data:
                xrobot_frame = _xrobot_body_frame_from_pose_dict(processed_human_data)

            if self.args.skip_retarget:
                with ScopedTimer(SKIP_MAP_TIMER_NAME):
                    robot_body_pose_dict = self._robot_body_pose_dict_from_human_pose_dict(processed_human_data)
                    body_pos_w, body_quat_w = self._canonical_body_arrays_from_pose_dict(robot_body_pose_dict)
                    joint_pos = np.full((len(self.robot_cfg.joint_names),), np.nan, dtype=np.float32)
                    qpos = self._skip_retarget_qpos_from_scaled_human_data(processed_human_data)
                return self._build_payload(
                    source_smplx_t_ns=int(source_smplx_t_ns),
                    pico_recv_time_ns=pico_recv_time_ns,
                    body_pos_w=body_pos_w,
                    body_quat_w=body_quat_w,
                    joint_pos=joint_pos,
                    qpos=qpos,
                    xrobot_frame=xrobot_frame,
                )
            else:
                configuration_data = self.retarget.configuration.data
                body_pos_w = np.asarray(configuration_data.xpos[self.body_ids], dtype=np.float32)
                body_quat_w = np.asarray(configuration_data.xquat[self.body_ids], dtype=np.float32)
                joint_pos = np.asarray(configuration_data.qpos[self.joint_qpos_indices], dtype=np.float32)
                return self._build_payload(
                    source_smplx_t_ns=int(source_smplx_t_ns),
                    pico_recv_time_ns=pico_recv_time_ns,
                    body_pos_w=body_pos_w,
                    body_quat_w=body_quat_w,
                    joint_pos=joint_pos,
                    qpos=np.asarray(configuration_data.qpos, dtype=np.float32),
                    xrobot_frame=xrobot_frame,
                )

    def close(self) -> None:
        self._controller_sock.close(0)


def run_publish(args: "PublisherArgs") -> None:
    worker = LiveRetargetPublisher(args)

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUB)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.SNDHWM, int(args.hwm))
    sock.setsockopt(zmq.CONFLATE, 1)
    sock.bind(args.bind)

    print(
        f"[publish] bind={args.bind} publish_hz={args.publish_hz} "
        f"controller_bind={args.controller_bind} "
        f"mjcf={worker.robot_cfg.mjcf_path} resolved={worker.mjcf_path}"
    )
    if args.startup_sleep_s > 0:
        time.sleep(float(args.startup_sleep_s))

    seq = 0
    try:
        while True:
            worker._reset_late_frame_timers()
            loop_start = time.perf_counter()
            payload = worker.sample_and_retarget()
            if payload is not None:
                payload[SEQ_KEY] = seq
                with ScopedTimer(SEND_TIMER_NAME):
                    sock.send_string(
                        json.dumps(payload, separators=(",", ":")),
                        flags=zmq.NOBLOCK,
                    )
                seq += 1
            worker._maybe_print_late_frame_breakdown(time.perf_counter() - loop_start)
            worker.rate.sleep()
    except KeyboardInterrupt:
        print("KeyboardInterrupt, exiting publisher.")
    finally:
        worker.close()
        sock.close(0)


@dataclass
class PublisherArgs:
    """Receive PICO/XRobot stream, retarget, and publish canonical motion over ZMQ."""

    robot: str = "g1"
    bind: str = "tcp://*:28701"
    controller_bind: str = f"tcp://*:{PORTS['pico_controller']}"
    publish_hz: float = 30.0
    hwm: int = 1
    controller_hwm: int = 1
    startup_sleep_s: float = 0.5
    actual_human_height: float = 1.6
    skip_retarget: bool = False
    min_link_height: float = 0.01
    min_link_height_align_strategy: Literal["none", "per_frame", "bootstrap"] = "bootstrap"
    min_link_height_bootstrap_frames: int = 30
    verbose: bool = False


if __name__ == "__main__":
    run_publish(tyro.cli(PublisherArgs))
