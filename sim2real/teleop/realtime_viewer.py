#!/usr/bin/env python3
"""
ZMQ viewer for retargeted G1 poses.

Subscribe to a live motion stream published by pico_retarget_pub.py and
render the robot in a MuJoCo viewer window.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import Optional

import mujoco
import mujoco.viewer
import numpy as np
import tyro
import zmq
from loop_rate_limiters import RateLimiter
from mjhub import temp_mjcf_with_floor

from sim2real.config.robots import RobotCfg, get_robot_cfg
from sim2real.config.robots.base import (
    BODY_POS_W_KEY,
    BODY_QUAT_W_KEY,
    JOINT_POS_KEY,
    XROBOT_BODY_NAMES_KEY,
    XROBOT_BODY_POS_W_KEY,
    XROBOT_BODY_QUAT_W_KEY,
    resolve_mjcf_joint_names,
    resolve_mjcf_root_body_name,
)
from sim2real.teleop.motion_legacy import motion_to_qpos

GROUND_RGB = (0.6, 0.7, 0.6)
AXIS_FRAME_LENGTH_M = 0.2
AXIS_FRAME_WIDTH_M = 0.01
AXIS_FRAME_RGBA = (
    np.array([1.0, 0.2, 0.2, 1.0], dtype=np.float32),
    np.array([0.2, 1.0, 0.2, 1.0], dtype=np.float32),
    np.array([0.2, 0.4, 1.0, 1.0], dtype=np.float32),
)


@dataclass(frozen=True)
class XRobotBodyFramePayload:
    names: tuple[str, ...]
    body_pos_w: np.ndarray
    body_quat_w: np.ndarray


def _quat_wxyz_to_rotmat(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(quat)
    if not np.isfinite(norm) or norm <= 1e-8:
        raise ValueError(f"Invalid quaternion norm: {norm}")
    w, x, y, z = quat / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _axis_endpoints_from_pose(
    pos_w: np.ndarray,
    quat_wxyz: np.ndarray,
    *,
    axis_length: float = AXIS_FRAME_LENGTH_M,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    origin = np.asarray(pos_w, dtype=np.float64).reshape(3)
    rotmat = _quat_wxyz_to_rotmat(quat_wxyz)
    return tuple(
        (origin, origin + rotmat[:, axis_idx] * float(axis_length))
        for axis_idx in range(3)
    )


def _parse_xrobot_body_payload(payload: dict[str, object]) -> Optional[XRobotBodyFramePayload]:
    names_raw = payload.get(XROBOT_BODY_NAMES_KEY)
    body_pos_raw = payload.get(XROBOT_BODY_POS_W_KEY)
    body_quat_raw = payload.get(XROBOT_BODY_QUAT_W_KEY)
    if names_raw is None or body_pos_raw is None or body_quat_raw is None:
        return None

    if not isinstance(names_raw, (list, tuple)):
        return None
    names = tuple(str(name) for name in names_raw)
    if not names:
        return None

    try:
        body_pos_w = np.asarray(body_pos_raw, dtype=np.float32)
        body_quat_w = np.asarray(body_quat_raw, dtype=np.float32)
    except (TypeError, ValueError):
        return None

    if body_pos_w.shape != (len(names), 3):
        return None
    if body_quat_w.shape != (len(names), 4):
        return None
    if not np.isfinite(body_pos_w).all() or not np.isfinite(body_quat_w).all():
        return None
    quat_norms = np.linalg.norm(body_quat_w, axis=1)
    if np.any(quat_norms <= 1e-8):
        return None

    return XRobotBodyFramePayload(
        names=names,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
    )


class NativeG1Viewer:
    def __init__(self, robot_cfg: RobotCfg) -> None:
        self.robot_cfg = robot_cfg
        self.mjcf_path = self.robot_cfg.resolve_mjcf_path()
        with temp_mjcf_with_floor(
            self.mjcf_path,
            ground_rgb=GROUND_RGB,
        ) as viewer_mjcf_path:
            self.model = mujoco.MjModel.from_xml_path(str(viewer_mjcf_path))
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco.viewer.launch_passive(
            self.model,
            self.data,
            show_left_ui=False,
            show_right_ui=False,
        )
        self.track_body_id = self._resolve_track_body_id()
        if self.track_body_id is not None:
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.viewer.cam.trackbodyid = self.track_body_id
        self.viewer.cam.distance = 3.0
        self.viewer.cam.elevation = -10

    def _resolve_track_body_id(self) -> Optional[int]:
        for body_name in self.robot_cfg.viewer_track_body_names:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id >= 0:
                return int(body_id)
        return None

    def is_running(self) -> bool:
        return bool(self.viewer.is_running())

    def _clear_user_scene(self) -> None:
        if self.viewer.user_scn is None:
            return
        self.viewer.user_scn.ngeom = 0

    def _append_axis_geom(
        self,
        start_w: np.ndarray,
        end_w: np.ndarray,
        rgba: np.ndarray,
    ) -> None:
        scene = self.viewer.user_scn
        if scene is None or scene.ngeom >= len(scene.geoms):
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_ARROW,
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            np.eye(3, dtype=np.float64).reshape(9),
            np.asarray(rgba, dtype=np.float32),
        )
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_ARROW,
            AXIS_FRAME_WIDTH_M,
            np.asarray(start_w, dtype=np.float64),
            np.asarray(end_w, dtype=np.float64),
        )
        geom.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
        geom.objtype = -1
        geom.objid = -1
        geom.segid = -1
        scene.ngeom += 1

    def _update_xrobot_frames(
        self,
        xrobot_frame: Optional[XRobotBodyFramePayload],
        *,
        show_xrobot_frames: bool,
    ) -> None:
        self._clear_user_scene()
        if not show_xrobot_frames or xrobot_frame is None:
            return

        for body_idx in range(len(xrobot_frame.names)):
            axis_endpoints = _axis_endpoints_from_pose(
                xrobot_frame.body_pos_w[body_idx],
                xrobot_frame.body_quat_w[body_idx],
            )
            for axis_idx, (start_w, end_w) in enumerate(axis_endpoints):
                self._append_axis_geom(start_w, end_w, AXIS_FRAME_RGBA[axis_idx])

    def render(
        self,
        qpos: np.ndarray,
        *,
        xrobot_frame: Optional[XRobotBodyFramePayload] = None,
        show_xrobot_frames: bool = True,
    ) -> None:
        qpos_arr = np.asarray(qpos, dtype=np.float32).reshape(-1)
        with self.viewer.lock():
            self.data.qpos[:] = 0.0
            self.data.qvel[:] = 0.0
            self.data.qpos[: min(self.model.nq, qpos_arr.shape[0])] = qpos_arr[: self.model.nq]
            mujoco.mj_forward(self.model, self.data)
            self._update_xrobot_frames(
                xrobot_frame,
                show_xrobot_frames=show_xrobot_frames,
            )
        self.viewer.sync()

    def close(self) -> None:
        self.viewer.close()


@dataclass
class ViewerArgs:
    """Receive a live ZMQ motion stream and visualize it in MuJoCo."""

    robot: str = "g1"
    connect: str = "tcp://127.0.0.1:28701"
    viewer_hz: float = 30.0
    hwm: int = 1
    show_xrobot_frames: bool = True


def run_viewer(args: ViewerArgs) -> None:
    robot_cfg = get_robot_cfg(args.robot)
    viewer = NativeG1Viewer(robot_cfg)
    mjcf_path = robot_cfg.resolve_mjcf_path()
    mjcf_root_body_name = resolve_mjcf_root_body_name(mjcf_path)
    mjcf_joint_names = resolve_mjcf_joint_names(mjcf_path)
    try:
        run_zmq_viewer(
            args,
            robot_cfg,
            viewer,
            mjcf_root_body_name,
            mjcf_joint_names,
        )
    finally:
        viewer.close()


def run_zmq_viewer(
    args: ViewerArgs,
    robot_cfg: RobotCfg,
    viewer: NativeG1Viewer,
    mjcf_root_body_name: str,
    mjcf_joint_names: tuple[str, ...],
) -> None:
    rate = RateLimiter(frequency=float(args.viewer_hz), warn=False)

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVHWM, int(args.hwm))
    sock.setsockopt(zmq.CONFLATE, 1)
    sock.connect(args.connect)
    sock.setsockopt(zmq.SUBSCRIBE, b"")

    latest_qpos = np.asarray(robot_cfg.default_qpos, dtype=np.float32).copy()
    latest_xrobot_frame: Optional[XRobotBodyFramePayload] = None
    last_recv_log = 0.0
    print(f"[viewer] backend=zmq connect={args.connect} viewer_hz={args.viewer_hz}")

    try:
        while viewer.is_running():
            try:
                while True:
                    raw = sock.recv_string(flags=zmq.NOBLOCK)
                    payload = json.loads(raw)
                    if not isinstance(payload, dict):
                        continue
                    latest_xrobot_frame = _parse_xrobot_body_payload(payload)
                    qpos = None
                    qpos = payload.get("qpos")
                    if qpos is not None:
                        qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)
                        if qpos.shape[0] < robot_cfg.qpos_size:
                            qpos = None
                    if qpos is None:
                        joint_pos = payload.get(JOINT_POS_KEY)
                        body_pos_w = payload.get(BODY_POS_W_KEY)
                        body_quat_w = payload.get(BODY_QUAT_W_KEY)
                        if joint_pos is None or body_pos_w is None or body_quat_w is None:
                            continue
                        qpos = motion_to_qpos(
                            body_pos_w,
                            body_quat_w,
                            joint_pos,
                            robot_cfg,
                            mjcf_root_body_name,
                            mjcf_joint_names,
                        )
                    if qpos is not None:
                        latest_qpos = qpos
                        last_recv_log = time.monotonic()
            except zmq.Again:
                pass
            except json.JSONDecodeError as exc:
                print(f"[viewer] bad JSON payload: {exc}")

            if time.monotonic() - last_recv_log > 2.0:
                last_recv_log = time.monotonic()

            viewer.render(
                latest_qpos,
                xrobot_frame=latest_xrobot_frame,
                show_xrobot_frames=bool(args.show_xrobot_frames),
            )
            rate.sleep()
    except KeyboardInterrupt:
        print("KeyboardInterrupt, exiting viewer.")
    finally:
        sock.close(0)


def main() -> None:
    run_viewer(tyro.cli(ViewerArgs))


if __name__ == "__main__":
    main()
