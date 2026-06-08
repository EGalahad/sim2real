from __future__ import annotations

import sched
import re
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Literal

import mujoco
import numpy as np
import tyro
from loguru import logger

from sim2real.config.robots import get_robot_cfg
from sim2real.rl_policy.controllers.passive import PassiveController
from sim2real.rl_policy.tracking import Tracking, TrackingArgs
from sim2real.sim_env.base_sim import BaseSimulator
from sim2real.utils.math import (
    projected_yaw_quat,
    quat_conjugate,
    quat_mul,
    quat_rotate_inverse_numpy,
)

try:
    import glfw
except Exception:  # pragma: no cover - glfw is only needed for interactive replay.
    glfw = None


TRACKING_BODY_PATTERNS = (
    "pelvis",
    "torso_link",
    ".*_hip_yaw_link",
    ".*_knee_link",
    ".*_toe_link",
    ".*_shoulder_yaw_link",
    ".*_elbow_link",
    ".*_wrist_yaw_link",
)
TERMINATION_ROOT_BODY_NAME = "torso_link"
ANCHOR_BODY_NAME = "pelvis"


class _IntegratedTracking(Tracking):
    def _build_controller(self):
        self.keyboard_controller = None
        self.joystick_controller = None
        self.pico_controller = None
        return PassiveController()


class IntegratedSim2Sim:
    def __init__(self, args: "IntegratedSim2SimArgs"):
        self.args = args
        self.restart_requested = Event()

        policy_args = TrackingArgs(
            policy_config=args.policy_config,
            robot=args.robot,
            rl_rate=args.rl_rate,
            inference_backend=args.inference_backend,
            controller="keyboard",
            record=args.record,
            record_output=args.record_output,
            motion_backend="npz",
            motion_path=args.motion_path,
            max_future=args.max_future,
        )
        self.policy = _IntegratedTracking(args=policy_args)
        self.policy.total_inference_cnt = 0
        self.policy.perf_dict = {}

        self.sim = BaseSimulator(
            get_robot_cfg(args.robot),
            sim_dt=args.sim_dt,
            decimation=args.decimation,
            enable_elastic_band=args.enable_elastic_band,
            headless=args.headless,
            key_callback=self._on_mujoco_key if not args.headless else None,
        )
        self.policy.state_processor._prepare_low_state = self._prepare_integrated_low_state
        self.root_trajectory: list[dict[str, np.ndarray | float | int]] = []
        self.trajectory: list[dict[str, np.ndarray | float | int]] = []
        self._trajectory_body_names: list[str] | None = None
        self._trajectory_robot_body_ids: list[int] | None = None
        self._trajectory_motion_body_indices: list[int] | None = None
        self._last_trajectory_motion_t: int | None = None
        self._tracking_failure_counts = {
            "root_ori_error": 0,
            "body_pos_error": 0,
            "body_ori_error": 0,
        }
        self._tracking_failure_detected = False
        self._tracking_failure_reason: str | None = None
        self._reset_playback()

    def _on_mujoco_key(self, key: int) -> None:
        if glfw is not None and key == glfw.KEY_SPACE:
            self.restart_requested.set()

    def _motion_frame(self, frame: int):
        state_processor = self.policy.state_processor
        return state_processor.motion_dataset.get_slice(
            state_processor.motion_ids,
            np.asarray([int(frame)], dtype=int),
            np.asarray([0], dtype=int),
        )

    def _set_robot_to_motion_frame(self, frame: int) -> None:
        motion_data = self._motion_frame(frame)
        state_processor = self.policy.state_processor
        bridge = self.sim.sim_bridge

        root_body_name = str(
            self.policy.policy_config.get("motion", {}).get("root_body_name", "pelvis")
        )
        root_body_idx = state_processor.motion_body_names.index(root_body_name)
        root_qpos = int(bridge.root_qpos_adr)
        root_qvel = int(bridge.root_qvel_adr)

        self.sim.mj_data.qpos[root_qpos : root_qpos + 3] = motion_data.body_pos_w[
            0, 0, root_body_idx
        ]
        self.sim.mj_data.qpos[root_qpos + 3 : root_qpos + 7] = motion_data.body_quat_w[
            0, 0, root_body_idx
        ]
        self.sim.mj_data.qvel[root_qvel : root_qvel + 6] = 0.0

        motion_joint_names = list(state_processor.motion_joint_names)
        for unitree_idx, qpos_addr, qvel_addr in zip(
            bridge.joint_indices_unitree,
            bridge.qpos_adrs,
            bridge.qvel_adrs,
        ):
            joint_name = self.sim.robot_cfg.joint_names[unitree_idx]
            if joint_name not in motion_joint_names:
                continue
            motion_idx = motion_joint_names.index(joint_name)
            self.sim.mj_data.qpos[qpos_addr] = motion_data.joint_pos[0, 0, motion_idx]
            self.sim.mj_data.qvel[qvel_addr] = 0.0

        mujoco.mj_forward(self.sim.mj_model, self.sim.mj_data)
        self.sim.sync_viewer()

    def _sync_policy_state_from_sim(self) -> None:
        state_processor = self.policy.state_processor
        bridge = self.sim.sim_bridge

        root_qpos = int(bridge.root_qpos_adr)
        root_qvel = int(bridge.root_qvel_adr)
        state_processor.root_pos_w[:] = self.sim.mj_data.qpos[root_qpos : root_qpos + 3]
        state_processor.root_quat_w[:] = self.sim.mj_data.qpos[
            root_qpos + 3 : root_qpos + 7
        ]
        state_processor.root_lin_vel_w[:] = self.sim.mj_data.qvel[
            root_qvel : root_qvel + 3
        ]
        state_processor.root_ang_vel_b[:] = self.sim.mj_data.qvel[
            root_qvel + 3 : root_qvel + 6
        ]

        joint_pos = np.zeros_like(state_processor.joint_pos)
        joint_vel = np.zeros_like(state_processor.joint_vel)
        for unitree_idx, qpos_addr, qvel_addr in zip(
            bridge.joint_indices_unitree,
            bridge.qpos_adrs,
            bridge.qvel_adrs,
        ):
            joint_pos[unitree_idx] = self.sim.mj_data.qpos[qpos_addr]
            joint_vel[unitree_idx] = self.sim.mj_data.qvel[qvel_addr]

        state_processor.joint_pos[:] = joint_pos
        state_processor.joint_vel[:] = joint_vel

    def _prepare_integrated_low_state(self) -> bool:
        self._sync_policy_state_from_sim()
        return True

    def _publish_initial_state(self) -> None:
        for _ in range(5):
            self.sim.sim_bridge.publish_low_state()
            if not self.args.headless:
                time.sleep(0.01)
            self.policy.state_processor._prepare_low_state()

    def _reset_playback(self) -> None:
        self.policy.state_dict = {
            "action": np.zeros(self.policy.num_actions, dtype=np.float32),
            "paused": True,
            "control_mode": "policy",
        }
        self.policy.state_processor.restart_motion()
        self._set_robot_to_motion_frame(0)
        self._publish_initial_state()
        self.policy.reset()
        self.policy.state_dict["control_mode"] = "policy"
        self.policy.state_dict["paused"] = True
        self.playback_started = False
        self.headless_elapsed_s = 0.0
        self.replay_start_time = time.perf_counter() + self.args.initial_pause_s
        self._last_trajectory_motion_t = None
        for key in self._tracking_failure_counts:
            self._tracking_failure_counts[key] = 0
        self._tracking_failure_detected = False
        self._tracking_failure_reason = None
        self.restart_requested.clear()
        logger.info(
            "Playback reset: robot set to motion frame 0; policy active; motion starts in {:.2f}s",
            self.args.initial_pause_s,
        )

    def _start_motion(self) -> None:
        self.policy.state_dict["paused"] = False
        self.playback_started = True
        logger.info("Motion playback started")

    def _maybe_start_motion(self) -> None:
        if self.playback_started:
            return
        if time.perf_counter() < self.replay_start_time:
            return
        self._start_motion()

    def _at_last_paused_frame(self) -> bool:
        state_processor = self.policy.state_processor
        if state_processor.motion_backend != "npz":
            return False
        return (
            bool(self.policy.state_dict.get("paused", False))
            and int(state_processor.motion_t[0]) >= int(state_processor.motion_length) - 1
        )

    def _motion_root_state(self) -> tuple[np.ndarray, np.ndarray]:
        state_processor = self.policy.state_processor
        root_body_name = str(
            self.policy.policy_config.get("motion", {}).get("root_body_name", "pelvis")
        )
        root_body_idx = state_processor.motion_body_names.index(root_body_name)
        motion_t = int(state_processor.motion_t[0])
        motion_data = self._motion_frame(motion_t)
        return (
            np.asarray(motion_data.body_pos_w[0, 0, root_body_idx], dtype=np.float32),
            np.asarray(motion_data.body_quat_w[0, 0, root_body_idx], dtype=np.float32),
        )

    def _prepare_trajectory_body_layout(self) -> None:
        if self._trajectory_body_names is not None:
            return
        state_processor = self.policy.state_processor
        configured_body_names = list(
            self.policy.policy_config.get("body_names_simulation")
            or self.policy.policy_config.get("motion", {}).get("body_names")
            or state_processor.motion_body_names
        )
        body_names: list[str] = []
        robot_body_ids: list[int] = []
        motion_body_indices: list[int] = []
        for body_name in configured_body_names:
            body_name = str(body_name)
            robot_body_id = mujoco.mj_name2id(
                self.sim.mj_model,
                mujoco.mjtObj.mjOBJ_BODY,
                body_name,
            )
            if robot_body_id < 0 or body_name not in state_processor.motion_body_names:
                continue
            body_names.append(body_name)
            robot_body_ids.append(int(robot_body_id))
            motion_body_indices.append(state_processor.motion_body_names.index(body_name))
        if not body_names:
            raise ValueError("Could not resolve any shared robot/motion body names for trajectory output")
        self._trajectory_body_names = body_names
        self._trajectory_robot_body_ids = robot_body_ids
        self._trajectory_motion_body_indices = motion_body_indices

    @staticmethod
    def _indices_for_patterns(names: list[str], patterns: tuple[str, ...]) -> list[int]:
        indices: list[int] = []
        for pattern in patterns:
            for idx, name in enumerate(names):
                if idx in indices:
                    continue
                if name == pattern or re.fullmatch(pattern, name):
                    indices.append(idx)
        if not indices:
            raise ValueError(f"No body names matched patterns: {patterns}")
        return indices

    @staticmethod
    def _quat_angle_magnitude(quat: np.ndarray, eps: float = 1.0e-9) -> np.ndarray:
        xyz_norm = np.linalg.norm(quat[..., 1:], axis=-1)
        return 2.0 * np.arctan2(xyz_norm, np.maximum(np.abs(quat[..., 0]), eps))

    @staticmethod
    def _local_tracking_state(
        body_pos_w: np.ndarray,
        body_quat_w: np.ndarray,
        anchor_idx: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        anchor_pos = body_pos_w[anchor_idx].copy()
        anchor_pos[2] = 0.0
        anchor_yaw = projected_yaw_quat(body_quat_w[anchor_idx].reshape(1, 4))[0]
        anchor_yaw_expanded = np.broadcast_to(anchor_yaw, body_quat_w.shape)
        body_pos_local = quat_rotate_inverse_numpy(
            anchor_yaw_expanded,
            body_pos_w - anchor_pos.reshape(1, 3),
        )
        body_quat_local = quat_mul(
            quat_conjugate(anchor_yaw_expanded),
            body_quat_w,
        )
        return body_pos_local, body_quat_local

    def _update_tracking_failure_state(
        self,
        robot_body_pos_w: np.ndarray,
        robot_body_quat_w: np.ndarray,
        motion_body_pos_w: np.ndarray,
        motion_body_quat_w: np.ndarray,
    ) -> None:
        if not self.args.stop_on_tracking_failure or self._tracking_failure_detected:
            return
        assert self._trajectory_body_names is not None
        names = self._trajectory_body_names
        tracking_indices = self._indices_for_patterns(names, TRACKING_BODY_PATTERNS)
        root_idx = names.index(TERMINATION_ROOT_BODY_NAME)
        anchor_idx = names.index(ANCHOR_BODY_NAME)

        robot_pos_local, robot_quat_local = self._local_tracking_state(
            robot_body_pos_w,
            robot_body_quat_w,
            anchor_idx,
        )
        motion_pos_local, motion_quat_local = self._local_tracking_state(
            motion_body_pos_w,
            motion_body_quat_w,
            anchor_idx,
        )
        root_ori_error = float(
            self._quat_angle_magnitude(
                quat_mul(
                    quat_conjugate(motion_body_quat_w[root_idx].reshape(1, 4)),
                    robot_body_quat_w[root_idx].reshape(1, 4),
                )
            )[0]
        )
        body_pos_error = float(
            np.linalg.norm(
                motion_pos_local[tracking_indices] - robot_pos_local[tracking_indices],
                axis=-1,
            ).max()
        )
        body_ori_error = float(
            self._quat_angle_magnitude(
                quat_mul(
                    quat_conjugate(motion_quat_local[tracking_indices]),
                    robot_quat_local[tracking_indices],
                )
            ).max()
        )

        checks = {
            "root_ori_error": (root_ori_error, 1.2, 25),
            "body_pos_error": (body_pos_error, 0.4, 5),
            "body_ori_error": (body_ori_error, 1.2, 5),
        }
        for name, (value, threshold, min_steps) in checks.items():
            if value >= threshold:
                self._tracking_failure_counts[name] += 1
            else:
                self._tracking_failure_counts[name] = 0
            if self._tracking_failure_counts[name] >= min_steps:
                self._tracking_failure_detected = True
                self._tracking_failure_reason = name
                logger.info(
                    "Stopping 统一 MuJoCo 评测链路 after tracking failure: {} "
                    "(value={:.4f}, threshold={:.4f}, min_steps={})",
                    name,
                    value,
                    threshold,
                    min_steps,
                )
                return

    def _append_trajectory_frame(self) -> None:
        if (
            self.args.root_trajectory_output is None
            and self.args.trajectory_output is None
        ) or not self.playback_started:
            return

        bridge = self.sim.sim_bridge
        root_qpos = int(bridge.root_qpos_adr)
        motion_t = int(self.policy.state_processor.motion_t[0])
        if self.args.trajectory_policy_frames_only and motion_t == self._last_trajectory_motion_t:
            return
        self._last_trajectory_motion_t = motion_t
        motion_root_pos, motion_root_quat = self._motion_root_state()
        root_frame = {
            "sim_time": float(self.sim.mj_data.time),
            "motion_t": motion_t,
            "robot_root_pos_w": np.asarray(
                self.sim.mj_data.qpos[root_qpos : root_qpos + 3],
                dtype=np.float32,
            ).copy(),
            "robot_root_quat_w": np.asarray(
                self.sim.mj_data.qpos[root_qpos + 3 : root_qpos + 7],
                dtype=np.float32,
            ).copy(),
            "motion_root_pos_w": motion_root_pos,
            "motion_root_quat_w": motion_root_quat,
        }
        if self.args.root_trajectory_output is not None:
            self.root_trajectory.append(root_frame)

        if self.args.trajectory_output is None:
            return

        self._prepare_trajectory_body_layout()
        assert self._trajectory_robot_body_ids is not None
        assert self._trajectory_motion_body_indices is not None
        motion_data = self._motion_frame(motion_t)
        robot_body_pos_w = np.asarray(
            self.sim.mj_data.xpos[self._trajectory_robot_body_ids],
            dtype=np.float32,
        ).copy()
        robot_body_quat_w = np.asarray(
            self.sim.mj_data.xquat[self._trajectory_robot_body_ids],
            dtype=np.float32,
        ).copy()
        motion_body_pos_w = np.asarray(
            motion_data.body_pos_w[0, 0, self._trajectory_motion_body_indices],
            dtype=np.float32,
        ).copy()
        motion_body_quat_w = np.asarray(
            motion_data.body_quat_w[0, 0, self._trajectory_motion_body_indices],
            dtype=np.float32,
        ).copy()
        self.trajectory.append(
            {
                **root_frame,
                "robot_body_pos_w": robot_body_pos_w,
                "robot_body_quat_w": robot_body_quat_w,
                "motion_body_pos_w": motion_body_pos_w,
                "motion_body_quat_w": motion_body_quat_w,
            }
        )
        self._update_tracking_failure_state(
            robot_body_pos_w,
            robot_body_quat_w,
            motion_body_pos_w,
            motion_body_quat_w,
        )

    @staticmethod
    def _relative_translation(end_pos: np.ndarray, start_pos: np.ndarray, start_quat: np.ndarray) -> np.ndarray:
        return quat_rotate_inverse_numpy(
            np.asarray(start_quat, dtype=np.float32).reshape(1, 4),
            (np.asarray(end_pos, dtype=np.float32) - np.asarray(start_pos, dtype=np.float32)).reshape(1, 3),
        )[0]

    def _save_root_trajectory(self) -> None:
        if self.args.root_trajectory_output is None or not self.root_trajectory:
            return

        output_path = Path(self.args.root_trajectory_output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        robot_root_pos_w = np.stack(
            [frame["robot_root_pos_w"] for frame in self.root_trajectory],
            axis=0,
        )
        robot_root_quat_w = np.stack(
            [frame["robot_root_quat_w"] for frame in self.root_trajectory],
            axis=0,
        )
        motion_root_pos_w = np.stack(
            [frame["motion_root_pos_w"] for frame in self.root_trajectory],
            axis=0,
        )
        motion_root_quat_w = np.stack(
            [frame["motion_root_quat_w"] for frame in self.root_trajectory],
            axis=0,
        )
        sim_time = np.asarray(
            [frame["sim_time"] for frame in self.root_trajectory],
            dtype=np.float32,
        )
        motion_t = np.asarray(
            [frame["motion_t"] for frame in self.root_trajectory],
            dtype=np.int32,
        )

        robot_relative_final_pos = self._relative_translation(
            robot_root_pos_w[-1],
            robot_root_pos_w[0],
            robot_root_quat_w[0],
        )
        motion_relative_final_pos = self._relative_translation(
            motion_root_pos_w[-1],
            motion_root_pos_w[0],
            motion_root_quat_w[0],
        )
        root_final_error = robot_relative_final_pos - motion_relative_final_pos

        np.savez_compressed(
            output_path,
            robot_root_pos_w=robot_root_pos_w,
            robot_root_quat_w=robot_root_quat_w,
            motion_root_pos_w=motion_root_pos_w,
            motion_root_quat_w=motion_root_quat_w,
            sim_time=sim_time,
            motion_t=motion_t,
            robot_start_pos_w=robot_root_pos_w[0],
            robot_end_pos_w=robot_root_pos_w[-1],
            motion_start_pos_w=motion_root_pos_w[0],
            motion_end_pos_w=motion_root_pos_w[-1],
            robot_relative_final_pos=robot_relative_final_pos.astype(np.float32),
            motion_relative_final_pos=motion_relative_final_pos.astype(np.float32),
            root_final_error=root_final_error.astype(np.float32),
            root_final_error_norm=np.asarray(
                np.linalg.norm(root_final_error),
                dtype=np.float32,
            ),
            root_final_error_xy_norm=np.asarray(
                np.linalg.norm(root_final_error[:2]),
                dtype=np.float32,
            ),
            motion_length=np.asarray(
                int(self.policy.state_processor.motion_length),
                dtype=np.int32,
            ),
            policy_config=np.asarray(str(self.args.policy_config)),
            motion_path=np.asarray(str(self.args.motion_path)),
            seed=np.asarray(-1 if self.args.seed is None else int(self.args.seed), dtype=np.int32),
        )
        logger.info("Saved root trajectory to {}", output_path)

    def _save_trajectory(self) -> None:
        if self.args.trajectory_output is None or not self.trajectory:
            return

        output_path = Path(self.args.trajectory_output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        assert self._trajectory_body_names is not None

        np.savez_compressed(
            output_path,
            robot_root_pos_w=np.stack(
                [frame["robot_root_pos_w"] for frame in self.trajectory],
                axis=0,
            ),
            robot_root_quat_w=np.stack(
                [frame["robot_root_quat_w"] for frame in self.trajectory],
                axis=0,
            ),
            motion_root_pos_w=np.stack(
                [frame["motion_root_pos_w"] for frame in self.trajectory],
                axis=0,
            ),
            motion_root_quat_w=np.stack(
                [frame["motion_root_quat_w"] for frame in self.trajectory],
                axis=0,
            ),
            robot_body_pos_w=np.stack(
                [frame["robot_body_pos_w"] for frame in self.trajectory],
                axis=0,
            ),
            robot_body_quat_w=np.stack(
                [frame["robot_body_quat_w"] for frame in self.trajectory],
                axis=0,
            ),
            motion_body_pos_w=np.stack(
                [frame["motion_body_pos_w"] for frame in self.trajectory],
                axis=0,
            ),
            motion_body_quat_w=np.stack(
                [frame["motion_body_quat_w"] for frame in self.trajectory],
                axis=0,
            ),
            body_names=np.asarray(self._trajectory_body_names),
            sim_time=np.asarray(
                [frame["sim_time"] for frame in self.trajectory],
                dtype=np.float32,
            ),
            motion_t=np.asarray(
                [frame["motion_t"] for frame in self.trajectory],
                dtype=np.int32,
            ),
            motion_length=np.asarray(
                int(self.policy.state_processor.motion_length),
                dtype=np.int32,
            ),
            policy_config=np.asarray(str(self.args.policy_config)),
            motion_path=np.asarray(str(self.args.motion_path)),
            seed=np.asarray(-1 if self.args.seed is None else int(self.args.seed), dtype=np.int32),
        )
        logger.info("Saved full trajectory to {}", output_path)

    def run(self) -> None:
        if self.args.headless:
            self._run_headless()
            return

        scheduler = sched.scheduler(time.perf_counter, time.sleep)
        next_sim_time = time.perf_counter()
        next_policy_time = next_sim_time
        stop_time = None
        if self.args.max_runtime_s is not None:
            stop_time = next_sim_time + float(self.args.max_runtime_s)
        sim_count = 0

        try:
            while self.sim.is_running():
                now = time.perf_counter()
                if stop_time is not None and now >= stop_time:
                    logger.info("Stopping 统一 MuJoCo 评测链路 after max_runtime_s")
                    break
                if self.restart_requested.is_set() and self._at_last_paused_frame():
                    self._reset_playback()
                    next_sim_time = time.perf_counter()
                    next_policy_time = next_sim_time
                    continue
                self.restart_requested.clear()

                self._maybe_start_motion()

                if now >= next_sim_time:
                    scheduler.enterabs(next_sim_time, 1, self.sim.sim_step, ())
                    scheduler.run()
                    next_sim_time += self.args.sim_dt
                    sim_count += 1
                    self._append_trajectory_frame()
                    if sim_count % self.args.decimation == 0:
                        self.sim.sync_viewer()

                if now >= next_policy_time:
                    self.policy.step()
                    self.policy.total_inference_cnt += 1
                    next_policy_time += 1.0 / self.args.rl_rate

                if self.args.run_once and self.playback_started and self._at_last_paused_frame():
                    logger.info("Motion reached final frame; exiting because run_once=True")
                    break
                if self.args.stop_on_tracking_failure and self._tracking_failure_detected:
                    break

                sleep_until = min(next_sim_time, next_policy_time)
                time.sleep(max(0.0, min(0.002, sleep_until - time.perf_counter())))
        except KeyboardInterrupt:
            pass
        finally:
            self._save_root_trajectory()
            self._save_trajectory()
            self.policy._save_recording()
            self.policy.controller.close()
            self.sim.stop()

    def _run_headless(self) -> None:
        sim_count = 0
        policy_interval_steps = max(1, int(round((1.0 / self.args.rl_rate) / self.args.sim_dt)))
        pause_steps = max(0, int(round(self.args.initial_pause_s / self.args.sim_dt)))
        max_steps = None
        if self.args.max_runtime_s is not None:
            max_steps = max(0, int(round(float(self.args.max_runtime_s) / self.args.sim_dt)))

        try:
            while self.sim.is_running():
                if max_steps is not None and sim_count >= max_steps:
                    logger.info("Stopping 统一 MuJoCo 评测链路 after max_runtime_s")
                    break

                if not self.playback_started and sim_count >= pause_steps:
                    self._start_motion()

                self.sim.sim_step()
                sim_count += 1
                self.headless_elapsed_s = sim_count * self.args.sim_dt
                self._append_trajectory_frame()

                if sim_count % policy_interval_steps == 0:
                    self.policy.step()
                    self.policy.total_inference_cnt += 1

                if self.args.run_once and self.playback_started and self._at_last_paused_frame():
                    logger.info("Motion reached final frame; exiting because run_once=True")
                    break
                if self.args.stop_on_tracking_failure and self._tracking_failure_detected:
                    break
        except KeyboardInterrupt:
            pass
        finally:
            self._save_root_trajectory()
            self._save_trajectory()
            self.policy._save_recording()
            self.policy.controller.close()
            self.sim.stop()


@dataclass
class IntegratedSim2SimArgs:
    policy_config: str
    motion_path: str
    robot: str = "g1"
    rl_rate: float = 50.0
    sim_dt: float = 0.005
    decimation: int = 4
    initial_pause_s: float = 5.0
    inference_backend: Literal["onnx-gpu", "onnx-cpu", "tensorrt"] = "onnx-cpu"
    headless: bool = False
    enable_elastic_band: bool = False
    max_future: int | None = None
    run_once: bool = False
    max_runtime_s: float | None = None
    record: bool = False
    record_output: str | None = None
    root_trajectory_output: str | None = None
    trajectory_output: str | None = None
    trajectory_policy_frames_only: bool = False
    stop_on_tracking_failure: bool = False
    seed: int | None = None

    def __post_init__(self) -> None:
        self.policy_config = str(Path(self.policy_config).expanduser())
        self.motion_path = str(Path(self.motion_path).expanduser())
        if self.root_trajectory_output is not None:
            self.root_trajectory_output = str(Path(self.root_trajectory_output).expanduser())
        if self.trajectory_output is not None:
            self.trajectory_output = str(Path(self.trajectory_output).expanduser())
        if self.seed is not None:
            np.random.seed(int(self.seed))


if __name__ == "__main__":
    IntegratedSim2Sim(tyro.cli(IntegratedSim2SimArgs)).run()
