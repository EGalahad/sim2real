from __future__ import annotations

import re
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import Any, Callable, Dict, Literal, Type

import torch  # Import torch before MuJoCo/ONNX native libs on aarch64 to avoid static TLS issues.
import mujoco
import numpy as np
import tyro
import yaml
from loguru import logger

from sim2real.config.robots import get_robot_cfg
from sim2real.config.robots.base import RobotCfg
from sim2real.rl_policy.observations import Observation, ObsGroup
from sim2real.rl_policy.utils.motion import (
    MotionData,
    MotionDataset,
    motion_dataset_first_motion,
)
from sim2real.rl_policy.inference import Timer, build_inference_module
from sim2real.sim_env.utils.mjcf import load_sim_model
from sim2real.utils.mjviser_viewer import MjviserMujocoViewer
from sim2real.utils.math import (
    projected_yaw_quat,
    quat_conjugate,
    quat_mul,
    quat_rotate_inverse_numpy,
)
from sim2real.utils.profiling import ScopedTimer
from sim2real.utils.strings import resolve_matching_names_values

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


def _record_array(values: list[Any]) -> Any:
    if not values:
        return np.empty((0,), dtype=np.float32)

    if any(value is None for value in values):
        return np.asarray(values, dtype=object)

    try:
        arrays = [np.asarray(value) for value in values]
        if arrays and all(array.shape == arrays[0].shape for array in arrays):
            if arrays[0].dtype.kind in "biufc?":
                return np.stack(arrays, axis=0)
    except Exception:
        pass

    try:
        return np.asarray(values)
    except Exception:
        return np.asarray(values, dtype=object)


def _copy_array(value: Any, dtype: Any | None = np.float32) -> np.ndarray:
    if dtype is None:
        return np.asarray(value).copy()
    return np.asarray(value, dtype=dtype).copy()


def _prepare_integrated_policy_config(
    policy_config: dict[str, Any],
    *,
    motion_path: str,
) -> dict[str, Any]:
    policy_config = deepcopy(policy_config)
    motion_cfg = policy_config.setdefault("motion", {})
    motion_cfg["motion_backend"] = "npz"
    motion_cfg["motion_path"] = motion_path
    return policy_config


class IntegratedMotionState:
    def __init__(self, robot_cfg: RobotCfg, policy_config: dict[str, Any]):
        self.robot_cfg = robot_cfg
        self.joint_names = list(robot_cfg.joint_names)
        self.num_dof = len(self.joint_names)

        self.qpos = np.zeros(3 + 4 + self.num_dof, dtype=np.float32)
        self.qvel = np.zeros(3 + 3 + self.num_dof, dtype=np.float32)
        self.root_pos_w = self.qpos[0:3]
        self.root_lin_vel_w = self.qvel[0:3]
        self.root_quat_w = self.qpos[3:7]
        self.root_ang_vel_b = self.qvel[3:6]
        self.joint_pos = self.qpos[7:]
        self.joint_vel = self.qvel[6:]
        self.joint_torque = np.zeros(self.num_dof, dtype=np.float32)
        self.low_state_tick = -1

        self.motion_config: Dict[str, Any] = dict(policy_config.get("motion", {}))
        self.motion_future_steps = np.asarray(
            self.motion_config.get("future_steps", []),
            dtype=int,
        )
        if self.motion_future_steps.ndim != 1:
            raise ValueError(
                f"motion.future_steps must be 1D, got shape={self.motion_future_steps.shape}"
            )

        self.motion_backend = str(
            self.motion_config.get("motion_backend", "npz")
        ).lower().strip()
        if self.motion_backend != "npz":
            raise ValueError(
                f"Integrated sim2sim requires motion_backend=npz, got {self.motion_backend}"
            )

        motion_path = self.motion_config.get("motion_path")
        if motion_path is None:
            raise ValueError("motion_path is required for integrated sim2sim")
        self.motion_dataset = MotionDataset.create_from_path(
            motion_path,
            robot_cfg=self.robot_cfg,
        )
        self.motion_dataset = motion_dataset_first_motion(self.motion_dataset)
        if self.motion_dataset.num_motions != 1:
            raise ValueError("Integrated sim2sim supports one motion per run")

        self.motion_ids = np.array([0], dtype=int)
        self.motion_t = np.array([0], dtype=int)
        self.motion_length = self.motion_dataset.num_steps
        self.motion_joint_names = list(self.motion_dataset.joint_names)
        self.motion_body_names = list(self.motion_dataset.body_names)
        self.motion_data: MotionData | None = None
        self._update_motion_data()

    def reset(self) -> None:
        self.motion_t[:] = 0
        self._update_motion_data()

    def restart_motion(self) -> None:
        self.reset()

    def update(self, data: Dict[str, Any] | None = None) -> None:
        data = data or {}
        paused = bool(data.get("paused", False))
        if not paused:
            self.motion_t += 1
            if int(self.motion_t[0]) >= int(self.motion_length):
                self.motion_t[:] = int(self.motion_length) - 1
                data["paused"] = True
        self._update_motion_data()

    def _update_motion_data(self) -> None:
        self.motion_data = self.motion_dataset.get_slice(
            self.motion_ids,
            self.motion_t,
            self.motion_future_steps,
        )

    def get_mocap_data(self, key: str) -> None:
        return None

    def register_subscriber(self, object_name: str, port: int | None = None) -> None:
        raise RuntimeError("Integrated sim2sim does not support external mocap subscribers")


class IntegratedPolicyRuntime:
    def __init__(
        self,
        *,
        args: "IntegratedSim2SimArgs",
        robot_cfg: RobotCfg,
    ):
        self.args = args
        self.robot_cfg = robot_cfg
        with open(args.policy_config) as file:
            raw_policy_config = yaml.load(file, Loader=yaml.FullLoader)
        self.policy_config = _prepare_integrated_policy_config(
            raw_policy_config,
            motion_path=args.motion_path,
        )
        self.model_path = args.policy_config.replace(".yaml", ".onnx")

        self.joint_names_simulation = list(self.policy_config["joint_names_simulation"])
        self.body_names_simulation = list(self.policy_config["body_names_simulation"])
        self.state_processor = IntegratedMotionState(self.robot_cfg, self.policy_config)
        self.action_joint_names = list(self.robot_cfg.joint_names)

        self.env_dt = float(args.env_dt)
        self.inference_backend = args.inference_backend
        self.num_dofs = len(self.joint_names_simulation)

        default_joint_pos_dict = self.policy_config["default_joint_pos"]
        joint_indices, _joint_names, default_joint_pos = resolve_matching_names_values(
            default_joint_pos_dict,
            self.action_joint_names,
            preserve_order=True,
            strict=False,
        )
        self.default_dof_angles = np.zeros(len(self.joint_names_simulation))
        self.default_dof_angles[joint_indices] = default_joint_pos

        self.policy_joint_names = self.policy_config["policy_joint_names"]
        self.num_actions = len(self.policy_joint_names)
        self.controlled_joint_indices = [
            self.action_joint_names.index(name)
            for name in self.policy_joint_names
        ]

        self.joint_kp_unitree = self._resolve_joint_array("joint_kp")
        self.joint_kd_unitree = self._resolve_joint_array("joint_kd")

        action_scale_cfg = self.policy_config["action_scale"]
        self.action_scale = np.ones((self.num_actions))
        if isinstance(action_scale_cfg, float):
            self.action_scale *= action_scale_cfg
        elif isinstance(action_scale_cfg, dict):
            joint_ids, _joint_names, action_scales = resolve_matching_names_values(
                action_scale_cfg, self.policy_joint_names, preserve_order=True
            )
            self.action_scale[joint_ids] = action_scales
        elif isinstance(action_scale_cfg, list):
            if len(action_scale_cfg) != self.num_actions:
                raise ValueError(
                    f"Action scale list length {len(action_scale_cfg)} does not match num actions {self.num_actions}"
                )
            self.action_scale[:] = np.array(action_scale_cfg)
        else:
            raise ValueError(f"Invalid action scale type: {type(action_scale_cfg)}")

        self.init_count = 0
        self.perf_dict: Dict[str, float] = {}
        self._obs_profile_totals: dict[str, float] = {}
        self._obs_profile_max: dict[str, float] = {}
        self._obs_profile_count = 0
        self.key_pressed: set[str] = set()
        self.use_joystick = False
        self.wc_msg = None
        self.state_dict = {
            "action": np.zeros(self.num_actions, dtype=np.float32),
            "paused": True,
            "control_mode": "zero",
        }
        self.total_inference_cnt = 0

        self._record_enabled = bool(args.record)
        self._record_output = self._resolve_record_output(args.record_output)
        self._record_frames: list[dict[str, Any]] = []
        self._record_start_time_ns: int | None = None

        self.setup_policy(self.model_path)
        self.setup_observations(self.policy_config["observation"])

    def _resolve_joint_array(self, key: str) -> np.ndarray:
        values_dict = self.policy_config[key]
        joint_indices, _joint_names, values = resolve_matching_names_values(
            values_dict,
            self.action_joint_names,
            preserve_order=True,
            strict=False,
        )
        out = np.zeros(len(self.action_joint_names), dtype=np.float32)
        out[joint_indices] = values
        return out

    def setup_policy(self, model_path: str) -> None:
        runtime_module = build_inference_module(model_path, self.inference_backend)
        runtime_label = self.inference_backend
        if self.inference_backend == "tensorrt":
            runtime_label = (
                "tensorrt"
                f"[fp16={runtime_module.use_fp16}, "
                f"workspace={runtime_module.workspace_size}]"
            )
        logger.info("Using policy inference backend {}", runtime_label)

        def policy(input_dict: dict[str, Any]):
            output_dict = runtime_module(input_dict)
            action = np.asarray(output_dict["action"], dtype=np.float32)
            next_state_dict = {
                k[1]: v
                for k, v in output_dict.items()
                if isinstance(k, tuple) and len(k) == 2 and k[0] == "next"
            }
            input_dict.update(next_state_dict)

            q_target = self.default_dof_angles.copy()
            q_target[self.controlled_joint_indices] += action * self.action_scale
            return action, q_target, input_dict

        self.policy = policy

    def setup_observations(self, obs_cfg: dict[str, Any]) -> None:
        self.observations: Dict[str, ObsGroup] = {}
        self.reset_callbacks: list[Callable[[], None]] = []
        self.update_callbacks: list[Callable[[dict[str, Any]], None]] = []
        self._obs_items: list[tuple[str, str, Observation]] = []

        self.reset_callbacks.append(self.state_processor.reset)
        self.update_callbacks.append(self.state_processor.update)

        for obs_group, obs_items in obs_cfg.items():
            print(f"obs_group: {obs_group}")
            obs_funcs = {}
            for obs_name, obs_config in obs_items.items():
                print(f"\t{obs_name}: {obs_config}")
                obs_config = dict(obs_config)
                obs_key = obs_config.pop("_target_", obs_name)
                if "." in obs_key:
                    obs_key = obs_key.split(".")[-1]
                obs_class: Type[Observation] = Observation.registry[obs_key]
                obs_func = obs_class(env=self, **obs_config)
                obs_funcs[obs_name] = obs_func
                self._obs_items.append((obs_group, obs_name, obs_func))
                self.reset_callbacks.append(obs_func.reset)
                self.update_callbacks.append(obs_func.update)
            self.observations[obs_group] = ObsGroup(obs_group, obs_funcs)

    def reset(self) -> None:
        self.state_dict["paused"] = True
        for reset_callback in self.reset_callbacks:
            reset_callback()

    def update(self) -> None:
        for update_callback in self.update_callbacks:
            update_callback(self.state_dict)

    def prepare_obs_for_rl(self) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]:
        obs_dict, obs_components, _timings = self._prepare_obs_for_rl(profile=False)
        return obs_dict, obs_components

    def _prepare_obs_for_rl(
        self,
        *,
        profile: bool,
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, dict[str, np.ndarray]],
        list[tuple[str, float]] | None,
    ]:
        timings: list[tuple[str, float]] | None = [] if profile else None

        if profile:
            start = time.perf_counter()
            self.state_processor.update(self.state_dict)
            assert timings is not None
            timings.append(("update:state_processor", time.perf_counter() - start))

            for obs_group, obs_name, obs_func in self._obs_items:
                start = time.perf_counter()
                obs_func.update(self.state_dict)
                timings.append((f"update:{obs_group}.{obs_name}", time.perf_counter() - start))
        else:
            self.update()

        obs_dict: dict[str, np.ndarray] = {}
        obs_components: dict[str, dict[str, np.ndarray]] = {}
        for obs_group in self.observations.values():
            group_components: dict[str, np.ndarray] = {}
            group_values: list[np.ndarray] = []
            for obs_name, obs_func in obs_group.funcs.items():
                start = time.perf_counter()
                obs = obs_func.compute().astype(np.float32)
                if timings is not None:
                    timings.append((f"compute:{obs_group.name}.{obs_name}", time.perf_counter() - start))
                group_components[obs_name] = obs
                group_values.append(obs)

            start = time.perf_counter()
            obs_components[obs_group.name] = group_components
            obs_dict[obs_group.name] = np.concatenate(group_values, axis=-1)
            if timings is not None:
                timings.append((f"concat:{obs_group.name}", time.perf_counter() - start))

        return obs_dict, obs_components, timings

    def get_init_target(self) -> np.ndarray:
        if self.init_count > 500:
            self.init_count = 500
        dof_pos = self.state_processor.joint_pos
        progress = self.init_count / 500
        q_target = dof_pos + (self.default_dof_angles - dof_pos) * progress
        self.init_count += 1
        return q_target

    def step(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        with ScopedTimer("integrated_policy.step") as step_timer:
            try:
                with ScopedTimer("integrated_policy.step.prepare_obs") as prepare_obs_timer:
                    with Timer(self.perf_dict, "prepare_obs"):
                        obs_dict, obs_components, obs_timings = self._prepare_obs_for_rl(
                            profile=bool(self.args.profile_obs),
                        )
                        self.state_dict.update(obs_dict)
                        self.state_dict["is_init"] = np.zeros(1, dtype=bool)

                with ScopedTimer("integrated_policy.step.policy") as policy_timer:
                    with Timer(self.perf_dict, "policy"):
                        action, q_target, self.state_dict = self.policy(self.state_dict)
                        self.state_dict["action"] = action
                        self.state_dict["q_target"] = q_target
            except Exception as exc:
                print(f"Error in policy inference: {exc}")
                import traceback

                traceback.print_exc()
                self.state_dict["action"] = np.zeros(self.num_actions)
                return None

            with ScopedTimer("integrated_policy.step.control_flow") as control_timer:
                with Timer(self.perf_dict, "rule_based_control_flow"):
                    control_mode = self.state_dict["control_mode"]
                    if control_mode == "init":
                        q_target = self.get_init_target()
                    elif control_mode == "zero":
                        q_target = self.state_processor.joint_pos
                    elif control_mode == "policy":
                        q_target = self.state_dict["q_target"]
                    else:
                        raise ValueError(f"Invalid control mode: {control_mode}")

                    cmd_q = np.asarray(q_target, dtype=np.float32)
                    cmd_dq = np.zeros(self.num_dofs, dtype=np.float32)
                    cmd_tau = np.zeros(self.num_dofs, dtype=np.float32)
                    self._append_record_frame(
                        obs_dict=obs_dict,
                        obs_components=obs_components,
                        action=action,
                        q_target=q_target,
                        cmd_q=cmd_q,
                        cmd_dq=cmd_dq,
                        cmd_tau=cmd_tau,
                    )

        elapsed = step_timer.last_time
        if bool(self.args.profile_obs) and obs_timings is not None:
            self._record_obs_profile(obs_timings)
        if elapsed > self.env_dt:
            if bool(self.args.profile_obs) and obs_timings is not None:
                top_timings = sorted(obs_timings, key=lambda item: item[1], reverse=True)[:8]
                logger.warning(
                    "Integrated prepare_obs breakdown: {}",
                    ", ".join(
                        f"{name}={duration_s * 1000.0:.3f} ms"
                        for name, duration_s in top_timings
                    ),
                )
            logger.warning(
                (
                    "Integrated policy step took {:.3f} ms, expected {:.3f} ms. "
                    "breakdown: prepare_obs={:.3f} ms, policy={:.3f} ms, "
                    "control_flow={:.3f} ms"
                ),
                elapsed * 1000.0,
                self.env_dt * 1000.0,
                prepare_obs_timer.last_time * 1000.0,
                policy_timer.last_time * 1000.0,
                control_timer.last_time * 1000.0,
            )
        return cmd_q, cmd_dq, cmd_tau, self.joint_kp_unitree, self.joint_kd_unitree

    def _record_obs_profile(self, obs_timings: list[tuple[str, float]]) -> None:
        self._obs_profile_count += 1
        for name, duration_s in obs_timings:
            self._obs_profile_totals[name] = self._obs_profile_totals.get(name, 0.0) + duration_s
            self._obs_profile_max[name] = max(self._obs_profile_max.get(name, 0.0), duration_s)

    def report_obs_profile(self) -> None:
        if not bool(self.args.profile_obs) or self._obs_profile_count == 0:
            return

        def format_items(items: list[tuple[str, float]]) -> str:
            return ", ".join(f"{name}={duration_s * 1000.0:.3f} ms" for name, duration_s in items)

        mean_items = sorted(
            (
                (name, total_s / self._obs_profile_count)
                for name, total_s in self._obs_profile_totals.items()
            ),
            key=lambda item: item[1],
            reverse=True,
        )[:10]
        max_items = sorted(
            self._obs_profile_max.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:10]
        logger.info(
            "Integrated prepare_obs profile over {} steps; top mean: {}",
            self._obs_profile_count,
            format_items(mean_items),
        )
        logger.info(
            "Integrated prepare_obs profile over {} steps; top max: {}",
            self._obs_profile_count,
            format_items(max_items),
        )

    def _resolve_record_output(self, record_output: str | None) -> Path:
        if record_output:
            return Path(record_output).expanduser()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        policy_stem = Path(self.args.policy_config).stem
        return Path.cwd() / f"policy_tracking_record_{policy_stem}_{timestamp}.npz"

    def _append_record_frame(
        self,
        *,
        obs_dict: Dict[str, np.ndarray],
        obs_components: dict[str, dict[str, np.ndarray]],
        action: np.ndarray,
        q_target: np.ndarray,
        cmd_q: np.ndarray,
        cmd_dq: np.ndarray,
        cmd_tau: np.ndarray,
    ) -> None:
        if not self._record_enabled:
            return

        now_ns = time.time_ns()
        if self._record_start_time_ns is None:
            self._record_start_time_ns = now_ns

        motion_t = getattr(self.state_processor, "motion_t", None)
        motion_t_value = -1
        if motion_t is not None:
            motion_t_value = int(np.asarray(motion_t).reshape(-1)[0])

        self._record_frames.append(
            {
                "step_index": int(self.total_inference_cnt),
                "time_ns": int(now_ns),
                "robot": {
                    "joint_pos": _copy_array(self.state_processor.joint_pos),
                    "joint_vel": _copy_array(self.state_processor.joint_vel),
                    "joint_torque": _copy_array(self.state_processor.joint_torque),
                    "root_quat_w": _copy_array(self.state_processor.root_quat_w),
                    "root_ang_vel_b": _copy_array(self.state_processor.root_ang_vel_b),
                    "low_state_tick": int(self.state_processor.low_state_tick),
                },
                "policy": {
                    "action": _copy_array(action),
                    "q_target": _copy_array(q_target),
                    "cmd_q": _copy_array(cmd_q),
                    "cmd_dq": _copy_array(cmd_dq),
                    "cmd_tau": _copy_array(cmd_tau),
                    "obs": {
                        group_name: {
                            obs_name: _copy_array(obs)
                            for obs_name, obs in group_obs.items()
                        }
                        for group_name, group_obs in obs_components.items()
                    },
                },
                "runtime": {
                    "motion_t": int(motion_t_value),
                    "control_mode": str(self.state_dict.get("control_mode", "")),
                    "paused": bool(self.state_dict.get("paused", False)),
                },
            }
        )

    def _build_record_data(self) -> dict[str, Any]:
        frames = self._record_frames

        def collect(group: str, key: str) -> Any:
            return _record_array([frame[group].get(key) for frame in frames])

        obs_groups = sorted(
            {
                group_name
                for frame in frames
                for group_name in frame["policy"].get("obs", {}).keys()
            }
        )
        obs = {}
        for group_name in obs_groups:
            obs_names = sorted(
                {
                    obs_name
                    for frame in frames
                    for obs_name in frame["policy"].get("obs", {}).get(group_name, {}).keys()
                }
            )
            obs[group_name] = {
                obs_name: _record_array(
                    [
                        frame["policy"].get("obs", {}).get(group_name, {}).get(obs_name)
                        for frame in frames
                    ]
                )
                for obs_name in obs_names
            }

        return {
            "metadata": {
                "schema": "policy_tracking_record_v1",
                "robot": str(self.args.robot),
                "policy_config": str(self.args.policy_config),
                "model_path": str(self.model_path),
                "env_dt": float(self.args.env_dt),
                "joint_names_simulation": list(self.joint_names_simulation),
                "body_names_simulation": list(self.body_names_simulation),
                "motion_backend": str(self.state_processor.motion_backend),
                "motion_path": str(self.state_processor.motion_config.get("motion_path", "")),
                "frame_count": int(len(frames)),
                "recorded_at_unix_ns": int(time.time_ns()),
                "record_start_unix_ns": int(self._record_start_time_ns or 0),
            },
            "robot": {
                "joint_names": list(self.state_processor.joint_names),
                "joint_pos": collect("robot", "joint_pos"),
                "joint_vel": collect("robot", "joint_vel"),
                "joint_torque": collect("robot", "joint_torque"),
                "root_quat_w": collect("robot", "root_quat_w"),
                "root_ang_vel_b": collect("robot", "root_ang_vel_b"),
                "low_state_tick": collect("robot", "low_state_tick"),
            },
            "policy": {
                "step_index": _record_array([frame["step_index"] for frame in frames]),
                "time_ns": _record_array([frame["time_ns"] for frame in frames]),
                "action": collect("policy", "action"),
                "q_target": collect("policy", "q_target"),
                "cmd_q": collect("policy", "cmd_q"),
                "cmd_dq": collect("policy", "cmd_dq"),
                "cmd_tau": collect("policy", "cmd_tau"),
                "obs": obs,
            },
            "runtime": {
                "motion_t": collect("runtime", "motion_t"),
                "control_mode": collect("runtime", "control_mode"),
                "paused": collect("runtime", "paused"),
            },
        }

    def save_recording(self) -> None:
        if not self._record_enabled:
            return
        data = self._build_record_data()
        self._record_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self._record_output, data=np.asarray(data, dtype=object))
        logger.info(
            "Saved {} policy tracking frames to {}",
            len(self._record_frames),
            self._record_output,
        )


class IntegratedSimRuntime:
    def __init__(
        self,
        robot_cfg: RobotCfg,
        *,
        sim_dt: float,
        headless: bool,
        key_callback: Callable[[int], None] | None,
    ):
        self.robot_cfg = robot_cfg
        self.sim_dt = float(sim_dt)
        self.headless = bool(headless)
        self._external_key_callback = key_callback
        self._stop_event = Event()

        self.mj_model = load_sim_model(self.robot_cfg)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = self.sim_dt

        self._init_mappings()
        self.torques = np.zeros(self.mj_model.nu, dtype=np.float32)
        self.cmd_q = np.zeros(len(self.robot_cfg.joint_names), dtype=np.float32)
        self.cmd_dq = np.zeros(len(self.robot_cfg.joint_names), dtype=np.float32)
        self.cmd_tau = np.zeros(len(self.robot_cfg.joint_names), dtype=np.float32)
        self.cmd_kp = np.zeros(len(self.robot_cfg.joint_names), dtype=np.float32)
        self.cmd_kd = np.zeros(len(self.robot_cfg.joint_names), dtype=np.float32)
        self.has_received_command = False

        self.pelvis_body_id = self._resolve_body_id(self.robot_cfg.viewer_track_body_names)
        self.viewer = None
        if not self.headless:
            self.viewer = MjviserMujocoViewer(
                self.mj_model,
                self.mj_data,
                label="sim2real-integrated-sim2sim",
                tracked_body_id=self.pelvis_body_id,
            )
            self._create_control_gui()

    def _create_control_gui(self) -> None:
        if self.viewer is None or self._external_key_callback is None or glfw is None:
            return

        with self.viewer.server.gui.add_folder("Integrated Sim2Sim"):
            restart = self.viewer.server.gui.add_button("Restart motion")

        @restart.on_click
        def _(_) -> None:
            self._external_key_callback(glfw.KEY_SPACE)

    def _init_mappings(self) -> None:
        joint_names_mujoco = [
            self.mj_model.joint(i).name for i in range(self.mj_model.njnt)
        ]
        actuator_names_mujoco = [
            self.mj_model.actuator(i).name for i in range(self.mj_model.nu)
        ]
        self.joint_indices_unitree: list[int] = []
        self.qpos_adrs: list[int] = []
        self.qvel_adrs: list[int] = []
        self.act_adrs: list[int] = []

        for name in self.robot_cfg.joint_names:
            if name not in joint_names_mujoco or name not in actuator_names_mujoco:
                continue
            print(f"shared_joint_names: {name}")
            self.joint_indices_unitree.append(self.robot_cfg.joint_names.index(name))
            joint_idx = joint_names_mujoco.index(name)
            self.qpos_adrs.append(int(self.mj_model.jnt_qposadr[joint_idx]))
            self.qvel_adrs.append(int(self.mj_model.jnt_dofadr[joint_idx]))
            self.act_adrs.append(actuator_names_mujoco.index(name))

        root_joint_idx = None
        for root_joint_name in self.robot_cfg.root_joint_names:
            if root_joint_name in joint_names_mujoco:
                root_joint_idx = joint_names_mujoco.index(root_joint_name)
                break
        if root_joint_idx is None:
            raise ValueError("No root joint found in the MuJoCo model.")
        self.root_qpos_adr = int(self.mj_model.jnt_qposadr[root_joint_idx])
        self.root_qvel_adr = int(self.mj_model.jnt_dofadr[root_joint_idx])

        joint_indices, joint_names_matched, joint_effort_limit = (
            resolve_matching_names_values(
                self.robot_cfg.joint_effort_limit,
                joint_names_mujoco,
                preserve_order=True,
                strict=False,
            )
        )
        del joint_indices
        self.joint_effort_limit_mjc = np.asarray(joint_effort_limit, dtype=np.float32)
        self.joint_idx_in_ctrl = np.asarray(
            [actuator_names_mujoco.index(name) for name in joint_names_matched],
            dtype=int,
        )

    def _resolve_body_id(self, body_names: tuple[str, ...]) -> int:
        for body_name in body_names:
            body_id = mujoco.mj_name2id(
                self.mj_model,
                mujoco.mjtObj.mjOBJ_BODY,
                body_name,
            )
            if body_id >= 0:
                return int(body_id)
        names = ", ".join(body_names)
        raise ValueError(f"Failed to resolve body from candidates: {names}")

    def sync_policy_state(self, state: IntegratedMotionState) -> None:
        root_qpos = int(self.root_qpos_adr)
        root_qvel = int(self.root_qvel_adr)
        state.root_pos_w[:] = self.mj_data.qpos[root_qpos : root_qpos + 3]
        state.root_quat_w[:] = self.mj_data.qpos[root_qpos + 3 : root_qpos + 7]
        state.root_lin_vel_w[:] = self.mj_data.qvel[root_qvel : root_qvel + 3]
        state.root_ang_vel_b[:] = self.mj_data.qvel[root_qvel + 3 : root_qvel + 6]

        joint_pos = np.zeros_like(state.joint_pos)
        joint_vel = np.zeros_like(state.joint_vel)
        joint_torque = np.zeros_like(state.joint_torque)
        for unitree_idx, qpos_addr, qvel_addr, act_addr in zip(
            self.joint_indices_unitree,
            self.qpos_adrs,
            self.qvel_adrs,
            self.act_adrs,
        ):
            joint_pos[unitree_idx] = self.mj_data.qpos[qpos_addr]
            joint_vel[unitree_idx] = self.mj_data.qvel[qvel_addr]
            joint_torque[unitree_idx] = self.mj_data.actuator_force[act_addr]
        state.joint_pos[:] = joint_pos
        state.joint_vel[:] = joint_vel
        state.joint_torque[:] = joint_torque
        state.low_state_tick = int(self.mj_data.time * 1e3)

    def apply_command(
        self,
        cmd_q: np.ndarray,
        cmd_dq: np.ndarray,
        cmd_tau: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
    ) -> None:
        expected = len(self.robot_cfg.joint_names)
        if np.asarray(cmd_q).size != expected:
            raise ValueError(f"Expected command size {expected}, got {np.asarray(cmd_q).size}")
        self.cmd_q[:] = np.asarray(cmd_q, dtype=np.float32)
        self.cmd_dq[:] = np.asarray(cmd_dq, dtype=np.float32)
        self.cmd_tau[:] = np.asarray(cmd_tau, dtype=np.float32)
        self.cmd_kp[:] = np.asarray(kp, dtype=np.float32)
        self.cmd_kd[:] = np.asarray(kd, dtype=np.float32)
        self.has_received_command = True

    def compute_torques(self) -> None:
        self.torques[:] = 0.0
        if self.has_received_command:
            for unitree_idx, qpos_addr, qvel_addr, act_addr in zip(
                self.joint_indices_unitree,
                self.qpos_adrs,
                self.qvel_adrs,
                self.act_adrs,
            ):
                q_des = self.cmd_q[unitree_idx]
                dq_des = self.cmd_dq[unitree_idx]
                tau_ff = self.cmd_tau[unitree_idx]
                kp = self.cmd_kp[unitree_idx]
                kd = self.cmd_kd[unitree_idx]
                self.torques[act_addr] = (
                    tau_ff
                    + kp * (q_des - self.mj_data.qpos[qpos_addr])
                    + kd * (dq_des - self.mj_data.qvel[qvel_addr])
                )
        self.torques[self.joint_idx_in_ctrl] = np.clip(
            self.torques[self.joint_idx_in_ctrl],
            -self.joint_effort_limit_mjc,
            self.joint_effort_limit_mjc,
        )

    def sim_step(self) -> None:
        self.compute_torques()
        self.mj_data.ctrl[:] = self.torques
        mujoco.mj_step(self.mj_model, self.mj_data)

    def is_running(self) -> bool:
        if self._stop_event.is_set():
            return False
        if self.viewer is None:
            return True
        return bool(self.viewer.is_running())

    def sync_viewer(self) -> None:
        if self.viewer is None or not self.viewer.has_clients():
            return
        self.viewer.sync()

    def stop(self) -> None:
        self._stop_event.set()
        if self.viewer is not None:
            self.viewer.close()


class IntegratedSim2Sim:
    def __init__(self, args: "IntegratedSim2SimArgs"):
        self.args = args
        self.restart_requested = Event()
        self.robot_cfg = get_robot_cfg(args.robot)
        self.policy = IntegratedPolicyRuntime(args=args, robot_cfg=self.robot_cfg)
        self.sim = IntegratedSimRuntime(
            self.robot_cfg,
            sim_dt=args.sim_dt,
            headless=args.headless,
            key_callback=self._on_mujoco_key if not args.headless else None,
        )
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

    @property
    def state_processor(self) -> IntegratedMotionState:
        return self.policy.state_processor

    def _motion_frame(self, frame: int) -> MotionData:
        state_processor = self.state_processor
        return state_processor.motion_dataset.get_slice(
            state_processor.motion_ids,
            np.asarray([int(frame)], dtype=int),
            np.asarray([0], dtype=int),
        )

    def _set_robot_to_motion_frame(self, frame: int) -> None:
        motion_data = self._motion_frame(frame)
        state_processor = self.state_processor

        root_body_name = str(
            self.policy.policy_config.get("motion", {}).get("root_body_name", "pelvis")
        )
        root_body_idx = state_processor.motion_body_names.index(root_body_name)
        root_qpos = int(self.sim.root_qpos_adr)
        root_qvel = int(self.sim.root_qvel_adr)

        self.sim.mj_data.qpos[root_qpos : root_qpos + 3] = motion_data.body_pos_w[
            0, 0, root_body_idx
        ]
        self.sim.mj_data.qpos[root_qpos + 3 : root_qpos + 7] = motion_data.body_quat_w[
            0, 0, root_body_idx
        ]
        self.sim.mj_data.qvel[root_qvel : root_qvel + 6] = 0.0

        motion_joint_names = list(state_processor.motion_joint_names)
        for unitree_idx, qpos_addr, qvel_addr in zip(
            self.sim.joint_indices_unitree,
            self.sim.qpos_adrs,
            self.sim.qvel_adrs,
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
        self.sim.sync_policy_state(self.state_processor)

    def _reset_playback(self) -> None:
        self.policy.state_dict = {
            "action": np.zeros(self.policy.num_actions, dtype=np.float32),
            "paused": True,
            "control_mode": "policy",
        }
        self.state_processor.restart_motion()
        self._set_robot_to_motion_frame(0)
        self._sync_policy_state_from_sim()
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

    def _maybe_start_motion(self, *, sim_elapsed_s: float | None = None) -> None:
        if self.playback_started:
            return
        if sim_elapsed_s is None:
            if time.perf_counter() < self.replay_start_time:
                return
        elif sim_elapsed_s < float(self.args.initial_pause_s):
            return
        self._start_motion()

    def _at_last_paused_frame(self) -> bool:
        state_processor = self.state_processor
        return (
            bool(self.policy.state_dict.get("paused", False))
            and int(state_processor.motion_t[0]) >= int(state_processor.motion_length) - 1
        )

    def _motion_root_state(self) -> tuple[np.ndarray, np.ndarray]:
        state_processor = self.state_processor
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
        state_processor = self.state_processor
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

        root_qpos = int(self.sim.root_qpos_adr)
        motion_t = int(self.state_processor.motion_t[0])
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
                int(self.state_processor.motion_length),
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
                int(self.state_processor.motion_length),
                dtype=np.int32,
            ),
            policy_config=np.asarray(str(self.args.policy_config)),
            motion_path=np.asarray(str(self.args.motion_path)),
            seed=np.asarray(-1 if self.args.seed is None else int(self.args.seed), dtype=np.int32),
        )
        logger.info("Saved full trajectory to {}", output_path)

    def _policy_step(self) -> None:
        self._sync_policy_state_from_sim()
        command = self.policy.step()
        if command is None:
            return
        self.sim.apply_command(*command)

    def run(self) -> None:
        self._run_synchronized(throttle=not self.args.headless)

    def _run_synchronized(self, *, throttle: bool) -> None:
        sim_count = 0
        tick_dt = float(self.args.env_dt)
        run_start_time = time.perf_counter()

        try:
            while self.sim.is_running():
                tick_start_time = time.perf_counter()
                if self.args.max_runtime_s is not None:
                    if throttle:
                        runtime_s = tick_start_time - run_start_time
                    else:
                        runtime_s = sim_count * self.args.sim_dt
                    if runtime_s >= float(self.args.max_runtime_s):
                        logger.info("Stopping 统一 MuJoCo 评测链路 after max_runtime_s")
                        break

                if self.restart_requested.is_set() and self._at_last_paused_frame():
                    self._reset_playback()
                    sim_count = 0
                    run_start_time = time.perf_counter()
                    continue
                self.restart_requested.clear()

                if throttle:
                    self._maybe_start_motion()
                else:
                    self._maybe_start_motion(sim_elapsed_s=sim_count * self.args.sim_dt)

                self._policy_step()
                self.policy.total_inference_cnt += 1

                for _ in range(self.args.decimation):
                    if not self.sim.is_running():
                        break
                    self.sim.sim_step()
                    sim_count += 1
                    self.headless_elapsed_s = sim_count * self.args.sim_dt
                    self._append_trajectory_frame()
                    if self.args.stop_on_tracking_failure and self._tracking_failure_detected:
                        break

                if throttle:
                    self.sim.sync_viewer()

                if self.args.run_once and self.playback_started and self._at_last_paused_frame():
                    logger.info("Motion reached final frame; exiting because run_once=True")
                    break
                if self.args.stop_on_tracking_failure and self._tracking_failure_detected:
                    break

                if throttle:
                    elapsed = time.perf_counter() - tick_start_time
                    sleep_s = tick_dt - elapsed
                    if sleep_s > 0.0:
                        time.sleep(sleep_s)
        except KeyboardInterrupt:
            pass
        finally:
            self.policy.report_obs_profile()
            self._save_root_trajectory()
            self._save_trajectory()
            self.policy.save_recording()
            self.sim.stop()


@dataclass
class IntegratedSim2SimArgs:
    policy_config: str
    motion_path: str
    robot: str = "g1"
    env_dt: float = 0.02
    sim_dt: float = 0.005
    initial_pause_s: float = 5.0
    inference_backend: Literal["onnx-gpu", "onnx-cpu", "tensorrt"] = "onnx-cpu"
    headless: bool = False
    profile_obs: bool = False
    run_once: bool = False
    max_runtime_s: float | None = None
    record: bool = False
    record_output: str | None = None
    root_trajectory_output: str | None = None
    trajectory_output: str | None = None
    trajectory_policy_frames_only: bool = False
    stop_on_tracking_failure: bool = False
    seed: int | None = None
    decimation: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.env_dt <= 0:
            raise ValueError(f"env_dt must be positive, got {self.env_dt}")
        if self.sim_dt <= 0:
            raise ValueError(f"sim_dt must be positive, got {self.sim_dt}")

        sim_steps = float(self.env_dt) / float(self.sim_dt)
        rounded_sim_steps = int(round(sim_steps))
        if rounded_sim_steps <= 0 or not np.isclose(
            sim_steps,
            rounded_sim_steps,
            rtol=1e-4,
            atol=1e-6,
        ):
            raise ValueError(
                "env_dt must be an integer multiple of sim_dt, got "
                f"env_dt={self.env_dt}, sim_dt={self.sim_dt}, ratio={sim_steps}"
            )
        self.decimation = rounded_sim_steps

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
