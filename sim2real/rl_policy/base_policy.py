import time
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Type
import sched

import tyro
import yaml
from loguru import logger

from sim2real.config.robots import get_robot_cfg
from sim2real.rl_policy.controllers.base import ControllerBase
from sim2real.rl_policy.controllers.keyboard import KeyboardController
from sim2real.rl_policy.controllers.pico import PicoController
from sim2real.rl_policy.controllers.unitree_joystick import UnitreeJoystickController
from sim2real.rl_policy.inference import Timer, build_inference_module
from sim2real.rl_policy.observations import Observation, ObsGroup
from sim2real.rl_policy.utils.command_sender import ActionManager
from sim2real.rl_policy.utils.state_processor import StateProcessor
from sim2real.rl_policy.utils.upstream_real_io import UpstreamG1IO
from sim2real.utils.common import PORTS
from sim2real.utils.profiling import ScopedTimer
from sim2real.utils.strings import resolve_matching_names_values


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


class BasePolicy:
    def __init__(
        self,
        args: "BasePolicyArgs",
    ):
        self.args = args
        self.robot_cfg = get_robot_cfg(args.robot)
        with open(args.policy_config) as file:
            policy_config = yaml.load(file, Loader=yaml.FullLoader)
        policy_config = self.prepare_policy_config(policy_config)
        model_path = args.policy_config.replace(".yaml", ".onnx")
        self.policy_config = policy_config
        self.model_path = model_path
        # initialize robot related processes
        self.joint_names_simulation = list(policy_config["joint_names_simulation"])
        self.body_names_simulation = list(policy_config["body_names_simulation"])
        self.real_io_backend = None
        if args.real_io_backend == "upstream":
            if args.robot != "g1":
                raise NotImplementedError(
                    "real_io_backend='upstream' is currently implemented only for robot='g1'."
                )
            self.real_io_backend = UpstreamG1IO(
                interface=args.robot_interface,
                joint_count=len(self.robot_cfg.joint_names),
            )
        self.state_processor = StateProcessor(
            self.robot_cfg,
            policy_config,
            real_io_backend=self.real_io_backend,
        )
        self.action_manager = ActionManager(
            self.robot_cfg,
            policy_config,
            real_io_backend=self.real_io_backend,
        )
        self.rl_dt = 1.0 / float(args.rl_rate)
        self.inference_backend = args.inference_backend

        self.num_dofs = len(self.joint_names_simulation)

        default_joint_pos_dict = policy_config["default_joint_pos"]
        joint_indices, joint_names, default_joint_pos = resolve_matching_names_values(
            default_joint_pos_dict,
            self.action_manager.joint_names,
            preserve_order=True,
            strict=False,
        )
        self.default_dof_angles = np.zeros(len(self.joint_names_simulation))
        self.default_dof_angles[joint_indices] = default_joint_pos

        self.policy_joint_names = policy_config["policy_joint_names"]
        self.num_actions = len(self.policy_joint_names)
        self.controlled_joint_indices = [
            self.action_manager.joint_names.index(name)
            for name in self.policy_joint_names
        ]

        action_scale_cfg = policy_config["action_scale"]
        self.action_scale = np.ones((self.num_actions))
        if isinstance(action_scale_cfg, float):
            self.action_scale *= action_scale_cfg
        elif isinstance(action_scale_cfg, dict):
            joint_ids, joint_names, action_scales = resolve_matching_names_values(
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
        # Perf metrics dict is reused; initialize early so background threads can record.
        self.perf_dict: Dict[str, float] = {}
        self.key_pressed: set[str] = set()
        self.state_dict = {
            "action": np.zeros(self.num_actions, dtype=np.float32),
            "paused": True,
            "control_mode": "zero",
        }

        # Joint limits
        joint_indices, joint_names, joint_pos_lower_limit = (
            resolve_matching_names_values(
                self.robot_cfg.joint_pos_lower_limit,
                self.joint_names_simulation,
                preserve_order=True,
                strict=False,
            )
        )
        self.joint_pos_lower_limit = np.zeros(self.num_dofs)
        self.joint_pos_lower_limit[joint_indices] = joint_pos_lower_limit

        joint_indices, joint_names, joint_pos_upper_limit = (
            resolve_matching_names_values(
                self.robot_cfg.joint_pos_upper_limit,
                self.joint_names_simulation,
                preserve_order=True,
                strict=False,
            )
        )
        self.joint_pos_upper_limit = np.zeros(self.num_dofs)
        self.joint_pos_upper_limit[joint_indices] = joint_pos_upper_limit

        self.controller_type = args.controller
        self.controller = self._build_controller()
        self.use_joystick = self.controller_type == "joystick"
        self.wc_msg = None

        # Setup observations after state processor is initialized
        self.setup_policy(model_path)
        self.setup_observations(policy_config["observation"])
        self._record_enabled = bool(args.record)
        self._record_output = self._resolve_record_output(args.record_output)
        self._record_frames: list[dict[str, Any]] = []
        self._record_start_time_ns: int | None = None

    def prepare_policy_config(self, policy_config):
        return policy_config

    def _build_controller(self) -> ControllerBase:
        self.keyboard_controller = None
        self.joystick_controller = None
        self.pico_controller = None

        controller_type = self.controller_type
        if controller_type == "keyboard":
            print("Using keyboard")
            self.keyboard_controller = KeyboardController()
            self.key_pressed = self.keyboard_controller.key_pressed
            return self.keyboard_controller

        if controller_type == "joystick":
            print("Using joystick")
            self.joystick_controller = UnitreeJoystickController(
                self.robot_cfg,
                self.perf_dict,
            )
            print("Wireless Controller Initialized")
            return self.joystick_controller

        if controller_type == "pico":
            self.pico_controller = PicoController(connect=self.args.pico_zmq_connect)
            return self.pico_controller

        raise ValueError(f"Unsupported controller_type: {controller_type}")

    def setup_policy(self, model_path):
        runtime_module = build_inference_module(model_path, self.inference_backend)
        runtime_label = self.inference_backend
        if self.inference_backend == "tensorrt":
            runtime_label = (
                "tensorrt"
                f"[fp16={runtime_module.use_fp16}, "
                f"workspace={runtime_module.workspace_size}]"
            )

        logger.info("Using policy inference backend {}", runtime_label)

        def policy(input_dict):
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

    def setup_observations(self, obs_cfg):
        """Setup observations for policy inference"""
        self.observations: Dict[str, ObsGroup] = {}
        self.reset_callbacks = []
        self.update_callbacks = []

        self.reset_callbacks.append(self.state_processor.reset)
        self.update_callbacks.append(self.state_processor.update)

        # Create observation instances based on config
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
                self.reset_callbacks.append(obs_func.reset)
                self.update_callbacks.append(obs_func.update)
            self.observations[obs_group] = ObsGroup(obs_group, obs_funcs)

    def reset(self):
        self.state_dict["paused"] = True
        [reset_callback() for reset_callback in self.reset_callbacks]

    def update(self):
        [update_callback(self.state_dict) for update_callback in self.update_callbacks]

    def prepare_obs_for_rl(self):
        """Prepare observation for policy inference using observation classes"""
        obs_dict: Dict[str, np.ndarray] = {}
        obs_components: dict[str, dict[str, np.ndarray]] = {}
        for obs_group in self.observations.values():
            group_components: dict[str, np.ndarray] = {}
            group_values: list[np.ndarray] = []
            for obs_name, obs_func in obs_group.funcs.items():
                obs = obs_func.compute().astype(np.float32)
                group_components[obs_name] = obs
                group_values.append(obs)

            obs_components[obs_group.name] = group_components
            obs_dict[obs_group.name] = np.concatenate(group_values, axis=-1)

        return obs_dict, obs_components

    def get_init_target(self):
        if self.init_count > 500:
            self.init_count = 500

        # interpolate from current dof_pos to default angles
        dof_pos = self.state_processor.joint_pos
        progress = self.init_count / 500
        q_target = dof_pos + (self.default_dof_angles - dof_pos) * progress
        self.init_count += 1
        return q_target

    def set_init_mode(self, *, source: str) -> None:
        self.init_count = 0
        self.state_dict["control_mode"] = "init"
        logger.info(f"Control mode set to init via {source}")

    def set_zero_mode(self, *, source: str) -> None:
        self.state_dict["control_mode"] = "zero"
        logger.info(f"Control mode set to zero via {source}")

    def set_policy_mode(self, *, source: str) -> None:
        self.reset()
        self.state_dict["control_mode"] = "policy"
        logger.info(f"Control mode set to policy via {source}")

    def process_controllers(self) -> None:
        if self.joystick_controller is not None:
            self.wc_msg = self.joystick_controller.state

        mode = self.controller.get_control_mode()
        if mode == "policy":
            self.set_policy_mode(source=self.controller.name)
        elif mode == "zero":
            self.set_zero_mode(source=self.controller.name)
        elif mode == "init":
            self.set_init_mode(source=self.controller.name)

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

        low_state = getattr(self.state_processor, "latest_low_state", None)
        joint_torque = getattr(low_state, "joint_torques", None) if low_state is not None else None
        if joint_torque is None:
            joint_torque = np.full(self.num_dofs, np.nan, dtype=np.float32)

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
                    "joint_torque": _copy_array(joint_torque),
                    "root_quat_w": _copy_array(self.state_processor.root_quat_w),
                    "root_ang_vel_b": _copy_array(self.state_processor.root_ang_vel_b),
                    "low_state_tick": int(getattr(low_state, "tick", -1)),
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
                "rl_rate": float(self.args.rl_rate),
                "joint_names_simulation": list(self.joint_names_simulation),
                "body_names_simulation": list(self.body_names_simulation),
                "motion_backend": str(
                    getattr(self.state_processor, "motion_backend", "")
                ),
                "motion_path": str(
                    getattr(self.state_processor, "motion_config", {}).get("motion_path", "")
                ),
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

    def _save_recording(self) -> None:
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

    def run(self):
        total_inference_cnt = 0

        self.state_dict = {
            "action": np.zeros(self.num_actions, dtype=np.float32),
            "paused": True,
            "control_mode": "zero",
        }
        self.total_inference_cnt = total_inference_cnt
        self.perf_dict = {}

        try:
            scheduler = sched.scheduler(time.perf_counter, time.sleep)
            next_run_time = time.perf_counter()
            
            while True:
                scheduler.enterabs(next_run_time, 1, self.step, ())
                scheduler.run()
                
                next_run_time += self.rl_dt
                self.total_inference_cnt += 1

                if self.total_inference_cnt % 100 == 0:
                    print(f"total_inference_cnt: {self.total_inference_cnt}")
                    for key, value in self.perf_dict.items():
                        print(f"\t{key}: {value/100*1000:.3f} ms")
                    self.perf_dict = {}
        except KeyboardInterrupt:
            pass
        finally:
            self._save_recording()
            self.controller.close()
            if self.real_io_backend is not None:
                self.real_io_backend.close()

    def step(self):
        with ScopedTimer("rl_policy.step") as step_timer:
            with ScopedTimer("rl_policy.step.prepare_low_state") as prepare_low_state_timer:
                with Timer(self.perf_dict, "prepare_low_state"):
                    with ScopedTimer(
                        "rl_policy.step.prepare_low_state.process_controllers"
                    ) as process_controllers_timer:
                        with Timer(self.perf_dict, "process_controllers"):
                            self.process_controllers()

                    with ScopedTimer(
                        "rl_policy.step.prepare_low_state.get_low_state"
                    ) as get_low_state_timer:
                        with Timer(self.perf_dict, "get_low_state"):
                            if not self.state_processor._prepare_low_state():
                                print("low state not ready.")
                                return

            try:
                with ScopedTimer("rl_policy.step.prepare_obs") as prepare_obs_timer:
                    with Timer(self.perf_dict, "prepare_obs"):
                        # Prepare observations
                        self.update()
                        obs_dict, obs_components = self.prepare_obs_for_rl()
                        self.state_dict.update(obs_dict)
                        self.state_dict["is_init"] = np.zeros(1, dtype=bool)

                with ScopedTimer("rl_policy.step.policy") as policy_timer:
                    with Timer(self.perf_dict, "policy"):
                        # Inference
                        action, q_target, self.state_dict = self.policy(self.state_dict)
                        # for key, value in self.state_dict.items():
                        #     if key.endswith("_ood_ratio"):
                        #         print(key, value)
                        # Clip policy action
                        # action = action.clip(-100, 100)
                        self.state_dict["action"] = action
                        self.state_dict["q_target"] = q_target
            except Exception as e:
                print(f"Error in policy inference: {e}")
                # print traceback for debugging
                import traceback
                traceback.print_exc()
                self.state_dict["action"] = np.zeros(self.num_actions)
                return

            with ScopedTimer("rl_policy.step.rule_based_control_flow") as rule_based_timer:
                with Timer(self.perf_dict, "rule_based_control_flow"):
                    with ScopedTimer(
                        "rl_policy.step.rule_based_control_flow.select_target"
                    ) as select_target_timer:
                        # rule based control flow
                        control_mode = self.state_dict["control_mode"]
                        if control_mode == "init":
                            q_target = self.get_init_target()
                        elif control_mode == "zero":
                            q_target = self.state_processor.joint_pos
                        elif control_mode == "policy":
                            q_target = self.state_dict["q_target"]
                        else:
                            raise ValueError(f"Invalid control mode: {control_mode}")

                        # # Clip q target
                        # q_target = np.clip(
                        #     q_target, self.joint_pos_lower_limit, self.joint_pos_upper_limit
                        # )

                        # Send command
                        cmd_q = q_target
                        cmd_dq = np.zeros(self.num_dofs)
                        cmd_tau = np.zeros(self.num_dofs)

                    with ScopedTimer(
                        "rl_policy.step.rule_based_control_flow.send_command"
                    ) as send_command_timer:
                        self.action_manager.send_command(cmd_q, cmd_dq, cmd_tau)
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
        if elapsed > self.rl_dt:
            logger.warning(
                (
                    "RL step took {:.3f} ms, expected {:.3f} ms. "
                    "breakdown: prepare_low_state={:.3f} ms "
                    "(process_controllers={:.3f}, get_low_state={:.3f}), "
                    "prepare_obs={:.3f} ms, policy={:.3f} ms, "
                    "rule_based_control_flow={:.3f} ms "
                    "(select_target={:.3f}, send_command={:.3f})"
                ),
                elapsed * 1000.0,
                self.rl_dt * 1000.0,
                prepare_low_state_timer.last_time * 1000.0,
                process_controllers_timer.last_time * 1000.0,
                get_low_state_timer.last_time * 1000.0,
                prepare_obs_timer.last_time * 1000.0,
                policy_timer.last_time * 1000.0,
                rule_based_timer.last_time * 1000.0,
                select_target_timer.last_time * 1000.0,
                send_command_timer.last_time * 1000.0,
            )


@dataclass
class BasePolicyArgs:
    """Robot."""

    policy_config: str
    robot: str = "g1"
    rl_rate: float = 50.0
    inference_backend: Literal["onnx-gpu", "onnx-cpu", "tensorrt"] = "onnx-cpu"
    real_io_backend: Literal["zmq", "upstream"] = "zmq"
    robot_interface: str = "eth0"
    controller: Literal["keyboard", "joystick", "pico"] = "keyboard"
    pico_zmq_connect: str = f"tcp://127.0.0.1:{PORTS['pico_controller']}"
    record: bool = False
    record_output: str | None = None

if __name__ == "__main__":
    args = tyro.cli(BasePolicyArgs)
    policy = BasePolicy(args=args)
    policy.run()
