import time
import numpy as np
from dataclasses import dataclass
from typing import Dict, Literal, Type
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
from sim2real.utils.common import PORTS
from sim2real.utils.profiling import ScopedTimer
from sim2real.utils.strings import resolve_matching_names_values


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
        self.state_processor = StateProcessor(self.robot_cfg, policy_config)
        self.action_manager = ActionManager(self.robot_cfg, policy_config)
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
        for obs_group in self.observations.values():
            obs = obs_group.compute()
            obs_dict[obs_group.name] = obs.astype(np.float32)
        return obs_dict

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
            self.controller.close()

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
                        obs_dict = self.prepare_obs_for_rl()
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
    controller: Literal["keyboard", "joystick", "pico"] = "keyboard"
    pico_zmq_connect: str = f"tcp://127.0.0.1:{PORTS['pico_controller']}"

if __name__ == "__main__":
    args = tyro.cli(BasePolicyArgs)
    policy = BasePolicy(args=args)
    policy.run()
