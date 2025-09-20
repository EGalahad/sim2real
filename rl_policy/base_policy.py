import time
import numpy as np
from typing import Dict, Type
import sched

from termcolor import colored
from loguru import logger

import sys
sys.path.append(".")
from utils.strings import resolve_matching_names_values
from rl_policy.utils.state_processor import StateProcessor
from rl_policy.utils.command_sender import CommandSender
from rl_policy.utils.onnx_module import Timer

# Import observation classes
from observations import Observation, ObsGroup


class BasePolicy:
    def __init__(
        self,
        robot_config,
        policy_config,
        model_path,
        rl_rate=50,
    ):
        # initialize robot related processes
        robot_type = robot_config["ROBOT_TYPE"]
        if robot_type != "g1_real":
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize
            if robot_config.get("INTERFACE", None):
                ChannelFactoryInitialize(robot_config["DOMAIN_ID"], robot_config["INTERFACE"])
            else:
                ChannelFactoryInitialize(robot_config["DOMAIN_ID"])
        else:
            sys.path.append("/home/elijah/Documents/projects/hdmi/unitree_sdk2/build/lib")
            import g1_interface
            network_interface = robot_config.get("INTERFACE", None)
            self.robot = g1_interface.G1Interface(network_interface)
            try:
                self.robot.set_control_mode(g1_interface.ControlMode.PR)
            except Exception:
                pass  # Ignore if firmware already in the correct mode
            robot_config["robot"] = self.robot

        self.state_processor = StateProcessor(robot_config, policy_config["isaac_joint_names"])
        self.command_sender = CommandSender(robot_config, policy_config)
        self.rl_dt = 1.0 / rl_rate

        self.policy_config = policy_config

        self.setup_policy(model_path)
        self.obs_cfg = policy_config["observation"]

        self.isaac_joint_names = policy_config["isaac_joint_names"]
        self.num_dofs = len(self.isaac_joint_names)

        default_joint_pos_dict = policy_config["default_joint_pos"]
        joint_indices, joint_names, default_joint_pos = resolve_matching_names_values(
            default_joint_pos_dict,
            self.isaac_joint_names,
            preserve_order=True,
            strict=False,
        )
        self.default_dof_angles = np.zeros(len(self.isaac_joint_names))
        self.default_dof_angles[joint_indices] = default_joint_pos

        self.policy_joint_names = policy_config["policy_joint_names"]
        self.num_actions = len(self.policy_joint_names)
        self.controlled_joint_indices = [
            self.isaac_joint_names.index(name)
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
        else:
            raise ValueError(f"Invalid action scale type: {type(action_scale_cfg)}")

        # Keypress control state
        self.use_policy_action = False

        self.first_time_init = True
        self.init_count = 0
        self.get_ready_state = False

        # Joint limits
        joint_indices, joint_names, joint_pos_lower_limit = (
            resolve_matching_names_values(
                robot_config["joint_pos_lower_limit"],
                self.isaac_joint_names,
                preserve_order=True,
                strict=False,
            )
        )
        self.joint_pos_lower_limit = np.zeros(self.num_dofs)
        self.joint_pos_lower_limit[joint_indices] = joint_pos_lower_limit

        joint_indices, joint_names, joint_pos_upper_limit = (
            resolve_matching_names_values(
                robot_config["joint_pos_upper_limit"],
                self.isaac_joint_names,
                preserve_order=True,
                strict=False,
            )
        )
        self.joint_pos_upper_limit = np.zeros(self.num_dofs)
        self.joint_pos_upper_limit[joint_indices] = joint_pos_upper_limit

        # joint_indices, joint_names, joint_vel_limit = resolve_matching_names_values(
        #     self.config["joint_vel_limit"], self.robot.isaac_joint_names, preserve_order=True, strict=False
        # )
        # self.joint_vel_limit = np.zeros(self.num_dofs)
        # self.joint_vel_limit[joint_indices] = joint_vel_limit

        # joint_indices, joint_names, joint_effort_limit = resolve_matching_names_values(
        #     self.config["joint_effort_limit"], self.robot.isaac_joint_names, preserve_order=True, strict=False
        # )
        # self.joint_effort_limit = np.zeros(self.num_dofs)
        # self.joint_effort_limit[joint_indices] = joint_effort_limit

        if robot_config.get("USE_JOYSTICK", False):
            # Yuanhang: pygame event can only run in main thread on Mac, so we need to implement it with rl inference
            assert robot_type == "g1_real", "Joystick control is only supported for g1_real"
            print("Using joystick")
            self.use_joystick = True
            self.wc_msg = None
            self.last_wc_msg = self.robot.read_wireless_controller()
            print("Wireless Controller Initialized")
        else:
            import threading
            print("Using keyboard")
            self.use_joystick = False
            self.key_listener_thread = threading.Thread(
                target=self.start_key_listener, daemon=True
            )
            self.key_listener_thread.start()

        # Setup observations after state processor is initialized
        self.setup_observations()

    def setup_policy(self, model_path):
        # load onnx policy
        from rl_policy.utils.onnx_module import ONNXModule
        onnx_module = ONNXModule(model_path)

        use_residual_action = self.policy_config.get("use_residual_action", False)
        if use_residual_action:
            def policy(input_dict):
                output_dict = onnx_module(input_dict)
                action = output_dict["action"].squeeze(0)
                next_state_dict = {k[1]: v for k, v in output_dict.items() if k[0] == "next"}
                input_dict.update(next_state_dict)

                ref_joint_pos = input_dict["_ref_joint_pos"].squeeze(0)
                q_target = self.default_dof_angles.copy()
                q_target[self.controlled_joint_indices] += \
                    ref_joint_pos - self.default_dof_angles[self.controlled_joint_indices] + \
                    action * self.action_scale

                return action, q_target, input_dict
        else:
            def policy(input_dict):
                output_dict = onnx_module(input_dict)
                action = output_dict["action"].squeeze(0)
                next_state_dict = {k[1]: v for k, v in output_dict.items() if k[0] == "next"}
                input_dict.update(next_state_dict)

                q_target = self.default_dof_angles.copy()
                q_target[self.controlled_joint_indices] += \
                    action * self.action_scale

                return action, q_target, input_dict

        self.policy = policy

    def setup_observations(self):
        """Setup observations for policy inference"""
        self.observations: Dict[str, ObsGroup] = {}
        self.reset_callbacks = []
        self.update_callbacks = []
        
        # Create observation instances based on config
        for obs_group, obs_items in self.obs_cfg.items():
            print(f"obs_group: {obs_group}")
            obs_funcs = {}
            for obs_name, obs_config in obs_items.items():
                print(f"\t{obs_name}: {obs_config}")
                obs_class: Type[Observation] = Observation.registry[obs_name]
                obs_func = obs_class(env=self, **obs_config)
                obs_funcs[obs_name] = obs_func
                self.reset_callbacks.append(obs_func.reset)
                self.update_callbacks.append(obs_func.update)
            self.observations[obs_group] = ObsGroup(obs_group, obs_funcs)

    def reset(self):
        self.state_dict["adapt_hx"][:] = 0.0
        self.state_dict["paused"] = False
        for reset_callback in self.reset_callbacks:
            reset_callback()

    def update(self):
        for update_callback in self.update_callbacks:
            update_callback(self.state_dict)

    def prepare_obs_for_rl(self):
        """Prepare observation for policy inference using observation classes"""
        obs_dict: Dict[str, np.ndarray] = {}
        for obs_group in self.observations.values():
            obs = obs_group.compute()
            obs_dict[obs_group.name] = obs[None, :].astype(np.float32)
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

    @property
    def command(self):
        return np.zeros(0)

    def start_key_listener(self):
        """Start a key listener using pynput."""

        self.key_pressed = set()
        def on_press(keycode):
            try:
                if keycode not in self.key_pressed:
                    self.key_pressed.add(keycode)
                    self.handle_keyboard_button(keycode)
            except AttributeError as e:
                logger.warning(
                    f"Keyboard key {keycode}. Error: {e}")
                pass  # Handle special keys if needed
        
        def on_release(keycode):
            try:
                if keycode in self.key_pressed:
                    self.key_pressed.remove(keycode)
            except AttributeError as e:
                logger.warning(
                    f"Keyboard key {keycode}. Error: {e}")
                pass

        from sshkeyboard import listen_keyboard
        listener = listen_keyboard(on_press=on_press, on_release=on_release)
        listener.start()
        listener.join()  # Keep the thread alive

    def handle_keyboard_button(self, keycode):
        """
        Rule:
        ]: Use policy actions
        o: Set actions to zero
        i: Set to init state
        5: Increase kp (coarse)
        6: Decrease kp (coarse)
        4: Decrease kp (fine)
        7: Increase kp (fine)
        0: Reset kp
        """
        if keycode == "]":
            self.reset()
            self.use_policy_action = True
            self.get_ready_state = False
            logger.info("Using policy actions")
            self.phase = 0.0
        elif keycode == "o":
            self.use_policy_action = False
            self.get_ready_state = False
            logger.info("Actions set to zero")
        elif keycode == "i":
            self.use_policy_action = False
            self.get_ready_state = True
            self.init_count = 0
            logger.info("Setting to init state")
        elif keycode == "5":
            self.command_sender.kp_level -= 0.01
        elif keycode == "6":
            self.command_sender.kp_level += 0.01
        elif keycode == "4":
            self.command_sender.kp_level -= 0.1
        elif keycode == "7":
            self.command_sender.kp_level += 0.1
        elif keycode == "0":
            self.command_sender.kp_level = 1.0

        if keycode in ["5", "6", "4", "7", "0"]:
            logger.info(
                colored(f"Debug kp level: {self.command_sender.kp_level}", "green")
            )

    def process_joystick_input(self):
        """Poll current wireless controller state and translate to high-level key events."""
        try:
            self.wc_msg = self.robot.read_wireless_controller()
        except Exception:
            return

        if self.wc_msg is None:
            return

        # print(f"wc_msg.A: {self.wc_msg.A}")
        if self.wc_msg.A and not self.last_wc_msg.A:
            self.handle_joystick_button("A")
        if self.wc_msg.B and not self.last_wc_msg.B:
            self.handle_joystick_button("B")
        if self.wc_msg.X and not self.last_wc_msg.X:
            self.handle_joystick_button("X")
        if self.wc_msg.Y and not self.last_wc_msg.Y:
            self.handle_joystick_button("Y")
        if self.wc_msg.L1 and not self.last_wc_msg.L1:
            self.handle_joystick_button("L1")
        if self.wc_msg.L2 and not self.last_wc_msg.L2:
            self.handle_joystick_button("L2")
        if self.wc_msg.R1 and not self.last_wc_msg.R1:
            self.handle_joystick_button("R1")
        if self.wc_msg.R2 and not self.last_wc_msg.R2:
            self.handle_joystick_button("R2")
        
        self.last_wc_msg = self.wc_msg
    
    def handle_joystick_button(self, cur_key):
        if cur_key == "R1":
            self.use_policy_action = True
            self.get_ready_state = False
            self.reset()
            logger.info(colored("Using policy actions", "blue"))
            self.phase = 0.0  # type: ignore
        elif cur_key == "R2":
            self.use_policy_action = False
            self.get_ready_state = False
            logger.info(colored("Actions set to zero", "blue"))
        elif cur_key == "A":
            self.get_ready_state = True
            self.init_count = 0
            logger.info(colored("Setting to init state", "blue"))
        # elif cur_key == "Y+left":
        #     self.command_sender.kp_level -= 0.1
        # elif cur_key == "Y+right":
        #     self.command_sender.kp_level += 0.1
        # elif cur_key == "A+left":
        #     self.command_sender.kp_level -= 0.01
        # elif cur_key == "A+right":
        #     self.command_sender.kp_level += 0.01

        # Debug print for kp level tuning
        if cur_key in ["Y+left", "Y+right", "A+left", "A+right"]:
            logger.info(colored(f"Debug kp level: {self.command_sender.kp_level}", "green"))

    def run(self):
        total_inference_cnt = 0
        
        # 初始化状态变量
        state_dict = {}
        state_dict["adapt_hx"] = np.zeros((1, 256), dtype=np.float32)
        state_dict["action"] = np.zeros(self.num_actions)
        self.state_dict = state_dict
        self.total_inference_cnt = total_inference_cnt
        self.perf_dict = {}

        try:
            # 使用scheduler进行精确时间控制
            scheduler = sched.scheduler(time.perf_counter, time.sleep)
            next_run_time = time.perf_counter()
            
            while True:
                # 调度下一次执行
                scheduler.enterabs(next_run_time, 1, self._rl_step_scheduled, ())
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

    def _rl_step_scheduled(self):
        loop_start = time.perf_counter()

        with Timer(self.perf_dict, "prepare_low_state"):
            if self.use_joystick:
                self.process_joystick_input()

            if not self.state_processor._prepare_low_state():
                print("low state not ready.")
                return
            
        try:
            with Timer(self.perf_dict, "prepare_obs"):
                # Prepare observations
                self.update()
                obs_dict = self.prepare_obs_for_rl()
                self.state_dict.update(obs_dict)
                self.state_dict["is_init"] = np.zeros(1, dtype=bool)

            with Timer(self.perf_dict, "policy"):   
                # Inference
                # print(self.state_dict.keys())
                action, q_target, self.state_dict = self.policy(self.state_dict)
                for key, value in self.state_dict.items():
                    if key.endswith("_ood_ratio"):
                        print(key, value)
                # Clip policy action
                action = action.clip(-100, 100)
                self.state_dict["action"] = action
                self.state_dict["q_target"] = q_target
        except Exception as e:
            print(f"Error in policy inference: {e}")
            self.state_dict["action"] = np.zeros(self.num_actions)
            return

        with Timer(self.perf_dict, "rule_based_control_flow"):
            # rule based control flow
            if self.get_ready_state:
                q_target = self.get_init_target()
            elif not self.use_policy_action:
                q_target = self.state_processor.joint_pos
            else:
                q_target = self.state_dict["q_target"]

            # # Clip q target
            # q_target = np.clip(
            #     q_target, self.joint_pos_lower_limit, self.joint_pos_upper_limit
            # )

            # Send command
            cmd_q = q_target
            cmd_dq = np.zeros(self.num_dofs)
            cmd_tau = np.zeros(self.num_dofs)
            self.command_sender.send_command(cmd_q, cmd_dq, cmd_tau)

        elapsed = time.perf_counter() - loop_start
        if elapsed > self.rl_dt:
            logger.warning(f"RL step took {elapsed:.6f} seconds, expected {self.rl_dt} seconds")

if __name__ == "__main__":
    import argparse
    import yaml
    parser = argparse.ArgumentParser(description="Robot")
    parser.add_argument(
        "--robot_config", type=str, default="config/robot/g1.yaml", help="robot config file"
    )
    parser.add_argument(
        "--policy_config", type=str, help="policy config file"
    )
    args = parser.parse_args()

    with open(args.policy_config) as file:
        policy_config = yaml.load(file, Loader=yaml.FullLoader)
    with open(args.robot_config) as file:
        robot_config = yaml.load(file, Loader=yaml.FullLoader)
    model_path = args.policy_config.replace(".yaml", ".onnx")

    policy = BasePolicy(
        robot_config=robot_config,
        policy_config=policy_config,
        model_path=model_path,
        rl_rate=50,
    )
    policy.run()
