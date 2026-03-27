import time
import numpy as np
from typing import Dict, Type
import sched
from types import SimpleNamespace
import subprocess
import threading
from copy import deepcopy

from termcolor import colored
from loguru import logger

from sim2real.config.robots import get_robot_cfg
from sim2real.config.robots.base import RobotCfg
from sim2real.rl_policy.observations import Observation, ObsGroup
from sim2real.rl_policy.utils.command_sender import ActionManager
from sim2real.rl_policy.utils.onnx_module import Timer
from sim2real.rl_policy.utils.state_processor import StateProcessor
from sim2real.utils.strings import resolve_matching_names_values


class BasePolicy:
    def __init__(
        self,
        robot_cfg: RobotCfg,
        policy_config,
        model_path,
        rl_rate=50,
        onnx_provider="cpu",
    ):
        self.robot_cfg = robot_cfg
        # initialize robot related processes
        self.joint_names_simulation = list(policy_config["joint_names_simulation"])
        self.body_names_simulation = list(policy_config["body_names_simulation"])
        self.state_processor = StateProcessor(self.robot_cfg, policy_config)
        self.action_manager = ActionManager(self.robot_cfg, policy_config)
        self.rl_dt = 1.0 / rl_rate
        self.onnx_provider = onnx_provider

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

        # Keypress control state
        self.use_policy_action = False

        self.first_time_init = True
        self.init_count = 0
        self.get_ready_state = False
        # Perf metrics dict is reused; initialize early so background threads can record.
        self.perf_dict: Dict[str, float] = {}

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

        if self.robot_cfg.use_joystick:
            print("Using joystick")
            self.use_joystick = True
            self.wc_msg = None
            from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_

            if self.robot_cfg.interface:
                ChannelFactoryInitialize(self.robot_cfg.domain_id, self.robot_cfg.interface)
            else:
                ChannelFactoryInitialize(self.robot_cfg.domain_id)

            self.wireless_controller_sub = ChannelSubscriber(
                "rt/wirelesscontroller", WirelessController_
            )
            self.wireless_controller_sub.Init(None, 0)
            self._wc_lock = threading.Lock()
            self.last_wc_msg = SimpleNamespace(
                A=False, B=False, X=False, Y=False,
                L1=False, L2=False, R1=False, R2=False,
                left_stick=(0.0, 0.0), right_stick=(0.0, 0.0)
            )
            self._joystick_thread_stop = threading.Event()
            self.joystick_thread = threading.Thread(
                target=self._poll_wireless_controller, daemon=True
            )
            self.joystick_thread.start()
            print("Wireless Controller Initialized")
        else:
            print("Using keyboard")
            self.use_joystick = False
            self.key_listener_thread = threading.Thread(
                target=self.start_key_listener, daemon=True
            )
            self.key_listener_thread.start()

        # Setup observations after state processor is initialized
        self.setup_policy(model_path)
        self.setup_observations(policy_config["observation"])

    def setup_policy(self, model_path):
        # load onnx policy
        from sim2real.rl_policy.utils.onnx_module import ONNXModule
        onnx_module = ONNXModule(model_path, providers=self.onnx_provider)

        def policy(input_dict):
            output_dict = onnx_module(input_dict)
            action = output_dict["action"] #.squeeze(0)
            next_state_dict = {k[1]: v for k, v in output_dict.items() if k[0] == "next"}
            input_dict.update(next_state_dict)

            q_target = self.default_dof_angles.copy()
            q_target[self.controlled_joint_indices] += \
                action * self.action_scale

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
                obs_class: Type[Observation] = Observation.registry[obs_name]
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

    def start_key_listener(self):
        """Start a key listener using sshkeyboard."""

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

        try:
            # listen_keyboard sets the TTY to raw/no-echo; wrapping in try/finally
            # and calling stop_listening in run() ensures the terminal gets restored
            # even if the main loop exits via KeyboardInterrupt.
            listen_keyboard(on_press=on_press, on_release=on_release)
        except Exception as e:
            logger.warning(f"Keyboard listener stopped unexpectedly: {e}")

    def stop_keyboard_listener(self):
        if self.use_joystick:
            return

        try:
            from sshkeyboard import stop_listening
            stop_listening()  # restores terminal settings
        except Exception as e:
            logger.debug(f"Failed to stop keyboard listener cleanly: {e}")
        finally:
            # Ensure TTY echo/canonical mode is restored even if listener cleanup failed
            try:
                subprocess.run(["stty", "sane"], check=False)
            except Exception as e:
                logger.debug(f"Failed to run stty sane: {e}")

    def stop_joystick_listener(self):
        if not self.use_joystick:
            return
        try:
            self._joystick_thread_stop.set()
            if hasattr(self, "joystick_thread") and self.joystick_thread.is_alive():
                self.joystick_thread.join(timeout=1.0)
        except Exception as e:
            logger.debug(f"Failed to stop joystick listener cleanly: {e}")

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
            self.action_manager.kp_level -= 0.01
        elif keycode == "6":
            self.action_manager.kp_level += 0.01
        elif keycode == "4":
            self.action_manager.kp_level -= 0.1
        elif keycode == "7":
            self.action_manager.kp_level += 0.1
        elif keycode == "0":
            self.action_manager.kp_level = 1.0

        if keycode in ["5", "6", "4", "7", "0"]:
            logger.info(
                colored(f"Debug kp level: {self.action_manager.kp_level}", "green")
            )

    def process_joystick_input(self):
        """Translate latest wireless controller state into high-level key events."""
        with self._wc_lock:
            wc_local = deepcopy(self.wc_msg)

        if wc_local is None:
            return

        if wc_local.A and not self.last_wc_msg.A:
            self.handle_joystick_button("A")
        if wc_local.B and not self.last_wc_msg.B:
            self.handle_joystick_button("B")
        if wc_local.X and not self.last_wc_msg.X:
            self.handle_joystick_button("X")
        if wc_local.Y and not self.last_wc_msg.Y:
            self.handle_joystick_button("Y")
        if wc_local.L1 and not self.last_wc_msg.L1:
            self.handle_joystick_button("L1")
        if wc_local.L2 and not self.last_wc_msg.L2:
            self.handle_joystick_button("L2")
        if wc_local.R1 and not self.last_wc_msg.R1:
            self.handle_joystick_button("R1")
        if wc_local.R2 and not self.last_wc_msg.R2:
            self.handle_joystick_button("R2")

        self.last_wc_msg = wc_local
    
    def _decode_wireless_controller(self, msg):
        key_bits = {
            "R1": 0,
            "L1": 1,
            "R2": 4,
            "L2": 5,
            "A": 8,
            "B": 9,
            "X": 10,
            "Y": 11,
        }
        keys = getattr(msg, "keys", 0)
        return SimpleNamespace(
            A=bool(keys & (1 << key_bits["A"])),
            B=bool(keys & (1 << key_bits["B"])),
            X=bool(keys & (1 << key_bits["X"])),
            Y=bool(keys & (1 << key_bits["Y"])),
            L1=bool(keys & (1 << key_bits["L1"])),
            L2=bool(keys & (1 << key_bits["L2"])),
            R1=bool(keys & (1 << key_bits["R1"])),
            R2=bool(keys & (1 << key_bits["R2"])),
            left_stick=(getattr(msg, "lx", 0.0), getattr(msg, "ly", 0.0)),
            right_stick=(getattr(msg, "rx", 0.0), getattr(msg, "ry", 0.0)),
        )

    def _poll_wireless_controller(self):
        """Background poller to read wireless controller at ~5 Hz to keep RL loop light."""
        poll_interval = 0.2  # 5 Hz
        while not self._joystick_thread_stop.is_set():
            try:
                with Timer(self.perf_dict, "read_wireless_controller"):
                    raw_msg = self.wireless_controller_sub.Read()
                if raw_msg is not None:
                    with Timer(self.perf_dict, "decode_wireless_controller"):
                        decoded = self._decode_wireless_controller(raw_msg)
                    with self._wc_lock:
                        self.wc_msg = decoded
            except Exception as e:
                logger.debug(f"Joystick poll error: {e}")
            finally:
                time.sleep(poll_interval)
    
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
        #     self.action_manager.kp_level -= 0.1
        # elif cur_key == "Y+right":
        #     self.action_manager.kp_level += 0.1
        # elif cur_key == "A+left":
        #     self.action_manager.kp_level -= 0.01
        # elif cur_key == "A+right":
        #     self.action_manager.kp_level += 0.01

        # Debug print for kp level tuning
        if cur_key in ["Y+left", "Y+right", "A+left", "A+right"]:
            logger.info(colored(f"Debug kp level: {self.action_manager.kp_level}", "green"))

    def run(self):
        total_inference_cnt = 0
        
        state_dict = {}
        state_dict["action"] = np.zeros(self.num_actions)
        self.state_dict = state_dict
        self.total_inference_cnt = total_inference_cnt
        self.perf_dict = {}

        try:
            scheduler = sched.scheduler(time.perf_counter, time.sleep)
            next_run_time = time.perf_counter()
            
            while True:
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
        finally:
            if self.use_joystick:
                self.stop_joystick_listener()
            else:
                self.stop_keyboard_listener()

    def _rl_step_scheduled(self):
        loop_start = time.perf_counter()

        with Timer(self.perf_dict, "prepare_low_state"):
            if self.use_joystick:
                with Timer(self.perf_dict, "process_joystick_input"):
                    self.process_joystick_input()

            with Timer(self.perf_dict, "get_low_state"):
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
            # print traceback for debugging
            import traceback
            traceback.print_exc()
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
            self.action_manager.send_command(cmd_q, cmd_dq, cmd_tau)

        elapsed = time.perf_counter() - loop_start
        if elapsed > self.rl_dt:
            logger.warning(f"RL step took {elapsed:.6f} seconds, expected {self.rl_dt} seconds")

if __name__ == "__main__":
    import argparse
    import yaml
    parser = argparse.ArgumentParser(description="Robot")
    parser.add_argument(
        "--robot", type=str, default="g1", help="robot name"
    )
    parser.add_argument(
        "--policy_config", type=str, help="policy config file"
    )
    parser.add_argument(
        "--onnx_provider",
        type=str,
        default="cpu",
        choices=["cpu", "gpu"],
        help="onnxruntime execution provider",
    )
    args = parser.parse_args()

    with open(args.policy_config) as file:
        policy_config = yaml.load(file, Loader=yaml.FullLoader)
    model_path = args.policy_config.replace(".yaml", ".onnx")

    policy = BasePolicy(
        robot_cfg=get_robot_cfg(args.robot),
        policy_config=policy_config,
        model_path=model_path,
        rl_rate=50,
        onnx_provider=args.onnx_provider,
    )
    policy.run()
