import time
import threading
from sshkeyboard import listen_keyboard
import numpy as np
from termcolor import colored
import sys
sys.path.append(".")
from typing import Dict
import sched

from loguru import logger

from utils.state_processor import StateProcessor
from utils.command_sender import CommandSender
from utils.math import quat_rotate_inverse_numpy
from utils.strings import resolve_matching_names_values

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_


class BasePolicy:
    def __init__(
        self,
        robot_config,
        policy_config,
        model_path,
        rl_rate=50,
    ):
        # initialize robot related processes
        if robot_config.get("INTERFACE", None):
            ChannelFactoryInitialize(robot_config["DOMAIN_ID"], robot_config["INTERFACE"])
        else:
            ChannelFactoryInitialize(robot_config["DOMAIN_ID"])

        self.state_processor = StateProcessor(robot_config["ROBOT_TYPE"], policy_config["isaac_joint_names"])
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

        action_scale_cfg = policy_config["action_scale"]
        self.action_scale = np.ones((self.num_dofs))
        if isinstance(action_scale_cfg, float):
            self.action_scale *= action_scale_cfg
        elif isinstance(action_scale_cfg, dict):
            joint_ids, joint_names, action_scales = resolve_matching_names_values(
                action_scale_cfg, self.isaac_joint_names, preserve_order=True
            )
            self.action_scale[joint_ids] = action_scales
        else:
            raise ValueError(f"Invalid action scale type: {type(action_scale_cfg)}")

        self.policy_joint_names = policy_config["policy_joint_names"]
        self.num_actions = len(self.policy_joint_names)
        self.controlled_joint_indices = [
            self.isaac_joint_names.index(name)
            for name in self.policy_joint_names
        ]

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

        if self.policy_config.get("USE_JOYSTICK", False):
            # Yuanhang: pygame event can only run in main thread on Mac, so we need to implement it with rl inference
            print("Using joystick")
            self.use_joystick = True
            self.wc_msg = None

            def wc_handler(msg: WirelessController_):
                self.wc_msg = msg
            self.wireless_controller_subscriber = ChannelSubscriber(
                "rt/wirelesscontroller", WirelessController_
            )
            self.wireless_controller_subscriber.Init(wc_handler, 1)
            self.wc_key_map = {
                1: "R1",
                2: "L1",
                3: "L1+R1",
                4: "start",
                8: "select",
                16: "R2",
                32: "L2",
                64: "F1",  # not used in sim2sim
                128: "F2",  # not used in sim2sim
                256: "A",
                512: "B",
                768: "A+B",
                1024: "X",
                1280: "A+X",
                2048: "Y",
                2304: "A+Y",
                2560: "B+Y",
                3072: "X+Y",
                4096: "up",
                4608: "B+up",
                8192: "right",
                8448: "A+right",
                10240: "Y+right",
                16384: "down",
                16896: "B+down",
                32768: "left",
                33024: "A+left",
                34816: "Y+left",
            }
            self._empty_key_states = {key: False for key in self.wc_key_map.values()}
            self.last_key_states = self._empty_key_states.copy()
            print("Wireless Controller Initialized")
        else:
            print("Using keyboard")
            self.use_joystick = False
            self.key_listener_thread = threading.Thread(
                target=self.start_key_listener, daemon=True
            )
            self.key_listener_thread.start()


    def setup_policy(self, model_path):
        # load onnx policy
        from utils.onnx_module import ONNXModule
        onnx_module = ONNXModule(model_path)

        def policy(input_dict):
            output_dict = onnx_module(input_dict)
            action = output_dict["action"].squeeze(0)
            carry = {k[1]: v for k, v in output_dict.items() if k[0] == "next"}
            return action, carry

        self.policy = policy

    def setup_observations(self):
        """Setup observations for policy inference"""
        # TODO: For future use of observation groups and observation classes
        pass

    def prepare_obs_for_rl(self):
        """Prepare observation for policy inference"""
        obs_dict: Dict[str, np.ndarray] = {}
        for obs_group in self.obs_cfg.keys():
            obs_keys = self.obs_cfg[obs_group].keys()
            obs_list = [getattr(self, f"_get_obs_{key}")() for key in obs_keys]
            obs_dict[obs_group] = np.concatenate(obs_list, axis=0)
        return {key: value[None, :].astype(np.float32) for key, value in obs_dict.items()}

    def get_init_target(self):
        if self.init_count > 500:
            self.init_count = 500

        # interpolate from current dof_pos to default angles
        dof_pos = self.state_processor.joint_pos
        progress = self.init_count / 500
        q_target = dof_pos + (self.default_dof_angles - dof_pos) * progress
        self.init_count += 1
        return q_target

    def pre_compute_obs_callback(self):
        pass

    def _get_obs_root_ang_vel_b(self):
        base_ang_vel = self.state_processor.root_ang_vel_b
        return base_ang_vel

    def _get_obs_projected_gravity_b(self):
        base_quat = self.state_processor.root_quat_b
        v = np.array([0, 0, -1])
        projected_gravity = quat_rotate_inverse_numpy(
            base_quat[None, :], 
            v[None, :]
        ).squeeze(0)
        return projected_gravity

    def _get_obs_joint_pos_multistep(self):
        return self.joint_pos_multistep.reshape(-1)

    def _get_obs_joint_vel_multistep(self):
        return self.joint_vel_multistep.reshape(-1)

    def _get_obs_prev_actions(self):
        return self.prev_actions.reshape(-1)

    @property
    def command(self):
        return np.zeros(0)

    def start_key_listener(self):
        """Start a key listener using pynput."""

        def on_press(keycode):
            try:
                self.handle_keyboard_button(keycode)
            except AttributeError as e:
                logger.warning(
                    f"Keyboard key {keycode}. Error: {e}")
                pass  # Handle special keys if needed

        listener = listen_keyboard(on_press=on_press)
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
        # Process stick
        cur_key = self.wc_key_map.get(self.wc_msg.keys, None)
        cur_key_states = self._empty_key_states.copy()
        if cur_key:
            cur_key_states[cur_key] = True

        for key, is_pressed in cur_key_states.items():
            if is_pressed and not self.last_key_states[key]:
                self.handle_joystick_button(key)

        self.last_key_states = cur_key_states

    def handle_joystick_button(self, cur_key):
        # Handle button press
        if cur_key == "start":
            self.use_policy_action = True
            self.get_ready_state = False
            self.node.get_logger().info(colored("Using policy actions", "blue"))
            self.phase = 0.0
        elif cur_key == "B+Y":
            self.use_policy_action = False
            self.get_ready_state = False
            self.node.get_logger().info(colored("Actions set to zero", "blue"))
        elif cur_key == "A+X":
            self.get_ready_state = True
            self.init_count = 0
            self.node.get_logger().info(colored("Setting to init state", "blue"))
        elif cur_key == "Y+left":
            self.command_sender.kp_level -= 0.1
        elif cur_key == "Y+right":
            self.command_sender.kp_level += 0.1
        elif cur_key == "A+left":
            self.command_sender.kp_level -= 0.01
        elif cur_key == "A+right":
            self.command_sender.kp_level += 0.01
        elif cur_key == "A+Y":
            self.command_sender.kp_level = 1.0

        if cur_key in ["5", "6", "4", "7", "0"]:
            self.node.get_logger().info(
                colored(f"Debug kp level: {self.command_sender.kp_level}", "green")
            )

    def run(self):
        total_inference_cnt = 0
        
        # 初始化状态变量
        state_dict = {}
        state_dict["adapt_hx"] = np.zeros((1, 128), dtype=np.float32)
        self.joint_pos_multistep = np.zeros((16, self.num_dofs))
        self.joint_vel_multistep = np.zeros((4, self.num_dofs))
        self.prev_actions = np.zeros((self.num_actions, 3))
        self.action = np.zeros(self.num_actions)
        self.state_dict = state_dict
        self.total_inference_cnt = total_inference_cnt
        
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
        except KeyboardInterrupt:
            pass

    def _rl_step_scheduled(self):
        """包装的RL推理步骤用于调度器"""
        loop_start = time.perf_counter()
        
        if self.use_joystick and self.wc_msg is not None:
            self.process_joystick_input()

        if not self.state_processor._prepare_low_state():
            print("low state not ready.")
            return
        
        self.joint_pos_multistep = np.roll(self.joint_pos_multistep, 1, axis=0)
        self.joint_vel_multistep = np.roll(self.joint_vel_multistep, 1, axis=0)

        self.joint_pos_multistep[0, :] = self.state_processor.joint_pos
        self.joint_vel_multistep[0, :] = self.state_processor.joint_vel

        try:
            # Prepare observations
            self.pre_compute_obs_callback()
            obs_dict = self.prepare_obs_for_rl()
            self.state_dict.update(obs_dict)
            self.state_dict["is_init"] = np.zeros(1, dtype=bool)

            # Inference
            action, self.state_dict = self.policy(self.state_dict)

            # Clip policy action
            action = action.clip(-100, 100)
            self.prev_actions = np.roll(self.prev_actions, 1, axis=1)
            self.prev_actions[:, 0] = action
            
            self.action = self.prev_actions[:, 0]
        except Exception as e:
            print(f"Error in policy inference: {e}")
            self.action = np.zeros(self.num_actions)
            return

        # rule based control flow
        if self.get_ready_state:
            q_target = self.get_init_target()
        elif not self.use_policy_action:
            q_target = self.state_processor.joint_pos
        else:
            policy_action = np.zeros((self.num_dofs))
            policy_action[self.controlled_joint_indices] = self.action
            policy_action = policy_action * self.action_scale
            q_target = policy_action + self.default_dof_angles

        # Clip q target
        q_target = np.clip(
            q_target, self.joint_pos_lower_limit, self.joint_pos_upper_limit
        )

        # Send command
        cmd_q = q_target
        cmd_dq = np.zeros(self.num_dofs)
        cmd_tau = np.zeros(self.num_dofs)
        self.command_sender.send_command(cmd_q, cmd_dq, cmd_tau)

        elapsed = time.perf_counter() - loop_start
        if elapsed > self.rl_dt:
            logger.warning(f"RL step took {elapsed:.6f} seconds, expected {self.rl_dt} seconds")
