import mujoco
import numpy as np
import glfw
import sys
from termcolor import colored
from loguru import logger
from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelPublisher
from unitree_sdk2py.idl.default import unitree_go_msg_dds__WirelessController_

import sys
sys.path.append(".")
from utils.strings import resolve_matching_names_values
from utils.strings import unitree_joint_names

import pygame

def yaw_quat(quat: np.ndarray) -> np.ndarray:
    """Extract the yaw component of a quaternion.

    Args:
        quat: The orientation in (w, x, y, z). Shape is (..., 4)

    Returns:
        A quaternion with only yaw component.
    """
    shape = quat.shape
    quat_yaw = quat.copy().reshape(-1, 4)
    qw = quat_yaw[:, 0]
    qx = quat_yaw[:, 1]
    qy = quat_yaw[:, 2]
    qz = quat_yaw[:, 3]
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    quat_yaw[:] = 0.0
    quat_yaw[:, 3] = np.sin(yaw / 2)
    quat_yaw[:, 0] = np.cos(yaw / 2)
    quat_yaw = quat_yaw / np.linalg.norm(quat_yaw, axis=1, keepdims=True)
    return quat_yaw.reshape(shape)

def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions together.

    Args:
        q1: The first quaternion in (w, x, y, z). Shape is (..., 4).
        q2: The second quaternion in (w, x, y, z). Shape is (..., 4).

    Returns:
        The product of the two quaternions in (w, x, y, z). Shape is (..., 4).

    Raises:
        ValueError: Input shapes of ``q1`` and ``q2`` are not matching.
    """
    # check input is correct
    if q1.shape != q2.shape:
        msg = f"Expected input quaternion shape mismatch: {q1.shape} != {q2.shape}."
        raise ValueError(msg)
    # reshape to (N, 4) for multiplication
    shape = q1.shape
    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)
    # extract components from quaternions
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    # perform multiplication
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    return np.stack([w, x, y, z], axis=-1).reshape(shape)

def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Computes the conjugate of a quaternion.

    Args:
        q: The quaternion orientation in (w, x, y, z). Shape is (..., 4).

    Returns:
        The conjugate quaternion in (w, x, y, z). Shape is (..., 4).
    """
    shape = q.shape
    q = q.reshape(-1, 4)
    return np.concatenate((q[:, 0:1], -q[:, 1:]), axis=-1).reshape(shape)


class UnitreeSdk2Bridge:

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        mj_data: mujoco.MjData,
        robot_config: dict,
        scene_config: dict,
    ):
        self.robot_config = robot_config
        self.scene_config = scene_config
        robot_type = robot_config["ROBOT_TYPE"]
        if "g1" in robot_type or "h1-2" in robot_type:
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
            from unitree_sdk2py.idl.default import (
                unitree_hg_msg_dds__LowState_ as LowState_default,
            )
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_

            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        elif "h1" == robot_type or "go2" == robot_type:
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
            from unitree_sdk2py.idl.default import (
                unitree_go_msg_dds__LowState_ as LowState_default,
            )
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
            from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_

            self.low_cmd = unitree_go_msg_dds__LowCmd_()
        else:
            # Raise an error if robot_type is not valid
            raise ValueError(
                f"Invalid robot type '{robot_type}'. Expected 'g1', 'h1', or 'go2'."
            )
        self.mj_model = mj_model
        self.mj_data = mj_data

        self.free_base = scene_config["FREE_BASE"]
        self.torques = np.zeros(self.mj_model.nu)

        self.use_sensor = scene_config["USE_SENSOR"]
        # True: use sensor data; False: use ground truth data
        # Check if the robot is using sensor data
        self.have_imu_ = False
        self.have_frame_sensor_ = False
        if self.use_sensor:
            MOTOR_SENSOR_NUM = 3
            self.dim_motor_sensor = MOTOR_SENSOR_NUM * self.num_dofs
            # Check sensor
            for i in range(self.dim_motor_sensor, self.mj_model.nsensor):
                name = mujoco.mj_id2name(
                    self.mj_model, mujoco._enums.mjtObj.mjOBJ_SENSOR, i
                )
                if name == "imu_quat":
                    self.have_imu_ = True
                if name == "frame_pos":
                    self.have_frame_sensor_ = True

        # Unitree sdk2 message
        self.low_state = LowState_default()
        self.low_state_puber = ChannelPublisher("rt/lowstate", LowState_)
        self.low_state_puber.Init()

        self.low_cmd_suber = ChannelSubscriber("rt/lowcmd", LowCmd_)
        self.low_cmd_suber.Init(self.LowCmdHandler, 1)

        self.wireless_controller = unitree_go_msg_dds__WirelessController_()
        self.wireless_controller_puber = ChannelPublisher(
            "rt/wirelesscontroller", WirelessController_
        )
        self.wireless_controller_puber.Init()

        # joystick
        self.key_map = {
            "R1": 0,
            "L1": 1,
            "start": 2,
            "select": 3,
            "R2": 4,
            "L2": 5,
            "F1": 6,
            "F2": 7,
            "A": 8,
            "B": 9,
            "X": 10,
            "Y": 11,
            "up": 12,
            "right": 13,
            "down": 14,
            "left": 15,
        }
        self.joystick = None

        self.init_joint_indices()

    def init_joint_indices(self):
        joint_names_mujoco = [
            self.mj_model.joint(i).name for i in range(self.mj_model.njnt)
        ]
        actuator_names_mujoco = [
            f"{self.mj_model.actuator(i).name}_joint" for i in range(self.mj_model.nu)
        ]
        self.joint_indices_unitree = []
        self.qpos_adrs = []
        self.qvel_adrs = []
        self.act_adrs = []
        for mjc_idx, mjc_name in enumerate(joint_names_mujoco):
            if mjc_name not in unitree_joint_names:
                continue
            unitree_idx = unitree_joint_names.index(mjc_name)
            qpos_addr = self.mj_model.jnt_qposadr[mjc_idx]
            qvel_addr = self.mj_model.jnt_dofadr[mjc_idx]
            act_addr = actuator_names_mujoco.index(mjc_name)

            self.joint_indices_unitree.append(unitree_idx)
            self.qpos_adrs.append(qpos_addr)
            self.qvel_adrs.append(qvel_addr)
            self.act_adrs.append(act_addr)

        joint_effort_limit_dict = self.robot_config["joint_effort_limit"]
        joint_indices, joint_names_matched, joint_effort_limit = (
            resolve_matching_names_values(
                joint_effort_limit_dict,
                joint_names_mujoco,
                preserve_order=True,
                strict=False,
            )
        )
        self.joint_effort_limit_mjc = np.array(joint_effort_limit)
        self.joint_idx_in_ctrl = np.array(
            [actuator_names_mujoco.index(name) for name in joint_names_matched]
        )

    def compute_torques(self):
        if self.low_cmd:
            if self.use_sensor:
                raise NotImplementedError("Sensor data has not been supported yet")
            for unitree_idx, qpos_addr, qvel_addr, act_addr in zip(
                self.joint_indices_unitree,
                self.qpos_adrs,
                self.qvel_adrs,
                self.act_adrs,
            ):
                self.torques[act_addr] = (
                    self.low_cmd.motor_cmd[unitree_idx].tau
                    + self.low_cmd.motor_cmd[unitree_idx].kp
                    * (
                        self.low_cmd.motor_cmd[unitree_idx].q
                        - self.mj_data.qpos[qpos_addr]
                    )
                    + self.low_cmd.motor_cmd[unitree_idx].kd
                    * (
                        self.low_cmd.motor_cmd[unitree_idx].dq
                        - self.mj_data.qvel[qvel_addr]
                    )
                )
        # Set the torque limit
        self.torques[self.joint_idx_in_ctrl] = np.clip(
            self.torques[self.joint_idx_in_ctrl],
            -self.joint_effort_limit_mjc,
            self.joint_effort_limit_mjc,
        )

    def LowCmdHandler(self, msg):
        self.low_cmd = msg

    def PublishLowState(self):
        if self.mj_data != None:
            if self.use_sensor:
                raise NotImplementedError("Sensor data has not been supported yet")
                for i in range(self.num_dofs):
                    self.low_state.motor_state[i].q = self.mj_data.sensordata[i]
                    self.low_state.motor_state[i].dq = self.mj_data.sensordata[
                        i + self.num_dofs
                    ]
                    self.low_state.motor_state[i].tau_est = self.mj_data.sensordata[
                        i + 2 * self.num_dofs
                    ]
                    # TODO: temperature (Default: human body temperature :))
                    # self.low_state.temperature[0] = 37.5
                    # self.low_state.temperature[1] = 37.5
            else:
                # for unitree_idx, (mjc_idx, qpos_addr, qvel_addr, act_addr) in enumerate(
                #     self.joint_unitree2mjc
                # ):
                #     if mjc_idx == -1:
                #         # this joint is not in the mujoco model
                #         qpos, qvel, act = 0, 0, 0
                #     else:
                #         qpos = self.mj_data.qpos[qpos_addr]
                #         qvel = self.mj_data.qvel[qvel_addr]
                #         act = self.mj_data.actuator_force[act_addr]
                #     self.low_state.motor_state[unitree_idx].q = qpos
                #     self.low_state.motor_state[unitree_idx].dq = qvel
                #     self.low_state.motor_state[unitree_idx].tau_est = act
                joint_pos = self.mj_data.qpos[self.qpos_adrs]
                joint_vel = self.mj_data.qvel[self.qvel_adrs]
                joint_torque = self.mj_data.actuator_force[self.act_adrs]
                for mjc_idx, unitree_idx in enumerate(self.joint_indices_unitree):
                    self.low_state.motor_state[unitree_idx].q = joint_pos[mjc_idx]
                    self.low_state.motor_state[unitree_idx].dq = joint_vel[mjc_idx]
                    self.low_state.motor_state[unitree_idx].tau_est = joint_torque[mjc_idx]

                    # TODO: temperature (Default: human body temperature :))
                    # self.low_state.temperature[0] = 37.5
                    # self.low_state.temperature[1] = 37.5

            # Get data from sensors
            if self.use_sensor and self.have_frame_sensor_:
                raise NotImplementedError("Sensor data has not been supported yet")

                self.low_state.imu_state.quaternion[0] = self.mj_data.sensordata[
                    self.dim_motor_sensor + 0
                ]
                self.low_state.imu_state.quaternion[1] = self.mj_data.sensordata[
                    self.dim_motor_sensor + 1
                ]
                self.low_state.imu_state.quaternion[2] = self.mj_data.sensordata[
                    self.dim_motor_sensor + 2
                ]
                self.low_state.imu_state.quaternion[3] = self.mj_data.sensordata[
                    self.dim_motor_sensor + 3
                ]
                self.low_state.imu_state.gyroscope[0] = self.mj_data.sensordata[
                    self.dim_motor_sensor + 4
                ]
                self.low_state.imu_state.gyroscope[1] = self.mj_data.sensordata[
                    self.dim_motor_sensor + 5
                ]
                self.low_state.imu_state.gyroscope[2] = self.mj_data.sensordata[
                    self.dim_motor_sensor + 6
                ]
            else:
                # quaternion: w, x, y, z
                root_quat_w = self.mj_data.qpos[3:7]
                root_quat_yaw_w = yaw_quat(root_quat_w)
                root_quat_b = quat_mul(quat_conjugate(root_quat_yaw_w), root_quat_w)
                # root_quat_b = root_quat_w
                # Note: 
                # does not matter, because we only use projected gravity,
                # which is same even if yaw is non zero and different
                for i in range(4):
                    self.low_state.imu_state.quaternion[i] = root_quat_b[i]

                # angular velocity: x, y, z
                root_ang_vel_b = self.mj_data.qvel[3:6]
                for i in range(3):
                    self.low_state.imu_state.gyroscope[i] = root_ang_vel_b[i]

            # acceleration: x, y, z (only available when frame sensor is enabled)
            if self.have_frame_sensor_:
                raise NotImplementedError("Sensor data has not been supported yet")
                self.low_state.imu_state.accelerometer[0] = self.mj_data.sensordata[
                    self.dim_motor_sensor + 7
                ]
                self.low_state.imu_state.accelerometer[1] = self.mj_data.sensordata[
                    self.dim_motor_sensor + 8
                ]
                self.low_state.imu_state.accelerometer[2] = self.mj_data.sensordata[
                    self.dim_motor_sensor + 9
                ]
            self.low_state.tick = int(self.mj_data.time * 1e3)
            self.low_state_puber.Write(self.low_state)

            return

    def PublishWirelessController(self):
        if self.joystick != None:
            pygame.event.get()
            key_state = [0] * 16
            key_state[self.key_map["R1"]] = self.joystick.get_button(
                self.button_id["RB"]
            )
            key_state[self.key_map["L1"]] = self.joystick.get_button(
                self.button_id["LB"]
            )
            key_state[self.key_map["start"]] = self.joystick.get_button(
                self.button_id["START"]
            )
            key_state[self.key_map["select"]] = self.joystick.get_button(
                self.button_id["SELECT"]
            )
            key_state[self.key_map["R2"]] = (
                self.joystick.get_axis(self.axis_id["RT"]) > 0
            )
            key_state[self.key_map["L2"]] = (
                self.joystick.get_axis(self.axis_id["LT"]) > 0
            )
            key_state[self.key_map["F1"]] = 0
            key_state[self.key_map["F2"]] = 0
            key_state[self.key_map["A"]] = self.joystick.get_button(self.button_id["A"])
            key_state[self.key_map["B"]] = self.joystick.get_button(self.button_id["B"])
            key_state[self.key_map["X"]] = self.joystick.get_button(self.button_id["X"])
            key_state[self.key_map["Y"]] = self.joystick.get_button(self.button_id["Y"])
            key_state[self.key_map["up"]] = self.joystick.get_hat(0)[1] > 0
            key_state[self.key_map["right"]] = self.joystick.get_hat(0)[0] > 0
            key_state[self.key_map["down"]] = self.joystick.get_hat(0)[1] < 0
            key_state[self.key_map["left"]] = self.joystick.get_hat(0)[0] < 0

            key_value = 0
            for i in range(16):
                key_value += key_state[i] << i

            self.wireless_controller.keys = key_value
            self.wireless_controller.lx = self.joystick.get_axis(self.axis_id["LX"])
            self.wireless_controller.ly = -self.joystick.get_axis(self.axis_id["LY"])
            self.wireless_controller.rx = self.joystick.get_axis(self.axis_id["RX"])
            self.wireless_controller.ry = -self.joystick.get_axis(self.axis_id["RY"])

            self.wireless_controller_puber.Write(self.wireless_controller)

    def SetupJoystick(self, device_id=0, js_type="xbox"):
        pygame.init()
        pygame.joystick.init()
        joystick_count = pygame.joystick.get_count()
        if joystick_count > 0:
            self.joystick = pygame.joystick.Joystick(device_id)
            self.joystick.init()
        else:
            print("No gamepad detected.")
            sys.exit()

        if js_type == "xbox":
            if sys.platform.startswith("linux"):
                self.axis_id = {
                    "LX": 0,  # Left stick axis x
                    "LY": 1,  # Left stick axis y
                    "RX": 3,  # Right stick axis x
                    "RY": 4,  # Right stick axis y
                    "LT": 2,  # Left trigger
                    "RT": 5,  # Right trigger
                    "DX": 6,  # Directional pad x
                    "DY": 7,  # Directional pad y
                }
                self.button_id = {
                    "X": 2,
                    "Y": 3,
                    "B": 1,
                    "A": 0,
                    "LB": 4,
                    "RB": 5,
                    "SELECT": 6,
                    "START": 7,
                    "XBOX": 8,
                    "LSB": 9,
                    "RSB": 10,
                }
            elif sys.platform == "darwin":
                self.axis_id = {
                    "LX": 0,  # Left stick axis x
                    "LY": 1,  # Left stick axis y
                    "RX": 2,  # Right stick axis x
                    "RY": 3,  # Right stick axis y
                    "LT": 4,  # Left trigger
                    "RT": 5,  # Right trigger
                }
                self.button_id = {
                    "X": 2,
                    "Y": 3,
                    "B": 1,
                    "A": 0,
                    "LB": 9,
                    "RB": 10,
                    "SELECT": 4,
                    "START": 6,
                    "XBOX": 5,
                    "LSB": 7,
                    "RSB": 8,
                    "DYU": 11,
                    "DYD": 12,
                    "DXL": 13,
                    "DXR": 14,
                }
            else:
                print("Unsupported OS. ")

        elif js_type == "switch":
            # Yuanhang: may differ for different OS, need to be checked
            self.axis_id = {
                "LX": 0,  # Left stick axis x
                "LY": 1,  # Left stick axis y
                "RX": 2,  # Right stick axis x
                "RY": 3,  # Right stick axis y
                "LT": 5,  # Left trigger
                "RT": 4,  # Right trigger
                "DX": 6,  # Directional pad x
                "DY": 7,  # Directional pad y
            }

            self.button_id = {
                "X": 3,
                "Y": 4,
                "B": 1,
                "A": 0,
                "LB": 6,
                "RB": 7,
                "SELECT": 10,
                "START": 11,
            }
        else:
            print("Unsupported gamepad. ")

    def PrintSceneInformation(self):
        print(" ")
        logger.info(colored("<<------------- Link ------------->>", "green"))
        for i in range(self.mj_model.nbody):
            name = mujoco.mj_id2name(self.mj_model, mujoco._enums.mjtObj.mjOBJ_BODY, i)
            if name:
                logger.info(f"link_index: {i}, name: {name}")
        print(" ")

        logger.info(colored("<<------------- Joint ------------->>", "green"))
        for i in range(self.mj_model.njnt):
            name = mujoco.mj_id2name(self.mj_model, mujoco._enums.mjtObj.mjOBJ_JOINT, i)
            if name:
                logger.info(f"joint_index: {i}, name: {name}")
        print(" ")

        logger.info(colored("<<------------- Actuator ------------->>", "green"))
        for i in range(self.mj_model.nu):
            name = mujoco.mj_id2name(
                self.mj_model, mujoco._enums.mjtObj.mjOBJ_ACTUATOR, i
            )
            if name:
                logger.info(f"actuator_index: {i}, name: {name}")
        print(" ")

        logger.info(colored("<<------------- Sensor ------------->>", "green"))
        index = 0
        for i in range(self.mj_model.nsensor):
            name = mujoco.mj_id2name(
                self.mj_model, mujoco._enums.mjtObj.mjOBJ_SENSOR, i
            )
            if name:
                logger.info(
                    f"sensor_index: {index}, name: {name}, dim: {self.mj_model.sensor_dim[i]}"
                )
            index = index + self.mj_model.sensor_dim[i]
        print(" ")


class ElasticBand:
    """
    ref: https://github.com/unitreerobotics/unitree_mujoco
    """

    def __init__(self):
        self.stiffness = 200
        self.damping = 100
        self.point = np.array([0, 0, 3])
        self.length = 0
        self.enable = True

    def Advance(self, x, dx):
        """
        Args:
          δx: desired position - current position
          dx: current velocity
        """
        δx = self.point - x
        distance = np.linalg.norm(δx)
        direction = δx / distance
        v = np.dot(dx, direction)
        f = (self.stiffness * (distance - self.length) - self.damping * v) * direction
        return f

    def MujuocoKeyCallback(self, key):
        if key == glfw.KEY_7:
            self.length -= 0.1
        if key == glfw.KEY_8:
            self.length += 0.1
        if key == glfw.KEY_9:
            self.enable = not self.enable
