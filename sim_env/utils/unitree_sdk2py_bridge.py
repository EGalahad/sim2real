import numpy as np
import glfw
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import mujoco
import mujoco

from termcolor import colored
from loguru import logger
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelPublisher

import sys
sys.path.append(".")
from utils.strings import resolve_matching_names_values, unitree_joint_names
from utils.math import quat_mul, quat_conjugate, yaw_quat


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

        self.torques = np.zeros(self.mj_model.nu)

        self.low_state = LowState_default()
        self.low_state_puber = ChannelPublisher("rt/lowstate", LowState_)
        self.low_state_puber.Init()

        self.low_cmd_suber = ChannelSubscriber("rt/lowcmd", LowCmd_)
        self.low_cmd_suber.Init(self.LowCmdHandler, 1)

        self.init_joint_indices()

    def init_joint_indices(self):
        joint_names_mujoco = [
            self.mj_model.joint(i).name for i in range(self.mj_model.njnt)
        ]
        actuator_names_mujoco = [
            self.mj_model.actuator(i).name for i in range(self.mj_model.nu)
        ]
        self.joint_indices_unitree = []
        self.qpos_adrs = []
        self.qvel_adrs = []
        self.act_adrs = []

        shared_joint_names = set(joint_names_mujoco) & set(unitree_joint_names)
        for name in shared_joint_names:
            print(f"shared_joint_names: {name}")
            self.joint_indices_unitree.append(unitree_joint_names.index(name))

            joint_idx = joint_names_mujoco.index(name)
            self.qpos_adrs.append(self.mj_model.jnt_qposadr[joint_idx])
            self.qvel_adrs.append(self.mj_model.jnt_dofadr[joint_idx])
            self.act_adrs.append(actuator_names_mujoco.index(name))
        
        if "floating_base_joint" in joint_names_mujoco:
            root_joint_idx = joint_names_mujoco.index("floating_base_joint")
        elif "pelvis_root" in joint_names_mujoco:
            root_joint_idx = joint_names_mujoco.index("pelvis_root")
        else:
            raise ValueError("No root joint found in the MuJoCo model.")
        self.root_qpos_adr = self.mj_model.jnt_qposadr[root_joint_idx]
        self.root_qvel_adr = self.mj_model.jnt_dofadr[root_joint_idx]

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
        if self.mj_data == None:
            return

        joint_pos = self.mj_data.qpos[self.qpos_adrs]
        joint_vel = self.mj_data.qvel[self.qvel_adrs]
        joint_torque = self.mj_data.actuator_force[self.act_adrs]
        for mjc_idx, unitree_idx in enumerate(self.joint_indices_unitree):
            self.low_state.motor_state[unitree_idx].q = joint_pos[mjc_idx]
            self.low_state.motor_state[unitree_idx].dq = joint_vel[mjc_idx]
            self.low_state.motor_state[unitree_idx].tau_est = joint_torque[mjc_idx]

        # quaternion: w, x, y, z
        root_quat_w = self.mj_data.qpos[self.root_qpos_adr + 3:self.root_qpos_adr+7]
        root_quat_yaw_w = yaw_quat(root_quat_w)
        root_quat_b = quat_mul(quat_conjugate(root_quat_yaw_w), root_quat_w)

        for i in range(4):
            self.low_state.imu_state.quaternion[i] = root_quat_b[i]

        # angular velocity: x, y, z
        root_ang_vel_b = self.mj_data.qvel[self.root_qvel_adr + 3:self.root_qvel_adr+6]
        for i in range(3):
            self.low_state.imu_state.gyroscope[i] = root_ang_vel_b[i]

        self.low_state.tick = int(self.mj_data.time * 1e3)
        self.low_state_puber.Write(self.low_state)

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

    def MujocoKeyCallback(self, key):
        if key == glfw.KEY_7:
            self.length -= 0.1
        if key == glfw.KEY_8:
            self.length += 0.1
        if key == glfw.KEY_9:
            self.enable = not self.enable
