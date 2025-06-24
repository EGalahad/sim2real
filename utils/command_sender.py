import numpy as np

from unitree_sdk2py.core.channel import ChannelPublisher

from unitree_sdk2py.utils.crc import CRC
from utils.strings import resolve_matching_names_values
from utils.strings import unitree_joint_names


class CommandSender:
    def __init__(self, robot_config, policy_config):
        self.robot_config = robot_config
        self.policy_config = policy_config
        if self.robot_config["ROBOT_TYPE"] == "h1" or self.robot_config["ROBOT_TYPE"] == "go2":
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
            from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_

            self.low_cmd = unitree_go_msg_dds__LowCmd_()
        elif (
            self.robot_config["ROBOT_TYPE"] == "g1_29dof"
            or self.robot_config["ROBOT_TYPE"] == "h1-2_21dof"
            or self.robot_config["ROBOT_TYPE"] == "h1-2_27dof"
        ):
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_

            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        else:
            raise NotImplementedError(
                f"Robot type {self.robot_config['ROBOT_TYPE']} is not supported yet"
            )
        # init robot and kp kd
        self._kp_level = 1.0  # 0.1

        joint_kp_dict = self.policy_config["joint_kp"]
        joint_indices, joint_names, joint_kp = resolve_matching_names_values(
            joint_kp_dict,
            unitree_joint_names,
            preserve_order=True,
            strict=False,
        )
        self.joint_kp_unitree_default = np.zeros(len(unitree_joint_names))
        self.joint_kp_unitree_default[joint_indices] = joint_kp
        self.joint_kp_unitree = self.joint_kp_unitree_default.copy()

        joint_kd_dict = self.policy_config["joint_kd"]
        joint_indices, joint_names, joint_kd = resolve_matching_names_values(
            joint_kd_dict,
            unitree_joint_names,
            preserve_order=True,
            strict=False,
        )
        self.joint_kd_unitree = np.zeros(len(unitree_joint_names))
        self.joint_kd_unitree[joint_indices] = joint_kd

        default_joint_pos_dict = self.policy_config["default_joint_pos"]
        joint_indices, joint_names, default_joint_pos = resolve_matching_names_values(
            default_joint_pos_dict,
            unitree_joint_names,
            preserve_order=True,
            strict=False,
        )
        self.default_joint_pos_unitree = np.zeros(len(unitree_joint_names))
        self.default_joint_pos_unitree[joint_indices] = default_joint_pos

        joint_names_isaac = self.policy_config["isaac_joint_names"]
        self.joint_indices_unitree = [unitree_joint_names.index(name) for name in joint_names_isaac]

        # init low cmd publisher
        self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher_.Init()
        self.InitLowCmd()
        self.low_state = None
        self.crc = CRC()

    @property
    def kp_level(self):
        return self._kp_level

    @kp_level.setter
    def kp_level(self, value):
        self._kp_level = value
        self.joint_kp_unitree[:] = self.joint_kp_unitree_default * self._kp_level

    def InitLowCmd(self):
        # h1/go2:
        if self.robot_config["ROBOT_TYPE"] == "h1" or self.robot_config["ROBOT_TYPE"] == "go2":
            self.low_cmd.head[0] = 0xFE
            self.low_cmd.head[1] = 0xEF
        else:
            pass

        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0
        unitree_legged_const = self.robot_config["UNITREE_LEGGED_CONST"]
        for unitree_idx in range(len(unitree_joint_names)):
            self.low_cmd.motor_cmd[unitree_idx].mode = 0x01
            # self.low_cmd.motor_cmd[unitree_motor_idx].mode = 0x0A
            self.low_cmd.motor_cmd[unitree_idx].q = (
                unitree_legged_const["PosStopF"]
            )
            self.low_cmd.motor_cmd[unitree_idx].kp = 0
            self.low_cmd.motor_cmd[unitree_idx].dq = (
                unitree_legged_const["VelStopF"]
            )
            self.low_cmd.motor_cmd[unitree_idx].kd = 0
            self.low_cmd.motor_cmd[unitree_idx].tau = 0
            if (
                self.robot_config["ROBOT_TYPE"] == "g1_29dof"
                or self.robot_config["ROBOT_TYPE"] == "h1-2_21dof"
                or self.robot_config["ROBOT_TYPE"] == "h1-2_27dof"
            ):
                self.low_cmd.mode_machine = unitree_legged_const["MODE_MACHINE"]
                self.low_cmd.mode_pr = unitree_legged_const["MODE_PR"]
            else:
                pass
    
        self.cmd_q = np.zeros(len(unitree_joint_names))
        self.cmd_dq = np.zeros(len(unitree_joint_names))
        self.cmd_tau = np.zeros(len(unitree_joint_names))

        self.cmd_q[:] = self.default_joint_pos_unitree

    def send_command(self, cmd_q, cmd_dq, cmd_tau):
        self.cmd_q[self.joint_indices_unitree] = cmd_q
        self.cmd_dq[self.joint_indices_unitree] = cmd_dq
        self.cmd_tau[self.joint_indices_unitree] = cmd_tau
        
        for unitree_idx in range(len(unitree_joint_names)):
            self.low_cmd.motor_cmd[unitree_idx].q = self.cmd_q[unitree_idx]
            self.low_cmd.motor_cmd[unitree_idx].dq = self.cmd_dq[unitree_idx]
            self.low_cmd.motor_cmd[unitree_idx].tau = self.cmd_tau[unitree_idx]

            self.low_cmd.motor_cmd[unitree_idx].kp = self.joint_kp_unitree[
                unitree_idx
            ]
            self.low_cmd.motor_cmd[unitree_idx].kd = self.joint_kd_unitree[
                unitree_idx
            ]

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher_.Write(self.low_cmd)
