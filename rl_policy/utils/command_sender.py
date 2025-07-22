import numpy as np


from utils.strings import resolve_matching_names_values
from utils.strings import unitree_joint_names


class CommandSender:
    def __init__(self, robot_config, policy_config):
        self.robot_type = robot_config["ROBOT_TYPE"]
        if self.robot_type == "h1" or self.robot_type == "go2":
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
            from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_

            self.low_cmd = unitree_go_msg_dds__LowCmd_()
        elif (
            self.robot_type == "g1_29dof"
            or self.robot_type == "h1-2_21dof"
            or self.robot_type == "h1-2_27dof"
        ):
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_

            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        elif self.robot_type == "g1_real":
            self.robot = robot_config["robot"]
        else:
            raise NotImplementedError(f"Robot type {self.robot_type} is not supported yet")

        # init low cmd publisher
        if self.robot_type != "g1_real":
            from unitree_sdk2py.core.channel import ChannelPublisher
            self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmd_)
            self.lowcmd_publisher_.Init()
            self.InitLowCmd()
            from unitree_sdk2py.utils.crc import CRC
            self.crc = CRC()

        # init robot and kp kd
        self._kp_level = 1.0  # 0.1

        self.policy_config = policy_config
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

    @property
    def kp_level(self):
        return self._kp_level

    @kp_level.setter
    def kp_level(self, value):
        self._kp_level = value
        self.joint_kp_unitree[:] = self.joint_kp_unitree_default * self._kp_level

    def InitLowCmd(self):
        # h1/go2:
        if self.robot_type == "h1" or self.robot_type == "go2":
            self.low_cmd.head[0] = 0xFE
            self.low_cmd.head[1] = 0xEF
        else:
            pass

        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0
        from utils.common import UNITREE_LEGGED_CONST
        unitree_legged_const = UNITREE_LEGGED_CONST
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
            # g1, h1-2:
            if (
                self.robot_type == "g1_29dof"
                or self.robot_type == "g1_real"
                or self.robot_type == "h1-2_21dof"
                or self.robot_type == "h1-2_27dof"
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
        if self.robot_type != "g1_real":
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
        else:
            cmd = self.robot.create_zero_command()

            # Apply kp_level scaling (kd remains constant, consistent with original implementation)
            kp_scaled = self.joint_kp_unitree * self._kp_level
            kd_scaled = self.joint_kd_unitree

            q_target = list(cmd.q_target)
            dq_target = list(cmd.dq_target)
            tau_ff = list(cmd.tau_ff)
            kp = list(cmd.kp)
            kd = list(cmd.kd)
            for i_policy, idx_unitree in enumerate(self.joint_indices_unitree):
                q_target[idx_unitree] = float(cmd_q[i_policy])
                dq_target[idx_unitree] = float(cmd_dq[i_policy])
                tau_ff[idx_unitree] = float(cmd_tau[i_policy])
                kp[idx_unitree] = float(kp_scaled[idx_unitree])
                kd[idx_unitree] = float(kd_scaled[idx_unitree])

            cmd.q_target = q_target
            cmd.dq_target = dq_target
            cmd.tau_ff = tau_ff
            cmd.kp = kp
            cmd.kd = kd

            self.robot.write_low_command(cmd)

