import numpy as np

from sim2real.config.robots.base import RobotCfg
from sim2real.rl_policy.robot_io import RobotIO
from sim2real.utils.strings import resolve_matching_names_values


class ActionManager:
    def __init__(
        self,
        robot_cfg: RobotCfg,
        policy_config,
        robot_io: RobotIO,
    ):
        self.robot_cfg = robot_cfg
        self.robot_io = robot_io

        self.policy_config = policy_config
        joint_kp_dict = self.policy_config["joint_kp"]
        joint_indices, joint_names, joint_kp = resolve_matching_names_values(
            joint_kp_dict,
            self.robot_cfg.joint_names,
            preserve_order=True,
            strict=False,
        )
        self.joint_kp_unitree = np.zeros(len(self.robot_cfg.joint_names))
        self.joint_kp_unitree[joint_indices] = joint_kp

        joint_kd_dict = self.policy_config["joint_kd"]
        joint_indices, joint_names, joint_kd = resolve_matching_names_values(
            joint_kd_dict,
            self.robot_cfg.joint_names,
            preserve_order=True,
            strict=False,
        )
        self.joint_kd_unitree = np.zeros(len(self.robot_cfg.joint_names))
        self.joint_kd_unitree[joint_indices] = joint_kd

        default_joint_pos_dict = self.policy_config["default_joint_pos"]
        joint_indices, joint_names, default_joint_pos = resolve_matching_names_values(
            default_joint_pos_dict,
            self.robot_cfg.joint_names,
            preserve_order=True,
            strict=False,
        )
        self.default_joint_pos_unitree = np.zeros(len(self.robot_cfg.joint_names))
        self.default_joint_pos_unitree[joint_indices] = default_joint_pos

        self.joint_names = list(self.robot_cfg.joint_names)
        # joint_names_simulation = self.policy_config["joint_names_simulation"]
        # # Policy q targets are expressed in simulation observation order.
        # self.joint_indices_unitree = [
        #     unitree_joint_names.index(name) for name in joint_names_simulation
        # ]

        self.InitLowCmd()

    def InitLowCmd(self):
        self.cmd_q = np.zeros(len(self.robot_cfg.joint_names))
        self.cmd_dq = np.zeros(len(self.robot_cfg.joint_names))
        self.cmd_tau = np.zeros(len(self.robot_cfg.joint_names))

        self.cmd_q[:] = self.default_joint_pos_unitree

    def send_command(self, cmd_q, cmd_dq, cmd_tau):
        self.cmd_q[:] = cmd_q
        self.cmd_dq[:] = cmd_dq
        self.cmd_tau[:] = cmd_tau

        self.robot_io.write_command(
            q_target=self.cmd_q,
            dq_target=self.cmd_dq,
            tau_ff=self.cmd_tau,
            kp=self.joint_kp_unitree,
            kd=self.joint_kd_unitree,
        )
