import numpy as np
import zmq

from sim2real.config.robots.base import RobotCfg
from sim2real.rl_policy.utils.upstream_real_io import UpstreamG1IO
from sim2real.utils.common import LowCmdMessage
from sim2real.utils.strings import resolve_matching_names_values


class ActionManager:
    def __init__(
        self,
        robot_cfg: RobotCfg,
        policy_config,
        real_io_backend: UpstreamG1IO | None = None,
    ):
        self.robot_cfg = robot_cfg
        self.real_io_backend = real_io_backend

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

        # init low cmd publisher
        if self.real_io_backend is None:
            self.zmq_context = zmq.Context.instance()
            self.low_cmd_port = self.robot_cfg.low_cmd_port
            bind_addr = self.robot_cfg.low_cmd_bind_addr
            bind_endpoint = f"tcp://{bind_addr}:{self.low_cmd_port}"

            self.lowcmd_socket: zmq.Socket = self.zmq_context.socket(zmq.PUB)
            self.lowcmd_socket.setsockopt(zmq.SNDHWM, 1)
            self.lowcmd_socket.setsockopt(zmq.LINGER, 0)
            self.lowcmd_socket.bind(bind_endpoint)
        else:
            self.lowcmd_socket = None

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

        if self.real_io_backend is not None:
            self.real_io_backend.send_command(
                cmd_q=self.cmd_q,
                cmd_dq=self.cmd_dq,
                cmd_tau=self.cmd_tau,
                kp=self.joint_kp_unitree,
                kd=self.joint_kd_unitree,
            )
            return

        message = LowCmdMessage(
            q_target=self.cmd_q,
            dq_target=self.cmd_dq,
            tau_ff=self.cmd_tau,
            kp=self.joint_kp_unitree,
            kd=self.joint_kd_unitree,
        )
        # print(self.joint_kp_unitree)
        try:
            self.lowcmd_socket.send(message.to_bytes(), flags=zmq.DONTWAIT)
        except zmq.Again:
            pass
