import time
import numpy as np
import zmq

from sim2real.utils.robot_defs import G1_JOINT_NAMES
from sim2real.utils.common import LowCmdMessage, PORTS
from sim2real.utils.strings import resolve_matching_names_values


class CommandSender:
    def __init__(self, robot_config, policy_config):
        # init robot and kp kd
        self._kp_level = 1.0  # 0.1

        self.policy_config = policy_config
        joint_kp_dict = self.policy_config["joint_kp"]
        joint_indices, joint_names, joint_kp = resolve_matching_names_values(
            joint_kp_dict,
            G1_JOINT_NAMES,
            preserve_order=True,
            strict=False,
        )
        self.joint_kp_unitree_default = np.zeros(len(G1_JOINT_NAMES))
        self.joint_kp_unitree_default[joint_indices] = joint_kp
        self.joint_kp_unitree = self.joint_kp_unitree_default.copy()

        joint_kd_dict = self.policy_config["joint_kd"]
        joint_indices, joint_names, joint_kd = resolve_matching_names_values(
            joint_kd_dict,
            G1_JOINT_NAMES,
            preserve_order=True,
            strict=False,
        )
        self.joint_kd_unitree = np.zeros(len(G1_JOINT_NAMES))
        self.joint_kd_unitree[joint_indices] = joint_kd

        default_joint_pos_dict = self.policy_config["default_joint_pos"]
        joint_indices, joint_names, default_joint_pos = resolve_matching_names_values(
            default_joint_pos_dict,
            G1_JOINT_NAMES,
            preserve_order=True,
            strict=False,
        )
        self.default_joint_pos_unitree = np.zeros(len(G1_JOINT_NAMES))
        self.default_joint_pos_unitree[joint_indices] = default_joint_pos

        self.joint_names = list(G1_JOINT_NAMES)
        # joint_names_simulation = self.policy_config["joint_names_simulation"]
        # # Policy q targets are expressed in simulation observation order.
        # self.joint_indices_unitree = [
        #     unitree_joint_names.index(name) for name in joint_names_simulation
        # ]

        # init low cmd publisher
        self.zmq_context = zmq.Context.instance()
        self.low_cmd_port = robot_config.get(
            "LOW_CMD_PORT", PORTS.get("low_cmd", 55901)
        )
        bind_addr = robot_config.get("LOW_CMD_BIND_ADDR", "*")
        bind_endpoint = f"tcp://{bind_addr}:{self.low_cmd_port}"

        self.lowcmd_socket: zmq.Socket = self.zmq_context.socket(zmq.PUB)
        self.lowcmd_socket.setsockopt(zmq.SNDHWM, 1)
        self.lowcmd_socket.setsockopt(zmq.LINGER, 0)
        self.lowcmd_socket.bind(bind_endpoint)

        self.InitLowCmd()

    @property
    def kp_level(self):
        return self._kp_level

    @kp_level.setter
    def kp_level(self, value):
        self._kp_level = value
        self.joint_kp_unitree[:] = self.joint_kp_unitree_default * self._kp_level

    def InitLowCmd(self):
        self.cmd_q = np.zeros(len(G1_JOINT_NAMES))
        self.cmd_dq = np.zeros(len(G1_JOINT_NAMES))
        self.cmd_tau = np.zeros(len(G1_JOINT_NAMES))

        self.cmd_q[:] = self.default_joint_pos_unitree

    def send_command(self, cmd_q, cmd_dq, cmd_tau):
        self.cmd_q[:] = cmd_q
        self.cmd_dq[:] = cmd_dq
        self.cmd_tau[:] = cmd_tau
        
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
