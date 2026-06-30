import numpy as np
import zmq
from typing import Any, Literal

from sim2real.config.robots.base import RobotCfg
from sim2real.utils.common import LowCmdMessage
from sim2real.utils.strings import resolve_matching_names_values


class ActionManager:
    def __init__(
        self,
        robot_cfg: RobotCfg,
        policy_config,
        robot_io: Literal["inline", "zmq"] = "zmq",
        robot: Any | None = None,
    ):
        self.robot_cfg = robot_cfg
        self.robot_io = robot_io
        self.robot = robot

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
        self.lowcmd_socket: zmq.Socket | None = None
        if self.robot_io == "inline":
            if self.robot is None:
                raise ValueError("robot is required when robot_io='inline'")
        elif self.robot_io == "zmq":
            self.zmq_context = zmq.Context.instance()
            self.low_cmd_port = self.robot_cfg.low_cmd_port
            bind_addr = self.robot_cfg.low_cmd_bind_addr
            bind_endpoint = f"tcp://{bind_addr}:{self.low_cmd_port}"

            self.lowcmd_socket = self.zmq_context.socket(zmq.PUB)
            self.lowcmd_socket.setsockopt(zmq.SNDHWM, 1)
            self.lowcmd_socket.setsockopt(zmq.LINGER, 0)
            self.lowcmd_socket.bind(bind_endpoint)
        else:
            raise ValueError(f"Unsupported robot_io: {self.robot_io}")

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

        if self.robot_io == "inline":
            self._send_inline_command()
            return

        if self.robot_io == "zmq":
            self._publish_zmq_command()
            return

        raise ValueError(f"Unsupported robot_io: {self.robot_io}")

    def _send_inline_command(self):
        if self.robot is None:
            raise RuntimeError("Inline robot is not initialized")

        cmd = self.robot.create_zero_command()
        cmd.q_target = np.asarray(self.cmd_q, dtype=np.float32).copy()
        cmd.dq_target = np.asarray(self.cmd_dq, dtype=np.float32).copy()
        cmd.tau_ff = np.asarray(self.cmd_tau, dtype=np.float32).copy()
        cmd.kp = np.asarray(self.joint_kp_unitree, dtype=np.float32).copy().tolist()
        cmd.kd = np.asarray(self.joint_kd_unitree, dtype=np.float32).copy().tolist()
        self.robot.write_low_command(cmd)

    def _publish_zmq_command(self):
        if self.lowcmd_socket is None:
            raise RuntimeError("ZMQ low command socket is not initialized")

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
