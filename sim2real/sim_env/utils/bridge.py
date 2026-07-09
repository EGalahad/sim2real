import numpy as np
import mujoco
import zmq

from loguru import logger

from sim2real.config.robots.base import RobotCfg
from sim2real.utils.common import LowStateMessage, LowCmdMessage
from sim2real.utils.strings import resolve_matching_names_values


class SimulationBridge:

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        mj_data: mujoco.MjData,
        robot_cfg: RobotCfg,
    ):
        self.robot_cfg = robot_cfg
        self.mj_model = mj_model
        self.mj_data = mj_data

        self.torques = np.zeros(self.mj_model.nu)

        # ZMQ communication setup
        self.zmq_context = zmq.Context.instance()

        self.low_state_port = self.robot_cfg.low_state_port
        low_state_bind_addr = self.robot_cfg.low_state_bind_addr
        low_state_endpoint = f"tcp://{low_state_bind_addr}:{self.low_state_port}"
        self.low_state_pub = self.zmq_context.socket(zmq.PUB)
        self.low_state_pub.setsockopt(zmq.SNDHWM, 1)
        self.low_state_pub.setsockopt(zmq.LINGER, 0)
        self.low_state_pub.bind(low_state_endpoint)

        self.low_cmd_port = self.robot_cfg.low_cmd_port
        low_cmd_host = self.robot_cfg.low_cmd_host
        low_cmd_endpoint = f"tcp://{low_cmd_host}:{self.low_cmd_port}"
        self.low_cmd_sub = self.zmq_context.socket(zmq.SUB)
        self.low_cmd_sub.setsockopt(zmq.SUBSCRIBE, b"")
        self.low_cmd_sub.setsockopt(zmq.CONFLATE, 1)
        self.low_cmd_sub.setsockopt(zmq.RCVTIMEO, 0)
        self.low_cmd_sub.setsockopt(zmq.LINGER, 0)
        self.low_cmd_sub.connect(low_cmd_endpoint)

        total_joints = len(self.robot_cfg.joint_names)
        self.cmd_q = np.zeros(total_joints, dtype=np.float32)
        self.cmd_dq = np.zeros(total_joints, dtype=np.float32)
        self.cmd_tau = np.zeros(total_joints, dtype=np.float32)
        self.cmd_kp = np.zeros(total_joints, dtype=np.float32)
        self.cmd_kd = np.zeros(total_joints, dtype=np.float32)
        self.has_received_command = False

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

        for name in self.robot_cfg.joint_names:
            if name not in joint_names_mujoco or name not in actuator_names_mujoco:
                continue
            self.joint_indices_unitree.append(self.robot_cfg.joint_names.index(name))

            joint_idx = joint_names_mujoco.index(name)
            self.qpos_adrs.append(self.mj_model.jnt_qposadr[joint_idx])
            self.qvel_adrs.append(self.mj_model.jnt_dofadr[joint_idx])
            self.act_adrs.append(actuator_names_mujoco.index(name))
        
        root_joint_idx = None
        for root_joint_name in self.robot_cfg.root_joint_names:
            if root_joint_name in joint_names_mujoco:
                root_joint_idx = joint_names_mujoco.index(root_joint_name)
                break
        if root_joint_idx is None:
            raise ValueError("No root joint found in the MuJoCo model.")
        self.root_qpos_adr = self.mj_model.jnt_qposadr[root_joint_idx]
        self.root_qvel_adr = self.mj_model.jnt_dofadr[root_joint_idx]

        joint_effort_limit_dict = self.robot_cfg.joint_effort_limit
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
        self.torques[:] = 0.0
        self._poll_low_cmd()

        if self.has_received_command:
            for unitree_idx, qpos_addr, qvel_addr, act_addr in zip(
                self.joint_indices_unitree,
                self.qpos_adrs,
                self.qvel_adrs,
                self.act_adrs,
            ):
                q_des = self.cmd_q[unitree_idx]
                dq_des = self.cmd_dq[unitree_idx]
                tau_ff = self.cmd_tau[unitree_idx]
                kp = self.cmd_kp[unitree_idx]
                kd = self.cmd_kd[unitree_idx]

                self.torques[act_addr] = (
                    tau_ff
                    + kp * (q_des - self.mj_data.qpos[qpos_addr])
                    + kd * (dq_des - self.mj_data.qvel[qvel_addr])
                )
        # Set the torque limit
        self.torques[self.joint_idx_in_ctrl] = np.clip(
            self.torques[self.joint_idx_in_ctrl],
            -self.joint_effort_limit_mjc,
            self.joint_effort_limit_mjc,
        )

    def _poll_low_cmd(self):
        """Non-blocking command subscriber that keeps the most recent message."""
        if self.low_cmd_sub is None:
            return

        updated = False
        while True:
            try:
                data = self.low_cmd_sub.recv(flags=zmq.DONTWAIT)
            except zmq.Again:
                break

            try:
                low_cmd = LowCmdMessage.from_bytes(data)
            except Exception as exc:
                logger.warning(f"Failed to decode low command message: {exc}")
                continue

            if low_cmd.q_target.size != len(self.robot_cfg.joint_names):
                logger.warning(
                    "Received low command with unexpected size {}",
                    low_cmd.q_target.size,
                )
                continue

            self.cmd_q[:] = low_cmd.q_target
            self.cmd_dq[:] = low_cmd.dq_target
            self.cmd_tau[:] = low_cmd.tau_ff
            self.cmd_kp[:] = low_cmd.kp
            self.cmd_kd[:] = low_cmd.kd
            updated = True

        if updated:
            self.has_received_command = True

    def publish_low_state(self):
        if self.mj_data is None:
            return

        joint_pos_partial = self.mj_data.qpos[self.qpos_adrs]
        joint_vel_partial = self.mj_data.qvel[self.qvel_adrs]
        joint_torque_partial = self.mj_data.actuator_force[self.act_adrs]

        joint_pos_full = np.zeros(len(self.robot_cfg.joint_names), dtype=np.float32)
        joint_vel_full = np.zeros(len(self.robot_cfg.joint_names), dtype=np.float32)
        joint_tau_full = np.zeros(len(self.robot_cfg.joint_names), dtype=np.float32)
        for mjc_idx, unitree_idx in enumerate(self.joint_indices_unitree):
            joint_pos_full[unitree_idx] = joint_pos_partial[mjc_idx]
            joint_vel_full[unitree_idx] = joint_vel_partial[mjc_idx]
            joint_tau_full[unitree_idx] = joint_torque_partial[mjc_idx]

        # quaternion: w, x, y, z
        root_quat_w = self.mj_data.qpos[self.root_qpos_adr + 3:self.root_qpos_adr+7]

        # angular velocity: x, y, z
        root_ang_vel_b = self.mj_data.qvel[self.root_qvel_adr + 3:self.root_qvel_adr+6]
        low_state_msg = LowStateMessage(
            quaternion=root_quat_w,
            gyroscope=root_ang_vel_b,
            joint_positions=joint_pos_full,
            joint_velocities=joint_vel_full,
            joint_torques=joint_tau_full,
            tick=int(self.mj_data.time * 1e3),
        )

        try:
            self.low_state_pub.send(low_state_msg.to_bytes(), flags=zmq.DONTWAIT)
        except zmq.Again:
            pass
