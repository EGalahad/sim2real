from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from loguru import logger

try:
    import unitree_interface
except ImportError:
    unitree_interface = None

try:
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
        MotionSwitcherClient,
    )
except ImportError:
    MotionSwitcherClient = None


@dataclass
class UpstreamLowStateSnapshot:
    quaternion: np.ndarray
    gyroscope: np.ndarray
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    joint_torques: np.ndarray
    tick: int


class UpstreamG1IO:
    """Direct Unitree I/O backend using the upstream unitree_interface binding."""

    def __init__(self, *, interface: str, joint_count: int):
        if unitree_interface is None:
            raise ImportError(
                "unitree_interface is required for real_io_backend='upstream' but is not installed."
            )

        self.interface = interface
        self.joint_count = int(joint_count)
        self.robot = unitree_interface.create_robot(
            interface,
            unitree_interface.RobotType.G1,
            unitree_interface.MessageType.HG,
        )
        self.robot.set_control_mode(unitree_interface.ControlMode.PR)
        self._release_motion_mode()
        self.robot.read_low_state()
        logger.info(
            "Initialized upstream G1 I/O backend on {} with {} joints",
            interface,
            joint_count,
        )

    def _release_motion_mode(self) -> None:
        if MotionSwitcherClient is None:
            logger.info("MotionSwitcherClient unavailable; skipping motion mode release.")
            return

        try:
            msc = MotionSwitcherClient()
            msc.SetTimeout(5.0)
            msc.Init()

            status, result = msc.CheckMode()
            logger.info("MotionSwitcher CheckMode: {}, {}", status, result)
            while result is not None and result.get("name", ""):
                logger.info("Releasing active motion mode: {}", result)
                msc.ReleaseMode()
                status, result = msc.CheckMode()
                logger.info("MotionSwitcher CheckMode: {}, {}", status, result)
                time.sleep(1.0)
        except Exception as exc:
            logger.warning("Motion mode release failed: {}", exc)

    def read_low_state(self) -> UpstreamLowStateSnapshot:
        low_state = self.robot.read_low_state()
        count = self.joint_count
        return UpstreamLowStateSnapshot(
            quaternion=np.asarray(low_state.imu.quat, dtype=np.float32).copy(),
            gyroscope=np.asarray(low_state.imu.omega, dtype=np.float32).copy(),
            joint_positions=np.asarray(low_state.motor.q[:count], dtype=np.float32).copy(),
            joint_velocities=np.asarray(low_state.motor.dq[:count], dtype=np.float32).copy(),
            joint_torques=np.asarray(low_state.motor.tau_est[:count], dtype=np.float32).copy(),
            tick=int(time.time_ns() // 1_000_000),
        )

    def send_command(
        self,
        *,
        cmd_q: np.ndarray,
        cmd_dq: np.ndarray,
        cmd_tau: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
    ) -> None:
        cmd = self.robot.create_zero_command()
        cmd.q_target = np.asarray(cmd_q, dtype=np.float32).copy()
        cmd.dq_target = np.asarray(cmd_dq, dtype=np.float32).copy()
        cmd.tau_ff = np.asarray(cmd_tau, dtype=np.float32).copy()
        cmd.kp = np.asarray(kp, dtype=np.float32).copy().tolist()
        cmd.kd = np.asarray(kd, dtype=np.float32).copy().tolist()
        self.robot.write_low_command(cmd)

    def close(self) -> None:
        return
