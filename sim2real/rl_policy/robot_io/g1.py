from __future__ import annotations

import importlib
import subprocess
import sys
import time
from types import ModuleType
from typing import Callable

import numpy as np
from loguru import logger

from sim2real.config.robots.base import RobotCfg
from sim2real.rl_policy.robot_io.base import RobotIO, RobotState


class G1RobotIO(RobotIO):
    """Inline G1 backend backed by the optional ``unitree_interface`` SDK."""

    def __init__(
        self,
        robot_cfg: RobotCfg,
        *,
        interface: str,
        sdk_module: ModuleType | None = None,
        motion_switcher_factory: Callable[[], object] | None = None,
        channel_factory_initialize: Callable[[int, str], None] | None = None,
        motion_release_delay_s: float = 1.0,
    ) -> None:
        self._joint_count = len(robot_cfg.joint_names)
        self._robot = None
        try:
            # The Python SDK2 motion-switcher and the C++ inline SDK share the
            # same DDS domain.  Release the high-level motion service before
            # even importing the inline SDK, because its module import creates
            # a DDS participant and can make CycloneDDS reject SDK2 RPC topics.
            if (
                motion_switcher_factory is None
                and channel_factory_initialize is None
            ):
                self._switch_to_debug_mode_in_subprocess(
                    domain_id=int(robot_cfg.domain_id),
                    interface=interface,
                    release_delay_s=motion_release_delay_s,
                )
            else:
                self._switch_to_debug_mode(
                    domain_id=int(robot_cfg.domain_id),
                    interface=interface,
                    motion_switcher_factory=motion_switcher_factory,
                    channel_factory_initialize=channel_factory_initialize,
                    release_delay_s=motion_release_delay_s,
                )
            if sdk_module is None:
                try:
                    sdk_module = importlib.import_module("unitree_interface")
                except ImportError as exc:
                    raise ImportError(
                        "unitree_interface is required for robot_io='inline' with "
                        "robot='g1'. Install the robot-g1 optional dependency."
                    ) from exc
            self._robot = sdk_module.create_robot(
                interface,
                sdk_module.RobotType.G1,
                sdk_module.MessageType.HG,
            )
            self._robot.set_control_mode(sdk_module.ControlMode.PR)
            # Preserve the existing inline startup behavior: prime the SDK once.
            self._robot.read_low_state()
        except Exception:
            self.close()
            raise

    @staticmethod
    def _switch_to_debug_mode_in_subprocess(
        *, domain_id: int, interface: str, release_delay_s: float
    ) -> None:
        command = [
            sys.executable,
            "-m",
            "sim2real.rl_policy.robot_io.g1_debug_mode",
            "--domain-id",
            str(domain_id),
            "--interface",
            interface,
            "--release-delay-s",
            str(release_delay_s),
        ]
        try:
            subprocess.run(command, check=True, timeout=30.0)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(
                "Failed to switch the G1 to debug mode before inline SDK startup"
            ) from exc

    def _switch_to_debug_mode(
        self,
        *,
        domain_id: int,
        interface: str,
        motion_switcher_factory: Callable[[], object] | None,
        channel_factory_initialize: Callable[[int, str], None] | None,
        release_delay_s: float,
    ) -> None:
        if release_delay_s < 0:
            raise ValueError("motion_release_delay_s must be >= 0")
        if motion_switcher_factory is None or channel_factory_initialize is None:
            try:
                motion_switcher_module = importlib.import_module(
                    "unitree_sdk2py.comm.motion_switcher.motion_switcher_client"
                )
                channel_module = importlib.import_module("unitree_sdk2py.core.channel")
            except ImportError as exc:
                raise ImportError(
                    "unitree_sdk2py is required to switch inline G1 control "
                    "to debug mode. Install the robot-g1 optional dependency."
                ) from exc
            if motion_switcher_factory is None:
                motion_switcher_factory = motion_switcher_module.MotionSwitcherClient
            if channel_factory_initialize is None:
                channel_factory_initialize = channel_module.ChannelFactoryInitialize

        channel_factory_initialize(domain_id, interface)
        self._motion_switcher = motion_switcher_factory()
        self._motion_switcher.SetTimeout(5.0)
        self._motion_switcher.Init()

        while True:
            status, result = self._motion_switcher.CheckMode()
            if status != 0 or result is None:
                raise RuntimeError(
                    "Failed to query the G1 motion mode before inline control: "
                    f"status={status} result={result!r}"
                )
            if not result.get("name"):
                logger.info("G1 inline robot is in debug mode")
                return

            logger.info("Releasing active G1 motion mode: {}", result.get("name"))
            release_status, _ = self._motion_switcher.ReleaseMode()
            if release_status != 0:
                raise RuntimeError(
                    "Failed to release the active G1 motion mode before inline "
                    f"control: status={release_status} mode={result.get('name')!r}"
                )
            if release_delay_s > 0:
                time.sleep(release_delay_s)

    def read_state(self) -> RobotState:
        low_state = self._robot.read_low_state()
        qpos = np.zeros(7 + self._joint_count, dtype=np.float32)
        qvel = np.zeros(6 + self._joint_count, dtype=np.float32)
        qpos[3:7] = np.asarray(low_state.imu.quat, dtype=np.float32)
        qpos[7:] = np.asarray(low_state.motor.q[: self._joint_count], dtype=np.float32)
        qvel[3:6] = np.asarray(low_state.imu.omega, dtype=np.float32)
        qvel[6:] = np.asarray(
            low_state.motor.dq[: self._joint_count], dtype=np.float32
        )
        joint_torque = np.asarray(
            low_state.motor.tau_est[: self._joint_count], dtype=np.float32
        ).copy()
        return RobotState(
            qpos=qpos,
            qvel=qvel,
            joint_torque=joint_torque,
            tick=int(time.time_ns() // 1_000_000),
        )

    def write_command(
        self,
        q_target: np.ndarray,
        dq_target: np.ndarray,
        tau_ff: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
    ) -> None:
        command = self._robot.create_zero_command()
        command.q_target = np.asarray(q_target, dtype=np.float32).copy()
        command.dq_target = np.asarray(dq_target, dtype=np.float32).copy()
        command.tau_ff = np.asarray(tau_ff, dtype=np.float32).copy()
        command.kp = np.asarray(kp, dtype=np.float32).copy().tolist()
        command.kd = np.asarray(kd, dtype=np.float32).copy().tolist()
        self._robot.write_low_command(command)

    def close(self) -> None:
        close = getattr(self._robot, "close", None)
        if callable(close):
            close()
