"""Bridge Unitree low state/command channels to the sim2real ZMQ interface.

This variant keeps the same ZMQ contract as scripts/real_bridge.py, but uses
the upstream C++ unitree_interface binding for robot I/O.
"""

from __future__ import annotations

import sched
import time
from dataclasses import dataclass

import tyro
import zmq
from loguru import logger

from sim2real.config.robots import get_robot_cfg
from sim2real.config.robots.base import RobotCfg
from sim2real.rl_policy.utils.upstream_real_io import UpstreamG1IO
from sim2real.utils.common import LowCmdMessage, LowStateMessage
from sim2real.utils.profiling import ScopedTimer


UNITREE_INTERFACE = "eth0"
UINT32_MASK = 0xFFFFFFFF


class RealBridgeCpp:
    """Bridge upstream C++ Unitree I/O to the sim2real ZMQ interface."""

    def __init__(
        self,
        robot_cfg: RobotCfg,
        rate_hz: float = 200.0,
        interface: str = UNITREE_INTERFACE,
    ):
        if robot_cfg.name != "g1":
            raise NotImplementedError(
                "real_bridge_cpp.py currently supports only robot='g1'."
            )

        self.robot_cfg = robot_cfg
        self.rate_hz = float(rate_hz)
        self.dt = 1.0 / self.rate_hz
        self.interface = interface
        self.joint_count = len(self.robot_cfg.joint_names)

        self._init_unitree_interface()
        self._init_zmq()

        self.has_received_command = False

    def _init_unitree_interface(self) -> None:
        self.real_io = UpstreamG1IO(
            interface=self.interface,
            joint_count=self.joint_count,
        )

    def _init_zmq(self) -> None:
        self.zmq_context = zmq.Context.instance()

        self.low_state_port = self.robot_cfg.low_state_port
        low_state_bind_addr = self.robot_cfg.low_state_bind_addr
        low_state_endpoint = f"tcp://{low_state_bind_addr}:{self.low_state_port}"
        self.low_state_zmq_pub: zmq.Socket = self.zmq_context.socket(zmq.PUB)
        self.low_state_zmq_pub.setsockopt(zmq.SNDHWM, 1)
        self.low_state_zmq_pub.setsockopt(zmq.LINGER, 0)
        self.low_state_zmq_pub.bind(low_state_endpoint)

        self.low_cmd_port = self.robot_cfg.low_cmd_port
        low_cmd_host = self.robot_cfg.low_cmd_host
        low_cmd_endpoint = f"tcp://{low_cmd_host}:{self.low_cmd_port}"
        self.low_cmd_zmq_sub: zmq.Socket = self.zmq_context.socket(zmq.SUB)
        self.low_cmd_zmq_sub.setsockopt(zmq.SUBSCRIBE, b"")
        self.low_cmd_zmq_sub.setsockopt(zmq.CONFLATE, 1)
        self.low_cmd_zmq_sub.setsockopt(zmq.RCVTIMEO, 0)
        self.low_cmd_zmq_sub.setsockopt(zmq.LINGER, 0)
        self.low_cmd_zmq_sub.connect(low_cmd_endpoint)

    def _low_state_unitree_to_zmq(self) -> dict[str, float]:
        with ScopedTimer("real_bridge_cpp.low_state") as total_timer:
            with ScopedTimer("real_bridge_cpp.low_state.read") as read_timer:
                low_state = self.real_io.read_low_state()

            with ScopedTimer("real_bridge_cpp.low_state.pack") as pack_timer:
                low_state_msg = LowStateMessage(
                    quaternion=low_state.quaternion,
                    gyroscope=low_state.gyroscope,
                    joint_positions=low_state.joint_positions,
                    joint_velocities=low_state.joint_velocities,
                    joint_torques=low_state.joint_torques,
                    tick=int(low_state.tick) & UINT32_MASK,
                )

            with ScopedTimer("real_bridge_cpp.low_state.publish") as publish_timer:
                try:
                    self.low_state_zmq_pub.send(
                        low_state_msg.to_bytes(),
                        flags=zmq.DONTWAIT,
                    )
                except zmq.Again:
                    pass

        return {
            "total_s": total_timer.last_time,
            "read_s": read_timer.last_time,
            "pack_s": pack_timer.last_time,
            "publish_s": publish_timer.last_time,
        }

    def _low_cmd_zmq_to_unitree(self) -> dict[str, float]:
        recv_s = 0.0
        decode_s = 0.0
        publish_s = 0.0
        command_count = 0

        with ScopedTimer("real_bridge_cpp.low_cmd") as total_timer:
            while True:
                with ScopedTimer("real_bridge_cpp.low_cmd.recv") as recv_timer:
                    try:
                        data = self.low_cmd_zmq_sub.recv(flags=zmq.DONTWAIT)
                    except zmq.Again:
                        data = None
                recv_s += recv_timer.last_time

                if data is None:
                    break

                with ScopedTimer("real_bridge_cpp.low_cmd.decode") as decode_timer:
                    try:
                        low_cmd = LowCmdMessage.from_bytes(data)
                    except Exception as exc:
                        logger.warning("Failed to decode low command message: {}", exc)
                        low_cmd = None
                decode_s += decode_timer.last_time

                if low_cmd is None:
                    continue

                if low_cmd.q_target.size != self.joint_count:
                    logger.warning(
                        "Received low command with unexpected size {}",
                        low_cmd.q_target.size,
                    )
                    continue

                with ScopedTimer("real_bridge_cpp.low_cmd.publish") as publish_timer:
                    self.real_io.send_command(
                        cmd_q=low_cmd.q_target,
                        cmd_dq=low_cmd.dq_target,
                        cmd_tau=low_cmd.tau_ff,
                        kp=low_cmd.kp,
                        kd=low_cmd.kd,
                    )
                publish_s += publish_timer.last_time

                self.has_received_command = True
                command_count += 1

        return {
            "total_s": total_timer.last_time,
            "recv_s": recv_s,
            "decode_s": decode_s,
            "publish_s": publish_s,
            "command_count": command_count,
        }

    def run(self) -> None:
        logger.info(
            "Real bridge cpp running: Unitree C++ interface <-> ZMQ "
            "(low_state pub on {}, low_cmd sub on {})",
            self.low_state_port,
            self.low_cmd_port,
        )

        scheduler = sched.scheduler(time.perf_counter, time.sleep)
        next_run_time = time.perf_counter()

        try:
            while True:
                scheduler.enterabs(next_run_time, 1, self._step, ())
                scheduler.run()
                next_run_time += self.dt
        except KeyboardInterrupt:
            logger.info("Real bridge cpp stopped.")
        finally:
            self.real_io.close()

    def _step(self) -> None:
        with ScopedTimer("real_bridge_cpp.step") as step_timer:
            low_cmd_profile = self._low_cmd_zmq_to_unitree()
            low_state_profile = self._low_state_unitree_to_zmq()

        elapsed = step_timer.last_time
        if elapsed > self.dt:
            logger.warning(
                (
                    "Bridge step took {:.3f} ms, expected {:.3f} ms. "
                    "breakdown: low_cmd={:.3f} ms "
                    "(poll={:.3f}, decode={:.3f}, publish={:.3f}, cmds={}), "
                    "low_state={:.3f} ms (read={:.3f}, pack={:.3f}, publish={:.3f})"
                ),
                elapsed * 1000.0,
                self.dt * 1000.0,
                low_cmd_profile["total_s"] * 1000.0,
                low_cmd_profile["recv_s"] * 1000.0,
                low_cmd_profile["decode_s"] * 1000.0,
                low_cmd_profile["publish_s"] * 1000.0,
                low_cmd_profile["command_count"],
                low_state_profile["total_s"] * 1000.0,
                low_state_profile["read_s"] * 1000.0,
                low_state_profile["pack_s"] * 1000.0,
                low_state_profile["publish_s"] * 1000.0,
            )


@dataclass
class Args:
    """Unitree C++ interface <-> ZMQ real bridge."""

    robot: str = "g1"
    rate: float = 100.0
    interface: str = UNITREE_INTERFACE


def main(args: Args) -> None:
    bridge = RealBridgeCpp(
        robot_cfg=get_robot_cfg(args.robot),
        rate_hz=args.rate,
        interface=args.interface,
    )
    bridge.run()


if __name__ == "__main__":
    main(tyro.cli(Args))
