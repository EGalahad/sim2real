"""Bridge Unitree low state/command channels to the sim2real ZMQ interface."""

import sched
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict

import numpy as np
import tyro
import zmq
from loguru import logger

from sim2real.config.robots import get_robot_cfg
from sim2real.config.robots.base import RobotCfg, get_unitree_dds_family
from sim2real.utils.common import LowCmdMessage, LowStateMessage, UNITREE_LEGGED_CONST
from sim2real.utils.profiling import ScopedTimer
if TYPE_CHECKING:
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmd_

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.utils.crc import CRC

from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowState_go
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowState_hg

from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

# UNITREE_INTERFACE = "enx00e04c680182"
# UNITREE_INTERFACE = "enP8p1s0"
UNITREE_INTERFACE = "eth0"
# UNITREE_INTERFACE = "enxf8e43b70ab8b"
# UNITREE_INTERFACE = "enx9c69d3560bd9"
UNITREE_DOMAIN_ID = 0

class RealBridge:
    """Bridge Unitree SDK2 channels to the sim2real ZMQ interface."""

    def __init__(self, robot_cfg: RobotCfg, rate_hz=200, interface: str = UNITREE_INTERFACE):
        self.robot_cfg = robot_cfg
        self.robot_name = robot_cfg.name
        self.rate_hz = rate_hz
        self.dt = 1.0 / rate_hz
        self.interface = interface

        self._init_unitree_channels()
        self._init_zmq()
        self._init_low_cmd_template()

        self.has_received_command = False

        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()

        status, result = self.msc.CheckMode()
        print(status, result)
        if result is not None:
            while result['name']:
                self.msc.ReleaseMode()
                status, result = self.msc.CheckMode()
                print(status, result)
                time.sleep(1)

    def _init_unitree_channels(self):
        domain_id = UNITREE_DOMAIN_ID
        dds_family = get_unitree_dds_family(self.robot_name)
        ChannelFactoryInitialize(domain_id, self.interface)
        print(
            f"ChannelFactory initialized with domain ID {domain_id} on interface {self.interface}"
        )

        if dds_family == "go":
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowState_go
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmd_go
            from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_

            self.low_state_unitree_sub = ChannelSubscriber("rt/lowstate", LowState_go)
            self.low_state_unitree_sub.Init(handler=None, queueLen=0)
            self.low_cmd_unitree_pub = ChannelPublisher("rt/lowcmd", LowCmd_go)
            self.low_cmd_unitree_pub.Init()
            self.low_cmd = unitree_go_msg_dds__LowCmd_()
        elif dds_family == "hg":
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowState_hg
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmd_hg
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_

            self.low_state_unitree_sub = ChannelSubscriber("rt/lowstate", LowState_hg)
            self.low_state_unitree_sub.Init(handler=None, queueLen=0)
            self.low_cmd_unitree_pub = ChannelPublisher("rt/lowcmd", LowCmd_hg)
            self.low_cmd_unitree_pub.Init()
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        else:
            raise NotImplementedError(
                f"Robot name {self.robot_name} is not supported for the real bridge."
            )

        self.crc = CRC()

    def _init_zmq(self):
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

    def _init_low_cmd_template(self):
        if get_unitree_dds_family(self.robot_name) == "go":
            self.low_cmd.head[0] = 0xFE
            self.low_cmd.head[1] = 0xEF

        self.low_cmd.level_flag = UNITREE_LEGGED_CONST["LOWLEVEL"]
        self.low_cmd.gpio = 0
        self.low_cmd.mode_machine = 5
        self.low_cmd.mode_pr = 0

        for cmd in self.low_cmd.motor_cmd:
            cmd: "MotorCmd_" = cmd
            cmd.mode = 1
            cmd.q = UNITREE_LEGGED_CONST["PosStopF"]
            cmd.kp = 0.0
            cmd.dq = UNITREE_LEGGED_CONST["VelStopF"]
            cmd.kd = 0.0
            cmd.tau = 0.0

    def _low_state_unitree_to_zmq(self) -> Dict[str, float]:
        with ScopedTimer("real_bridge.low_state") as total_timer:
            with ScopedTimer("real_bridge.low_state.read") as read_timer:
                msg: LowState_hg | LowState_go = self.low_state_unitree_sub.Read()
                imu = msg.imu_state
                motor_state = msg.motor_state

            with ScopedTimer("real_bridge.low_state.pack") as pack_timer:
                joint_count = len(self.robot_cfg.joint_names)
                joint_pos = np.zeros(joint_count, dtype=np.float32)
                joint_vel = np.zeros(joint_count, dtype=np.float32)
                joint_tau = np.zeros(joint_count, dtype=np.float32)
                for idx in range(joint_count):
                    joint_pos[idx] = motor_state[idx].q
                    joint_vel[idx] = motor_state[idx].dq
                    joint_tau[idx] = motor_state[idx].tau_est

                low_state_msg = LowStateMessage(
                    quaternion=np.array(imu.quaternion, dtype=np.float32),
                    gyroscope=np.array(imu.gyroscope, dtype=np.float32),
                    joint_positions=joint_pos,
                    joint_velocities=joint_vel,
                    joint_torques=joint_tau,
                    tick=int(getattr(msg, "tick", 0)),
                )

            with ScopedTimer("real_bridge.low_state.publish") as publish_timer:
                try:
                    self.low_state_zmq_pub.send(low_state_msg.to_bytes(), flags=zmq.DONTWAIT)
                except zmq.Again:
                    pass

        return {
            "total_s": total_timer.last_time,
            "read_s": read_timer.last_time,
            "pack_s": pack_timer.last_time,
            "publish_s": publish_timer.last_time,
        }

    def _low_cmd_zmq_to_unitree(self) -> Dict[str, float]:
        recv_s = 0.0
        decode_s = 0.0
        apply_s = 0.0
        publish_s = 0.0
        updated = False
        command_count = 0

        with ScopedTimer("real_bridge.low_cmd") as total_timer:
            while True:
                with ScopedTimer("real_bridge.low_cmd.recv") as recv_timer:
                    try:
                        data = self.low_cmd_zmq_sub.recv(flags=zmq.DONTWAIT)
                    except zmq.Again:
                        data = None
                recv_s += recv_timer.last_time

                if data is None:
                    break

                with ScopedTimer("real_bridge.low_cmd.decode") as decode_timer:
                    try:
                        low_cmd = LowCmdMessage.from_bytes(data)
                    except Exception as exc:
                        logger.warning(f"Failed to decode low command message: {exc}")
                        low_cmd = None
                decode_s += decode_timer.last_time

                if low_cmd is None:
                    continue

                if low_cmd.q_target.size != len(self.robot_cfg.joint_names):
                    logger.warning(
                        "Received low command with unexpected size {}",
                        low_cmd.q_target.size,
                    )
                    continue

                with ScopedTimer("real_bridge.low_cmd.apply") as apply_timer:
                    motor_cmd = self.low_cmd.motor_cmd
                    count = min(len(self.robot_cfg.joint_names), len(motor_cmd))
                    for i in range(count):
                        cmd: "MotorCmd_" = motor_cmd[i]
                        cmd.q = float(low_cmd.q_target[i])
                        cmd.dq = float(low_cmd.dq_target[i])
                        cmd.tau = float(low_cmd.tau_ff[i])
                        cmd.kp = float(low_cmd.kp[i])
                        cmd.kd = float(low_cmd.kd[i])
                apply_s += apply_timer.last_time

                with ScopedTimer("real_bridge.low_cmd.publish") as publish_timer:
                    self.low_cmd.crc = self.crc.Crc(self.low_cmd)
                    self.low_cmd_unitree_pub.Write(self.low_cmd)
                publish_s += publish_timer.last_time

                updated = True
                command_count += 1

        if updated:
            self.has_received_command = True

        return {
            "total_s": total_timer.last_time,
            "recv_s": recv_s,
            "decode_s": decode_s,
            "apply_s": apply_s,
            "publish_s": publish_s,
            "command_count": command_count,
        }

    def run(self):
        logger.info(
            "Real bridge running: Unitree <-> ZMQ (low_state pub on {}, low_cmd sub on {})",
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
            logger.info("Real bridge stopped.")

    def _step(self):
        with ScopedTimer("real_bridge.step") as step_timer:
            low_cmd_profile = self._low_cmd_zmq_to_unitree()
            low_state_profile = self._low_state_unitree_to_zmq()

        elapsed = step_timer.last_time
        if elapsed > self.dt:
            logger.warning(
                (
                    "Bridge step took {:.3f} ms, expected {:.3f} ms. "
                    "breakdown: low_cmd={:.3f} ms "
                    "(poll={:.3f}, decode={:.3f}, apply={:.3f}, publish={:.3f}, cmds={}), "
                    "low_state={:.3f} ms (read={:.3f}, pack={:.3f}, publish={:.3f})"
                ),
                elapsed * 1000.0,
                self.dt * 1000.0,
                low_cmd_profile["total_s"] * 1000.0,
                low_cmd_profile["recv_s"] * 1000.0,
                low_cmd_profile["decode_s"] * 1000.0,
                low_cmd_profile["apply_s"] * 1000.0,
                low_cmd_profile["publish_s"] * 1000.0,
                low_cmd_profile["command_count"],
                low_state_profile["total_s"] * 1000.0,
                low_state_profile["read_s"] * 1000.0,
                low_state_profile["pack_s"] * 1000.0,
                low_state_profile["publish_s"] * 1000.0,
            )


@dataclass
class Args:
    """Unitree <-> ZMQ real bridge."""

    robot: str = "g1"
    rate: float = 100.0
    interface: str = UNITREE_INTERFACE


def main(args: Args) -> None:

    bridge = RealBridge(robot_cfg=get_robot_cfg(args.robot), rate_hz=args.rate, interface=args.interface)
    bridge.run()


if __name__ == "__main__":
    main(tyro.cli(Args))
