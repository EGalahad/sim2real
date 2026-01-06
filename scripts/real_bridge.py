"""Bridge Unitree low state/command channels to the sim2real ZMQ interface."""

import argparse
import sched
import time
import threading

import numpy as np
import yaml
import zmq
from loguru import logger

import sys

sys.path.append(".")

from utils.common import LowCmdMessage, LowStateMessage, PORTS, UNITREE_LEGGED_CONST
from utils.strings import unitree_joint_names
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmd_

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.utils.crc import CRC

from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowState_go
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmd_go
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowState_hg
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmd_hg

from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

UNITREE_INTERFACE = "eth0"
UNITREE_DOMAIN_ID = 0

class RealBridge:
    """Bridge Unitree SDK2 channels to the sim2real ZMQ interface."""

    def __init__(self, robot_config, rate_hz=200):
        self.robot_config = robot_config
        self.robot_type = robot_config["ROBOT_TYPE"]
        self.rate_hz = rate_hz
        self.dt = 1.0 / rate_hz

        self._init_unitree_channels()
        self._init_zmq()
        self._init_low_cmd_template()

        self.has_received_command = False

        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()

        status, result = self.msc.CheckMode()
        print(result)
        while result['name']:
            self.msc.ReleaseMode()
            status, result = self.msc.CheckMode()
            print(result)
            time.sleep(1)

    def _init_unitree_channels(self):
        self.robot_config["DOMAIN_ID"] = UNITREE_DOMAIN_ID
        self.robot_config["INTERFACE"] = UNITREE_INTERFACE
        ChannelFactoryInitialize(
            self.robot_config["DOMAIN_ID"], self.robot_config["INTERFACE"]
        )
        print(f"ChannelFactory initialized with domain ID {self.robot_config['DOMAIN_ID']} on interface {self.robot_config['INTERFACE']}")

        if self.robot_type in ("h1", "go2"):
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowState_go
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmd_go
            from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_

            self.low_state_unitree_sub = ChannelSubscriber("rt/lowstate", LowState_go)
            self.low_state_unitree_sub.Init(handler=None, queueLen=0)
            self.low_cmd_unitree_pub = ChannelPublisher("rt/lowcmd", LowCmd_go)
            self.low_cmd_unitree_pub.Init()
            self.low_cmd = unitree_go_msg_dds__LowCmd_()
        elif self.robot_type in ("g1_29dof", "h1-2_21dof", "h1-2_27dof"):
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
                f"Robot type {self.robot_type} is not supported for the real bridge."
            )

        self.crc = CRC()

    def _init_zmq(self):
        self.zmq_context = zmq.Context.instance()

        self.low_state_port = self.robot_config.get(
            "LOW_STATE_PORT", PORTS.get("low_state", 5590)
        )
        low_state_bind_addr = self.robot_config.get("LOW_STATE_BIND_ADDR", "*")
        low_state_endpoint = f"tcp://{low_state_bind_addr}:{self.low_state_port}"
        self.low_state_zmq_pub: zmq.Socket = self.zmq_context.socket(zmq.PUB)
        self.low_state_zmq_pub.setsockopt(zmq.SNDHWM, 1)
        self.low_state_zmq_pub.setsockopt(zmq.LINGER, 0)
        self.low_state_zmq_pub.bind(low_state_endpoint)

        self.low_cmd_port = self.robot_config.get(
            "LOW_CMD_PORT", PORTS.get("low_cmd", 5591)
        )
        low_cmd_host = self.robot_config.get("LOW_CMD_HOST", "127.0.0.1")
        low_cmd_endpoint = f"tcp://{low_cmd_host}:{self.low_cmd_port}"
        self.low_cmd_zmq_sub: zmq.Socket = self.zmq_context.socket(zmq.SUB)
        self.low_cmd_zmq_sub.setsockopt(zmq.SUBSCRIBE, b"")
        self.low_cmd_zmq_sub.setsockopt(zmq.CONFLATE, 1)
        self.low_cmd_zmq_sub.setsockopt(zmq.RCVTIMEO, 0)
        self.low_cmd_zmq_sub.setsockopt(zmq.LINGER, 0)
        self.low_cmd_zmq_sub.connect(low_cmd_endpoint)

    def _init_low_cmd_template(self):
        if self.robot_type in ("h1", "go2"):
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

    def _low_state_unitree_to_zmq(self):
        msg: LowState_hg | LowState_go = self.low_state_unitree_sub.Read()
        imu = msg.imu_state
        motor_state = msg.motor_state
        # print(f"low state mode machine: {msg.mode_machine}")

        joint_pos = np.zeros(len(unitree_joint_names), dtype=np.float32)
        joint_vel = np.zeros(len(unitree_joint_names), dtype=np.float32)
        joint_tau = np.zeros(len(unitree_joint_names), dtype=np.float32)
        for idx in range(len(unitree_joint_names)):
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

        try:
            self.low_state_zmq_pub.send(low_state_msg.to_bytes(), flags=zmq.DONTWAIT)
            print(f"published low state msg tick={low_state_msg.tick}")
        except zmq.Again:
            pass

    def _low_cmd_zmq_to_unitree(self):
        updated = False
        while True:
            try:
                data = self.low_cmd_zmq_sub.recv(flags=zmq.DONTWAIT)
            except zmq.Again:
                break

            try:
                low_cmd = LowCmdMessage.from_bytes(data)
            except Exception as exc:
                logger.warning(f"Failed to decode low command message: {exc}")
                continue

            if low_cmd.q_target.size != len(unitree_joint_names):
                logger.warning(
                    "Received low command with unexpected size {}",
                    low_cmd.q_target.size,
                )
                continue

            motor_cmd = self.low_cmd.motor_cmd
            count = min(len(unitree_joint_names), len(motor_cmd))
            for i in range(count):
                cmd: "MotorCmd_" = motor_cmd[i]
                cmd.q = float(low_cmd.q_target[i])
                cmd.dq = float(low_cmd.dq_target[i])
                cmd.tau = float(low_cmd.tau_ff[i])
                cmd.kp = float(low_cmd.kp[i])
                cmd.kd = float(low_cmd.kd[i])

            self.low_cmd.crc = self.crc.Crc(self.low_cmd)
            self.low_cmd_unitree_pub.Write(self.low_cmd)
            print(f"command q_: {low_cmd.q_target}")
            print(f"command kp: {low_cmd.kp}")
            print(f"sent low cmd msg")

            updated = True

        if updated:
            self.has_received_command = True

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
        loop_start = time.perf_counter()
        self._low_cmd_zmq_to_unitree()
        self._low_state_unitree_to_zmq()
        elapsed = time.perf_counter() - loop_start
        if elapsed > self.dt:
            logger.warning(
                f"Bridge step took {elapsed:.6f} seconds, expected {self.dt:.6f}"
            )


def main():
    parser = argparse.ArgumentParser(description="Unitree <-> ZMQ real bridge")
    parser.add_argument(
        "--robot_config",
        type=str,
        default="config/robot/g1.yaml",
        help="Robot config file",
    )
    parser.add_argument(
        "--rate", type=float, default=200.0, help="Bridge loop rate (Hz)"
    )
    args = parser.parse_args()

    with open(args.robot_config) as file:
        robot_config = yaml.load(file, Loader=yaml.FullLoader)

    bridge = RealBridge(robot_config=robot_config, rate_hz=args.rate)
    bridge.run()


if __name__ == "__main__":
    main()
