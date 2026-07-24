import numpy as np
import zmq
import threading
import time

from loguru import logger
from typing import Dict, Optional

from sim2real.config.robots.base import RobotCfg
from sim2real.rl_policy.robot_io import RobotIO, RobotState
from sim2real.utils.common import ZMQSubscriber, PORTS


class StateProcessor:
    """Copies backend robot state into the stable arrays consumed by observations."""

    def __init__(
        self,
        robot_cfg: RobotCfg,
        robot_io: RobotIO,
    ):
        self.robot_cfg = robot_cfg
        self.mocap_ip = self.robot_cfg.mocap_ip
        self.robot_io = robot_io
        self.latest_state: RobotState | None = None

        # Initialize joint mapping
        self.joint_names = list(self.robot_cfg.joint_names)
        self.num_dof = len(self.joint_names)

        self.qpos = np.zeros(3 + 4 + self.num_dof)
        self.qvel = np.zeros(3 + 3 + self.num_dof)

        # create views of qpos and qvel
        self.root_pos_w = self.qpos[0:3]
        self.root_lin_vel_w = self.qvel[0:3]

        self.root_quat_w = self.qpos[3:7]
        self.root_ang_vel_b = self.qvel[3:6]

        self.joint_pos = self.qpos[7:]
        self.joint_vel = self.qvel[6:]

        self.mocap_subscribers: Dict[str, ZMQSubscriber] = {}  # Dictionary to store ZMQ subscribers
        self.mocap_threads = {}      # Dictionary to store subscriber threads
        self.mocap_data = {}         # Dictionary to store received mocap data
        self.mocap_data_lock = threading.Lock()  # Lock for thread-safe access

    def register_subscriber(self, object_name: str, port: Optional[int] = None):
        if object_name in self.mocap_subscribers:
            return

        # init ZMQ subscriber
        port = PORTS.get(f"{object_name}_pose", port)
        subscriber = ZMQSubscriber(port)
        self.mocap_subscribers[object_name] = subscriber

        def _sub_thread(obj_name: str):
            while True:
                try:
                    pose_msg = self.mocap_subscribers[obj_name].receive_pose()
                    if pose_msg:
                        with self.mocap_data_lock:
                            self.mocap_data[f"{obj_name}_pos"] = pose_msg.position
                            self.mocap_data[f"{obj_name}_quat"] = pose_msg.quaternion
                except zmq.Again:
                    time.sleep(0.001)
                except Exception as e:
                    logger.warning(f"{obj_name} subscriber error: {e}")
                    time.sleep(0.01)

        # start subscriber thread
        th = threading.Thread(target=_sub_thread, args=(object_name,), daemon=True)
        th.start()
        self.mocap_threads[object_name] = th


    def get_mocap_data(self, key: str):
        """Thread-safe method to get mocap data"""
        with self.mocap_data_lock:
            return self.mocap_data.get(key, None)

    def _prepare_low_state(self) -> bool:
        state = self.robot_io.read_state()
        if state is None:
            return False
        self._apply_state(state)
        return True

    def _apply_state(self, state: RobotState) -> None:
        qpos = np.asarray(state.qpos, dtype=np.float32)
        qvel = np.asarray(state.qvel, dtype=np.float32)
        joint_torque = np.asarray(state.joint_torque, dtype=np.float32)
        if qpos.shape != self.qpos.shape:
            raise ValueError(
                f"Robot qpos shape {qpos.shape} != expected {self.qpos.shape}"
            )
        if qvel.shape != self.qvel.shape:
            raise ValueError(
                f"Robot qvel shape {qvel.shape} != expected {self.qvel.shape}"
            )
        if joint_torque.shape != (self.num_dof,):
            raise ValueError(
                f"Robot joint_torque shape {joint_torque.shape} != expected {(self.num_dof,)}"
            )
        self.qpos[:] = qpos
        self.qvel[:] = qvel
        self.latest_state = RobotState(
            qpos=qpos.copy(),
            qvel=qvel.copy(),
            joint_torque=joint_torque.copy(),
            tick=int(state.tick),
        )
