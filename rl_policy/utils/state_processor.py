import numpy as np
import zmq
import threading
import time


from utils.strings import unitree_joint_names
from loguru import logger
from typing import Dict
from utils.common import ZMQSubscriber, PORTS

class StateProcessor:
    """Listens to the unitree sdk channels and converts observation into isaac compatible order.
    Assumes the message in the channel follows the joint order of unitree_joint_names.
    """
    def __init__(self, robot_config, dest_joint_names):
        self.robot_type = robot_config["ROBOT_TYPE"]
        self.mocap_ip = robot_config.get("MOCAP_IP", "localhost")
        # Initialize channel subscriber
        if self.robot_type != "g1_real":
            from unitree_sdk2py.core.channel import ChannelSubscriber
        if self.robot_type == "h1" or self.robot_type == "go2":
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowState_go
            self.robot_low_state = None
            def LowStateHandler_go(msg: LowState_go):
                self.robot_low_state = msg
            self.robot_lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_go)
            self.robot_lowstate_subscriber.Init(LowStateHandler_go, 1)
        elif self.robot_type == "g1_29dof" or self.robot_type == "h1-2_27dof" or self.robot_type == "h1-2_21dof":
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowState_hg
            self.robot_low_state = None
            def LowStateHandler_hg(msg: LowState_hg):
                self.robot_low_state = msg
            self.robot_lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_hg)
            self.robot_lowstate_subscriber.Init(LowStateHandler_hg, 1)
        elif self.robot_type == "g1_real":
            self.robot = robot_config["robot"]
        else: 
            raise NotImplementedError(f"Robot type {self.robot_type} is not supported")

        # Initialize joint mapping
        self.num_dof = len(dest_joint_names)
        self.joint_indices_in_source = [unitree_joint_names.index(name) for name in dest_joint_names]
        self.joint_names = dest_joint_names

        self.qpos = np.zeros(3 + 4 + self.num_dof)
        self.qvel = np.zeros(3 + 3 + self.num_dof)

        # create views of qpos and qvel
        self.root_pos_w = self.qpos[0:3]
        self.root_lin_vel_w = self.qvel[0:3]

        self.root_quat_b = self.qpos[3:7]
        self.root_ang_vel_b = self.qvel[3:6]

        self.joint_pos = self.qpos[7:]
        self.joint_vel = self.qvel[6:]

        # Initialize ZMQ context and mocap data management
        self.zmq_context = zmq.Context()
        self.mocap_subscribers: Dict[str, ZMQSubscriber] = {}  # Dictionary to store ZMQ subscribers
        self.mocap_threads = {}      # Dictionary to store subscriber threads
        self.mocap_data = {}         # Dictionary to store received mocap data
        self.mocap_data_lock = threading.Lock()  # Lock for thread-safe access

    def register_subscriber(self, object_name: str, port: int | None = None):
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

    def _prepare_low_state(self):
        if hasattr(self, "robot_low_state"):
            if not self.robot_low_state:
                return False

            # imu sensor
            imu_state = self.robot_low_state.imu_state
            self.root_quat_b[:] = imu_state.quaternion # w, x, y, z
            self.root_ang_vel_b[:] = imu_state.gyroscope

            # joint encoder
            source_joint_state = self.robot_low_state.motor_state
            for dst_idx, src_idx in enumerate(self.joint_indices_in_source):
                self.joint_pos[dst_idx] = source_joint_state[src_idx].q
                self.joint_vel[dst_idx] = source_joint_state[src_idx].dq
            
            return True
        elif hasattr(self, "robot"):
            try:
                state = self.robot.read_low_state()
            except Exception as e:
                logger.warning(f"Failed to read G1 low state: {e}")
                return False

            if state is None:
                return False

            # IMU
            self.root_quat_b[:] = state.imu.quat  # [w, x, y, z]
            self.root_ang_vel_b[:] = state.imu.omega

            # Joints
            for dst_idx, src_idx in enumerate(self.joint_indices_in_source):
                self.joint_pos[dst_idx] = state.motor.q[src_idx]
                self.joint_vel[dst_idx] = state.motor.dq[src_idx]
            return True