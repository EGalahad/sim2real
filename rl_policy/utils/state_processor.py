import numpy as np
import zmq
import threading
import time


from utils.strings import unitree_joint_names
from loguru import logger
from typing import Dict, Iterable, List, Optional
from utils.common import ZMQSubscriber, PORTS, LowStateMessage
from rl_policy.utils.motion import MotionDataset, MotionData

class StateProcessor:
    """Listens to the unitree sdk channels and converts observation into isaac compatible order.
    Assumes the message in the channel follows the joint order of unitree_joint_names.
    """
    def __init__(self, robot_config, dest_joint_names):
        self.mocap_ip = robot_config.get("MOCAP_IP", "localhost")

        self.low_state_port = PORTS["low_state"]
        state_host = robot_config.get("LOW_STATE_HOST", "127.0.0.1")
        state_endpoint = f"tcp://{state_host}:{self.low_state_port}"

        self.zmq_context = zmq.Context.instance()
        self.low_state_socket: zmq.Socket = self.zmq_context.socket(zmq.SUB)
        self.low_state_socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.low_state_socket.setsockopt(zmq.CONFLATE, 1)
        self.low_state_socket.setsockopt(zmq.RCVTIMEO, 10)
        self.low_state_socket.connect(state_endpoint)
        self.latest_low_state: LowStateMessage | None = None

        # Initialize joint mapping
        self.num_dof = len(dest_joint_names)
        self.joint_indices_in_source = [unitree_joint_names.index(name) for name in dest_joint_names]
        self.joint_names = dest_joint_names

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

        # Motion data management
        self.motion_dataset: Optional[MotionDataset] = None
        self.motion_requests: Dict[str, Dict] = {}
        self.motion_cache: Dict[str, Dict] = {}
        self.motion_ids = np.array([0], dtype=int)
        self.motion_t = np.array([0], dtype=int)
        self.motion_length = 0
    
    def reset(self):
        # Reset motion playback to the first frame (standing pose)
        self.motion_t[:] = 0
        self._update_motion_cache(paused=False)

    def update(self, data: Optional[Dict] = None):
        data = data or {}
        paused = data.get("paused", False)
        if self.motion_dataset is not None and not paused:
            self.motion_t += 1
            if self.motion_t[0] >= self.motion_length:
                self.motion_t[:] = 0
                data["paused"] = True
        self._update_motion_cache(paused=paused)

    # ---- Motion utilities ----
    def register_motion_request(
        self,
        name: str,
        motion_path: str,
        future_steps: Iterable[int],
        joint_names: List[str],
        body_names: List[str],
        root_body_name: str = "pelvis",
        anchor_body_name: str = "torso_link",
    ):
        """Register a motion slice request. Loading happens once here."""
        if self.motion_dataset is None:
            self.motion_dataset = MotionDataset.create_from_path(motion_path)
            assert self.motion_dataset.num_motions == 1, "Only one motion is supported"
            self.motion_length = self.motion_dataset.num_steps

        future_steps = np.array(list(future_steps), dtype=int)
        joint_indices = [self.motion_dataset.joint_names.index(name) for name in joint_names]
        body_indices = [self.motion_dataset.body_names.index(name) for name in body_names]
        root_body_idx = self.motion_dataset.body_names.index(root_body_name)
        anchor_body_idx = self.motion_dataset.body_names.index(anchor_body_name)

        self.motion_requests[name] = dict(
            future_steps=future_steps,
            joint_indices=joint_indices,
            body_indices=body_indices,
            root_body_idx=root_body_idx,
            anchor_body_idx=anchor_body_idx,
        )
        # ensure cache initialized
        self._update_motion_cache(paused=False)

    def _update_motion_cache(self, paused: bool):
        if self.motion_dataset is None:
            return
        for name, req in self.motion_requests.items():
            motion_data: MotionData = self.motion_dataset.get_slice(
                self.motion_ids, self.motion_t, req["future_steps"]
            )
            self.motion_cache[name] = {"data": motion_data, "req": req, "paused": paused}

    def get_motion_packet(self, name: str) -> Dict:
        """Return cached motion data and request metadata."""
        return self.motion_cache[name]

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
        if hasattr(self, "low_state_socket"):
            self._receive_low_state()
            if not self.latest_low_state:
                return False

            low_state = self.latest_low_state
            self.root_quat_w[:] = low_state.quaternion
            self.root_ang_vel_b[:] = low_state.gyroscope

            source_joint_pos = low_state.joint_positions
            source_joint_vel = low_state.joint_velocities
            for dst_idx, src_idx in enumerate(self.joint_indices_in_source):
                self.joint_pos[dst_idx] = source_joint_pos[src_idx]
                self.joint_vel[dst_idx] = source_joint_vel[src_idx]

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
            self.root_quat_w[:] = state.imu.quat  # [w, x, y, z]
            self.root_ang_vel_b[:] = state.imu.omega

            # Joints
            for dst_idx, src_idx in enumerate(self.joint_indices_in_source):
                self.joint_pos[dst_idx] = state.motor.q[src_idx]
                self.joint_vel[dst_idx] = state.motor.dq[src_idx]
            return True

    def _receive_low_state(self):
        """Fetch the most recent low state message from the ZMQ socket."""
        if not hasattr(self, "low_state_socket"):
            return

        while True:
            try:
                data = self.low_state_socket.recv(flags=zmq.DONTWAIT)
            except zmq.Again:
                break
            try:
                self.latest_low_state = LowStateMessage.from_bytes(data)
            except Exception as exc:
                logger.warning(f"Failed to decode low state message: {exc}")
