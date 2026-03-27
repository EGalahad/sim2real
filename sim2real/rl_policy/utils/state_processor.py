import numpy as np
import zmq
import threading
import time


from loguru import logger
from typing import Any, Dict, Optional
from sim2real.config.robots.base import RobotCfg
from sim2real.rl_policy.utils.motion import MotionDataset, MotionData
from sim2real.rl_policy.utils.motion_buffer import RealtimeMotionBuffer
from sim2real.utils.common import ZMQSubscriber, PORTS, LowStateMessage

class StateProcessor:
    """Listens to the unitree sdk channels and converts observation into isaac compatible order.
    Assumes the message in the channel follows the joint order of robot_cfg.joint_names.
    """
    def __init__(self, robot_cfg: RobotCfg, policy_config):
        self.robot_cfg = robot_cfg
        self.mocap_ip = self.robot_cfg.mocap_ip

        self.low_state_port = self.robot_cfg.low_state_port
        state_host = self.robot_cfg.low_state_host
        state_endpoint = f"tcp://{state_host}:{self.low_state_port}"

        self.zmq_context = zmq.Context.instance()
        self.low_state_socket: zmq.Socket = self.zmq_context.socket(zmq.SUB)
        self.low_state_socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.low_state_socket.setsockopt(zmq.CONFLATE, 1)
        self.low_state_socket.setsockopt(zmq.RCVTIMEO, 10)
        self.low_state_socket.connect(state_endpoint)
        self.latest_low_state: LowStateMessage | None = None

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

        # Motion data management
        self.motion_config: Dict[str, Any] = dict(policy_config.get("motion", {}))
        self.motion_data: Optional[MotionData] = None
        self._init_motion_backend()
    
    def reset(self):
        # Reset motion playback to the first frame (standing pose)
        if self.motion_backend == "npz":
            self.motion_t[:] = 0
        self._update_motion_data()

    def update(self, data: Optional[Dict] = None):
        data = data or {}
        paused = data.get("paused", False)
        if not paused and self.motion_backend == "npz":
            self.motion_t += 1
            if self.motion_backend == "npz" and self.motion_dataset is not None and self.motion_length > 0:
                if self.motion_t[0] >= self.motion_length:
                    self.motion_t[:] = 0
                    data["paused"] = True
        self._update_motion_data()

    def _init_motion_backend(self) -> None:
        self.motion_future_steps = np.array(
            self.motion_config.get("future_steps", []),
            dtype=int,
        )
        if self.motion_future_steps.ndim != 1:
            raise ValueError(
                f"motion.future_steps must be 1D, got shape={self.motion_future_steps.shape}"
            )

        motion_backend = str(self.motion_config.get("motion_backend", "npz")).lower().strip()
        self.motion_config["motion_backend"] = self.motion_backend = motion_backend

        if motion_backend == "npz":
            motion_path = self.motion_config.get("motion_path")
            if motion_path is None:
                raise ValueError("motion_path is required for npz motion backend")
            self.motion_dataset = MotionDataset.create_from_path(
                motion_path,
                robot_cfg=self.robot_cfg,
            )
            assert self.motion_dataset.num_motions == 1, "Only one motion is supported"
            self.motion_ids = np.array([0], dtype=int)
            self.motion_t = np.array([0], dtype=int)
            self.motion_length = self.motion_dataset.num_steps

            self.motion_joint_names = list(self.motion_dataset.joint_names)
            self.motion_body_names = list(self.motion_dataset.body_names)
        elif self.motion_backend == "zmq":
            self.motion_buffer = RealtimeMotionBuffer(
                robot_cfg=self.robot_cfg,
                future_steps=self.motion_future_steps,
                motion_zmq_connect=self.motion_config.get("motion_zmq_connect", "tcp://127.0.0.1:28701"),
                motion_zmq_topic=self.motion_config.get("motion_zmq_topic", ""),
                motion_zmq_hwm=int(self.motion_config.get("motion_zmq_hwm", 1)),
                dt_s=float(self.motion_config.get("motion_dt_s", 0.02)),
                tolerance_s=float(self.motion_config.get("motion_tolerance_s", 0.04)),
            )

            self.motion_joint_names = list(self.motion_buffer.joint_names)
            self.motion_body_names = list(self.motion_buffer.body_names)
        else:
            raise ValueError(f"Unsupported motion_backend: {motion_backend}")

    def _update_motion_data(self):
        if self.motion_backend == "npz":
            self.motion_data = self.motion_dataset.get_slice(
                self.motion_ids,
                self.motion_t,
                self.motion_future_steps,
            )
        elif self.motion_backend == "zmq":
            self.motion_data = self.motion_buffer.get_obs()

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

            self.joint_pos[:] = low_state.joint_positions
            self.joint_vel[:] = low_state.joint_velocities

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
