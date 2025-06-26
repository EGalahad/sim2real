import numpy as np
import zmq
import threading
import time

from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowState_go
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowState_hg

from unitree_sdk2py.core.channel import ChannelSubscriber

from utils.strings import unitree_joint_names

class StateProcessor:
    """Listens to the unitree sdk channels and converts observation into isaac compatible order.
    Assumes the message in the channel follows the joint order of unitree_joint_names.
    """
    def __init__(self, robot_type, dest_joint_names, source_joint_names=unitree_joint_names):
        # Initialize channel subscriber
        if robot_type == "h1" or robot_type == "go2":
            self.robot_lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_go)
            self.robot_lowstate_subscriber.Init(self.LowStateHandler_go, 1)
        elif robot_type == "g1_29dof" or robot_type == "h1-2_27dof" or robot_type == "h1-2_21dof":
            self.robot_lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_hg)
            self.robot_lowstate_subscriber.Init(self.LowStateHandler_hg, 1)
        else:
            raise NotImplementedError(f"Robot type {robot_type} is not supported")

        # Initialize joint mapping
        self.num_dof = len(dest_joint_names)
        self.joint_indices_in_source = [source_joint_names.index(name) for name in dest_joint_names]

        self.qpos = np.zeros(3 + 4 + self.num_dof)
        self.qvel = np.zeros(3 + 3 + self.num_dof)

        # create views of qpos and qvel
        self.root_pos_w = self.qpos[0:3]
        self.root_lin_vel_w = self.qvel[0:3]

        self.root_quat_b = self.qpos[3:7]
        self.root_ang_vel_b = self.qvel[3:6]

        self.joint_pos = self.qpos[7:]
        self.joint_vel = self.qvel[6:]

        # self.tau_est = np.zeros(self.num_dof)
        # self.temp_first = np.zeros(self.num_dof)
        # self.temp_second = np.zeros(self.num_dof)
        self.robot_low_state = None

        # Initialize ZMQ context and mocap data management
        self.zmq_context = zmq.Context()
        self.mocap_subscribers = {}  # Dictionary to store ZMQ subscribers
        self.mocap_threads = {}      # Dictionary to store subscriber threads
        self.mocap_data = {}         # Dictionary to store received mocap data
        self.mocap_data_lock = threading.Lock()  # Lock for thread-safe access

    def register_subscriber(self, object_name: str, port: int):
        """Register a ZMQ subscriber for a specific object
        
        Args:
            object_name: Name of the object to subscribe to
            port: ZMQ port to connect to
        """
        if object_name in self.mocap_subscribers:
            print(f"Subscriber for {object_name} already exists")
            return

        # Create ZMQ subscriber socket
        socket = self.zmq_context.socket(zmq.SUB)
        socket.connect(f"tcp://localhost:{port}")
        socket.setsockopt(zmq.SUBSCRIBE, object_name.encode('utf-8'))
        socket.setsockopt(zmq.RCVTIMEO, 100)  # 100ms timeout
        
        self.mocap_subscribers[object_name] = socket
        print(f"Registered subscriber for {object_name} on port {port}")

        # Start subscriber thread
        thread = threading.Thread(
            target=self._mocap_subscriber_thread, 
            args=(object_name,), 
            daemon=True
        )
        thread.start()
        self.mocap_threads[object_name] = thread

    def _mocap_subscriber_thread(self, obj_name: str):
        """Thread function to continuously receive pose data for a specific object"""
        socket = self.mocap_subscribers[obj_name]
        
        while True:
            try:
                # Receive multipart message [object_name, pose_data]
                message = socket.recv_multipart(zmq.NOBLOCK)
                if len(message) == 2:
                    received_obj_name = message[0].decode('utf-8')
                    pose_bytes = message[1]
                    
                    # Convert bytes back to numpy array with explicit dtype
                    pose_data = np.frombuffer(pose_bytes, dtype=np.float64)
                    
                    if len(pose_data) == 7:  # [x, y, z, qw, qx, qy, qz]
                        position = pose_data[:3].copy()
                        quaternion = pose_data[3:].copy()  # [qw, qx, qy, qz]
                        
                        # Store in mocap_data with thread-safe access
                        with self.mocap_data_lock:
                            self.mocap_data[f"{obj_name}_pos"] = position
                            self.mocap_data[f"{obj_name}_quat"] = quaternion
                        
                    else:
                        print(f"Invalid pose data length for {obj_name}: {len(pose_data)}")
                        
            except zmq.Again:
                # No message available, continue
                time.sleep(0.001)
            except Exception as e:
                print(f"Error in {obj_name} subscriber thread: {str(e)}")
                time.sleep(0.01)

    def get_mocap_data(self, key: str):
        """Thread-safe method to get mocap data"""
        with self.mocap_data_lock:
            return self.mocap_data.get(key, None)

    def _prepare_low_state(self):
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

    def LowStateHandler_go(self, msg: LowState_go):
        self.robot_low_state = msg
    
    def LowStateHandler_hg(self, msg: LowState_hg):
        self.robot_low_state = msg
