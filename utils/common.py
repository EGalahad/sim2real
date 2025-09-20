#!/usr/bin/env python3
"""
Common ZMQ communication configuration for motion data visualization.
Defines ports and message formats for body poses and joint states.
"""

UNITREE_LEGGED_CONST = dict(
  HIGHLEVEL = 0xEE,
  LOWLEVEL = 0xFF,
  TRIGERLEVEL = 0xF0,
  PosStopF = 2146000000.0,
  VelStopF = 16000.0,
  MODE_MACHINE = 5,
  MODE_PR = 0,
)


import zmq
import numpy as np
import struct

# ZMQ Port Configuration
PORTS = {
    "joint_names": 5550,
    "body_names": 5551,
    "torso_link_pose": 5552, # for T1
    'pelvis_pose': 5555, # for G1
    'box_pose': 5556,
    'joint_pos': 5559,
    'joint_vel': 5560,  # Reserved for future use
    "suitcase_pose": 5561,
    "plasticbox_pose": 5562,
    "stool_pose": 5563,
    "ball_pose": 5564,
    "foldchair_pose": 5565,
    "foldchair_joint_pos": 5566,
    "door_pose": 5567,
    "door_panel_pose": 5568,
    "door_joint_pos": 5569,
    "box_small_pose": 5570,
    "box_target_pose": 5571,
    "stool_low_pose": 5564,
    "foam_pose": 5565,
    "bread_box_pose": 5566,
    "stair_pose": 5572,
}

class PoseMessage:
    """Message format for body pose (position + quaternion)"""
    def __init__(self, position: np.ndarray, quaternion: np.ndarray):
        """
        Args:
            position: 3D position [x, y, z]
            quaternion: Quaternion [w, x, y, z]
        """
        self.position = np.array(position, dtype=np.float32)
        self.quaternion = np.array(quaternion, dtype=np.float32)
    
    def to_bytes(self) -> bytes:
        """Convert to binary format for ZMQ transmission"""
        # Pack as 7 float32 values: [px, py, pz, qw, qx, qy, qz]
        data = np.concatenate([self.position, self.quaternion]).astype(np.float32)
        return data.tobytes()
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'PoseMessage':
        """Create from binary data"""
        values = np.frombuffer(data, dtype=np.float32)
        if len(values) != 7:
            raise ValueError(f"Expected 7 float32 values, got {len(values)}")
        return cls(values[:3], values[3:])

class JointStateMessage:
    """Message format for joint state (positions and optionally velocities)"""
    def __init__(self, positions: np.ndarray, velocities: np.ndarray | None = None):
        """
        Args:
            positions: Joint positions array
            velocities: Joint velocities array (optional)
        """
        self.positions = np.array(positions, dtype=np.float32)
        self.velocities = np.array(velocities, dtype=np.float32) if velocities is not None else None
    
    def to_bytes(self) -> bytes:
        """Convert to binary format for ZMQ transmission"""
        # Pack header with number of positions and whether velocities are included
        header = struct.pack('II', len(self.positions), 1 if self.velocities is not None else 0)
        
        # Pack positions
        pos_data = self.positions.astype(np.float32).tobytes()
        
        # Pack velocities if available
        if self.velocities is not None:
            vel_data = self.velocities.astype(np.float32).tobytes()
            return header + pos_data + vel_data
        else:
            return header + pos_data
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'JointStateMessage':
        """Create from binary data"""
        # Unpack header
        num_positions, has_velocities = struct.unpack('II', data[:8])
        
        # Extract positions
        pos_size = num_positions * 4  # 4 bytes per float32
        positions = np.frombuffer(data[8:8+pos_size], dtype=np.float32)
        
        # Extract velocities if present
        velocities = None
        if has_velocities:
            velocities = np.frombuffer(data[8+pos_size:8+pos_size*2], dtype=np.float32)
        
        return cls(positions, velocities)

class ZMQPublisher:
    """ZMQ Publisher wrapper"""
    def __init__(self, port: int):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{port}")
        
    def publish_pose(self, position: np.ndarray, quaternion: np.ndarray):
        """Publish a pose message"""
        msg = PoseMessage(position, quaternion)
        self.socket.send(msg.to_bytes())

    def publish_joint_state(self, positions: np.ndarray, velocities: np.ndarray | None = None):
        """Publish joint state message"""
        msg = JointStateMessage(positions, velocities)
        self.socket.send(msg.to_bytes())
    
    def publish_names(self, joint_names: list[str]):
        """Publish a list of joint names"""
        # Convert list to bytes
        names_bytes = '\n'.join(joint_names).encode('utf-8')
        self.socket.send(names_bytes)
    
    def close(self):
        """Close the publisher"""
        self.socket.close()
        self.context.term()

class ZMQSubscriber:
    """ZMQ Subscriber wrapper"""
    def __init__(self, port: int, ip: str = "localhost"):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.connect(f"tcp://{ip}:{port}")
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")  # Subscribe to all messages
        self.socket.setsockopt(zmq.RCVTIMEO, 10)  # 10ms timeout
        
    def receive_pose(self) -> PoseMessage | None:
        """Receive a pose message"""
        try:
            data = self.socket.recv()
            return PoseMessage.from_bytes(data)
        except zmq.Again:
            return None  # No message available
        except Exception as e:
            print(f"Error receiving pose: {e}")
            return None
    
    def receive_joint_state(self) -> JointStateMessage | None:
        """Receive joint state message"""
        try:
            data = self.socket.recv()
            return JointStateMessage.from_bytes(data)
        except zmq.Again:
            return None  # No message available
        except Exception as e:
            print(f"Error receiving joint state: {e}")
            return None
    
    def receive_names(self) -> list[str] | None:
        """Receive a list of joint names"""
        try:
            data = self.socket.recv()
            return data.decode('utf-8').split('\n')
        except zmq.Again:
            return None  # No message available
        except Exception as e:
            print(f"Error receiving joint names: {e}")
            return None

    def close(self):
        """Close the subscriber"""
        self.socket.close()
        self.context.term()
