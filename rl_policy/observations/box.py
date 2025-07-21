from .base import Observation

import numpy as np
from typing import Dict, Any
from utils.math import quat_rotate_numpy, yaw_quat, quat_rotate_inverse_numpy, wrap_to_pi, yaw_from_quat
from utils.common import PORTS
import time

class eef_target_pos_b(Observation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.state_processor.register_subscriber("box", PORTS["box"])
        self.state_processor.register_subscriber("pelvis", PORTS["pelvis"])

        box_height = 0.78
        self.contact_pos_offset = np.array([
            [0.0, -0.15, box_height],
            [0.0, 0.15, box_height],
        ])
        
        # Give time for connections to establish
        time.sleep(0.5)

    def compute(self) -> np.ndarray:
        box_pos = self.state_processor.get_mocap_data("box_pos")
        box_quat = self.state_processor.get_mocap_data("box_quat")
        if box_pos is None or box_quat is None:
            raise ValueError("Box position or quaternion data not available")

        box_pos = box_pos[None, :]
        box_quat = box_quat[None, :]
        box_pos = box_pos.repeat(2, axis=0)
        box_quat = box_quat.repeat(2, axis=0)
        
        contact_pos = box_pos + quat_rotate_numpy(box_quat, self.contact_pos_offset)
        # print(f"contact pos: {contact_pos[0]}")

        pelvis_pos = self.state_processor.get_mocap_data("pelvis_pos")
        pelvis_quat = self.state_processor.get_mocap_data("pelvis_quat")
        pelvis_pos = pelvis_pos[None, :]
        pelvis_quat = pelvis_quat[None, :]
        pelvis_pos = pelvis_pos.repeat(2, axis=0)
        pelvis_quat = pelvis_quat.repeat(2, axis=0)

        contact_pos_b = quat_rotate_inverse_numpy(yaw_quat(pelvis_quat), contact_pos - pelvis_pos)
        contact_pos_b += np.array([-0.0, 0.0, 0.0])  # Adjust for the robot's base position
        print(f"contact pos b: {contact_pos_b}")
        return contact_pos_b.reshape(-1)

class box_yaw(Observation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.state_processor.register_subscriber("box", PORTS["box"])
        self.state_processor.register_subscriber("pelvis", PORTS["pelvis"])

        # Give time for connections to establish
        time.sleep(0.5)

    def compute(self) -> np.ndarray:
        box_quat = self.state_processor.get_mocap_data("box_quat")
        if box_quat is None:
            raise ValueError("Box quaternion data not available")
        box_yaw = yaw_from_quat(box_quat[None, :]).squeeze(0)
        pelvis_quat = self.state_processor.get_mocap_data("pelvis_quat")
        pelvis_yaw = yaw_from_quat(pelvis_quat[None, :]).squeeze(0)
        box_yaw = wrap_to_pi(box_yaw + np.pi - pelvis_yaw)
        print(f"box yaw: {box_yaw}")
        return box_yaw

class box_contact(Observation):
    def __init__(self, motion_path, **kwargs):
        super().__init__(**kwargs)
        from pathlib import Path
        motion_path = Path(motion_path) / "motion.npz"
        motion = np.load(motion_path)
        self.box_contact = motion["box_contact"].astype(np.bool)
        self.motion_length = self.box_contact.shape[0]

        self.t = np.array([0])
    
    def reset(self):
        self.t[:] = 0
    
    def update(self, data: Dict[str, Any]) -> None:
        self.t += 1
        if self.t[0] == self.motion_length:
            self.t[:] = 0

    def compute(self) -> np.ndarray:
        return self.box_contact[self.t].astype(np.float32).reshape(-1)

        