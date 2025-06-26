import time
from typing import Any, Dict
import numpy as np
from rl_policy.observations.base import Observation
from utils.math import quat_rotate_inverse_numpy, yaw_from_quat, wrap_to_pi, yaw_quat
from utils.common import PORTS


class ref_motion_phase(Observation):
    def __init__(self, motion_duration_second: float, **kwargs):
        super().__init__(**kwargs)
        self.ref_motion_phase = np.zeros(1)
        self.start_time = time.time()
        self.motion_duration_second = motion_duration_second
    
    def reset(self):
        """Reset the motion phase to start from 0"""
        self.start_time = time.time()
        self.ref_motion_phase[:] = 0

    def compute(self) -> np.ndarray:
        t = time.time()
        self.ref_motion_phase[:] = (t - self.start_time) / self.motion_duration_second
        self.ref_motion_phase %= 1.0
        print(f"ref_motion_phase: {self.ref_motion_phase}")
        return self.ref_motion_phase


class door_pos_b(Observation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Register required subscribers
        self.state_processor.register_subscriber("Wall", PORTS["Wall"])
        self.state_processor.register_subscriber("pelvis", PORTS["pelvis"])
        
        # Give time for connections to establish
        time.sleep(0.5)

    def compute(self) -> np.ndarray:
        # Get mocap data from state processor
        door_pos_w = self.state_processor.get_mocap_data("Wall_pos")
        root_pos_w = self.state_processor.get_mocap_data("pelvis_pos")
        root_quat_w = self.state_processor.get_mocap_data("pelvis_quat")
        
        if door_pos_w is None or root_pos_w is None or root_quat_w is None:
            raise ValueError("Missing mocap data for door_pos_b computation")
            
        root_quat_yaw_w = yaw_quat(root_quat_w)
        door_pos_b = quat_rotate_inverse_numpy(
            root_quat_yaw_w[None, :], (door_pos_w - root_pos_w)[None, :]
        ).squeeze(0)
        return door_pos_b


class root_yaw(Observation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Register required subscribers
        self.state_processor.register_subscriber("Wall", PORTS["Wall"])
        self.state_processor.register_subscriber("pelvis", PORTS["pelvis"])
        
        # Give time for connections to establish
        time.sleep(0.5)

    def compute(self) -> np.ndarray:
        # Get mocap data from state processor
        door_quat_w = self.state_processor.get_mocap_data("Wall_quat")
        root_quat_w = self.state_processor.get_mocap_data("pelvis_quat")
        
        if door_quat_w is None or root_quat_w is None:
            raise ValueError("Missing mocap data for root_yaw computation")
            
        root_yaw_w = yaw_from_quat(root_quat_w[None, :]).squeeze(0)
        door_yaw_w = yaw_from_quat(door_quat_w[None, :]).squeeze(0)
        root_yaw = wrap_to_pi(root_yaw_w - door_yaw_w - np.pi)
        return np.array(root_yaw)

