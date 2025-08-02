from .base import Observation

from typing import List, Tuple
import numpy as np
from utils.math import quat_rotate_inverse_numpy, yaw_quat, quat_rotate_numpy

class ref_contact_pos_b(Observation):
    def __init__(self, object_name: str, root_body_name: str, contact_target_pos_offset: List[Tuple[float, float, float]], yaw_only: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.object_name = object_name
        self.root_body_name = root_body_name
        self.yaw_only = yaw_only

        self.state_processor.register_subscriber(self.object_name)
        self.state_processor.register_subscriber(self.root_body_name)

        self.contact_target_pos_offset = np.array(contact_target_pos_offset)
        self.n_eef = len(contact_target_pos_offset)
        self.ref_contact_pos_b = np.zeros((self.n_eef, 3))

    def update(self, data):
        object_pos_w = self.state_processor.get_mocap_data(f"{self.object_name}_pos")
        object_quat_w = self.state_processor.get_mocap_data(f"{self.object_name}_quat")
        if object_pos_w is None or object_quat_w is None:
            raise ValueError(f"{self.object_name} position or quaternion data not available")

        object_pos_w = object_pos_w[None, :].repeat(self.n_eef, axis=0)
        object_quat_w = object_quat_w[None, :].repeat(self.n_eef, axis=0)
        contact_pos_w = object_pos_w + quat_rotate_numpy(object_quat_w, np.array(self.contact_target_pos_offset))

        pelvis_pos_w = self.state_processor.get_mocap_data(f"{self.root_body_name}_pos")
        pelvis_quat_w = self.state_processor.get_mocap_data(f"{self.root_body_name}_quat")
        if pelvis_pos_w is None or pelvis_quat_w is None:
            raise ValueError(f"{self.root_body_name} position or quaternion data not available")
        pelvis_pos_w = pelvis_pos_w[None, :].repeat(self.n_eef, axis=0)
        pelvis_quat_w = pelvis_quat_w[None, :].repeat(self.n_eef, axis=0)

        if self.yaw_only:
            pelvis_quat_w = yaw_quat(pelvis_quat_w)
        contact_pos_b = quat_rotate_inverse_numpy(pelvis_quat_w, contact_pos_w - pelvis_pos_w)
        self.ref_contact_pos_b[:] = contact_pos_b
    
    def compute(self) -> np.ndarray:
        return self.ref_contact_pos_b.reshape(-1)
