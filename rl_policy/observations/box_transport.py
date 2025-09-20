from .base import Observation

from typing import List, Tuple
import numpy as np
from utils.math import quat_rotate_inverse_numpy, yaw_quat, quat_rotate_numpy, quat_mul, quat_conjugate, matrix_from_quat

class target_box_pos_b(Observation):
    def __init__(self, body_target_mocap_body_name: str = "box_target", root_body_name: str = "pelvis", **kwargs):
        super().__init__(**kwargs)
        self.mocap_body_name = body_target_mocap_body_name
        self.root_body_name = root_body_name

        self.state_processor.register_subscriber(self.mocap_body_name)
        self.state_processor.register_subscriber(self.root_body_name)

        self.target_pos_b = np.zeros(3)

    def update(self, data):
        object_pos_w = self.state_processor.get_mocap_data(f"{self.mocap_body_name}_pos")
        object_quat_w = self.state_processor.get_mocap_data(f"{self.mocap_body_name}_quat")
        if object_pos_w is None or object_quat_w is None:
            raise ValueError(f"{self.mocap_body_name} position or quaternion data not available")

        pelvis_pos_w = self.state_processor.get_mocap_data(f"{self.root_body_name}_pos")
        pelvis_quat_w = self.state_processor.get_mocap_data(f"{self.root_body_name}_quat")
        if pelvis_pos_w is None or pelvis_quat_w is None:
            raise ValueError(f"{self.root_body_name} position or quaternion data not available")

        # if self.yaw_only:
        #     pelvis_quat_w = yaw_quat(pelvis_quat_w)
        target_pos_b = quat_rotate_inverse_numpy(pelvis_quat_w, object_pos_w - pelvis_pos_w)
        self.target_pos_b[:] = target_pos_b
    
    def compute(self) -> np.ndarray:
        return self.target_pos_b

class target_box_ori_b(Observation):
    def __init__(self, body_target_mocap_body_name: str = "box_target", root_body_name: str = "pelvis", **kwargs):
        super().__init__(**kwargs)
        self.mocap_body_name = body_target_mocap_body_name
        self.root_body_name = root_body_name

        self.state_processor.register_subscriber(self.mocap_body_name)
        self.state_processor.register_subscriber(self.root_body_name)

        self.target_ori_b = np.zeros(6)

    def update(self, data):
        object_pos_w = self.state_processor.get_mocap_data(f"{self.mocap_body_name}_pos")
        object_quat_w = self.state_processor.get_mocap_data(f"{self.mocap_body_name}_quat")
        if object_pos_w is None or object_quat_w is None:
            raise ValueError(f"{self.mocap_body_name} position or quaternion data not available")

        pelvis_pos_w = self.state_processor.get_mocap_data(f"{self.root_body_name}_pos")
        pelvis_quat_w = self.state_processor.get_mocap_data(f"{self.root_body_name}_quat")
        if pelvis_pos_w is None or pelvis_quat_w is None:
            raise ValueError(f"{self.root_body_name} position or quaternion data not available")

        target_quat_b = quat_mul(quat_conjugate(pelvis_quat_w), object_quat_w)
        target_ori_b = matrix_from_quat(target_quat_b)
        self.target_ori_b[:] = target_ori_b[:2, :3].reshape(-1)
    
    def compute(self) -> np.ndarray:
        return self.target_ori_b

class target_box_xy_b(Observation):
    def __init__(self, body_target_mocap_body_name: str = "box_target", root_body_name: str = "pelvis", **kwargs):
        super().__init__(**kwargs)
        self.mocap_body_name = body_target_mocap_body_name
        self.root_body_name = root_body_name

        self.state_processor.register_subscriber(self.mocap_body_name)
        self.state_processor.register_subscriber(self.root_body_name)

        self.target_pos_b = np.zeros(2)

    def update(self, data):
        object_pos_w = self.state_processor.get_mocap_data(f"{self.mocap_body_name}_pos")
        object_quat_w = self.state_processor.get_mocap_data(f"{self.mocap_body_name}_quat")
        if object_pos_w is None or object_quat_w is None:
            raise ValueError(f"{self.mocap_body_name} position or quaternion data not available")

        pelvis_pos_w = self.state_processor.get_mocap_data(f"{self.root_body_name}_pos")
        pelvis_quat_w = self.state_processor.get_mocap_data(f"{self.root_body_name}_quat")
        if pelvis_pos_w is None or pelvis_quat_w is None:
            raise ValueError(f"{self.root_body_name} position or quaternion data not available")

        # if self.yaw_only:
        #     pelvis_quat_w = yaw_quat(pelvis_quat_w)
        target_pos_b = quat_rotate_inverse_numpy(pelvis_quat_w, object_pos_w - pelvis_pos_w)
        self.target_pos_b[:] = target_pos_b[:2]
    
    def compute(self) -> np.ndarray:
        return self.target_pos_b

class target_box_heading_b(Observation):
    def __init__(self, body_target_mocap_body_name: str = "box_target", root_body_name: str = "pelvis", **kwargs):
        super().__init__(**kwargs)
        self.mocap_body_name = body_target_mocap_body_name
        self.root_body_name = root_body_name

        self.state_processor.register_subscriber(self.mocap_body_name)
        self.state_processor.register_subscriber(self.root_body_name)

        self.target_heading_b = np.zeros(2)
        self.x_vec = np.array([1.0, 0.0, 0.0])

    def update(self, data):
        object_pos_w = self.state_processor.get_mocap_data(f"{self.mocap_body_name}_pos")
        object_quat_w = self.state_processor.get_mocap_data(f"{self.mocap_body_name}_quat")
        if object_pos_w is None or object_quat_w is None:
            raise ValueError(f"{self.mocap_body_name} position or quaternion data not available")

        pelvis_pos_w = self.state_processor.get_mocap_data(f"{self.root_body_name}_pos")
        pelvis_quat_w = self.state_processor.get_mocap_data(f"{self.root_body_name}_quat")
        if pelvis_pos_w is None or pelvis_quat_w is None:
            raise ValueError(f"{self.root_body_name} position or quaternion data not available")

        target_heading_w = quat_rotate_numpy(yaw_quat(object_quat_w), self.x_vec)
        target_heading_b = quat_rotate_inverse_numpy(yaw_quat(pelvis_quat_w), target_heading_w)
        self.target_heading_b[:] = target_heading_b[:2]
    
    def compute(self) -> np.ndarray:
        return self.target_heading_b