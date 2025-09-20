from .base import Observation

import time
from typing import Any, Dict, List
import numpy as np
from utils.math import quat_rotate_inverse_numpy, yaw_quat, quat_mul, quat_conjugate, matrix_from_quat
from rl_policy.utils.motion import MotionDataset, MotionData



class _motion_obs(Observation):
    def __init__(self, motion_path: str, future_steps: List[int], joint_names: List[str], body_names: List[str], root_body_name: str = "pelvis", **kwargs):
        super().__init__(**kwargs)
        self.motion_dataset = MotionDataset.create_from_path(motion_path)
        assert self.motion_dataset.num_motions == 1, "Only one motion is supported"
        self.motion_ids = np.array([0])
        self.motion_length = self.motion_dataset.num_steps

        self.t = np.array([0])
        self.future_steps = np.array(future_steps)

        self.joint_indices = [self.motion_dataset.joint_names.index(name) for name in joint_names]
        self.body_indices = [self.motion_dataset.body_names.index(name) for name in body_names]
        self.root_body_idx = self.motion_dataset.body_names.index(root_body_name)

        self.n_future_steps = len(self.future_steps)
        self.n_bodies = len(self.body_indices)
    
    def reset(self):
        self.t[:] = 0
    
    def update(self, data: Dict[str, Any]) -> None:
        if data.get("paused", False):
            return
        
        self.t += 1
        if self.t[0] == self.motion_length:
            self.t[:] = 0
        motion_data: MotionData = self.motion_dataset.get_slice(self.motion_ids, self.t, self.future_steps)
        self.ref_joint_pos_future = motion_data.joint_pos[:, :, self.joint_indices]
        self.ref_joint_vel_future = motion_data.joint_vel[:, :, self.joint_indices]
        self.ref_body_pos_future_w = motion_data.body_pos_w[:, :, self.body_indices]
        self.ref_body_lin_vel_future_w = motion_data.body_lin_vel_w[:, :, self.body_indices]
        self.ref_body_quat_future_w = motion_data.body_quat_w[:, :, self.body_indices]
        self.ref_body_ang_vel_future_w = motion_data.body_ang_vel_w[:, :, self.body_indices]
        self.ref_root_pos_w = motion_data.body_pos_w[:, [0], [self.root_body_idx], :]
        self.ref_root_quat_w = motion_data.body_quat_w[:, [0], [self.root_body_idx], :]

class ref_motion_phase(_motion_obs):
    def __init__(self, motion_duration_second: float, **kwargs):
        super().__init__(**kwargs)
        self.motion_steps = int(motion_duration_second * 50)
    
    def compute(self) -> np.ndarray:
        ref_motion_phase = (self.t % self.motion_steps) / self.motion_steps
        return ref_motion_phase.reshape(-1)
        


class ref_joint_pos_future(_motion_obs):
    def compute(self) -> np.ndarray:
        print(f"t: {self.t.item()}")
        return self.ref_joint_pos_future.reshape(-1)
    
class ref_joint_vel_future(_motion_obs):
    def compute(self) -> np.ndarray:
        return self.ref_joint_vel_future.reshape(-1)
    
class ref_body_pos_future_local(_motion_obs):
    """
    Reference body position in motion root frame
    """
    def update(self, data: Dict[str, Any]) -> None:
        super().update(data)
        ref_body_pos_future_w = self.ref_body_pos_future_w
        ref_root_pos_w: np.ndarray = self.ref_root_pos_w # [batch, 1, 1, 3]
        ref_root_quat_w: np.ndarray = self.ref_root_quat_w  # [batch, 1, 1, 4]

        # Expand dimensions to match ref_body_pos_future_w
        ref_root_pos_w = np.tile(ref_root_pos_w, (1, self.n_future_steps, self.n_bodies, 1))  # [batch, future_steps, n_bodies, 3]
        ref_root_quat_w = np.tile(ref_root_quat_w, (1, self.n_future_steps, self.n_bodies, 1))  # [batch, future_steps, n_bodies, 4]

        ref_root_pos_w[..., 2] = 0.0
        ref_root_quat_w = yaw_quat(ref_root_quat_w)

        ref_body_pos_future_local = quat_rotate_inverse_numpy(ref_root_quat_w, ref_body_pos_future_w - ref_root_pos_w)
        self.ref_body_pos_future_local = ref_body_pos_future_local
    
    def compute(self):
        return self.ref_body_pos_future_local.reshape(-1)
    
class ref_body_ori_future_local(_motion_obs):
    def update(self, data: Dict[str, Any]) -> None:
        super().update(data)
        ref_body_quat_future_w = self.ref_body_quat_future_w
        ref_root_quat_w = self.ref_root_quat_w

        ref_root_quat_w = np.tile(ref_root_quat_w, (1, self.n_future_steps, self.n_bodies, 1))
        
        ref_root_quat_w = yaw_quat(ref_root_quat_w)

        ref_body_quat_future_local = quat_mul(
            quat_conjugate(ref_root_quat_w),
            ref_body_quat_future_w
        )
        self.ref_body_ori_future_local = matrix_from_quat(ref_body_quat_future_local)
    
    def compute(self):
        return self.ref_body_ori_future_local[:, :, :, :2, :3].reshape(-1)

class ref_body_lin_vel_future_local(_motion_obs):
    def __init__(self, root_body_name: str = "pelvis", **kwargs):
        super().__init__(**kwargs)
        self.root_body_idx = self.motion_dataset.body_names.index(root_body_name)
    
    def update(self):
        super().update()
        ref_body_lin_vel_future_w = self.motion_data.body_lin_vel_w[:, :, :, :]
        ref_root_quat_w = self.motion_data.body_quat_w[:, :, self.root_body_idx, :]

        ref_root_quat_w = yaw_quat(ref_root_quat_w)

        ref_body_lin_vel_future_local = quat_rotate_inverse_numpy(ref_root_quat_w, ref_body_lin_vel_future_w)
        self.ref_body_lin_vel_future_local = ref_body_lin_vel_future_local
    
    def compute(self):
        return self.ref_body_lin_vel_future_local.reshape(-1)
