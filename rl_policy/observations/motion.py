from .base import Observation

from typing import Any, Dict, List
import numpy as np
from utils.math import quat_rotate_inverse_numpy, yaw_quat, quat_mul, quat_conjugate, matrix_from_quat
from rl_policy.utils.motion import MotionData

from utils.math import yaw_from_quat


class _motion_obs(Observation):
    def __init__(
        self,
        motion_path: str,
        future_steps: List[int],
        joint_names: List[str],
        body_names: List[str],
        root_body_name: str = "pelvis",
        anchor_body_name: str = "torso_link",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.name = self.__class__.__name__
        self.future_steps = np.array(future_steps)
        self.n_future_steps = len(self.future_steps)
        self.n_bodies = len(body_names)

        # Register motion usage with the state processor; loading happens there
        self.state_processor.register_motion_request(
            name=self.name,
            motion_path=motion_path,
            future_steps=future_steps,
            joint_names=joint_names,
            body_names=body_names,
            root_body_name=root_body_name,
            anchor_body_name=anchor_body_name,
        )
    
    def reset(self):
        # state processor reset handles motion timing; we only refresh cache
        motion_packet = self.state_processor.get_motion_packet(self.name)
        self._assign_motion_views(motion_packet)
    
    def update(self, data: Dict[str, Any]) -> None:
        motion_packet = self.state_processor.get_motion_packet(self.name)
        self._assign_motion_views(motion_packet)

    def _assign_motion_views(self, motion_packet: Dict[str, Any]):
        motion_data: MotionData = motion_packet["data"]
        req = motion_packet["req"]
        self.ref_joint_pos_future = motion_data.joint_pos[:, :, req["joint_indices"]]
        self.ref_joint_vel_future = motion_data.joint_vel[:, :, req["joint_indices"]]
        self.ref_body_pos_future_w = motion_data.body_pos_w[:, :, req["body_indices"]]
        self.ref_body_lin_vel_future_w = motion_data.body_lin_vel_w[:, :, req["body_indices"]]
        self.ref_body_quat_future_w = motion_data.body_quat_w[:, :, req["body_indices"]]
        self.ref_body_ang_vel_future_w = motion_data.body_ang_vel_w[:, :, req["body_indices"]]
        self.ref_root_pos_future_w = motion_data.body_pos_w[:, :, [req["root_body_idx"]], :]
        self.ref_root_quat_future_w = motion_data.body_quat_w[:, :, [req["root_body_idx"]], :]
        self.ref_root_pos_w = motion_data.body_pos_w[:, [0], [req["root_body_idx"]], :]
        self.ref_root_quat_w = motion_data.body_quat_w[:, [0], [req["root_body_idx"]], :]
        self.ref_anchor_pos_w = motion_data.body_pos_w[:, [0], [req["anchor_body_idx"]], :]
        self.ref_anchor_quat_w = motion_data.body_quat_w[:, [0], [req["anchor_body_idx"]], :]

class ref_motion_phase(_motion_obs):
    def __init__(self, motion_duration_second: float, **kwargs):
        super().__init__(**kwargs)
        self.motion_steps = int(motion_duration_second * 50)
    
    def compute(self) -> np.ndarray:
        t = self.state_processor.motion_t
        ref_motion_phase = (t % self.motion_steps) / self.motion_steps
        return ref_motion_phase.reshape(-1)
        


class ref_joint_pos_future(_motion_obs):
    def compute(self) -> np.ndarray:
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
        ref_anchor_pos_w: np.ndarray = self.ref_anchor_pos_w  # [batch, 1, 1, 3]
        ref_anchor_quat_w: np.ndarray = self.ref_anchor_quat_w  # [batch, 1, 1, 4]

        # Expand dimensions to match ref_body_pos_future_w
        ref_anchor_pos_w = np.tile(ref_anchor_pos_w, (1, self.n_future_steps, self.n_bodies, 1))  # [batch, future_steps, n_bodies, 3]
        ref_anchor_quat_w = np.tile(ref_anchor_quat_w, (1, self.n_future_steps, self.n_bodies, 1))  # [batch, future_steps, n_bodies, 4]

        ref_anchor_pos_w[..., 2] = 0.0
        ref_anchor_quat_w = yaw_quat(ref_anchor_quat_w)

        ref_body_pos_future_local = quat_rotate_inverse_numpy(
            ref_anchor_quat_w, ref_body_pos_future_w - ref_anchor_pos_w
        )
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
    def update(self, data: Dict[str, Any]) -> None:
        super().update(data)
        ref_body_lin_vel_future_w = self.ref_body_lin_vel_future_w
        ref_root_quat_future_w = self.ref_root_quat_future_w

        ref_root_quat_future_w = yaw_quat(ref_root_quat_future_w)
        ref_root_quat_future_w = np.tile(ref_root_quat_future_w, (1, 1, self.n_bodies, 1))

        ref_body_lin_vel_future_local = quat_rotate_inverse_numpy(
            ref_root_quat_future_w,
            ref_body_lin_vel_future_w,
        )
        self.ref_body_lin_vel_future_local = ref_body_lin_vel_future_local
    
    def compute(self):
        return self.ref_body_lin_vel_future_local.reshape(-1)


class ref_root_ori_future_b(_motion_obs):
    def __init__(self, motion_path, future_steps, joint_names, body_names, root_body_name = "pelvis", anchor_body_name: str = "torso_link", **kwargs):
        super().__init__(
            motion_path,
            future_steps,
            joint_names,
            body_names,
            root_body_name,
            anchor_body_name=anchor_body_name,
            **kwargs,
        )
        self.root_quat_offset = np.array([1.0, 0.0, 0.0, 0.0])  # identity quaternion

    def reset(self):
        super().reset()
        motion_packet = self.state_processor.get_motion_packet(self.name)
        req = motion_packet["req"]
        motion_root_quat_w = motion_packet["data"].body_quat_w[0, 0, req["root_body_idx"], :]
        robot_root_quat_w = self.state_processor.root_quat_w

        motion_root_quat_w = yaw_quat(motion_root_quat_w)
        robot_root_quat_w = yaw_quat(robot_root_quat_w)
        self.root_quat_offset = quat_mul(motion_root_quat_w, quat_conjugate(robot_root_quat_w))

    def update(self, data: Dict[str, Any]) -> None:
        super().update(data)
        ref_root_quat_future_w = self.ref_root_quat_future_w
        robot_root_quat_w = self.state_processor.root_quat_w
        robot_root_quat_w = quat_mul(self.root_quat_offset, robot_root_quat_w)

        robot_root_quat_w = np.tile(robot_root_quat_w, (1, self.n_future_steps, 1, 1))  # [batch, future_steps, 1, 4]

        ref_root_quat_future_b = quat_mul(
            quat_conjugate(robot_root_quat_w),
            ref_root_quat_future_w
        )
        self.ref_root_ori_future_b = matrix_from_quat(ref_root_quat_future_b)
    
    def compute(self):
        return self.ref_root_ori_future_b[:, :, :, :2, :3].reshape(-1)
