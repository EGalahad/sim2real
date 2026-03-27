from .base import Observation
from .common import sort_names_by_preferred_order

from typing import Any, Dict, List, Sequence
import numpy as np
from sim2real.rl_policy.utils.motion import MotionData
from sim2real.utils.math import quat_rotate_inverse_numpy, yaw_quat, quat_mul, quat_conjugate, matrix_from_quat


class _motion_obs(Observation):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        motion_cfg = self.state_processor.motion_config
        if not motion_cfg:
            raise ValueError("policy_config.motion is required for motion observations")

        future_steps = motion_cfg.get("future_steps")
        joint_names = motion_cfg.get("joint_names")
        body_names = motion_cfg.get("body_names")
        if future_steps is None or joint_names is None or body_names is None:
            raise ValueError("policy_config.motion must define future_steps, joint_names, and body_names")

        self.future_steps = np.array(future_steps)
        self.n_future_steps = len(self.future_steps)
        self.joint_names = sort_names_by_preferred_order(
            joint_names,
            self.env.joint_names_simulation,
        )
        self.body_names = sort_names_by_preferred_order(
            body_names,
            self.env.body_names_simulation,
        )
        self.root_body_name = str(motion_cfg.get("root_body_name", "pelvis"))
        self.anchor_body_name = str(motion_cfg.get("anchor_body_name", "torso_link"))
        self.n_bodies = len(self.body_names)
        self._cached_motion_layout: tuple[tuple[str, ...], tuple[str, ...]] | None = None
    
    def reset(self):
        # state processor reset handles motion timing; we only refresh cache
        self._assign_motion_views()
    
    def update(self, data: Dict[str, Any]) -> None:
        self._assign_motion_views()

    def _refresh_motion_indices(self) -> None:
        joint_names = tuple(self.state_processor.motion_joint_names)
        body_names = tuple(self.state_processor.motion_body_names)
        layout = (joint_names, body_names)
        if self._cached_motion_layout == layout:
            return
        if not joint_names or not body_names:
            raise ValueError("Motion source names are not ready")

        self._joint_indices = [joint_names.index(name) for name in self.joint_names]
        self._body_indices = [body_names.index(name) for name in self.body_names]
        self._root_body_idx = body_names.index(self.root_body_name)
        self._anchor_body_idx = body_names.index(self.anchor_body_name)
        self._cached_motion_layout = layout

    def _assign_motion_views(self):
        motion_data: MotionData = self.state_processor.motion_data
        self._refresh_motion_indices()

        self.ref_joint_pos_future = motion_data.joint_pos[:, :, self._joint_indices]
        self.ref_body_pos_future_w = motion_data.body_pos_w[:, :, self._body_indices]
        self.ref_body_quat_future_w = motion_data.body_quat_w[:, :, self._body_indices]

        self.ref_root_pos_future_w = motion_data.body_pos_w[:, :, self._root_body_idx, :]
        self.ref_root_quat_future_w = motion_data.body_quat_w[:, :, self._root_body_idx, :]

        self.ref_root_pos_w = motion_data.body_pos_w[:, 0, self._root_body_idx, :]
        self.ref_root_quat_w = motion_data.body_quat_w[:, 0, self._root_body_idx, :]

        self.ref_anchor_pos_w = motion_data.body_pos_w[:, 0, self._anchor_body_idx, :]
        self.ref_anchor_quat_w = motion_data.body_quat_w[:, 0, self._anchor_body_idx, :]

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

# class ref_joint_vel_future(_motion_obs):
#     def compute(self) -> np.ndarray:
#         return self.ref_joint_vel_future.reshape(-1)
    
class ref_body_pos_future_local(_motion_obs):
    """
    Reference body position in motion root frame
    """
    def update(self, data: Dict[str, Any]) -> None:
        super().update(data)
        ref_body_pos_future_w = self.ref_body_pos_future_w
        ref_anchor_pos_w: np.ndarray = self.ref_anchor_pos_w[:, None, None, :]  # [batch, 1, 1, 3]
        ref_anchor_quat_w: np.ndarray = self.ref_anchor_quat_w[:, None, None, :]  # [batch, 1, 1, 4]

        # Expand dimensions to match ref_body_pos_future_w
        ref_anchor_pos_w = np.tile(ref_anchor_pos_w, (1, self.n_future_steps, self.n_bodies, 1))  # [batch, future_steps, n_bodies, 3]
        ref_anchor_quat_w = np.tile(ref_anchor_quat_w, (1, self.n_future_steps, self.n_bodies, 1))  # [batch, future_steps, n_bodies, 4]

        ref_anchor_pos_w[..., 2] = 0.0
        # ref_anchor_quat_w = yaw_quat(ref_anchor_quat_w)

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
        ref_anchor_quat_w = self.ref_anchor_quat_w[:, None, None, :]  # [batch, 1, 1, 4]

        ref_anchor_quat_w = np.tile(ref_anchor_quat_w, (1, self.n_future_steps, self.n_bodies, 1))
        
        # ref_anchor_quat_w = yaw_quat(ref_anchor_quat_w)

        ref_body_quat_future_local = quat_mul(
            quat_conjugate(ref_anchor_quat_w),
            ref_body_quat_future_w
        )
        self.ref_body_ori_future_local = matrix_from_quat(ref_body_quat_future_local)
    
    def compute(self):
        return self.ref_body_ori_future_local[:, :, :, :2, :3].reshape(-1)

# class ref_body_lin_vel_future_local(_motion_obs):
#     def update(self, data: Dict[str, Any]) -> None:
#         super().update(data)
#         ref_body_lin_vel_future_w = self.ref_body_lin_vel_future_w
#         ref_root_quat_future_w = self.ref_root_quat_future_w

#         ref_root_quat_future_w = yaw_quat(ref_root_quat_future_w)
#         ref_root_quat_future_w = np.tile(ref_root_quat_future_w, (1, 1, self.n_bodies, 1))

#         ref_body_lin_vel_future_local = quat_rotate_inverse_numpy(
#             ref_root_quat_future_w,
#             ref_body_lin_vel_future_w,
#         )
#         self.ref_body_lin_vel_future_local = ref_body_lin_vel_future_local
    
#     def compute(self):
#         return self.ref_body_lin_vel_future_local.reshape(-1)


class ref_root_ori_future_b(_motion_obs):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.root_quat_offset = np.array([1.0, 0.0, 0.0, 0.0])  # identity quaternion

    def reset(self):
        super().reset()
        motion_root_quat_w = self.ref_root_quat_w[0]
        robot_root_quat_w = self.state_processor.root_quat_w

        motion_root_quat_w = yaw_quat(motion_root_quat_w)
        robot_root_quat_w = yaw_quat(robot_root_quat_w)
        self.root_quat_offset = quat_mul(motion_root_quat_w, quat_conjugate(robot_root_quat_w))

    def update(self, data: Dict[str, Any]) -> None:
        super().update(data)
        ref_root_quat_future_w = self.ref_root_quat_future_w
        robot_root_quat_w = self.state_processor.root_quat_w
        robot_root_quat_w = quat_mul(self.root_quat_offset, robot_root_quat_w)

        robot_root_quat_w = np.tile(robot_root_quat_w, (1, self.n_future_steps, 1))  # [batch, future_steps, 1, 4]

        ref_root_quat_future_b = quat_mul(
            quat_conjugate(robot_root_quat_w),
            ref_root_quat_future_w
        )
        self.ref_root_ori_future_b = matrix_from_quat(ref_root_quat_future_b)
    
    def compute(self):
        return self.ref_root_ori_future_b[:, :, :2, :3].reshape(-1)
