from .base import Observation
from .common import sort_names_by_preferred_order

from typing import Any, Dict, Optional, Sequence, Tuple, Union
import numpy as np
from sim2real.rl_policy.utils.motion import MotionData
from sim2real.utils.math import (
    matrix_from_quat,
    projected_yaw_quat,
    quat_conjugate,
    quat_mul,
    quat_rotate_inverse_numpy,
)
from sim2real.utils.strings import resolve_matching_names


class _motion_obs(Observation):
    def __init__(
        self,
        future_steps: Optional[Union[Sequence[int], int]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        selected_future_steps = future_steps
        motion_cfg = self.state_processor.motion_config
        if not motion_cfg:
            raise ValueError("policy_config.motion is required for motion observations")

        motion_future_steps = motion_cfg.get("future_steps")
        joint_names = motion_cfg.get("joint_names")
        body_names = motion_cfg.get("body_names")
        if motion_future_steps is None or joint_names is None or body_names is None:
            raise ValueError("policy_config.motion must define future_steps, joint_names, and body_names")

        self.motion_future_steps = np.asarray(motion_future_steps, dtype=int)
        if self.motion_future_steps.ndim != 1:
            raise ValueError(f"motion.future_steps must be 1D, got shape={self.motion_future_steps.shape}")
        self.available_future_steps = [int(step) for step in self.motion_future_steps.tolist()]
        if 0 not in self.available_future_steps:
            raise ValueError("motion.future_steps must include 0 to compute current observation")
        self.obs_current_step_index = int(self.available_future_steps.index(0))
        self.future_step_indices, self.future_steps = self._resolve_future_steps(selected_future_steps)
        self.selected_future_steps = self.future_steps
        self.n_future_steps = len(self.future_steps)
        self.n_selected_future_steps = self.n_future_steps
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
        self._cached_motion_layout: Optional[Tuple[Tuple[str, ...], Tuple[str, ...]]] = None

    def _resolve_future_steps(
        self,
        future_steps: Optional[Union[Sequence[int], int]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        if future_steps is None:
            requested_future_steps = self.available_future_steps
        elif isinstance(future_steps, (int, np.integer)):
            requested_future_steps = [int(future_steps)]
        else:
            requested_future_steps = [int(step) for step in future_steps]

        if not requested_future_steps:
            raise ValueError("future_steps must select at least one step")

        future_step_indices = []
        for step in requested_future_steps:
            if step not in self.available_future_steps:
                raise ValueError(
                    f"future step {step} not in motion.future_steps={self.available_future_steps}"
                )
            future_step_indices.append(self.available_future_steps.index(step))

        return (
            np.asarray(future_step_indices, dtype=int),
            np.asarray(requested_future_steps, dtype=int),
        )

    def _select(self, x: np.ndarray) -> np.ndarray:
        return np.take(x, self.future_step_indices, axis=1)
    
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

        self.ref_root_pos_w = motion_data.body_pos_w[
            :, self.obs_current_step_index, self._root_body_idx, :
        ]
        self.ref_root_quat_w = motion_data.body_quat_w[
            :, self.obs_current_step_index, self._root_body_idx, :
        ]

        self.ref_anchor_pos_w = motion_data.body_pos_w[
            :, self.obs_current_step_index, self._anchor_body_idx, :
        ]
        self.ref_anchor_quat_w = motion_data.body_quat_w[
            :, self.obs_current_step_index, self._anchor_body_idx, :
        ]


class _motion_body_obs(_motion_obs):
    def __init__(
        self,
        body_names: Optional[Union[Sequence[str], str]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if body_names is None:
            body_names = self.body_names

        body_indices, matched_body_names = resolve_matching_names(
            body_names,
            self.body_names,
        )
        if not matched_body_names:
            raise ValueError("No tracking body matched for observation.")

        self.body_indices_tracking = np.asarray(body_indices, dtype=int)
        self.selected_body_names = matched_body_names
        self.n_selected_bodies = len(self.body_indices_tracking)

    def _select(self, x: np.ndarray) -> np.ndarray:
        x = super()._select(x)
        return np.take(x, self.body_indices_tracking, axis=2)


class _motion_joint_obs(_motion_obs):
    def __init__(
        self,
        joint_names: Optional[Union[Sequence[str], str]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if joint_names is None:
            joint_names = self.joint_names

        joint_indices, matched_joint_names = resolve_matching_names(
            joint_names,
            self.joint_names,
        )
        if not matched_joint_names:
            raise ValueError("No tracking joint matched for observation.")

        self.joint_indices_tracking = np.asarray(joint_indices, dtype=int)
        self.selected_joint_names = matched_joint_names
        self.n_selected_joints = len(self.joint_indices_tracking)

    def _select(self, x: np.ndarray) -> np.ndarray:
        x = super()._select(x)
        return np.take(x, self.joint_indices_tracking, axis=2)

class ref_motion_phase(_motion_obs):
    def __init__(self, motion_duration_second: float, **kwargs):
        super().__init__(**kwargs)
        self.motion_steps = int(motion_duration_second * 50)
    
    def compute(self) -> np.ndarray:
        t = self.state_processor.motion_t
        ref_motion_phase = (t % self.motion_steps) / self.motion_steps
        return ref_motion_phase.reshape(-1)
        


class ref_joint_pos_future(_motion_joint_obs):
    def compute(self) -> np.ndarray:
        return self._select(self.ref_joint_pos_future).reshape(-1)

# class ref_joint_vel_future(_motion_obs):
#     def compute(self) -> np.ndarray:
#         return self.ref_joint_vel_future.reshape(-1)
    
class ref_body_pos_future_local(_motion_body_obs):
    """
    Reference body position in motion anchor frame
    """
    def update(self, data: Dict[str, Any]) -> None:
        super().update(data)
        ref_body_pos_future_w = self._select(self.ref_body_pos_future_w)
        ref_anchor_pos_w: np.ndarray = self.ref_anchor_pos_w[:, None, None, :].copy()
        ref_anchor_quat_w: np.ndarray = self.ref_anchor_quat_w[:, None, None, :]

        ref_anchor_pos_w[..., 2] = 0.0
        ref_anchor_quat_w = projected_yaw_quat(ref_anchor_quat_w)
        delta = ref_body_pos_future_w - ref_anchor_pos_w
        qw = ref_anchor_quat_w[..., 0:1]
        qz = ref_anchor_quat_w[..., 3:4]
        cos_yaw = qw * qw - qz * qz
        sin_yaw = 2.0 * qw * qz
        ref_body_pos_future_local = np.empty_like(delta)
        ref_body_pos_future_local[..., 0:1] = cos_yaw * delta[..., 0:1] + sin_yaw * delta[..., 1:2]
        ref_body_pos_future_local[..., 1:2] = -sin_yaw * delta[..., 0:1] + cos_yaw * delta[..., 1:2]
        ref_body_pos_future_local[..., 2:3] = delta[..., 2:3]
        self.ref_body_pos_future_local = ref_body_pos_future_local
    
    def compute(self):
        return self.ref_body_pos_future_local.reshape(-1)
    
class ref_body_ori_future_local(_motion_body_obs):
    """
    Reference body orientation in motion anchor frame
    """
    def update(self, data: Dict[str, Any]) -> None:
        super().update(data)
        ref_body_quat_future_w = self._select(self.ref_body_quat_future_w)
        ref_anchor_quat_w = self.ref_anchor_quat_w[:, None, None, :]

        ref_anchor_quat_w = projected_yaw_quat(ref_anchor_quat_w)
        ref_anchor_quat_w = np.broadcast_to(
            ref_anchor_quat_w,
            ref_body_quat_future_w.shape,
        )

        ref_body_quat_future_local = quat_mul(
            quat_conjugate(ref_anchor_quat_w),
            ref_body_quat_future_w
        )
        self.ref_body_ori_future_local = matrix_from_quat(ref_body_quat_future_local)
    
    def compute(self):
        return self.ref_body_ori_future_local[:, :, :, :2, :3].reshape(-1)

class ref_root_ori_future_b(_motion_obs):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.root_quat_offset = np.array([1.0, 0.0, 0.0, 0.0])  # identity quaternion

    def _align_root_yaw(self) -> None:
        motion_root_quat_w = self.ref_root_quat_w[0]
        robot_root_quat_w = self.state_processor.root_quat_w

        motion_root_quat_w = projected_yaw_quat(motion_root_quat_w)
        robot_root_quat_w = projected_yaw_quat(robot_root_quat_w)
        self.root_quat_offset = quat_mul(motion_root_quat_w, quat_conjugate(robot_root_quat_w))

    def reset(self):
        super().reset()
        self._align_root_yaw()

    def update(self, data: Dict[str, Any]) -> None:
        super().update(data)
        ref_root_quat_future_w = self._select(self.ref_root_quat_future_w)
        robot_root_quat_w = self.state_processor.root_quat_w
        robot_root_quat_w = quat_mul(self.root_quat_offset, robot_root_quat_w)

        robot_root_quat_w = np.broadcast_to(
            robot_root_quat_w.reshape(1, 1, 4),
            ref_root_quat_future_w.shape,
        )

        ref_root_quat_future_b = quat_mul(
            quat_conjugate(robot_root_quat_w),
            ref_root_quat_future_w
        )
        self.ref_root_ori_future_b = matrix_from_quat(ref_root_quat_future_b)
    
    def compute(self):
        return self.ref_root_ori_future_b[:, :, :2, :3].reshape(-1)
