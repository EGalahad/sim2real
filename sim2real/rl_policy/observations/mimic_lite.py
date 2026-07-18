from __future__ import annotations

from typing import Any, Dict

import numpy as np

from sim2real.rl_policy.observations.motion import motion_body_obs, motion_joint_obs, motion_obs
from sim2real.utils.math import (
    matrix_from_quat,
    projected_yaw_quat,
    quat_conjugate,
    quat_mul,
    quat_rotate_inverse_numpy,
)


class ref_motion_phase(motion_obs, namespace=("mimic_lite", "hdmi")):
    def __init__(self, motion_duration_second: float, **kwargs):
        super().__init__(**kwargs)
        self.motion_steps = int(motion_duration_second * 50)

    def compute(self) -> np.ndarray:
        t = self.state_processor.motion_t
        ref_motion_phase = (t % self.motion_steps) / self.motion_steps
        return ref_motion_phase.reshape(-1)


class ref_joint_pos_future(motion_joint_obs, namespace=("mimic_lite", "hdmi")):
    def compute(self) -> np.ndarray:
        return self._select(self.ref_joint_pos_future).reshape(-1)


class root_pos_future_diff(motion_obs, namespace=("mimic_lite", "hdmi")):
    """
    Reference root displacements in the current motion root frame.
    """

    def __init__(
        self,
        noise_std: float = 0.0,
        exclude_current_step: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.noise_std = float(noise_std)
        keep_indices = [
            idx
            for idx, step in enumerate(self.future_steps.tolist())
            if not exclude_current_step or int(step) != 0
        ]
        if not keep_indices:
            raise ValueError("root_pos_future_diff selected no future steps")
        self.keep_indices = np.asarray(keep_indices, dtype=int)

    def compute(self) -> np.ndarray:
        root_pos_future_w = self._select(self.ref_root_pos_future_w)
        diff_w = root_pos_future_w - self.ref_root_pos_w[:, None, :]
        root_quat_w = np.broadcast_to(
            self.ref_root_quat_w[:, None, :],
            root_pos_future_w.shape[:-1] + (4,),
        )
        diff_b = quat_rotate_inverse_numpy(root_quat_w, diff_w)
        return np.take(diff_b, self.keep_indices, axis=1).reshape(-1)


class root_pos_future_z(motion_obs, namespace=("mimic_lite", "hdmi")):
    """
    Reference root z trajectory in world frame.
    """

    def __init__(self, noise_std: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.noise_std = float(noise_std)

    def compute(self) -> np.ndarray:
        return self._select(self.ref_root_pos_future_w)[..., 2].reshape(-1)


class ref_body_pos_future_local(motion_body_obs, namespace=("mimic_lite", "hdmi")):
    """
    Reference body position in motion anchor frame.
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


class ref_body_ori_future_local(motion_body_obs, namespace=("mimic_lite", "hdmi")):
    """
    Reference body orientation in motion anchor frame.
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
            ref_body_quat_future_w,
        )
        self.ref_body_ori_future_local = matrix_from_quat(ref_body_quat_future_local)

    def compute(self):
        return self.ref_body_ori_future_local[:, :, :, :2, :3].reshape(-1)


class ref_root_ori_future_b(motion_obs, namespace=("mimic_lite", "hdmi")):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.root_quat_offset = np.array([1.0, 0.0, 0.0, 0.0])

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
            ref_root_quat_future_w,
        )
        self.ref_root_ori_future_b = matrix_from_quat(ref_root_quat_future_b)

    def compute(self):
        return self.ref_root_ori_future_b[:, :, :2, :3].reshape(-1)
