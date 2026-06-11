from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np

from .base import Observation
from .common import sort_names_by_preferred_order
from sim2real.utils.math import quat_rotate_inverse_numpy


def _quat_to_roll_pitch(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float32).reshape(4)
    qw, qx, qy, qz = quat
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1.0:
        pitch = np.copysign(np.pi / 2.0, sinp)
    else:
        pitch = np.arcsin(sinp)

    return np.asarray([roll, pitch], dtype=np.float32)


class twist2_input(Observation):
    """TWIST2 student-future ONNX input.

    Matches upstream TWIST2 deployment/training layout:
    current `[mimic_obs(35), proprio(92)]`, then 10 frames of that same
    127-dim observation history, then one 35-dim future mimic observation.
    """

    def __init__(
        self,
        joint_names: Sequence[str],
        history_len: int = 10,
        current_step: int = 0,
        future_step: int = 0,
        ang_vel_scale: float = 0.25,
        dof_pos_scale: float = 1.0,
        dof_vel_scale: float = 0.05,
        ankle_joint_names: Sequence[str] = (
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
        ),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.joint_names = sort_names_by_preferred_order(
            joint_names,
            self.env.joint_names_simulation,
        )
        self.joint_ids = [
            self.state_processor.joint_names.index(joint_name)
            for joint_name in self.joint_names
        ]
        self.motion_joint_ids: list[int] | None = None
        self.motion_root_body_idx: int | None = None
        self.cached_motion_layout: tuple[tuple[str, ...], tuple[str, ...]] | None = None
        self.root_body_name = str(
            self.env.policy_config.get("motion", {}).get("root_body_name", "pelvis")
        )
        self.history_len = int(history_len)
        self.current_step = int(current_step)
        self.future_step = int(future_step)
        self.ang_vel_scale = float(ang_vel_scale)
        self.dof_pos_scale = float(dof_pos_scale)
        self.dof_vel_scale = float(dof_vel_scale)
        self.default_joint_pos = self.env.default_dof_angles[self.joint_ids].astype(np.float32)
        self.ankle_indices = [
            self.joint_names.index(joint_name)
            for joint_name in ankle_joint_names
            if joint_name in self.joint_names
        ]
        self.history = np.zeros((self.history_len, 127), dtype=np.float32)
        self.input = np.zeros((1, 1432), dtype=np.float32)

    def _refresh_motion_indices(self) -> None:
        joint_names = tuple(self.state_processor.motion_joint_names)
        body_names = tuple(self.state_processor.motion_body_names)
        layout = (joint_names, body_names)
        if self.cached_motion_layout == layout:
            return

        self.motion_joint_ids = [joint_names.index(name) for name in self.joint_names]
        self.motion_root_body_idx = body_names.index(self.root_body_name)
        self.cached_motion_layout = layout

    def _motion_mimic_obs(self, step_index: int) -> np.ndarray:
        self._refresh_motion_indices()
        assert self.motion_joint_ids is not None
        assert self.motion_root_body_idx is not None
        motion_data = self.state_processor.motion_data

        root_pos = motion_data.body_pos_w[0, step_index, self.motion_root_body_idx]
        root_quat = motion_data.body_quat_w[0, step_index, self.motion_root_body_idx]
        root_lin_vel_w = motion_data.body_lin_vel_w[0, step_index, self.motion_root_body_idx]
        root_ang_vel_w = motion_data.body_ang_vel_w[0, step_index, self.motion_root_body_idx]
        joint_pos = motion_data.joint_pos[0, step_index, self.motion_joint_ids]

        root_lin_vel_local = quat_rotate_inverse_numpy(
            root_quat.reshape(1, 4),
            root_lin_vel_w.reshape(1, 3),
        )[0]
        root_ang_vel_local = quat_rotate_inverse_numpy(
            root_quat.reshape(1, 4),
            root_ang_vel_w.reshape(1, 3),
        )[0]
        roll_pitch = _quat_to_roll_pitch(root_quat)

        return np.concatenate(
            [
                root_lin_vel_local[:2],
                root_pos[2:3],
                roll_pitch,
                root_ang_vel_local[2:3],
                joint_pos,
            ]
        ).astype(np.float32)

    def _robot_proprio_obs(self, data: Dict[str, Any]) -> np.ndarray:
        joint_pos = self.state_processor.joint_pos[self.joint_ids]
        joint_vel = self.state_processor.joint_vel[self.joint_ids].copy()
        if self.ankle_indices:
            joint_vel[self.ankle_indices] = 0.0
        previous_action = np.asarray(
            data.get("action", np.zeros(self.env.num_actions, dtype=np.float32)),
            dtype=np.float32,
        )
        return np.concatenate(
            [
                self.state_processor.root_ang_vel_b * self.ang_vel_scale,
                _quat_to_roll_pitch(self.state_processor.root_quat_w),
                (joint_pos - self.default_joint_pos) * self.dof_pos_scale,
                joint_vel * self.dof_vel_scale,
                previous_action,
            ]
        ).astype(np.float32)

    def reset(self) -> None:
        self._refresh_motion_indices()
        current_step_index = self._step_to_motion_index(self.current_step)
        future_step_index = self._step_to_motion_index(self.future_step)
        current = np.concatenate(
            [
                self._motion_mimic_obs(current_step_index),
                self._robot_proprio_obs({"action": np.zeros(self.env.num_actions, dtype=np.float32)}),
            ]
        ).astype(np.float32)
        self.history[:] = current
        self.input[0, :] = np.concatenate(
            [current, self.history.reshape(-1), self._motion_mimic_obs(future_step_index)]
        )

    def _step_to_motion_index(self, step: int) -> int:
        available_steps = [int(step) for step in self.state_processor.motion_future_steps.tolist()]
        if step not in available_steps:
            raise ValueError(
                f"TWIST2 step={step} not in motion.future_steps={available_steps}"
            )
        return available_steps.index(step)

    def update(self, data: Dict[str, Any]) -> None:
        current_step_index = self._step_to_motion_index(self.current_step)
        future_step_index = self._step_to_motion_index(self.future_step)

        current = np.concatenate(
            [
                self._motion_mimic_obs(current_step_index),
                self._robot_proprio_obs(data),
            ]
        ).astype(np.float32)
        future = self._motion_mimic_obs(future_step_index)

        self.input[0, :] = np.concatenate([current, self.history.reshape(-1), future])
        self.history[:-1] = self.history[1:]
        self.history[-1] = current

    def compute(self) -> np.ndarray:
        return self.input
