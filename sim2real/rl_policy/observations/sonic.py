from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np

from sim2real.rl_policy.observations.base import Observation
from sim2real.rl_policy.observations.common import _get_simulation_joint_selection
from sim2real.utils.math import (
    matrix_from_quat,
    projected_yaw_quat,
    quat_conjugate,
    quat_mul,
    quat_rotate_inverse_numpy,
)
from sim2real.utils.strings import resolve_matching_names


class sonic_encoder_select(Observation):
    def compute(self) -> np.ndarray:
        return np.zeros(1, dtype=np.float32)


class sonic_encoder_index(Observation):
    def __init__(self, width: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.width = int(width)

    def compute(self) -> np.ndarray:
        return np.zeros(self.width, dtype=np.float32)


class _SonicMotionObservation(Observation):
    def __init__(
        self,
        future_steps: Sequence[int],
        joint_names: Sequence[str] | str = ".*",
        root_body_name: str = "pelvis",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.future_steps = np.asarray([int(step) for step in future_steps], dtype=int)
        if self.future_steps.ndim != 1 or self.future_steps.size == 0:
            raise ValueError("future_steps must be a non-empty 1D sequence")
        self.root_body_name = root_body_name
        self.requested_joint_names = joint_names
        self._cached_motion_layout: tuple[tuple[str, ...], tuple[str, ...]] | None = None

    def _refresh_motion_indices(self) -> None:
        joint_names = tuple(self.state_processor.motion_joint_names)
        body_names = tuple(self.state_processor.motion_body_names)
        layout = (joint_names, body_names)
        if self._cached_motion_layout == layout:
            return

        _, matched_joint_names = resolve_matching_names(
            self.requested_joint_names,
            joint_names,
            preserve_order=True,
        )
        preferred_joint_names = list(self.env.policy_joint_names)
        ordered_joint_names = [
            name for name in preferred_joint_names if name in matched_joint_names
        ]
        if len(ordered_joint_names) != len(matched_joint_names):
            missing = [name for name in matched_joint_names if name not in ordered_joint_names]
            raise ValueError(f"Failed to order SONIC motion joints: {missing}")

        self._joint_indices = [joint_names.index(name) for name in ordered_joint_names]
        self._root_body_idx = body_names.index(self.root_body_name)
        self._cached_motion_layout = layout

    def _motion_slice(self, steps: np.ndarray):
        if self.state_processor.motion_backend != "npz":
            raise ValueError("SONIC motion observations currently require motion_backend=npz")
        self._refresh_motion_indices()
        return self.state_processor.motion_dataset.get_slice(
            self.state_processor.motion_ids,
            self.state_processor.motion_t,
            steps,
        )

    def _playback_steps(self) -> np.ndarray:
        if bool(self.env.state_dict.get("paused", False)):
            return np.zeros_like(self.future_steps)
        return self.future_steps


class sonic_command_multi_future_nonflat(_SonicMotionObservation):
    def update(self, data: Dict[str, Any]) -> None:
        steps = self._playback_steps()
        motion_data = self._motion_slice(steps)
        joint_pos = motion_data.joint_pos[:, :, self._joint_indices]
        if bool(data.get("paused", False)):
            joint_vel = np.zeros_like(joint_pos)
        else:
            joint_vel = motion_data.joint_vel[:, :, self._joint_indices]
        self.command = np.concatenate([joint_pos, joint_vel], axis=1)

    def compute(self) -> np.ndarray:
        return self.command.reshape(-1)


class sonic_motion_anchor_ori_b_mf_nonflat(_SonicMotionObservation):
    def reset(self) -> None:
        steps = np.zeros(1, dtype=int)
        motion_data = self._motion_slice(steps)
        ref_root_quat_w = motion_data.body_quat_w[0, 0, self._root_body_idx]
        robot_root_quat_w = self.state_processor.root_quat_w
        self._heading_offset = quat_mul(
            projected_yaw_quat(robot_root_quat_w[None, :]),
            quat_conjugate(projected_yaw_quat(ref_root_quat_w[None, :])),
        )[0]

    def update(self, data: Dict[str, Any]) -> None:
        if not hasattr(self, "_heading_offset"):
            self.reset()
        steps = self._playback_steps()
        motion_data = self._motion_slice(steps)
        ref_root_quat_w = motion_data.body_quat_w[:, :, self._root_body_idx, :]
        heading_offset = np.broadcast_to(
            self._heading_offset.reshape(1, 1, 4),
            ref_root_quat_w.shape,
        )
        ref_root_quat_w = quat_mul(heading_offset, ref_root_quat_w)
        robot_root_quat_w = np.broadcast_to(
            self.state_processor.root_quat_w.reshape(1, 1, 4),
            ref_root_quat_w.shape,
        )
        rel_quat = quat_mul(quat_conjugate(robot_root_quat_w), ref_root_quat_w)
        self.anchor_ori = matrix_from_quat(rel_quat)[..., :, :2]

    def compute(self) -> np.ndarray:
        return self.anchor_ori.reshape(-1)


class _SonicHistoryObservation(Observation):
    def __init__(self, history_steps: Sequence[int], **kwargs):
        super().__init__(**kwargs)
        self.history_steps = [int(step) for step in history_steps]
        self.max_lag = max(self.history_steps)
        self._history_indices = [
            self.max_lag - lag for lag in sorted(self.history_steps, reverse=True)
        ]
        self._history_initialized = False

    def reset(self) -> None:
        self.history[:] = 0.0
        self._history_initialized = False

    def _append_history(self, value: np.ndarray) -> None:
        if not self._history_initialized:
            self.history[:] = value
            self._history_initialized = True
            return
        self.history = np.roll(self.history, -1, axis=0)
        self.history[-1, :] = value

    def _history_flat(self) -> np.ndarray:
        return self.history[self._history_indices].reshape(-1)


class sonic_root_ang_vel_history(_SonicHistoryObservation):
    def __init__(self, history_steps: Sequence[int], **kwargs):
        super().__init__(history_steps=history_steps, **kwargs)
        self.history = np.zeros((self.max_lag + 1, 3), dtype=np.float32)

    def update(self, data: Dict[str, Any]) -> None:
        self._append_history(self.state_processor.root_ang_vel_b)

    def compute(self) -> np.ndarray:
        return self._history_flat()


class sonic_projected_gravity_history(_SonicHistoryObservation):
    def __init__(self, history_steps: Sequence[int], **kwargs):
        super().__init__(history_steps=history_steps, **kwargs)
        self.history = np.zeros((self.max_lag + 1, 3), dtype=np.float32)
        self.down = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    def update(self, data: Dict[str, Any]) -> None:
        gravity = quat_rotate_inverse_numpy(
            self.state_processor.root_quat_w[None, :], self.down[None, :]
        )[0]
        self._append_history(gravity)

    def compute(self) -> np.ndarray:
        return self._history_flat()


class sonic_joint_pos_rel_history(_SonicHistoryObservation):
    def __init__(
        self,
        history_steps: Sequence[int],
        joint_names: Sequence[str] | str = ".*",
        **kwargs,
    ):
        super().__init__(history_steps=history_steps, **kwargs)
        self.joint_ids, _ = _get_simulation_joint_selection(self.env, joint_names)
        self.default = self.env.default_dof_angles[self.joint_ids]
        self.history = np.zeros(
            (self.max_lag + 1, len(self.joint_ids)),
            dtype=np.float32,
        )

    def update(self, data: Dict[str, Any]) -> None:
        self._append_history(self.state_processor.joint_pos[self.joint_ids] - self.default)

    def compute(self) -> np.ndarray:
        return self._history_flat()


class sonic_joint_vel_history(_SonicHistoryObservation):
    def __init__(
        self,
        history_steps: Sequence[int],
        joint_names: Sequence[str] | str = ".*",
        **kwargs,
    ):
        super().__init__(history_steps=history_steps, **kwargs)
        self.joint_ids, _ = _get_simulation_joint_selection(self.env, joint_names)
        self.history = np.zeros(
            (self.max_lag + 1, len(self.joint_ids)),
            dtype=np.float32,
        )

    def update(self, data: Dict[str, Any]) -> None:
        self._append_history(self.state_processor.joint_vel[self.joint_ids])

    def compute(self) -> np.ndarray:
        return self._history_flat()


class sonic_prev_actions_history(_SonicHistoryObservation):
    def __init__(self, history_steps: Sequence[int], **kwargs):
        super().__init__(history_steps=history_steps, **kwargs)
        self.history = np.zeros(
            (self.max_lag + 1, self.env.num_actions),
            dtype=np.float32,
        )

    def update(self, data: Dict[str, Any]) -> None:
        self._append_history(data["action"])

    def compute(self) -> np.ndarray:
        return self._history_flat()
