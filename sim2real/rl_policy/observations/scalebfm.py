from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from sim2real.rl_policy.observations.base import Observation
from sim2real.rl_policy.utils.motion import MotionData
from sim2real.utils.math import (
    projected_yaw_quat,
    quat_conjugate,
    quat_mul,
    quat_rotate_inverse_numpy,
    quat_rotate_numpy,
)


def _normalize_quat_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.maximum(norm, 1.0e-8)


def _identity_quat() -> np.ndarray:
    return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def _safe_root_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32).reshape(4)
    if float(np.linalg.norm(q)) < 1.0e-6:
        return _identity_quat()
    return _normalize_quat_wxyz(q)


def _quat_inverse_left(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    q1 = np.asarray(q1, dtype=np.float32)
    q2 = np.asarray(q2, dtype=np.float32)
    return _normalize_quat_wxyz(quat_mul(quat_conjugate(q1), q2))


@dataclass(frozen=True)
class _ScaleBFMLayout:
    joint_names: tuple[str, ...]
    action_names: tuple[str, ...]
    body_names: tuple[str, ...]
    selected_body_names: tuple[str, ...]


class _ScaleBFMCore:
    def __init__(
        self,
        env,
        *,
        joint_names: Sequence[str],
        action_names: Sequence[str],
        body_names: Sequence[str],
        selected_body_names: Sequence[str],
        future_idx: Sequence[int],
        history_buffer_size: int = 3,
        control_mode: int = 7,
        reference_forcing: bool = True,
    ):
        self.env = env
        self.joint_names = tuple(str(name) for name in joint_names)
        self.action_names = tuple(str(name) for name in action_names)
        self.body_names = tuple(str(name) for name in body_names)
        self.selected_body_names = tuple(str(name) for name in selected_body_names)
        self.future_idx = np.asarray([int(idx) for idx in future_idx], dtype=np.int64)
        self.history_buffer_size = int(history_buffer_size)
        self.control_mode_value = int(control_mode)
        self.reference_forcing = bool(reference_forcing)
        if self.history_buffer_size <= 0:
            raise ValueError("ScaleBFM history_buffer_size must be positive")
        if self.future_idx.ndim != 1 or self.future_idx.size == 0:
            raise ValueError("ScaleBFM future_idx must be a non-empty 1D sequence")

        self.root_quat_buffer = np.zeros((1, self.history_buffer_size, 4), dtype=np.float32)
        self.base_ang_vel_buffer = np.zeros((1, self.history_buffer_size, 3), dtype=np.float32)
        self.dof_pos_buffer = np.zeros((1, self.history_buffer_size, len(self.joint_names)), dtype=np.float32)
        self.dof_vel_buffer = np.zeros((1, self.history_buffer_size, len(self.joint_names)), dtype=np.float32)
        self.actions_buffer = np.zeros((1, self.history_buffer_size, len(self.action_names)), dtype=np.float32)
        self.target_body_pos_future_to_robot_base = np.zeros(
            (1, len(self.future_idx), len(self.selected_body_names), 3),
            dtype=np.float32,
        )
        self.target_body_rot_future_to_robot_base = np.zeros(
            (1, len(self.future_idx), len(self.selected_body_names), 4),
            dtype=np.float32,
        )
        self.control_mode = np.asarray([self.control_mode_value], dtype=np.int64)
        self.future_time_offsets = self.future_idx.reshape(1, -1, 1).astype(np.int64)

        self._layout: _ScaleBFMLayout | None = None
        self._joint_indices: list[int] = []
        self._selected_body_indices: list[int] = []
        self._motion_future_step_indices: list[int] = []
        self._root_alignment_quat = _identity_quat()
        self._root_alignment_pos = np.zeros(3, dtype=np.float32)
        self._last_update_token: tuple[int, int] | None = None
        self._fill_empty_targets()

    def reset(self) -> None:
        self._refresh_layout(require_motion=False)
        root_quat, base_ang_vel, dof_pos, dof_vel = self._current_robot_state()
        self.root_quat_buffer[:] = root_quat.reshape(1, 1, 4)
        self.base_ang_vel_buffer[:] = base_ang_vel.reshape(1, 1, 3)
        self.dof_pos_buffer[:] = dof_pos.reshape(1, 1, -1)
        self.dof_vel_buffer[:] = dof_vel.reshape(1, 1, -1)
        self.actions_buffer.fill(0.0)
        if self._has_motion_data():
            self._refresh_layout(require_motion=True)
            self._reset_motion_alignment()
            self._update_motion_targets()
        else:
            self._fill_empty_targets()
        self._last_update_token = None

    def update_once(self, data: dict[str, Any]) -> None:
        token = (id(data), int(getattr(self.env, "total_inference_cnt", -1)))
        if self._last_update_token == token:
            return
        self._last_update_token = token

        self._refresh_layout(require_motion=False)
        root_quat, base_ang_vel, dof_pos, dof_vel = self._current_robot_state()
        self.root_quat_buffer = np.roll(self.root_quat_buffer, -1, axis=1)
        self.base_ang_vel_buffer = np.roll(self.base_ang_vel_buffer, -1, axis=1)
        self.dof_pos_buffer = np.roll(self.dof_pos_buffer, -1, axis=1)
        self.dof_vel_buffer = np.roll(self.dof_vel_buffer, -1, axis=1)
        self.actions_buffer = np.roll(self.actions_buffer, -1, axis=1)

        self.root_quat_buffer[:, -1] = root_quat
        self.base_ang_vel_buffer[:, -1] = base_ang_vel
        self.dof_pos_buffer[:, -1] = dof_pos
        self.dof_vel_buffer[:, -1] = dof_vel
        self.actions_buffer[:, -1] = np.asarray(
            data.get("action", np.zeros(len(self.action_names), dtype=np.float32)),
            dtype=np.float32,
        ).reshape(-1)
        if self._has_motion_data():
            self._refresh_layout(require_motion=True)
            self._update_motion_targets()
        else:
            self._fill_empty_targets()

    def output(self, name: str) -> np.ndarray:
        return getattr(self, name)

    def _current_robot_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        state = self.env.state_processor
        root_quat = _safe_root_quat(state.root_quat_w)
        base_ang_vel = np.asarray(state.root_ang_vel_b, dtype=np.float32).reshape(3)
        joint_pos = np.asarray(state.joint_pos, dtype=np.float32).reshape(-1)
        joint_vel = np.asarray(state.joint_vel, dtype=np.float32).reshape(-1)
        dof_pos = joint_pos[self._joint_indices]
        dof_vel = joint_vel[self._joint_indices]
        return root_quat, base_ang_vel, dof_pos, dof_vel

    def _refresh_layout(self, *, require_motion: bool) -> None:
        layout = _ScaleBFMLayout(
            joint_names=tuple(self.env.state_processor.joint_names),
            action_names=tuple(self.env.policy_joint_names),
            body_names=tuple(getattr(self.env, "motion_body_names", ())),
            selected_body_names=self.selected_body_names,
        )
        if self._layout == layout and (not require_motion or (self._selected_body_indices and self._motion_future_step_indices)):
            return

        missing_joints = [name for name in self.joint_names if name not in layout.joint_names]
        if missing_joints:
            raise ValueError(f"ScaleBFM robot state missing joints: {missing_joints}")
        missing_actions = [name for name in self.action_names if name not in layout.action_names]
        if missing_actions:
            raise ValueError(f"ScaleBFM policy action order missing actions: {missing_actions}")

        motion_steps = [int(step) for step in np.asarray(getattr(self.env, "motion_future_steps", []), dtype=int)]
        self._joint_indices = [layout.joint_names.index(name) for name in self.joint_names]

        if not layout.body_names or not motion_steps:
            if require_motion:
                raise ValueError("ScaleBFM observations require motion_body_names and motion_future_steps")
            self._layout = layout
            return

        missing_bodies = [name for name in self.selected_body_names if name not in layout.body_names]
        if missing_bodies:
            raise ValueError(f"ScaleBFM motion source missing selected bodies: {missing_bodies}")

        missing_steps = [int(step) for step in self.future_idx.tolist() if int(step) not in motion_steps]
        if missing_steps:
            raise ValueError(
                f"ScaleBFM motion.future_steps missing required future_idx={missing_steps}; "
                f"available={motion_steps}"
            )

        self._selected_body_indices = [layout.body_names.index(name) for name in self.selected_body_names]
        self._motion_future_step_indices = [motion_steps.index(int(step)) for step in self.future_idx.tolist()]
        self._layout = layout

    def _reset_motion_alignment(self) -> None:
        motion_data = self._motion_data()
        body_pos = np.asarray(motion_data.body_pos_w[0], dtype=np.float32)
        body_quat = np.asarray(motion_data.body_quat_w[0], dtype=np.float32)
        selected_root_pos = body_pos[self._motion_future_step_indices[0], self._selected_body_indices[0]].copy()
        selected_root_quat = _safe_root_quat(
            body_quat[self._motion_future_step_indices[0], self._selected_body_indices[0]]
        )
        selected_root_pos[2] = 0.0

        robot_heading = projected_yaw_quat(_safe_root_quat(self.env.state_processor.root_quat_w).reshape(1, 4))[0]
        source_heading = projected_yaw_quat(selected_root_quat.reshape(1, 4))[0]
        self._root_alignment_quat = quat_mul(robot_heading.reshape(1, 4), quat_conjugate(source_heading).reshape(1, 4))[0]
        self._root_alignment_pos = selected_root_pos.astype(np.float32)

    def _motion_data(self) -> MotionData:
        motion_data: MotionData | None = getattr(self.env, "motion_data", None)
        if motion_data is None:
            raise ValueError("ScaleBFM observations require motion_data")
        return motion_data

    def _has_motion_data(self) -> bool:
        return getattr(self.env, "motion_data", None) is not None

    def _fill_empty_targets(self) -> None:
        self.target_body_pos_future_to_robot_base.fill(0.0)
        self.target_body_rot_future_to_robot_base.fill(0.0)
        self.target_body_rot_future_to_robot_base[..., 0] = 1.0

    def _update_motion_targets(self) -> None:
        motion_data = self._motion_data()
        body_pos = np.asarray(motion_data.body_pos_w[0], dtype=np.float32)[
            self._motion_future_step_indices
        ][:, self._selected_body_indices]
        body_quat = np.asarray(motion_data.body_quat_w[0], dtype=np.float32)[
            self._motion_future_step_indices
        ][:, self._selected_body_indices]

        align_quat = self._root_alignment_quat.reshape(1, 1, 4)
        aligned_pos = quat_rotate_numpy(
            np.broadcast_to(align_quat, body_pos.shape[:-1] + (4,)),
            body_pos - self._root_alignment_pos.reshape(1, 1, 3),
        )
        aligned_quat = quat_mul(
            np.broadcast_to(align_quat, body_quat.shape),
            _normalize_quat_wxyz(body_quat),
        )

        root_quat = self.root_quat_buffer[:, -1]
        if self.reference_forcing:
            root_pos = aligned_pos[0, 0].reshape(1, 3)
        else:
            root_pos = np.asarray(self.env.state_processor.root_pos_w, dtype=np.float32).reshape(1, 3)

        pos_rel = quat_rotate_inverse_numpy(
            np.broadcast_to(root_quat[:, None, None, :], (1,) + aligned_pos.shape[:-1] + (4,)),
            aligned_pos.reshape(1, *aligned_pos.shape) - root_pos[:, None, None, :],
        )
        root_quat_expand = np.broadcast_to(root_quat[:, None, None, :], (1,) + aligned_quat.shape)
        rot_rel = _quat_inverse_left(root_quat_expand, aligned_quat.reshape(1, *aligned_quat.shape))
        self.target_body_pos_future_to_robot_base[:] = pos_rel
        self.target_body_rot_future_to_robot_base[:] = rot_rel


def _core_key(
    *,
    joint_names: Sequence[str],
    action_names: Sequence[str],
    body_names: Sequence[str],
    selected_body_names: Sequence[str],
    future_idx: Sequence[int],
    history_buffer_size: int,
    control_mode: int,
    reference_forcing: bool,
) -> tuple[Any, ...]:
    return (
        "scalebfm",
        tuple(joint_names),
        tuple(action_names),
        tuple(body_names),
        tuple(selected_body_names),
        tuple(int(v) for v in future_idx),
        int(history_buffer_size),
        int(control_mode),
        bool(reference_forcing),
    )


def _get_core(env, **kwargs) -> _ScaleBFMCore:
    key = _core_key(**kwargs)
    cache = getattr(env, "_scalebfm_core_cache", None)
    if cache is None:
        cache = {}
        setattr(env, "_scalebfm_core_cache", cache)
    if key not in cache:
        cache[key] = _ScaleBFMCore(env, **kwargs)
    return cache[key]


class _ScaleBFMObservation(Observation, namespace="scalebfm"):
    output_name: str

    def __init__(
        self,
        joint_names: Sequence[str],
        action_names: Sequence[str],
        body_names: Sequence[str],
        selected_body_names: Sequence[str],
        future_idx: Sequence[int],
        history_buffer_size: int = 3,
        control_mode: int = 7,
        reference_forcing: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.core = _get_core(
            self.env,
            joint_names=joint_names,
            action_names=action_names,
            body_names=body_names,
            selected_body_names=selected_body_names,
            future_idx=future_idx,
            history_buffer_size=history_buffer_size,
            control_mode=control_mode,
            reference_forcing=reference_forcing,
        )

    def reset(self) -> None:
        self.core.reset()

    def update(self, data: dict[str, Any]) -> None:
        self.core.update_once(data)

    def compute(self) -> np.ndarray:
        return self.core.output(self.output_name)


class scalebfm_root_quat_buffer(_ScaleBFMObservation, namespace="scalebfm"):
    output_name = "root_quat_buffer"


class scalebfm_base_ang_vel_buffer(_ScaleBFMObservation, namespace="scalebfm"):
    output_name = "base_ang_vel_buffer"


class scalebfm_dof_pos_buffer(_ScaleBFMObservation, namespace="scalebfm"):
    output_name = "dof_pos_buffer"


class scalebfm_dof_vel_buffer(_ScaleBFMObservation, namespace="scalebfm"):
    output_name = "dof_vel_buffer"


class scalebfm_actions_buffer(_ScaleBFMObservation, namespace="scalebfm"):
    output_name = "actions_buffer"


class scalebfm_target_body_pos_future_to_robot_base(_ScaleBFMObservation, namespace="scalebfm"):
    output_name = "target_body_pos_future_to_robot_base"


class scalebfm_target_body_rot_future_to_robot_base(_ScaleBFMObservation, namespace="scalebfm"):
    output_name = "target_body_rot_future_to_robot_base"


class scalebfm_control_mode(_ScaleBFMObservation, namespace="scalebfm"):
    output_name = "control_mode"


class scalebfm_future_time_offsets(_ScaleBFMObservation, namespace="scalebfm"):
    output_name = "future_time_offsets"
