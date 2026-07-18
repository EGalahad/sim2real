from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np

from sim2real.rl_policy.observations.base import Observation
from sim2real.rl_policy.observations.common import _get_simulation_joint_selection
from sim2real.rl_policy.observations.motion import motion_obs
from sim2real.utils.math import (
    matrix_from_quat,
    projected_yaw_quat,
    quat_conjugate,
    quat_mul,
    quat_rotate_inverse_numpy,
)


class sonic_encoder_select(Observation, namespace="sonic"):
    def compute(self) -> np.ndarray:
        return np.zeros(1, dtype=np.float32)


class sonic_encoder_index(Observation, namespace="sonic"):
    def __init__(self, width: int = 4, mode_id: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.width = int(width)
        self.mode_id = float(mode_id)

    def compute(self) -> np.ndarray:
        out = np.zeros(self.width, dtype=np.float32)
        if self.width:
            out[0] = self.mode_id
        return out


class sonic_smpl_official_encoder_input(Observation, namespace="sonic"):
    """Full 1762D official encoder input with SMPL-mode fields populated.

    Official deployment exports one encoder input containing all enabled encoder
    observations. In SMPL mode only a subset is semantically required; the other
    encoder fields are left as zero.
    """

    ENCODER_DIM = 1762
    SMPL_MODE_ID = 2.0
    SMPL_JOINT_POS_ROOT_OFFSET = 922
    SMPL_ANCHOR_OFFSET = 1642
    WRIST_OFFSET = 1702
    WRIST_JOINT_NAMES = (
        "left_wrist_roll_joint",
        "right_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "right_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_wrist_yaw_joint",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._cached_joint_names: tuple[str, ...] | None = None
        self._wrist_indices: np.ndarray | None = None
        self._heading_offset: np.ndarray | None = None

    def reset(self) -> None:
        self._cached_joint_names = None
        self._wrist_indices = None
        self._heading_offset = None
        motion_data = getattr(self.state_processor, "motion_data", None)
        if motion_data is not None:
            ref_root_quat_w = np.asarray(motion_data.smpl_root_quat_w[0], dtype=np.float32)
            self._ensure_heading_offset(ref_root_quat_w)

    def _refresh_wrist_indices(self) -> np.ndarray:
        joint_names = tuple(self.state_processor.motion_joint_names)
        if self._cached_joint_names == joint_names and self._wrist_indices is not None:
            return self._wrist_indices
        missing = [name for name in self.WRIST_JOINT_NAMES if name not in joint_names]
        if missing:
            raise ValueError(f"SMPL motion joint_names missing wrist joints: {missing}")
        self._wrist_indices = np.asarray(
            [joint_names.index(name) for name in self.WRIST_JOINT_NAMES],
            dtype=int,
        )
        self._cached_joint_names = joint_names
        return self._wrist_indices

    def _ensure_heading_offset(self, ref_root_quat_w: np.ndarray) -> None:
        if self._heading_offset is None:
            first_ref_root_quat_w = np.asarray(
                ref_root_quat_w[0],
                dtype=np.float32,
            ).reshape(1, 4)
            robot_root_quat_w = np.asarray(
                self.state_processor.root_quat_w,
                dtype=np.float32,
            ).reshape(1, 4)
            self._heading_offset = quat_mul(
                projected_yaw_quat(robot_root_quat_w),
                quat_conjugate(projected_yaw_quat(first_ref_root_quat_w)),
            )[0].astype(np.float32, copy=False)

    def _align_root_yaw(self, ref_root_quat_w: np.ndarray) -> np.ndarray:
        self._ensure_heading_offset(ref_root_quat_w)
        assert self._heading_offset is not None
        heading_offset = np.broadcast_to(
            self._heading_offset.reshape(1, 4),
            ref_root_quat_w.shape,
        )
        return quat_mul(heading_offset, ref_root_quat_w)

    def compute(self) -> np.ndarray:
        motion_data = self.state_processor.motion_data
        out = np.zeros(self.ENCODER_DIM, dtype=np.float32)
        if motion_data is None:
            return out
        out[0] = self.SMPL_MODE_ID

        smpl_joint_pos_root = np.asarray(motion_data.smpl_joint_pos_root[0], dtype=np.float32)
        out[
            self.SMPL_JOINT_POS_ROOT_OFFSET : self.SMPL_JOINT_POS_ROOT_OFFSET + 720
        ] = smpl_joint_pos_root.reshape(-1)

        ref_root_quat_w = np.asarray(motion_data.smpl_root_quat_w[0], dtype=np.float32)
        ref_root_quat_w = self._align_root_yaw(ref_root_quat_w)
        robot_root_quat_w = np.broadcast_to(
            self.state_processor.root_quat_w.reshape(1, 4),
            ref_root_quat_w.shape,
        )
        rel_quat = quat_mul(quat_conjugate(robot_root_quat_w), ref_root_quat_w)
        anchor_ori = matrix_from_quat(rel_quat)[..., :, :2].reshape(-1)
        out[self.SMPL_ANCHOR_OFFSET : self.SMPL_ANCHOR_OFFSET + 60] = anchor_ori

        joint_pos = np.asarray(motion_data.joint_pos[0], dtype=np.float32)
        wrists = joint_pos[:, self._refresh_wrist_indices()]
        out[self.WRIST_OFFSET : self.WRIST_OFFSET + 60] = wrists.reshape(-1)
        return out


class _SonicSmplFutureObservation(Observation):
    def __init__(self, future_steps: Sequence[int], **kwargs):
        super().__init__(**kwargs)
        self.future_steps = tuple(int(step) for step in future_steps)
        if not self.future_steps:
            raise ValueError("future_steps must be non-empty")

    @property
    def num_future_frames(self) -> int:
        return len(self.future_steps)

    def _validate_frames(self, values: np.ndarray, name: str) -> np.ndarray:
        if values.shape[0] != self.num_future_frames:
            raise ValueError(
                f"{name} has {values.shape[0]} future frames, expected "
                f"{self.num_future_frames} from future_steps={self.future_steps}"
            )
        return values


class sonic_smpl_joints_multi_future_local(
    _SonicSmplFutureObservation,
    namespace="sonic",
):
    """Root-local SMPL joint positions without a full-encoder layout."""

    NUM_SMPL_JOINTS = 24

    def compute(self) -> np.ndarray:
        motion_data = self.state_processor.motion_data
        if motion_data is None:
            return np.zeros(
                self.num_future_frames * self.NUM_SMPL_JOINTS * 3,
                dtype=np.float32,
            )
        joints = np.asarray(motion_data.smpl_joint_pos_root[0], dtype=np.float32)
        self._validate_frames(joints, "smpl_joint_pos_root")
        if joints.shape[1:] != (self.NUM_SMPL_JOINTS, 3):
            raise ValueError(
                "smpl_joint_pos_root must have per-frame shape "
                f"({self.NUM_SMPL_JOINTS}, 3), got {joints.shape[1:]}"
            )
        return joints.reshape(-1)


class sonic_smpl_root_ori_b_multi_future(
    _SonicSmplFutureObservation,
    namespace="sonic",
):
    """Future SMPL root orientations relative to the robot root."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._heading_offset: np.ndarray | None = None

    def reset(self) -> None:
        self._heading_offset = None
        motion_data = getattr(self.state_processor, "motion_data", None)
        if motion_data is not None:
            ref_root_quat_w = np.asarray(
                motion_data.smpl_root_quat_w[0], dtype=np.float32
            )
            self._ensure_heading_offset(ref_root_quat_w)

    def _ensure_heading_offset(self, ref_root_quat_w: np.ndarray) -> None:
        if self._heading_offset is not None:
            return
        robot_root_quat_w = np.asarray(
            self.state_processor.root_quat_w, dtype=np.float32
        ).reshape(1, 4)
        self._heading_offset = quat_mul(
            projected_yaw_quat(robot_root_quat_w),
            quat_conjugate(projected_yaw_quat(ref_root_quat_w[:1])),
        )[0].astype(np.float32, copy=False)

    def compute(self) -> np.ndarray:
        motion_data = self.state_processor.motion_data
        if motion_data is None:
            return np.zeros(self.num_future_frames * 6, dtype=np.float32)
        ref_root_quat_w = np.asarray(
            motion_data.smpl_root_quat_w[0], dtype=np.float32
        )
        self._validate_frames(ref_root_quat_w, "smpl_root_quat_w")
        self._ensure_heading_offset(ref_root_quat_w)
        assert self._heading_offset is not None
        heading_offset = np.broadcast_to(
            self._heading_offset.reshape(1, 4), ref_root_quat_w.shape
        )
        aligned_ref_root = quat_mul(heading_offset, ref_root_quat_w)
        robot_root_quat_w = np.broadcast_to(
            np.asarray(self.state_processor.root_quat_w, dtype=np.float32).reshape(1, 4),
            aligned_ref_root.shape,
        )
        rel_quat = quat_mul(quat_conjugate(robot_root_quat_w), aligned_ref_root)
        return matrix_from_quat(rel_quat)[..., :, :2].reshape(-1).astype(np.float32)


class sonic_joint_pos_multi_future_wrist_for_smpl(
    _SonicSmplFutureObservation,
    namespace="sonic",
):
    """Future G1 wrist joint targets paired with an SMPL reference."""

    WRIST_JOINT_NAMES = sonic_smpl_official_encoder_input.WRIST_JOINT_NAMES

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._cached_joint_names: tuple[str, ...] | None = None
        self._wrist_indices: np.ndarray | None = None

    def reset(self) -> None:
        self._cached_joint_names = None
        self._wrist_indices = None

    def _refresh_wrist_indices(self) -> np.ndarray:
        joint_names = tuple(self.state_processor.motion_joint_names)
        if self._cached_joint_names == joint_names and self._wrist_indices is not None:
            return self._wrist_indices
        missing = [name for name in self.WRIST_JOINT_NAMES if name not in joint_names]
        if missing:
            raise ValueError(f"SMPL motion joint_names missing wrist joints: {missing}")
        self._wrist_indices = np.asarray(
            [joint_names.index(name) for name in self.WRIST_JOINT_NAMES], dtype=int
        )
        self._cached_joint_names = joint_names
        return self._wrist_indices

    def compute(self) -> np.ndarray:
        motion_data = self.state_processor.motion_data
        if motion_data is None:
            return np.zeros(
                self.num_future_frames * len(self.WRIST_JOINT_NAMES), dtype=np.float32
            )
        joint_pos = np.asarray(motion_data.joint_pos[0], dtype=np.float32)
        self._validate_frames(joint_pos, "joint_pos")
        return joint_pos[:, self._refresh_wrist_indices()].reshape(-1)


class _SonicMotionObservation(motion_obs):
    def __init__(
        self,
        future_steps: Sequence[int],
        joint_names: Sequence[str] | str = ".*",
        root_body_name: str = "pelvis",
        **kwargs,
    ):
        super().__init__(
            future_steps=future_steps,
            joint_names=joint_names,
            body_names=[root_body_name],
            root_body_name=root_body_name,
            anchor_body_name=root_body_name,
            joint_order="policy",
            body_order="given",
            **kwargs,
        )
        if self.future_steps.ndim != 1 or self.future_steps.size == 0:
            raise ValueError("future_steps must be a non-empty 1D sequence")

    def _playback_steps(self) -> np.ndarray:
        return self.future_steps


class sonic_command_multi_future_nonflat(_SonicMotionObservation, namespace="sonic"):
    def update(self, data: Dict[str, Any]) -> None:
        super().update(data)
        joint_pos = self._select(self.ref_joint_pos_future)
        joint_vel = self._select(self.ref_joint_vel_future)
        self.command = np.concatenate([joint_pos, joint_vel], axis=1)

    def compute(self) -> np.ndarray:
        return self.command.reshape(-1)


class sonic_motion_anchor_ori_b_mf_nonflat(_SonicMotionObservation, namespace="sonic"):
    def reset(self) -> None:
        super().reset()
        ref_root_quat_w = self.ref_root_quat_w[0]
        robot_root_quat_w = self.state_processor.root_quat_w
        self._heading_offset = quat_mul(
            projected_yaw_quat(robot_root_quat_w[None, :]),
            quat_conjugate(projected_yaw_quat(ref_root_quat_w[None, :])),
        )[0]

    def update(self, data: Dict[str, Any]) -> None:
        if not hasattr(self, "_heading_offset"):
            self.reset()
        super().update(data)
        ref_root_quat_w = self._select(self.ref_root_quat_future_w)
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


class sonic_root_ang_vel_history(_SonicHistoryObservation, namespace="sonic"):
    def __init__(self, history_steps: Sequence[int], **kwargs):
        super().__init__(history_steps=history_steps, **kwargs)
        self.history = np.zeros((self.max_lag + 1, 3), dtype=np.float32)

    def update(self, data: Dict[str, Any]) -> None:
        self._append_history(self.state_processor.root_ang_vel_b)

    def compute(self) -> np.ndarray:
        return self._history_flat()


class sonic_projected_gravity_history(_SonicHistoryObservation, namespace="sonic"):
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


class sonic_joint_pos_rel_history(_SonicHistoryObservation, namespace="sonic"):
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


class sonic_joint_vel_history(_SonicHistoryObservation, namespace="sonic"):
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


class sonic_prev_actions_history(_SonicHistoryObservation, namespace="sonic"):
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
