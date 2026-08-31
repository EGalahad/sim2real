from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from sim2real.rl_policy.observations.motion import motion_obs
from sim2real.utils.math import (
    matrix_from_quat,
    quat_conjugate,
    quat_mul,
    quat_rotate_inverse_numpy,
)
from sim2real.utils.strings import resolve_matching_names_values


class grit_reference_context(motion_obs, namespace="grit"):
    """GRIT's nine 70D reference frames at its 30 Hz model offsets."""

    def __init__(
        self,
        future_steps: Sequence[int] = (0, 2, 3, 5, 7, 8, 10, 12, 13),
        joint_names: Sequence[str] | None = None,
        root_body_name: str = "pelvis",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            future_steps=future_steps,
            joint_names=joint_names or kwargs["env"].policy_joint_names,
            body_names=[root_body_name],
            root_body_name=root_body_name,
            anchor_body_name=root_body_name,
            joint_order="given",
            body_order="given",
            **kwargs,
        )
        if len(self.joint_names) != 29 or len(self.future_steps) != 9:
            raise ValueError("GRIT expects 29 joints and nine reference frames")

    def compute(self) -> np.ndarray:
        joint_pos = np.asarray(self._select(self.ref_joint_pos_future), dtype=np.float32)
        joint_vel = np.asarray(self._select(self.ref_joint_vel_future), dtype=np.float32)
        root_quat = np.asarray(self._select(self.ref_root_quat_future_w), dtype=np.float32)
        root_lin_vel_w = np.asarray(
            self._select(self.ref_root_lin_vel_future_w), dtype=np.float32
        )
        root_ang_vel_w = np.asarray(
            self._select(self.ref_root_ang_vel_future_w), dtype=np.float32
        )
        root_lin_vel_b = quat_rotate_inverse_numpy(root_quat, root_lin_vel_w)
        root_ang_vel_b = quat_rotate_inverse_numpy(root_quat, root_ang_vel_w)
        root_rot6d = matrix_from_quat(root_quat)[..., :, :2].reshape(1, 9, 6)
        return np.concatenate(
            [joint_pos, joint_vel, root_rot6d, root_lin_vel_b, root_ang_vel_b],
            axis=-1,
        ).astype(np.float32)


class grit_proprio_history(motion_obs, namespace="grit"):
    """GRIT's term-major, oldest-to-newest ten-frame proprio history."""

    TERM_DIMS = (3, 3, 6, 29, 29, 29)
    HISTORY_LENGTH = 10

    def __init__(
        self,
        joint_names: Sequence[str] | None = None,
        root_body_name: str = "pelvis",
        **kwargs: Any,
    ) -> None:
        names = list(joint_names or kwargs["env"].policy_joint_names)
        super().__init__(
            future_steps=[0],
            joint_names=names,
            body_names=[root_body_name],
            root_body_name=root_body_name,
            anchor_body_name=root_body_name,
            joint_order="given",
            body_order="given",
            **kwargs,
        )
        if len(self.joint_names) != 29:
            raise ValueError(f"GRIT expects 29 joints, got {len(self.joint_names)}")
        self._state_joint_indices = [
            self.state_processor.joint_names.index(name) for name in self.joint_names
        ]
        joint_ids, _, default_values = resolve_matching_names_values(
            self.env.policy_config["default_joint_pos"],
            self.joint_names,
            preserve_order=True,
            strict=False,
        )
        self._default_joint_pos = np.zeros(29, dtype=np.float32)
        self._default_joint_pos[joint_ids] = np.asarray(default_values, dtype=np.float32)
        self._history = [
            np.zeros((self.HISTORY_LENGTH, dim), dtype=np.float32)
            for dim in self.TERM_DIMS
        ]

    def _frame(self, action: np.ndarray) -> tuple[np.ndarray, ...]:
        robot_quat = np.asarray(self.state_processor.root_quat_w, dtype=np.float32)
        gravity = quat_rotate_inverse_numpy(
            robot_quat.reshape(1, 4),
            np.asarray([[0.0, 0.0, -1.0]], dtype=np.float32),
        )[0]
        ref_quat = np.asarray(self.ref_root_quat_w[0], dtype=np.float32)
        relative_quat = quat_mul(
            quat_conjugate(robot_quat.reshape(1, 4)), ref_quat.reshape(1, 4)
        )
        reference_orientation_b = matrix_from_quat(relative_quat)[0, :, :2].reshape(6)
        joint_pos = np.asarray(
            self.state_processor.joint_pos[self._state_joint_indices], dtype=np.float32
        )
        joint_vel = np.asarray(
            self.state_processor.joint_vel[self._state_joint_indices], dtype=np.float32
        )
        return (
            np.asarray(self.state_processor.root_ang_vel_b, dtype=np.float32),
            gravity,
            reference_orientation_b.astype(np.float32),
            joint_pos - self._default_joint_pos,
            joint_vel,
            action,
        )

    def reset(self) -> None:
        super().reset()
        for history in self._history:
            history.fill(0.0)
        robot_quat = np.asarray(self.state_processor.root_quat_w)
        if np.linalg.norm(robot_quat) > 0.0:
            frames = self._frame(np.zeros(29, dtype=np.float32))
            for history, frame in zip(self._history, frames):
                history[:] = frame

    def update(self, data: dict[str, Any]) -> None:
        super().update(data)
        action = np.asarray(data.get("action", np.zeros(29)), dtype=np.float32).reshape(-1)
        if action.shape != (29,):
            raise ValueError(f"GRIT previous action shape {action.shape} != (29,)")
        for history, frame in zip(self._history, self._frame(action)):
            history[:-1] = history[1:]
            history[-1] = frame

    def compute(self) -> np.ndarray:
        return np.concatenate([history.reshape(-1) for history in self._history]).reshape(1, -1)
