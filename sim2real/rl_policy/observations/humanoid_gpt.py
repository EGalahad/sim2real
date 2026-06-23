from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np

from sim2real.rl_policy.observations.base import Observation
from sim2real.rl_policy.utils.motion import MotionData
from sim2real.utils.math import matrix_from_quat, quat_rotate_inverse_numpy


def _quat_to_yaw(q_wxyz: np.ndarray) -> np.ndarray:
    w = q_wxyz[..., 0]
    x = q_wxyz[..., 1]
    y = q_wxyz[..., 2]
    z = q_wxyz[..., 3]
    yaw = np.arctan2(2.0 * (x * y + w * z), 1.0 - 2.0 * (y * y + z * z))
    return _wrap_pi(yaw)


def _wrap_pi(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


def _batch_base_to_navi(base_to_world: np.ndarray, eps: float = 1.0e-8) -> np.ndarray:
    """Match Humanoid-GPT's yaw-aligned gravity-view frame construction."""

    x_proj = np.asarray(base_to_world[..., :3, 0], dtype=np.float32)
    z_axis = np.zeros_like(x_proj)
    z_axis[..., 2] = 1.0

    y_axis = np.cross(z_axis, x_proj)
    y_norm = np.linalg.norm(y_axis, axis=-1, keepdims=True)
    fallback_y = np.zeros_like(y_axis)
    fallback_y[..., 1] = 1.0
    y_axis = np.where(y_norm > eps, y_axis / np.clip(y_norm, eps, None), fallback_y)
    x_axis = np.cross(y_axis, z_axis)
    return np.stack((x_axis, y_axis, z_axis), axis=-1).astype(np.float32)


class humanoid_gpt_pns_obs(Observation):
    """Humanoid-GPT PNS non-privileged 136D observation.

    This mirrors GalaxyGeneralRobotics/Humanoid-GPT
    ``G1TrackInferFn.get_nn_state()`` for the released
    ``pns_wo_priv216.onnx`` policy.
    """

    OBS_DIM = 136

    def __init__(
        self,
        joint_names: Sequence[str] | None = None,
        root_body_name: str = "pelvis",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.joint_names = list(joint_names or self.env.policy_joint_names)
        self.root_body_name = str(root_body_name)
        self._cached_motion_layout: tuple[tuple[str, ...], tuple[str, ...]] | None = None
        self._obs = np.zeros((1, self.OBS_DIM), dtype=np.float32)

        self._state_joint_indices = [
            self.state_processor.joint_names.index(name) for name in self.joint_names
        ]
        self._default_joint_pos = np.asarray(
            [
                self.env.default_dof_angles[self.env.joint_names_simulation.index(name)]
                for name in self.joint_names
            ],
            dtype=np.float32,
        )

        if len(self.joint_names) != 29:
            raise ValueError(
                "Humanoid-GPT PNS expects 29 policy joints, "
                f"got {len(self.joint_names)}"
            )

    def reset(self) -> None:
        self._cached_motion_layout = None
        self._obs[:] = 0.0

    def _refresh_motion_indices(self) -> None:
        joint_names = tuple(self.state_processor.motion_joint_names)
        body_names = tuple(self.state_processor.motion_body_names)
        layout = (joint_names, body_names)
        if self._cached_motion_layout == layout:
            return

        self._motion_joint_indices = [joint_names.index(name) for name in self.joint_names]
        self._root_body_idx = body_names.index(self.root_body_name)
        self._cached_motion_layout = layout

    def _frame(self, frame: int) -> MotionData:
        return self.state_processor.motion_dataset.get_slice(
            self.state_processor.motion_ids,
            np.asarray([int(frame)], dtype=np.int64),
            np.asarray([0], dtype=np.int64),
        )

    def _current_next_frames(self, data: Dict[str, Any]) -> tuple[int, int]:
        motion_t = int(np.asarray(self.state_processor.motion_t).reshape(-1)[0])
        motion_t = min(max(motion_t, 0), int(self.state_processor.motion_length) - 1)
        if bool(data.get("paused", False)):
            return motion_t, motion_t
        return max(motion_t - 1, 0), motion_t

    def update(self, data: Dict[str, Any]) -> None:
        self._refresh_motion_indices()
        curr_frame, next_frame = self._current_next_frames(data)
        curr_motion = self._frame(curr_frame)
        next_motion = self._frame(next_frame)

        robot_root_quat = np.asarray(self.state_processor.root_quat_w, dtype=np.float32)
        robot_root_pos = np.asarray(self.state_processor.root_pos_w, dtype=np.float32)
        robot_joint_pos = np.asarray(
            self.state_processor.joint_pos[self._state_joint_indices],
            dtype=np.float32,
        )
        robot_joint_vel = np.asarray(
            self.state_processor.joint_vel[self._state_joint_indices],
            dtype=np.float32,
        )
        robot_gyro = np.asarray(self.state_processor.root_ang_vel_b, dtype=np.float32)
        robot_gravity = quat_rotate_inverse_numpy(
            robot_root_quat.reshape(1, 4),
            np.asarray([[0.0, 0.0, -1.0]], dtype=np.float32),
        )[0]

        last_action = np.asarray(
            data.get("action", np.zeros(len(self.joint_names), dtype=np.float32)),
            dtype=np.float32,
        ).reshape(-1)
        if last_action.size != len(self.joint_names):
            raise ValueError(
                "Humanoid-GPT previous action size mismatch: "
                f"expected {len(self.joint_names)}, got {last_action.size}"
            )

        curr_root_pos = np.asarray(
            curr_motion.body_pos_w[0, 0, self._root_body_idx],
            dtype=np.float32,
        )
        curr_root_quat = np.asarray(
            curr_motion.body_quat_w[0, 0, self._root_body_idx],
            dtype=np.float32,
        )
        next_root_pos = np.asarray(
            next_motion.body_pos_w[0, 0, self._root_body_idx],
            dtype=np.float32,
        )
        next_root_quat = np.asarray(
            next_motion.body_quat_w[0, 0, self._root_body_idx],
            dtype=np.float32,
        )
        next_joint_pos = np.asarray(
            next_motion.joint_pos[0, 0, self._motion_joint_indices],
            dtype=np.float32,
        )

        next_root_rot_w = matrix_from_quat(next_root_quat.reshape(1, 4))[0]
        next_gv_to_world = _batch_base_to_navi(next_root_rot_w.reshape(1, 3, 3))[0]
        next_root_rot_gv = next_gv_to_world.T @ next_root_rot_w
        next_ref_gravity = -next_root_rot_gv.T[:, 2]

        next_root_cvel_w = np.concatenate(
            [
                np.asarray(
                    next_motion.body_ang_vel_w[0, 0, self._root_body_idx],
                    dtype=np.float32,
                ),
                np.asarray(
                    next_motion.body_lin_vel_w[0, 0, self._root_body_idx],
                    dtype=np.float32,
                ),
            ],
            axis=0,
        )
        next_root_cvel_gv = np.concatenate(
            [
                next_gv_to_world.T @ next_root_cvel_w[:3],
                next_gv_to_world.T @ next_root_cvel_w[3:],
            ],
            axis=0,
        ).astype(np.float32)

        yaw_d = _wrap_pi(
            _quat_to_yaw(curr_root_quat.reshape(1, 4))[0]
            - _quat_to_yaw(robot_root_quat.reshape(1, 4))[0]
        )
        xy_d = curr_root_pos[:2] - robot_root_pos[:2]
        yaw_curr = _quat_to_yaw(robot_root_quat.reshape(1, 4))[0]
        c, s = np.cos(-yaw_curr), np.sin(-yaw_curr)
        xy_d = np.asarray(
            [c * xy_d[0] - s * xy_d[1], s * xy_d[0] + c * xy_d[1]],
            dtype=np.float32,
        )

        obs = np.concatenate(
            [
                robot_gyro,
                robot_gravity,
                robot_joint_pos - self._default_joint_pos,
                robot_joint_vel,
                last_action,
                next_joint_pos - self._default_joint_pos,
                np.asarray([next_root_pos[2]], dtype=np.float32),
                next_ref_gravity.astype(np.float32),
                next_root_cvel_gv,
                np.asarray([np.cos(yaw_d), np.sin(yaw_d)], dtype=np.float32),
                xy_d,
            ],
            axis=0,
        ).astype(np.float32)

        if obs.size != self.OBS_DIM:
            raise ValueError(f"Humanoid-GPT obs dim mismatch: {obs.size} != {self.OBS_DIM}")
        self._obs[0, :] = obs

    def compute(self) -> np.ndarray:
        return self._obs
