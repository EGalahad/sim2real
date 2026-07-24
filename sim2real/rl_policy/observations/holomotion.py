from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from sim2real.rl_policy.observations.base import Observation
from sim2real.rl_policy.observations.motion import TrackingObservation
from sim2real.utils.math import matrix_from_quat, quat_conjugate, quat_mul, quat_rotate_inverse_numpy
from sim2real.utils.strings import resolve_matching_names_values


def _normalize_quat_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    q = q / np.maximum(norm, 1.0e-9)
    return np.where(q[..., 0:1] < 0.0, -q, q)


def _yaw_from_quat_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    qw = q[..., 0]
    qx = q[..., 1]
    qy = q[..., 2]
    qz = q[..., 3]
    return np.arctan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    ).astype(np.float32)


def _yaw_quat_wxyz(yaw: float) -> np.ndarray:
    half = np.float32(0.5 * yaw)
    return np.asarray([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float32)


def _projected_gravity_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    qw = q[..., 0]
    qx = q[..., 1]
    qy = q[..., 2]
    qz = q[..., 3]
    out = np.empty(q.shape[:-1] + (3,), dtype=np.float32)
    out[..., 0] = 2.0 * (-qz * qx + qw * qy)
    out[..., 1] = -2.0 * (qz * qy + qw * qx)
    out[..., 2] = 1.0 - 2.0 * (qw * qw + qz * qz)
    return out


def _rot6d_from_quat_wxyz(q: np.ndarray) -> np.ndarray:
    return matrix_from_quat(_normalize_quat_wxyz(q))[..., :, :2].reshape(q.shape[:-1] + (6,))


def _runtime_key(namespace: str, name: str) -> str:
    return f"_holomotion_{namespace}_{name}"


def _reset_at(rope_max_seq_len: int, rope_reset_margin: int) -> int:
    rope_max_seq_len = int(rope_max_seq_len)
    if rope_max_seq_len <= 0:
        return 0
    reset_at = rope_max_seq_len - int(rope_reset_margin)
    if reset_at <= 0:
        reset_at = rope_max_seq_len
    return reset_at


def _maybe_request_rope_reset(
    state_dict: dict[str, Any] | None,
    *,
    namespace: str,
    rope_max_seq_len: int,
    rope_reset_margin: int,
) -> bool:
    reset_at = _reset_at(rope_max_seq_len, rope_reset_margin)
    if state_dict is None or reset_at <= 0:
        return False

    step_key = _runtime_key(namespace, "step_idx")
    reset_key = _runtime_key(namespace, "rope_reset_requested")
    step_idx = int(state_dict.get(step_key, 0))
    if step_idx < reset_at:
        return bool(state_dict.get(reset_key, False))

    state_dict[step_key] = 0
    state_dict[reset_key] = True
    return True


class holomotion_zero_state(Observation, namespace="holomotion"):
    def __init__(
        self,
        shape: Sequence[int],
        dtype: str = "float32",
        state_key: str = "past_key_values",
        namespace: str = "motion",
        rope_max_seq_len: int = 0,
        rope_reset_margin: int = 64,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.shape = tuple(int(dim) for dim in shape)
        self.dtype = np.dtype(dtype)
        self.state_key = str(state_key)
        self.namespace = str(namespace)
        self.rope_max_seq_len = int(rope_max_seq_len)
        self.rope_reset_margin = int(rope_reset_margin)
        self._value = np.zeros(self.shape, dtype=self.dtype)

    def reset(self) -> None:
        self._value.fill(0)
        state_dict = getattr(self.env, "state_dict", None)
        if isinstance(state_dict, dict):
            state_dict[self.state_key] = self._value.copy()
            state_dict[_runtime_key(self.namespace, "rope_reset_requested")] = False

    def compute(self) -> np.ndarray:
        state_dict = getattr(self.env, "state_dict", None)
        if isinstance(state_dict, dict):
            _maybe_request_rope_reset(
                state_dict,
                namespace=self.namespace,
                rope_max_seq_len=self.rope_max_seq_len,
                rope_reset_margin=self.rope_reset_margin,
            )
            reset_key = _runtime_key(self.namespace, "rope_reset_requested")
            if bool(state_dict.get(reset_key, False)):
                self._value.fill(0)
                state_dict[self.state_key] = self._value.copy()
                state_dict[reset_key] = False
                return self._value

        if isinstance(state_dict, dict) and self.state_key in state_dict:
            value = np.asarray(state_dict[self.state_key], dtype=self.dtype)
            if value.shape == self.shape:
                return value
            if value.shape == self.shape[1:]:
                return value.reshape(self.shape)
            raise ValueError(
                f"HoloMotion state {self.state_key!r} has shape {value.shape}, "
                f"expected {self.shape} or {self.shape[1:]}"
            )
        return self._value


class holomotion_step_idx(Observation, namespace="holomotion"):
    def __init__(
        self,
        dtype: str = "int64",
        namespace: str = "motion",
        rope_max_seq_len: int = 0,
        rope_reset_margin: int = 64,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dtype = np.dtype(dtype)
        self.namespace = str(namespace)
        self.rope_max_seq_len = int(rope_max_seq_len)
        self.rope_reset_margin = int(rope_reset_margin)
        self._step = 0
        self._value = np.zeros((1,), dtype=self.dtype)

    def reset(self) -> None:
        self._step = 0
        state_dict = getattr(self.env, "state_dict", None)
        if isinstance(state_dict, dict):
            state_dict[_runtime_key(self.namespace, "step_idx")] = 0
            state_dict[_runtime_key(self.namespace, "rope_reset_requested")] = False

    def compute(self) -> np.ndarray:
        state_dict = getattr(self.env, "state_dict", None)
        if isinstance(state_dict, dict):
            _maybe_request_rope_reset(
                state_dict,
                namespace=self.namespace,
                rope_max_seq_len=self.rope_max_seq_len,
                rope_reset_margin=self.rope_reset_margin,
            )
            step_key = _runtime_key(self.namespace, "step_idx")
            step_idx = int(state_dict.get(step_key, self._step))
            self._value[0] = step_idx
            state_dict[step_key] = step_idx + 1
            self._step = step_idx + 1
            return self._value

        reset_at = _reset_at(self.rope_max_seq_len, self.rope_reset_margin)
        if reset_at > 0 and self._step >= reset_at:
            self._step = 0
        self._value[0] = self._step
        self._step += 1
        return self._value


class holomotion_motion_actor_obs(TrackingObservation, namespace="holomotion"):
    """HoloMotion v1.4 G1 motion-tracking actor observation.

    The release policy uses one flat observation input with:
    current terms 134D + 10 future frames * 47D = 604D.
    """

    CURRENT_DIM = 134
    FUTURE_DIM = 47

    def __init__(
        self,
        joint_names: Sequence[str],
        root_body_name: str = "pelvis",
        num_future_frames: int = 10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.joint_names = [str(name) for name in joint_names]
        self.root_body_name = str(root_body_name)
        self.num_future_frames = int(num_future_frames)
        self.obs_dim = self.CURRENT_DIM + self.num_future_frames * self.FUTURE_DIM
        self._out = np.zeros((1, self.obs_dim), dtype=np.float32)
        self._last_action = np.zeros((len(self.joint_names),), dtype=np.float32)
        self._yaw_alignment: np.ndarray | None = None
        self._cached_layout: tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]] | None = None

    def reset(self) -> None:
        self._last_action.fill(0.0)
        self._yaw_alignment = None
        self._cached_layout = None

    def update(self, data: dict[str, Any]) -> None:
        action = np.asarray(data.get("action", self._last_action), dtype=np.float32).reshape(-1)
        if action.shape[0] == self._last_action.shape[0]:
            self._last_action[:] = action

    def _refresh_indices(self) -> None:
        motion_joint_names = tuple(self.env.motion_joint_names)
        motion_body_names = tuple(self.env.motion_body_names)
        robot_joint_names = tuple(self.state_processor.joint_names)
        layout = (motion_joint_names, motion_body_names, robot_joint_names)
        if layout == self._cached_layout:
            return

        self._motion_joint_indices = [motion_joint_names.index(name) for name in self.joint_names]
        self._root_body_idx = motion_body_names.index(self.root_body_name)
        self._robot_joint_indices = [robot_joint_names.index(name) for name in self.joint_names]

        default_cfg = self.env.policy_config["default_joint_pos"]
        joint_ids, _, default_joint_pos = resolve_matching_names_values(
            default_cfg,
            self.joint_names,
            preserve_order=True,
            strict=False,
        )
        self._default_joint_pos = np.zeros((len(self.joint_names),), dtype=np.float32)
        self._default_joint_pos[joint_ids] = np.asarray(default_joint_pos, dtype=np.float32)
        self._cached_layout = layout

    def _ensure_yaw_alignment(self, ref_root_quat_w: np.ndarray) -> None:
        if self._yaw_alignment is not None:
            return
        ref_yaw = float(_yaw_from_quat_wxyz(_normalize_quat_wxyz(ref_root_quat_w)))
        robot_yaw = float(_yaw_from_quat_wxyz(_normalize_quat_wxyz(self.state_processor.root_quat_w)))
        self._yaw_alignment = _yaw_quat_wxyz(robot_yaw - ref_yaw)

    def compute(self) -> np.ndarray:
        motion_data = self.env.motion_data
        if motion_data is None:
            return self._out
        self._refresh_indices()

        joint_pos = np.asarray(
            np.take(motion_data.joint_pos[0], self._motion_joint_indices, axis=1),
            dtype=np.float32,
        )
        body_pos = np.asarray(motion_data.body_pos_w[0], dtype=np.float32)
        body_quat = _normalize_quat_wxyz(np.asarray(motion_data.body_quat_w[0], dtype=np.float32))
        body_lin_vel = np.asarray(motion_data.body_lin_vel_w[0], dtype=np.float32)
        body_ang_vel = np.asarray(motion_data.body_ang_vel_w[0], dtype=np.float32)

        required_frames = self.num_future_frames + 1
        if joint_pos.shape[0] < required_frames:
            raise ValueError(
                f"HoloMotion observation needs {required_frames} motion frames "
                f"(current + {self.num_future_frames} future), got {joint_pos.shape[0]}"
            )

        root_pos = body_pos[:required_frames, self._root_body_idx]
        root_quat = body_quat[:required_frames, self._root_body_idx]
        root_lin_vel = body_lin_vel[:required_frames, self._root_body_idx]
        root_ang_vel = body_ang_vel[:required_frames, self._root_body_idx]
        root_lin_vel_local = quat_rotate_inverse_numpy(root_quat, root_lin_vel)
        root_ang_vel_local = quat_rotate_inverse_numpy(root_quat, root_ang_vel)
        ref_gravity = _projected_gravity_wxyz(root_quat)

        robot_root_quat = _normalize_quat_wxyz(self.state_processor.root_quat_w)
        robot_joint_pos = self.state_processor.joint_pos[self._robot_joint_indices].astype(np.float32)
        robot_joint_vel = self.state_processor.joint_vel[self._robot_joint_indices].astype(np.float32)
        self._ensure_yaw_alignment(root_quat[0])
        assert self._yaw_alignment is not None

        out = self._out[0]
        offset = 0

        def put(value: np.ndarray) -> None:
            nonlocal offset
            value = np.asarray(value, dtype=np.float32).reshape(-1)
            out[offset : offset + value.size] = value
            offset += value.size

        aligned_ref_cur = _normalize_quat_wxyz(
            quat_mul(self._yaw_alignment.reshape(1, 4), root_quat[0:1])[0]
        )
        yaw_error = _yaw_from_quat_wxyz(aligned_ref_cur) - _yaw_from_quat_wxyz(robot_root_quat)
        future_quat_aligned = _normalize_quat_wxyz(
            quat_mul(
                np.broadcast_to(self._yaw_alignment.reshape(1, 4), root_quat[1:required_frames].shape),
                root_quat[1:required_frames],
            )
        )
        robot_quat_future = np.broadcast_to(robot_root_quat.reshape(1, 4), future_quat_aligned.shape)
        rel_future_quat = quat_mul(quat_conjugate(robot_quat_future), future_quat_aligned)
        yaw_delta = _yaw_from_quat_wxyz(root_quat[1:required_frames]) - _yaw_from_quat_wxyz(root_quat[0:1])

        put(ref_gravity[0])
        put(root_lin_vel_local[0])
        put(root_ang_vel_local[0])
        put(joint_pos[0])
        put(root_pos[0, 2:3])
        put(np.asarray([np.sin(yaw_error), np.cos(yaw_error)], dtype=np.float32))
        put(_projected_gravity_wxyz(robot_root_quat))
        put(self.state_processor.root_ang_vel_b)
        put(robot_joint_pos - self._default_joint_pos)
        put(robot_joint_vel)
        put(self._last_action)
        put(joint_pos[1:required_frames].reshape(-1))
        put(root_pos[1:required_frames, 2])
        put(ref_gravity[1:required_frames].reshape(-1))
        put(root_lin_vel_local[1:required_frames].reshape(-1))
        put(root_ang_vel_local[1:required_frames].reshape(-1))
        put(np.stack([np.sin(yaw_delta), np.cos(yaw_delta)], axis=-1).reshape(-1))
        put(_rot6d_from_quat_wxyz(rel_future_quat).reshape(-1))

        if offset != self.obs_dim:
            raise RuntimeError(f"HoloMotion observation wrote {offset} values, expected {self.obs_dim}")
        return self._out
