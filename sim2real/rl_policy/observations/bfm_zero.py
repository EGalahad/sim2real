from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np

from sim2real.rl_policy.observations.base import Observation
from sim2real.rl_policy.utils.motion import MotionData
from sim2real.utils.math import (
    matrix_from_quat,
    quat_conjugate,
    quat_mul,
    quat_rotate_inverse_numpy,
    quat_rotate_numpy,
)


BFM_ZERO_STATE_DIM = 64
BFM_ZERO_ACTION_DIM = 29
BFM_ZERO_HISTORY_DIM = 372
BFM_ZERO_Z_DIM = 256
BFM_ZERO_PRIVILEGED_STATE_DIM = 463
BFM_ZERO_BASE_ANG_VEL_SCALE = 0.25
BFM_ZERO_ACTION_OBS_SCALE = 5.0
BFM_ZERO_ACTION_OBS_CLIP = 5.0
BFM_ZERO_HEAD_LINK_OFFSET = np.asarray([0.0, 0.0, 0.35], dtype=np.float32)
BFM_ZERO_MINIMAL_BODY_NAMES = (
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
    "head_link",
)


def _batched(value: np.ndarray) -> np.ndarray:
    return np.asarray(value, dtype=np.float32).reshape(1, -1)


def _safe_root_quat_w(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).reshape(4)
    norm = np.linalg.norm(quat)
    if norm < 1.0e-8:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return quat / norm


def _projected_gravity(root_quat_w: np.ndarray) -> np.ndarray:
    return quat_rotate_inverse_numpy(
        _safe_root_quat_w(root_quat_w).reshape(1, 4),
        np.asarray([[0.0, 0.0, -1.0]], dtype=np.float32),
    )[0].astype(np.float32)


def _scaled_base_ang_vel(root_ang_vel_b: np.ndarray) -> np.ndarray:
    return (
        np.asarray(root_ang_vel_b, dtype=np.float32).reshape(3)
        * BFM_ZERO_BASE_ANG_VEL_SCALE
    )


def _action_obs(action: np.ndarray, expected_dim: int) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.size != expected_dim:
        raise ValueError(
            f"BFM-Zero action dim mismatch: {action.size} != {expected_dim}"
        )
    return np.clip(
        action * BFM_ZERO_ACTION_OBS_SCALE,
        -BFM_ZERO_ACTION_OBS_CLIP,
        BFM_ZERO_ACTION_OBS_CLIP,
    ).astype(np.float32)


def _heading_from_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    ref_dir = np.zeros(quat.shape[:-1] + (3,), dtype=np.float32)
    ref_dir[..., 0] = 1.0
    rot_dir = quat_rotate_numpy(quat, ref_dir)
    return np.arctan2(rot_dir[..., 1], rot_dir[..., 0])


def _quat_from_yaw(yaw: np.ndarray) -> np.ndarray:
    yaw = np.asarray(yaw, dtype=np.float32)
    quat = np.zeros(yaw.shape + (4,), dtype=np.float32)
    quat[..., 0] = np.cos(yaw / 2.0)
    quat[..., 3] = np.sin(yaw / 2.0)
    return quat


def _quat_to_tan_norm(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    flat = quat.reshape(-1, 4)
    ref_tan = np.zeros((flat.shape[0], 3), dtype=np.float32)
    ref_tan[:, 0] = 1.0
    ref_norm = np.zeros((flat.shape[0], 3), dtype=np.float32)
    ref_norm[:, 2] = 1.0
    tan = quat_rotate_numpy(flat, ref_tan)
    norm = quat_rotate_numpy(flat, ref_norm)
    return np.concatenate([tan, norm], axis=-1).reshape(quat.shape[:-1] + (6,))


def _quat_mul_xyzw(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    q1 = np.asarray(q1, dtype=np.float32)
    q2 = np.asarray(q2, dtype=np.float32)
    shape = q1.shape
    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)
    x1, y1, z1, w1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    x2, y2, z2, w2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    return np.stack([x, y, z, w], axis=-1).reshape(shape).astype(np.float32)


def _quat_rotate_xyzw(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    shape = v.shape
    q = q.reshape(-1, 4)
    v = v.reshape(-1, 3)
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w**2 - 1.0)[:, None]
    b = np.cross(q_vec, v) * q_w[:, None] * 2.0
    c = q_vec * np.sum(q_vec * v, axis=1, keepdims=True) * 2.0
    return (a + b + c).reshape(shape).astype(np.float32)


def _quat_to_tan_norm_xyzw(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    flat = quat.reshape(-1, 4)
    ref_tan = np.zeros((flat.shape[0], 3), dtype=np.float32)
    ref_tan[:, 0] = 1.0
    ref_norm = np.zeros((flat.shape[0], 3), dtype=np.float32)
    ref_norm[:, 2] = 1.0
    tan = _quat_rotate_xyzw(flat, ref_tan)
    norm = _quat_rotate_xyzw(flat, ref_norm)
    return np.concatenate([tan, norm], axis=-1).reshape(quat.shape[:-1] + (6,))


def _heading_inv_quat_xyzw(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    heading = np.arctan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    ).astype(np.float32)
    half = -heading / 2.0
    out = np.zeros(quat.shape, dtype=np.float32)
    out[..., 2] = np.sin(half)
    out[..., 3] = np.cos(half)
    return out


def _minimal_calc_angular_velocity_wxyz(
    quat_cur: np.ndarray,
    quat_prev: np.ndarray,
    dt: float,
) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R

    quat_cur = np.asarray(quat_cur, dtype=np.float32)
    quat_prev = np.asarray(quat_prev, dtype=np.float32)
    original_shape = quat_cur.shape
    if quat_cur.ndim == 1:
        quat_cur = quat_cur.reshape(1, 4)
        quat_prev = quat_prev.reshape(1, 4)
    flat_cur = quat_cur.reshape(-1, 4)
    flat_prev = quat_prev.reshape(-1, 4)
    quat_cur_xyzw = flat_cur[:, [1, 2, 3, 0]]
    quat_prev_xyzw = flat_prev[:, [1, 2, 3, 0]]
    delta = R.from_quat(quat_prev_xyzw).inv() * R.from_quat(quat_cur_xyzw)
    ang_vel = (delta.as_rotvec() / float(dt)).astype(np.float32)
    if original_shape == (4,):
        return ang_vel[0]
    return ang_vel.reshape(original_shape[:-1] + (3,))


def _minimal_projected_gravity_and_ang_vel(
    root_quat_wxyz: np.ndarray,
    root_ang_vel_local: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial.transform import Rotation as R

    root_quat_wxyz = np.asarray(root_quat_wxyz, dtype=np.float32)
    rotation = R.from_quat(root_quat_wxyz[[1, 2, 3, 0]])
    projected_gravity = rotation.inv().apply(np.asarray([0.0, 0.0, -1.0], dtype=np.float32))
    ang_vel = rotation.apply(np.asarray(root_ang_vel_local, dtype=np.float32))
    return projected_gravity.astype(np.float32), ang_vel.astype(np.float32)


def _minimal_local_root_ang_vel(
    root_quat_wxyz: np.ndarray,
    root_ang_vel_w: np.ndarray,
) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R

    rotation = R.from_quat(np.asarray(root_quat_wxyz, dtype=np.float32)[[1, 2, 3, 0]])
    return rotation.inv().apply(np.asarray(root_ang_vel_w, dtype=np.float32)).astype(np.float32)


def _minimal_privileged_state(
    body_pos: np.ndarray,
    body_quat_wxyz: np.ndarray,
    body_vel: np.ndarray,
    body_ang_vel: np.ndarray,
) -> np.ndarray:
    body_pos = np.asarray(body_pos, dtype=np.float32)
    body_rot_xyzw = np.asarray(body_quat_wxyz, dtype=np.float32)[:, [1, 2, 3, 0]]
    body_vel = np.asarray(body_vel, dtype=np.float32)
    body_ang_vel = np.asarray(body_ang_vel, dtype=np.float32)

    root_pos = body_pos[0:1]
    root_rot = body_rot_xyzw[0:1]
    heading_inv = _heading_inv_quat_xyzw(root_rot)
    heading_inv_expand = np.broadcast_to(heading_inv, body_rot_xyzw.shape)

    local_body_pos = body_pos - root_pos
    local_body_pos = _quat_rotate_xyzw(heading_inv_expand, local_body_pos).reshape(1, -1)[:, 3:]
    local_body_rot = _quat_mul_xyzw(heading_inv_expand, body_rot_xyzw)
    local_body_rot_obs = _quat_to_tan_norm_xyzw(local_body_rot).reshape(1, -1)
    local_body_vel = _quat_rotate_xyzw(heading_inv_expand, body_vel).reshape(1, -1)
    local_body_ang_vel = _quat_rotate_xyzw(heading_inv_expand, body_ang_vel).reshape(1, -1)
    root_h = root_pos[:, 2:3]

    privileged_state = np.concatenate(
        [
            root_h,
            local_body_pos,
            local_body_rot_obs,
            local_body_vel,
            local_body_ang_vel,
        ],
        axis=-1,
    ).reshape(-1)
    if privileged_state.size != BFM_ZERO_PRIVILEGED_STATE_DIM:
        raise ValueError(
            "BFM-Zero minimal privileged_state dim mismatch: "
            f"{privileged_state.size} != {BFM_ZERO_PRIVILEGED_STATE_DIM}"
        )
    return privileged_state.astype(np.float32)


def _append_synthetic_head_link(
    *,
    body_pos: np.ndarray,
    body_quat: np.ndarray,
    body_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    body_names = list(body_names)
    if "head_link" in body_names:
        return body_pos, body_quat
    torso_idx = body_names.index("torso_link")
    torso_pos = np.asarray(body_pos[:, torso_idx], dtype=np.float32)
    torso_quat = np.asarray(body_quat[:, torso_idx], dtype=np.float32)
    head_offset = np.einsum(
        "tij,j->ti",
        matrix_from_quat(torso_quat),
        BFM_ZERO_HEAD_LINK_OFFSET,
    ).astype(np.float32)
    head_pos = torso_pos + head_offset
    body_pos = np.concatenate([body_pos, head_pos[:, None, :]], axis=1)
    body_quat = np.concatenate([body_quat, torso_quat[:, None, :]], axis=1)
    return body_pos.astype(np.float32, copy=False), body_quat.astype(np.float32, copy=False)


def _compute_minimal_backward_observations_from_motion_arrays(
    *,
    root_quat: np.ndarray,
    dof_pos: np.ndarray,
    body_pos: np.ndarray,
    body_quat: np.ndarray,
    default_joint_pos: np.ndarray,
    target_fps: float = 50.0,
    target_frame_indices: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    root_quat = np.asarray(root_quat, dtype=np.float32)
    dof_pos = np.asarray(dof_pos, dtype=np.float32)
    body_pos = np.asarray(body_pos, dtype=np.float32)
    body_quat = np.asarray(body_quat, dtype=np.float32)
    if root_quat.ndim != 2 or root_quat.shape[1] != 4:
        raise ValueError(f"root_quat must have shape [T, 4], got {root_quat.shape}")
    if dof_pos.ndim != 2 or dof_pos.shape[1] != BFM_ZERO_ACTION_DIM:
        raise ValueError(
            f"dof_pos must have shape [T, {BFM_ZERO_ACTION_DIM}], got {dof_pos.shape}"
        )
    if body_pos.ndim != 3 or body_pos.shape[1:] != (len(BFM_ZERO_MINIMAL_BODY_NAMES), 3):
        raise ValueError(
            "body_pos must have shape "
            f"[T, {len(BFM_ZERO_MINIMAL_BODY_NAMES)}, 3], got {body_pos.shape}"
        )
    if body_quat.ndim != 3 or body_quat.shape[1:] != (len(BFM_ZERO_MINIMAL_BODY_NAMES), 4):
        raise ValueError(
            "body_quat must have shape "
            f"[T, {len(BFM_ZERO_MINIMAL_BODY_NAMES)}, 4], got {body_quat.shape}"
        )
    length = int(dof_pos.shape[0])
    if root_quat.shape[0] != length or body_pos.shape[0] != length or body_quat.shape[0] != length:
        raise ValueError("root_quat, dof_pos, body_pos, and body_quat time lengths must match")
    if target_frame_indices is None:
        target_frame_indices = np.arange(length, dtype=np.int64)
    else:
        target_frame_indices = np.asarray(target_frame_indices, dtype=np.int64).reshape(-1)
    if target_frame_indices.size == 0:
        raise ValueError("target_frame_indices must not be empty")
    if np.any(target_frame_indices < 0) or np.any(target_frame_indices >= length):
        raise ValueError(
            f"target_frame_indices out of range for length {length}: "
            f"{target_frame_indices.tolist()}"
        )

    dt = 1.0 / float(target_fps)
    dof_vel = np.zeros_like(dof_pos, dtype=np.float32)
    root_ang_vel = np.zeros((length, 3), dtype=np.float32)
    body_vel = np.zeros_like(body_pos, dtype=np.float32)
    body_ang_vel = np.zeros_like(body_pos, dtype=np.float32)
    if length > 1:
        dof_vel[1:] = (dof_pos[1:] - dof_pos[:-1]) / dt
        root_ang_vel[1:] = _minimal_calc_angular_velocity_wxyz(root_quat[1:], root_quat[:-1], dt)
        body_vel[1:] = ((body_pos[1:] - body_pos[:-1]) / dt).astype(np.float32)
        body_ang_vel[1:] = _minimal_calc_angular_velocity_wxyz(body_quat[1:], body_quat[:-1], dt)

    states = np.empty((target_frame_indices.size, BFM_ZERO_STATE_DIM), dtype=np.float32)
    privileged = np.empty(
        (target_frame_indices.size, BFM_ZERO_PRIVILEGED_STATE_DIM),
        dtype=np.float32,
    )
    default_joint_pos = np.asarray(default_joint_pos, dtype=np.float32).reshape(BFM_ZERO_ACTION_DIM)
    for out_idx, frame_idx in enumerate(target_frame_indices):
        frame_idx = int(frame_idx)
        root_ang_vel_local = _minimal_local_root_ang_vel(
            root_quat[frame_idx],
            root_ang_vel[frame_idx],
        )
        projected_gravity, ang_vel = _minimal_projected_gravity_and_ang_vel(
            root_quat[frame_idx],
            root_ang_vel_local,
        )
        states[out_idx] = np.concatenate(
            [
                dof_pos[frame_idx] - default_joint_pos,
                dof_vel[frame_idx],
                projected_gravity,
                ang_vel,
            ],
            axis=0,
        ).astype(np.float32)
        privileged[out_idx] = _minimal_privileged_state(
            body_pos[frame_idx],
            body_quat[frame_idx],
            body_vel[frame_idx],
            body_ang_vel[frame_idx],
        )

    return {
        "state": states,
        "privileged_state": privileged,
    }


def _resolve_motion_data_step_indices(
    *,
    motion_data: MotionData | None,
    motion_future_steps: np.ndarray,
    required_steps: np.ndarray,
) -> tuple[MotionData, np.ndarray]:
    if motion_data is None:
        raise ValueError("BFM-Zero window obs requires state_processor.motion_data")
    available_steps = [int(step) for step in np.asarray(motion_future_steps, dtype=np.int64).reshape(-1)]
    if not available_steps:
        raise ValueError("BFM-Zero window obs requires motion.future_steps")
    step_to_index = {step: idx for idx, step in enumerate(available_steps)}
    missing_steps = [int(step) for step in required_steps if int(step) not in step_to_index]
    if missing_steps:
        raise ValueError(
            "BFM-Zero window obs requires motion.future_steps to contain "
            f"{[int(step) for step in required_steps]}, got {available_steps}; "
            f"missing {missing_steps}"
        )
    indices = np.asarray([step_to_index[int(step)] for step in required_steps], dtype=np.int64)
    return motion_data, indices


def _backward_window_query(
    *,
    state_processor: Any,
    seq_length: int,
    motion_t_offset: int,
) -> tuple[np.ndarray, list[str], list[str]]:
    required_steps = int(motion_t_offset) - 1 + np.arange(
        int(seq_length) + 1,
        dtype=np.int64,
    )
    source_joint_names = list(getattr(state_processor, "motion_joint_names", ()))
    source_body_names = list(getattr(state_processor, "motion_body_names", ()))
    if not source_joint_names or not source_body_names:
        raise ValueError(
            "BFM-Zero window obs requires motion_joint_names and "
            "motion_body_names on state_processor"
        )
    return required_steps, source_joint_names, source_body_names


def _get_motion_data_backward_observation_window(
    *,
    state_processor: Any,
    joint_names: Sequence[str],
    default_joint_pos: np.ndarray,
    target_fps: float,
    seq_length: int,
    motion_t_offset: int,
    clamp_to_final: bool,
) -> dict[str, np.ndarray]:
    if not clamp_to_final:
        raise ValueError("BFM-Zero MotionData window obs currently requires clamp_to_final=True")

    required_steps, source_joint_names, source_body_names = _backward_window_query(
        state_processor=state_processor,
        seq_length=seq_length,
        motion_t_offset=motion_t_offset,
    )

    motion_data, support_indices = _resolve_motion_data_step_indices(
        motion_data=getattr(state_processor, "motion_data", None),
        motion_future_steps=getattr(
            state_processor,
            "motion_future_steps",
            np.asarray([], dtype=np.int64),
        ),
        required_steps=required_steps,
    )

    selected_timestamps = np.asarray(
        getattr(motion_data, "timestamps_ns", np.zeros((1, 0), dtype=np.int64)),
        dtype=np.int64,
    )
    timestamp_key = ()
    if selected_timestamps.ndim >= 2 and selected_timestamps.shape[1] > int(np.max(support_indices)):
        timestamp_key = tuple(selected_timestamps[0, support_indices].reshape(-1).tolist())
    selected_steps = np.asarray(
        getattr(motion_data, "step", np.zeros((1, 0), dtype=np.int64)),
        dtype=np.int64,
    )
    step_key = ()
    if selected_steps.ndim >= 2 and selected_steps.shape[1] > int(np.max(support_indices)):
        step_key = tuple(selected_steps[0, support_indices].reshape(-1).tolist())
    cache_key = (
        tuple(required_steps.tolist()),
        step_key,
        timestamp_key,
        float(target_fps),
        int(seq_length),
        int(motion_t_offset),
        tuple(joint_names),
        tuple(np.asarray(default_joint_pos, dtype=np.float32).round(8).tolist()),
    )
    if getattr(state_processor, "_bfm_zero_motion_data_window_obs_cache_key", None) == cache_key:
        return getattr(state_processor, "_bfm_zero_motion_data_window_obs_cache_value")

    joint_indices = [source_joint_names.index(name) for name in joint_names]
    body_indices = [
        source_body_names.index(name)
        for name in BFM_ZERO_MINIMAL_BODY_NAMES
        if name != "head_link"
    ]
    selected_body_names = [
        name for name in BFM_ZERO_MINIMAL_BODY_NAMES if name != "head_link"
    ]
    if "head_link" in source_body_names:
        body_indices.append(source_body_names.index("head_link"))
        selected_body_names.append("head_link")

    dof_pos = np.asarray(
        motion_data.joint_pos[0, support_indices][:, joint_indices],
        dtype=np.float32,
    )
    body_pos = np.asarray(
        motion_data.body_pos_w[0, support_indices][:, body_indices],
        dtype=np.float32,
    )
    body_quat = np.asarray(
        motion_data.body_quat_w[0, support_indices][:, body_indices],
        dtype=np.float32,
    )
    body_pos, body_quat = _append_synthetic_head_link(
        body_pos=body_pos,
        body_quat=body_quat,
        body_names=selected_body_names,
    )
    root_idx = BFM_ZERO_MINIMAL_BODY_NAMES.index("pelvis")
    root_quat = body_quat[:, root_idx]

    obs = _compute_minimal_backward_observations_from_motion_arrays(
        root_quat=root_quat,
        dof_pos=dof_pos,
        body_pos=body_pos,
        body_quat=body_quat,
        default_joint_pos=default_joint_pos,
        target_fps=target_fps,
        target_frame_indices=np.arange(1, int(seq_length) + 1, dtype=np.int64),
    )
    setattr(state_processor, "_bfm_zero_motion_data_window_obs_cache_key", cache_key)
    setattr(state_processor, "_bfm_zero_motion_data_window_obs_cache_value", obs)
    return obs


def _gaussian_filter_time(values: np.ndarray, sigma: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if sigma <= 0.0 or values.shape[0] <= 1:
        return values
    from scipy.ndimage import gaussian_filter1d

    return gaussian_filter1d(values, sigma, axis=0, mode="nearest").astype(np.float32)


def _quat_angle_axis_humanoidverse_wxyz(quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    quat = np.asarray(quat, dtype=np.float32)
    quat = quat / np.clip(np.linalg.norm(quat, axis=-1, keepdims=True), 1.0e-9, None)
    w = quat[..., 0]
    axis = quat[..., 1:] / np.clip(
        np.linalg.norm(quat[..., 1:], axis=-1, keepdims=True),
        1.0e-9,
        None,
    )
    angle = np.arccos(np.clip(2.0 * w * w - 1.0, -1.0, 1.0)).astype(np.float32)
    return angle, axis.astype(np.float32)


def _compute_humanoidverse_motion_velocities(
    *,
    body_pos_w: np.ndarray,
    body_quat_w: np.ndarray,
    joint_pos: np.ndarray,
    fps: float,
    sigma: float,
) -> dict[str, np.ndarray]:
    """Match HumanoidVerse motion-lib velocity construction for BFM-Zero targets."""
    body_pos_w = np.asarray(body_pos_w, dtype=np.float32)
    body_quat_w = np.asarray(body_quat_w, dtype=np.float32)
    joint_pos = np.asarray(joint_pos, dtype=np.float32)
    dt = 1.0 / float(fps)

    body_lin_vel_w = _gaussian_filter_time(np.gradient(body_pos_w, axis=0) / dt, sigma)

    quat_delta = np.zeros_like(body_quat_w)
    if body_quat_w.shape[0] > 1:
        quat_delta[:-1] = quat_mul(body_quat_w[1:], quat_conjugate(body_quat_w[:-1]))
    quat_delta[-1, ..., 0] = 1.0
    angle, axis = _quat_angle_axis_humanoidverse_wxyz(quat_delta)
    body_ang_vel_w = _gaussian_filter_time(axis * angle[..., None] / dt, sigma)

    joint_vel = np.zeros_like(joint_pos, dtype=np.float32)
    if joint_pos.shape[0] > 1:
        joint_diff = (joint_pos[1:] - joint_pos[:-1]) / dt
        joint_vel[:-1] = joint_diff
        if joint_diff.shape[0] > 1:
            joint_vel[-1] = joint_diff[-2]
        else:
            joint_vel[-1] = joint_diff[-1]

    return {
        "body_lin_vel_w": body_lin_vel_w.astype(np.float32, copy=False),
        "body_ang_vel_w": body_ang_vel_w.astype(np.float32, copy=False),
        "joint_vel": joint_vel.astype(np.float32, copy=False),
    }


class _BFMZeroJointSelection(Observation):
    def __init__(
        self,
        joint_names: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.joint_names = list(joint_names or self.env.policy_joint_names)
        if len(self.joint_names) != BFM_ZERO_ACTION_DIM:
            raise ValueError(
                f"BFM-Zero expects {BFM_ZERO_ACTION_DIM} joints, got {len(self.joint_names)}"
            )
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

    def _dof_pos(self) -> np.ndarray:
        joint_pos = np.asarray(
            self.state_processor.joint_pos[self._state_joint_indices],
            dtype=np.float32,
        )
        return joint_pos - self._default_joint_pos

    def _dof_vel(self) -> np.ndarray:
        return np.asarray(
            self.state_processor.joint_vel[self._state_joint_indices],
            dtype=np.float32,
        )


class _BFMZeroMotionSelection(_BFMZeroJointSelection):
    def __init__(
        self,
        body_names: Sequence[str] | None = None,
        root_body_name: str = "pelvis",
        motion_t_offset: int = 0,
        motion_velocity_source: str = "motion_data",
        humanoidverse_velocity_sigma: float = 2.0,
        humanoidverse_velocity_fps: float = 50.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.body_names = list(body_names or self.env.body_names_simulation)
        self.root_body_name = str(root_body_name)
        self.motion_t_offset = int(motion_t_offset)
        self.motion_velocity_source = str(motion_velocity_source)
        self.humanoidverse_velocity_sigma = float(humanoidverse_velocity_sigma)
        self.humanoidverse_velocity_fps = float(humanoidverse_velocity_fps)
        self._cached_motion_layout: tuple[tuple[str, ...], tuple[str, ...]] | None = None

    def _refresh_motion_indices(self) -> None:
        joint_names = tuple(self.state_processor.motion_joint_names)
        body_names = tuple(self.state_processor.motion_body_names)
        layout = (joint_names, body_names)
        if self._cached_motion_layout == layout:
            return

        self._motion_joint_indices = [joint_names.index(name) for name in self.joint_names]
        self._motion_body_indices = [
            body_names.index(name)
            for name in self.body_names
            if name != "head_link"
        ]
        self._has_source_head_link = "head_link" in body_names
        if self._has_source_head_link and "head_link" in self.body_names:
            self._motion_body_indices.append(body_names.index("head_link"))
        self._root_body_idx = self.body_names.index(self.root_body_name)
        self._torso_body_idx = self.body_names.index("torso_link")
        self._cached_motion_layout = layout

    def _motion_data(self) -> MotionData:
        motion_data = self.state_processor.motion_data
        if motion_data is None:
            raise ValueError("BFM-Zero fused observations require motion_data")
        motion_ids, motion_t, motion_steps = self._motion_query()
        if self.motion_t_offset != 0:
            motion_data = self.state_processor.motion_dataset.get_slice(
                motion_ids,
                motion_t,
                motion_steps,
            )
        if self.motion_velocity_source == "humanoidverse":
            return self._with_humanoidverse_velocities(
                motion_data,
                motion_ids=motion_ids,
                motion_t=motion_t,
                motion_steps=motion_steps,
            )
        if self.motion_velocity_source != "motion_data":
            raise ValueError(
                "BFM-Zero motion_velocity_source must be 'motion_data' or "
                f"'humanoidverse', got {self.motion_velocity_source!r}"
            )
        return motion_data

    def _motion_query(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        motion_ids = np.asarray(self.state_processor.motion_ids, dtype=np.int64).reshape(-1)
        motion_t = np.asarray(self.state_processor.motion_t, dtype=np.int64).reshape(-1)
        motion_steps = np.asarray(
            getattr(self.state_processor, "motion_future_steps", np.asarray([0], dtype=int)),
            dtype=np.int64,
        ).reshape(-1)
        return motion_ids, motion_t + self.motion_t_offset, motion_steps

    def _with_humanoidverse_velocities(
        self,
        motion_data: MotionData,
        *,
        motion_ids: np.ndarray,
        motion_t: np.ndarray,
        motion_steps: np.ndarray,
    ) -> MotionData:
        dataset = getattr(self.state_processor, "motion_dataset", None)
        if dataset is None:
            return motion_data

        result = {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in motion_data.__dict__.items()
        }
        body_lin_vel = np.empty_like(np.asarray(motion_data.body_lin_vel_w))
        body_ang_vel = np.empty_like(np.asarray(motion_data.body_ang_vel_w))
        joint_vel = np.empty_like(np.asarray(motion_data.joint_vel))

        for env_idx, local_motion_id in enumerate(motion_ids):
            cache = self._humanoidverse_velocity_cache(int(local_motion_id))
            idx = np.clip(
                int(motion_t[env_idx]) + motion_steps,
                0,
                cache["body_lin_vel_w"].shape[0] - 1,
            )
            body_lin_vel[env_idx] = cache["body_lin_vel_w"][idx]
            body_ang_vel[env_idx] = cache["body_ang_vel_w"][idx]
            joint_vel[env_idx] = cache["joint_vel"][idx]

        result["body_lin_vel_w"] = body_lin_vel.astype(np.float32, copy=False)
        result["body_ang_vel_w"] = body_ang_vel.astype(np.float32, copy=False)
        result["joint_vel"] = joint_vel.astype(np.float32, copy=False)
        return MotionData(**result)

    def _humanoidverse_velocity_cache(self, local_motion_id: int) -> dict[str, np.ndarray]:
        dataset = self.state_processor.motion_dataset
        cache = getattr(dataset, "_bfm_zero_humanoidverse_velocity_cache", None)
        if cache is None:
            cache = {}
            setattr(dataset, "_bfm_zero_humanoidverse_velocity_cache", cache)

        key = (
            local_motion_id,
            self.humanoidverse_velocity_sigma,
            self.humanoidverse_velocity_fps,
        )
        if key not in cache:
            length = int(np.asarray(dataset.lengths).reshape(-1)[local_motion_id])
            full_motion = dataset.get_slice(
                np.asarray([local_motion_id], dtype=np.int64),
                np.asarray([0], dtype=np.int64),
                np.arange(length, dtype=np.int64),
            )
            cache[key] = _compute_humanoidverse_motion_velocities(
                body_pos_w=np.asarray(full_motion.body_pos_w[0], dtype=np.float32),
                body_quat_w=np.asarray(full_motion.body_quat_w[0], dtype=np.float32),
                joint_pos=np.asarray(full_motion.joint_pos[0], dtype=np.float32),
                fps=self.humanoidverse_velocity_fps,
                sigma=self.humanoidverse_velocity_sigma,
            )
        return cache[key]

    def _motion_joint_pos_vel(self) -> tuple[np.ndarray, np.ndarray]:
        self._refresh_motion_indices()
        motion_data = self._motion_data()
        joint_pos = np.asarray(
            motion_data.joint_pos[0, 0, self._motion_joint_indices],
            dtype=np.float32,
        )
        joint_vel = np.asarray(
            motion_data.joint_vel[0, 0, self._motion_joint_indices],
            dtype=np.float32,
        )
        return joint_pos, joint_vel

    def _motion_body_state(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self._refresh_motion_indices()
        motion_data = self._motion_data()
        body_pos = np.asarray(
            motion_data.body_pos_w[0, 0, self._motion_body_indices],
            dtype=np.float32,
        )
        body_quat = np.asarray(
            motion_data.body_quat_w[0, 0, self._motion_body_indices],
            dtype=np.float32,
        )
        body_lin_vel = np.asarray(
            motion_data.body_lin_vel_w[0, 0, self._motion_body_indices],
            dtype=np.float32,
        )
        body_ang_vel = np.asarray(
            motion_data.body_ang_vel_w[0, 0, self._motion_body_indices],
            dtype=np.float32,
        )

        if "head_link" in self.body_names and not self._has_source_head_link:
            torso_pos = body_pos[self._torso_body_idx]
            torso_quat = body_quat[self._torso_body_idx]
            torso_lin_vel = body_lin_vel[self._torso_body_idx]
            torso_ang_vel = body_ang_vel[self._torso_body_idx]
            offset_w = matrix_from_quat(torso_quat.reshape(1, 4))[0] @ BFM_ZERO_HEAD_LINK_OFFSET
            head_pos = torso_pos + offset_w
            # Match HumanoidVerse's extended-body velocity path exactly: it uses
            # the configured parent-local offset in the cross product.
            head_lin_vel = torso_lin_vel + np.cross(
                torso_ang_vel,
                BFM_ZERO_HEAD_LINK_OFFSET,
            ).astype(np.float32)
            body_pos = np.concatenate([body_pos, head_pos.reshape(1, 3)], axis=0)
            body_quat = np.concatenate([body_quat, torso_quat.reshape(1, 4)], axis=0)
            body_lin_vel = np.concatenate([body_lin_vel, head_lin_vel.reshape(1, 3)], axis=0)
            body_ang_vel = np.concatenate([body_ang_vel, torso_ang_vel.reshape(1, 3)], axis=0)

        return body_pos, body_quat, body_lin_vel, body_ang_vel


class bfm_zero_state(_BFMZeroJointSelection):
    """BFM-Zero 64D ``state`` input.

    Mirrors ``HumanoidVerseVectorEnv._get_g1env_observation``:
    dof_pos, dof_vel, projected_gravity, base_ang_vel.

    The source dict is named ``obs_buf_dict_raw``, but its writer already applies
    ``obs_scales``. For BFM-Zero, ``base_ang_vel`` is scaled by 0.25 before the
    exported ONNX's internal observation normalizer.
    """

    def compute(self) -> np.ndarray:
        state = np.concatenate(
            [
                self._dof_pos(),
                self._dof_vel(),
                _projected_gravity(self.state_processor.root_quat_w),
                _scaled_base_ang_vel(self.state_processor.root_ang_vel_b),
            ],
            axis=0,
        )
        if state.size != BFM_ZERO_STATE_DIM:
            raise ValueError(
                f"BFM-Zero state dim mismatch: {state.size} != {BFM_ZERO_STATE_DIM}"
            )
        return _batched(state)


class bfm_zero_encoder_state(_BFMZeroMotionSelection):
    """Target-motion state consumed by the BFM-Zero backward encoder."""

    def compute(self) -> np.ndarray:
        joint_pos, joint_vel = self._motion_joint_pos_vel()
        body_pos, body_quat, _, body_ang_vel = self._motion_body_state()
        root_quat = body_quat[self._root_body_idx]
        state = np.concatenate(
            [
                joint_pos - self._default_joint_pos,
                joint_vel,
                _projected_gravity(root_quat),
                body_ang_vel[self._root_body_idx],
            ],
            axis=0,
        )
        if state.size != BFM_ZERO_STATE_DIM:
            raise ValueError(
                f"BFM-Zero encoder_state dim mismatch: {state.size} != {BFM_ZERO_STATE_DIM}"
            )
        return _batched(state)


class bfm_zero_privileged_state(_BFMZeroMotionSelection):
    """Target-motion ``max_local_self`` input for BFM-Zero's backward encoder."""

    def compute(self) -> np.ndarray:
        body_pos, body_quat, body_lin_vel, body_ang_vel = self._motion_body_state()
        if body_pos.shape[0] != 31:
            raise ValueError(f"BFM-Zero privileged_state expects 31 bodies, got {body_pos.shape[0]}")

        root_pos = body_pos[self._root_body_idx]
        root_quat = body_quat[self._root_body_idx]
        heading_inv = _quat_from_yaw(-_heading_from_quat_wxyz(root_quat))
        heading_inv_expand = np.broadcast_to(heading_inv.reshape(1, 4), body_quat.shape)

        local_body_pos = body_pos - root_pos.reshape(1, 3)
        local_body_pos = quat_rotate_numpy(heading_inv_expand, local_body_pos).reshape(-1)[3:]
        local_body_quat = quat_mul(heading_inv_expand, body_quat)
        local_body_rot = _quat_to_tan_norm(local_body_quat).reshape(-1)
        local_body_vel = quat_rotate_numpy(heading_inv_expand, body_lin_vel).reshape(-1)
        local_body_ang_vel = quat_rotate_numpy(heading_inv_expand, body_ang_vel).reshape(-1)
        privileged_state = np.concatenate(
            [
                root_pos[2:3],
                local_body_pos,
                local_body_rot,
                local_body_vel,
                local_body_ang_vel,
            ],
            axis=0,
        ).astype(np.float32)

        if privileged_state.size != BFM_ZERO_PRIVILEGED_STATE_DIM:
            raise ValueError(
                "BFM-Zero privileged_state dim mismatch: "
                f"{privileged_state.size} != {BFM_ZERO_PRIVILEGED_STATE_DIM}"
            )
        return _batched(privileged_state)


class bfm_zero_last_action(_BFMZeroJointSelection):
    """Previous normalized env action in BFM-Zero policy joint order.

    Source training multiplies the raw actor output by ``normalize_action_to=5``
    and clips to ``[-5, 5]`` before storing ``actions`` in observations. The
    deploy PD target still uses the raw actor output with YAML ``action_scale``.
    """

    def update(self, data: Dict[str, Any]) -> None:
        self._last_action = _action_obs(
            data.get("action", np.zeros(len(self.joint_names), dtype=np.float32)),
            len(self.joint_names),
        )

    def reset(self) -> None:
        self._last_action = np.zeros(len(self.joint_names), dtype=np.float32)

    def compute(self) -> np.ndarray:
        return _batched(getattr(self, "_last_action", np.zeros(len(self.joint_names), dtype=np.float32)))


class bfm_zero_history_actor(_BFMZeroJointSelection):
    """BFM-Zero 4-step actor history.

    Upstream ``_get_obs_history_actor`` sorts keys alphabetically, so the layout is:
    actions, base_ang_vel, dof_pos, dof_vel, projected_gravity. Each key is flattened
    newest-to-oldest from the env-internal history buffer. The current frame is inserted
    only after this observation is computed, matching HumanoidVerse's post-observation
    history add.
    Stored values are post-``obs_scales``/post-action-normalization, matching
    the source ``obs_buf_dict_raw`` writer and environment action path.
    """

    def __init__(self, history_length: int = 4, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.history_length = int(history_length)
        if self.history_length <= 0:
            raise ValueError("history_length must be positive")
        self._history: dict[str, np.ndarray] = {
            "actions": np.zeros((self.history_length, BFM_ZERO_ACTION_DIM), dtype=np.float32),
            "base_ang_vel": np.zeros((self.history_length, 3), dtype=np.float32),
            "dof_pos": np.zeros((self.history_length, BFM_ZERO_ACTION_DIM), dtype=np.float32),
            "dof_vel": np.zeros((self.history_length, BFM_ZERO_ACTION_DIM), dtype=np.float32),
            "projected_gravity": np.zeros((self.history_length, 3), dtype=np.float32),
        }
        self._pending_current: dict[str, np.ndarray] | None = None
        self._pending_written = True

    def reset(self) -> None:
        for value in self._history.values():
            value[:] = 0.0
        self._pending_current = None
        self._pending_written = True

    def update(self, data: Dict[str, Any]) -> None:
        action = _action_obs(
            data.get("action", np.zeros(len(self.joint_names), dtype=np.float32)),
            len(self.joint_names),
        )

        self._pending_current = {
            "actions": action,
            "base_ang_vel": _scaled_base_ang_vel(self.state_processor.root_ang_vel_b),
            "dof_pos": self._dof_pos(),
            "dof_vel": self._dof_vel(),
            "projected_gravity": _projected_gravity(self.state_processor.root_quat_w),
        }
        self._pending_written = False

    def _append_pending_current(self) -> None:
        if self._pending_written or self._pending_current is None:
            return
        for key in sorted(self._history):
            hist = self._history[key]
            hist[1:] = hist[:-1].copy()
            hist[0] = self._pending_current[key]
        self._pending_written = True

    def compute(self) -> np.ndarray:
        history = np.concatenate(
            [self._history[key].reshape(-1) for key in sorted(self._history)],
            axis=0,
        )
        if history.size != BFM_ZERO_HISTORY_DIM:
            raise ValueError(
                f"BFM-Zero history dim mismatch: {history.size} != {BFM_ZERO_HISTORY_DIM}"
            )
        self._append_pending_current()
        return _batched(history)


class _BFMZeroMinimalBackwardFutureWindow(_BFMZeroJointSelection):
    """Future-window backward obs for fused BFM-Zero streaming graphs."""

    obs_key: str
    obs_dim: int

    def __init__(
        self,
        seq_length: int = 8,
        target_fps: float = 50.0,
        clamp_to_final: bool = True,
        motion_t_offset: int = -1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.seq_length = int(seq_length)
        self.target_fps = float(target_fps)
        self.clamp_to_final = bool(clamp_to_final)
        self.motion_t_offset = int(motion_t_offset)
        if self.seq_length <= 0:
            raise ValueError("BFM-Zero future window seq_length must be positive")

    def compute(self) -> np.ndarray:
        obs = _get_motion_data_backward_observation_window(
            state_processor=self.state_processor,
            joint_names=self.joint_names,
            default_joint_pos=self._default_joint_pos,
            target_fps=self.target_fps,
            seq_length=self.seq_length,
            motion_t_offset=self.motion_t_offset,
            clamp_to_final=self.clamp_to_final,
        )
        value = np.asarray(obs[self.obs_key], dtype=np.float32)
        if value.ndim != 2 or value.shape[1] != self.obs_dim:
            raise ValueError(
                f"BFM-Zero {self.obs_key} window must have shape [T, {self.obs_dim}], "
                f"got {value.shape}"
            )
        return value


class bfm_zero_minimal_encoder_state_future(_BFMZeroMinimalBackwardFutureWindow):
    obs_key = "state"
    obs_dim = BFM_ZERO_STATE_DIM


class bfm_zero_minimal_privileged_state_future(_BFMZeroMinimalBackwardFutureWindow):
    obs_key = "privileged_state"
    obs_dim = BFM_ZERO_PRIVILEGED_STATE_DIM


class bfm_zero_minimal_future_window_weight(Observation):
    """Validity weights for BFM-Zero future-window latent smoothing."""

    def __init__(
        self,
        seq_length: int = 8,
        clamp_to_final: bool = True,
        motion_t_offset: int = -1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.seq_length = int(seq_length)
        self.clamp_to_final = bool(clamp_to_final)
        self.motion_t_offset = int(motion_t_offset)
        if self.seq_length <= 0:
            raise ValueError("BFM-Zero future window weight seq_length must be positive")

    def compute(self) -> np.ndarray:
        if not self.clamp_to_final:
            return np.ones((self.seq_length, 1), dtype=np.float32)

        required_steps, _, _ = _backward_window_query(
            state_processor=self.state_processor,
            seq_length=self.seq_length,
            motion_t_offset=self.motion_t_offset,
        )
        motion_data, support_indices = _resolve_motion_data_step_indices(
            motion_data=getattr(self.state_processor, "motion_data", None),
            motion_future_steps=getattr(
                self.state_processor,
                "motion_future_steps",
                np.asarray([], dtype=np.int64),
            ),
            required_steps=required_steps,
        )

        step = np.asarray(getattr(motion_data, "step"), dtype=np.int64)
        if step.ndim != 2 or step.shape[0] != 1:
            raise ValueError(f"BFM-Zero future-window weight expects step shape [1, T], got {step.shape}")
        support_step = step[0, support_indices]
        output_step = support_step[1:]
        output_required_steps = required_steps[1:]

        weights = np.ones(self.seq_length, dtype=np.float32)
        if output_step.shape[0] > 1:
            repeated_from_previous = output_step[1:] == output_step[:-1]
            positive_future = output_required_steps[1:] > 0
            weights[1:] = np.where(
                repeated_from_previous & positive_future,
                0.0,
                1.0,
            ).astype(np.float32)
        return weights.reshape(self.seq_length, 1)
