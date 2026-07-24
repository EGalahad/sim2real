from __future__ import annotations

import numpy as np

from sim2real.rl_policy.observations import bfm_zero


def _normalized_quaternions(
    rng: np.random.Generator,
    shape: tuple[int, ...],
) -> np.ndarray:
    quaternions = rng.normal(size=shape).astype(np.float32)
    return quaternions / np.linalg.norm(quaternions, axis=-1, keepdims=True)


def _reference_minimal_privileged_state(
    body_pos: np.ndarray,
    body_quat_wxyz: np.ndarray,
    body_vel: np.ndarray,
    body_ang_vel: np.ndarray,
) -> np.ndarray:
    body_rot_xyzw = np.asarray(body_quat_wxyz, dtype=np.float32)[:, [1, 2, 3, 0]]
    root_pos = body_pos[0:1]
    root_rot = body_rot_xyzw[0:1]
    heading_inv = bfm_zero._heading_inv_quat_xyzw(root_rot)
    heading_inv_expand = np.broadcast_to(heading_inv, body_rot_xyzw.shape)

    local_body_pos = body_pos - root_pos
    local_body_pos = bfm_zero._quat_rotate_xyzw(
        heading_inv_expand,
        local_body_pos,
    ).reshape(1, -1)[:, 3:]
    local_body_rot = bfm_zero._quat_mul_xyzw(heading_inv_expand, body_rot_xyzw)
    local_body_rot_obs = bfm_zero._quat_to_tan_norm_xyzw(local_body_rot).reshape(1, -1)
    local_body_vel = bfm_zero._quat_rotate_xyzw(
        heading_inv_expand,
        body_vel,
    ).reshape(1, -1)
    local_body_ang_vel = bfm_zero._quat_rotate_xyzw(
        heading_inv_expand,
        body_ang_vel,
    ).reshape(1, -1)

    return np.concatenate(
        [
            root_pos[:, 2:3],
            local_body_pos,
            local_body_rot_obs,
            local_body_vel,
            local_body_ang_vel,
        ],
        axis=-1,
    ).reshape(-1).astype(np.float32)


def _reference_calc_angular_velocity_wxyz(
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
    delta = (
        R.from_quat(flat_prev[:, [1, 2, 3, 0]]).inv()
        * R.from_quat(flat_cur[:, [1, 2, 3, 0]])
    )
    ang_vel = (delta.as_rotvec() / float(dt)).astype(np.float32)
    if original_shape == (4,):
        return ang_vel[0]
    return ang_vel.reshape(original_shape[:-1] + (3,))


def _reference_minimal_backward_observations(
    *,
    root_quat: np.ndarray,
    dof_pos: np.ndarray,
    body_pos: np.ndarray,
    body_quat: np.ndarray,
    default_joint_pos: np.ndarray,
    target_fps: float,
    target_frame_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    dt = 1.0 / float(target_fps)
    dof_vel = np.zeros_like(dof_pos, dtype=np.float32)
    root_ang_vel = np.zeros((root_quat.shape[0], 3), dtype=np.float32)
    body_vel = np.zeros_like(body_pos, dtype=np.float32)
    body_ang_vel = np.zeros_like(body_pos, dtype=np.float32)
    dof_vel[1:] = (dof_pos[1:] - dof_pos[:-1]) / dt
    root_ang_vel[1:] = bfm_zero._minimal_calc_angular_velocity_wxyz(
        root_quat[1:],
        root_quat[:-1],
        dt,
    )
    body_vel[1:] = ((body_pos[1:] - body_pos[:-1]) / dt).astype(np.float32)
    body_ang_vel[1:] = bfm_zero._minimal_calc_angular_velocity_wxyz(
        body_quat[1:],
        body_quat[:-1],
        dt,
    )

    states = []
    privileged = []
    for frame_idx in target_frame_indices:
        root_ang_vel_local = bfm_zero._minimal_local_root_ang_vel(
            root_quat[frame_idx],
            root_ang_vel[frame_idx],
        )
        projected_gravity, ang_vel = bfm_zero._minimal_projected_gravity_and_ang_vel(
            root_quat[frame_idx],
            root_ang_vel_local,
        )
        states.append(
            np.concatenate(
                [
                    dof_pos[frame_idx] - default_joint_pos,
                    dof_vel[frame_idx],
                    projected_gravity,
                    ang_vel,
                ]
            ).astype(np.float32)
        )
        privileged.append(
            _reference_minimal_privileged_state(
                body_pos[frame_idx],
                body_quat[frame_idx],
                body_vel[frame_idx],
                body_ang_vel[frame_idx],
            )
        )
    return {
        "state": np.stack(states),
        "privileged_state": np.stack(privileged),
    }


def test_minimal_angular_velocity_matches_scipy_reference() -> None:
    rng = np.random.default_rng(5)
    quat_prev = _normalized_quaternions(rng, (8, 31, 4))
    quat_cur = _normalized_quaternions(rng, (8, 31, 4))

    expected = _reference_calc_angular_velocity_wxyz(quat_cur, quat_prev, 0.02)
    actual = bfm_zero._minimal_calc_angular_velocity_wxyz(quat_cur, quat_prev, 0.02)

    np.testing.assert_array_equal(actual, expected)


def test_minimal_privileged_state_matches_scalar_reference() -> None:
    rng = np.random.default_rng(7)
    body_pos = rng.normal(size=(31, 3)).astype(np.float32)
    body_quat = _normalized_quaternions(rng, (31, 4))
    body_vel = rng.normal(size=(31, 3)).astype(np.float32)
    body_ang_vel = rng.normal(size=(31, 3)).astype(np.float32)

    expected = _reference_minimal_privileged_state(
        body_pos,
        body_quat,
        body_vel,
        body_ang_vel,
    )
    actual = bfm_zero._minimal_privileged_state(
        body_pos,
        body_quat,
        body_vel,
        body_ang_vel,
    )

    assert actual.shape == (bfm_zero.BFM_ZERO_PRIVILEGED_STATE_DIM,)
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, expected)


def test_minimal_privileged_state_batches_frames_without_semantic_drift() -> None:
    rng = np.random.default_rng(11)
    body_pos = rng.normal(size=(8, 31, 3)).astype(np.float32)
    body_quat = _normalized_quaternions(rng, (8, 31, 4))
    body_vel = rng.normal(size=(8, 31, 3)).astype(np.float32)
    body_ang_vel = rng.normal(size=(8, 31, 3)).astype(np.float32)

    expected = np.stack(
        [
            _reference_minimal_privileged_state(
                body_pos[index],
                body_quat[index],
                body_vel[index],
                body_ang_vel[index],
            )
            for index in range(body_pos.shape[0])
        ]
    )
    actual = bfm_zero._minimal_privileged_state(
        body_pos,
        body_quat,
        body_vel,
        body_ang_vel,
    )

    assert actual.shape == (8, bfm_zero.BFM_ZERO_PRIVILEGED_STATE_DIM)
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, expected)


def test_minimal_backward_observation_window_matches_scalar_reference() -> None:
    rng = np.random.default_rng(17)
    root_quat = _normalized_quaternions(rng, (9, 4))
    body_quat = _normalized_quaternions(rng, (9, 31, 4))
    dof_pos = rng.normal(size=(9, 29)).astype(np.float32)
    body_pos = rng.normal(size=(9, 31, 3)).astype(np.float32)
    default_joint_pos = rng.normal(size=29).astype(np.float32)
    target_frame_indices = np.arange(1, 9, dtype=np.int64)

    expected = _reference_minimal_backward_observations(
        root_quat=root_quat,
        dof_pos=dof_pos,
        body_pos=body_pos,
        body_quat=body_quat,
        default_joint_pos=default_joint_pos,
        target_fps=50.0,
        target_frame_indices=target_frame_indices,
    )
    actual = bfm_zero._compute_minimal_backward_observations_from_motion_arrays(
        root_quat=root_quat,
        dof_pos=dof_pos,
        body_pos=body_pos,
        body_quat=body_quat,
        default_joint_pos=default_joint_pos,
        target_fps=50.0,
        target_frame_indices=target_frame_indices,
    )

    np.testing.assert_array_equal(actual["state"], expected["state"])
    np.testing.assert_array_equal(
        actual["privileged_state"],
        expected["privileged_state"],
    )
