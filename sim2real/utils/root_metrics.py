"""Root endpoint error after fixed initial XY/yaw alignment."""

import numpy as np


ROOT_ERROR_CONVENTION = "initial_xy_yaw_aligned_endpoint_xyz_v1"


def root_endpoint_index(motion_t: np.ndarray, motion_length: int) -> int:
    """Return the last valid same-time endpoint in a recorded rollout.

    Full-motion recordings include the dataset terminal sentinel at
    ``motion_length - 1``. That sample may already contain the next clip's root
    and is not a motion pose. Early-terminated recordings have no sentinel and
    use their last recorded frame.
    """
    frames = np.asarray(motion_t)
    if frames.ndim != 1 or not len(frames):
        raise ValueError("motion_t must be a nonempty 1D array")
    if not np.issubdtype(frames.dtype, np.integer):
        raise ValueError("motion_t must contain integer frame indices")
    if int(motion_length) < 2:
        raise ValueError("motion_length must be at least 2")
    valid = np.flatnonzero(frames < int(motion_length) - 1)
    if not len(valid):
        raise ValueError("Full-motion recording has no frame before its terminal sentinel")
    return int(valid[-1])


def align_root_positions(
    robot_pos: np.ndarray,
    robot_quat: np.ndarray,
    motion_pos: np.ndarray,
    motion_quat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned robot positions and reference positions in motion world frame.

    Inputs are matching, nonempty (T, 3) positions and (T, 4) wxyz
    quaternions. Align only initial XY and yaw; preserve absolute world Z.
    The endpoint is the last supplied same-time frame, which may precede
    motion completion. Initial roll/pitch never rotate the displacement.
    """
    arrays = []
    for name, value, width in (
        ("robot_pos", robot_pos, 3),
        ("robot_quat", robot_quat, 4),
        ("motion_pos", motion_pos, 3),
        ("motion_quat", motion_quat, 4),
    ):
        array = np.asarray(value, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != width or not len(array):
            raise ValueError(f"{name} must have nonempty shape (T, {width})")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must contain only finite values")
        arrays.append(array)
    robot_pos, robot_quat, motion_pos, motion_quat = arrays
    if len({len(array) for array in arrays}) != 1:
        raise ValueError("Root trajectories must have equal frame counts")

    initial_yaws = []
    for quat in (robot_quat, motion_quat):
        scales = np.max(np.abs(quat), axis=1)
        if np.any(scales == 0):
            raise ValueError("Root quaternions must have nonzero norm")
        # Scale first so even large finite quaternion magnitudes normalize safely.
        initial = quat[0] / scales[0]
        w, x, y, z = initial / np.linalg.norm(initial)
        initial_yaws.append(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    yaw_delta = initial_yaws[1] - initial_yaws[0]
    c, s = np.cos(yaw_delta), np.sin(yaw_delta)
    aligned = robot_pos.copy()
    aligned[:, :2] = (
        (robot_pos[:, :2] - robot_pos[0, :2]) @ np.array([[c, s], [-s, c]])
        + motion_pos[0, :2]
    )
    return aligned, motion_pos


def root_final_error(
    robot_pos: np.ndarray,
    robot_quat: np.ndarray,
    motion_pos: np.ndarray,
    motion_quat: np.ndarray,
    *,
    endpoint_index: int = -1,
) -> np.ndarray:
    """Signed XYZ error at the last supplied same-time frame, in metres.

    Use fixed initial XY/yaw alignment and absolute world Z. Last supplied
    frame is not necessarily the full motion endpoint (e.g. early failure).
    """
    aligned, reference = align_root_positions(robot_pos, robot_quat, motion_pos, motion_quat)
    if not -len(aligned) <= endpoint_index < len(aligned):
        raise IndexError("endpoint_index is outside the root trajectory")
    return aligned[endpoint_index] - reference[endpoint_index]
