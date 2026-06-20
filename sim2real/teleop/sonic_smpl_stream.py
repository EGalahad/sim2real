from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.spatial.transform import Rotation as R

from sim2real.utils.math import quat_conjugate, quat_mul, quat_rotate_inverse_numpy


HEADER_SIZE = 1280
SMPL_NUM_JOINTS = 24
SMPL_NUM_POSE_JOINTS = 21
SMPL_WAIST_JOINT_NAMES = ("Spine1", "Spine2", "Spine3")
DEFAULT_HUMAN_JOINTS_INFO_PATH = (
    "checkpoints/sonic_official_release/human/human_joints_info.pkl"
)
_HUMAN_JOINTS_INFO_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}

DEFAULT_STANDING_SMPL_JOINTS = np.asarray(
    [
        [0.00, 0.00, 0.00],   # Pelvis
        [0.00, 0.10, -0.10],  # Left_Hip
        [0.00, -0.10, -0.10], # Right_Hip
        [0.00, 0.00, 0.12],   # Spine1
        [0.00, 0.10, -0.48],  # Left_Knee
        [0.00, -0.10, -0.48], # Right_Knee
        [0.00, 0.00, 0.28],   # Spine2
        [0.00, 0.10, -0.86],  # Left_Ankle
        [0.00, -0.10, -0.86], # Right_Ankle
        [0.00, 0.00, 0.44],   # Spine3
        [0.12, 0.10, -0.92],  # Left_Foot
        [0.12, -0.10, -0.92], # Right_Foot
        [0.00, 0.00, 0.58],   # Neck
        [0.00, 0.14, 0.50],   # Left_Collar
        [0.00, -0.14, 0.50],  # Right_Collar
        [0.00, 0.00, 0.76],   # Head
        [0.00, 0.24, 0.46],   # Left_Shoulder
        [0.00, -0.24, 0.46],  # Right_Shoulder
        [0.02, 0.26, 0.18],   # Left_Elbow
        [0.02, -0.26, 0.18],  # Right_Elbow
        [0.04, 0.25, -0.08],  # Left_Wrist
        [0.04, -0.25, -0.08], # Right_Wrist
        [0.05, 0.25, -0.16],  # Left_Hand
        [0.05, -0.25, -0.16], # Right_Hand
    ],
    dtype=np.float32,
)

ISAAC_G1_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)

SMPL_PARENT_INDICES = (
    -1,
    0,
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    9,
    9,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
    20,
    22,
)


def dtype_code(array: np.ndarray) -> str:
    dtype = np.dtype(array.dtype)
    if dtype == np.float32:
        return "f32"
    if dtype == np.float64:
        return "f64"
    if dtype == np.int64:
        return "i64"
    if dtype == np.int32:
        return "i32"
    if dtype == np.bool_:
        return "bool"
    raise TypeError(f"Unsupported packed dtype: {dtype}")


def pack_pose_message(
    fields: dict[str, np.ndarray],
    *,
    topic: str = "pose",
    version: int = 3,
) -> bytes:
    payload_parts: list[bytes] = []
    field_headers: list[dict[str, object]] = []
    for name, value in fields.items():
        array = np.ascontiguousarray(value)
        field_headers.append(
            {
                "name": str(name),
                "dtype": dtype_code(array),
                "shape": list(array.shape),
            }
        )
        payload_parts.append(array.tobytes(order="C"))

    header = {
        "v": int(version),
        "endian": "le" if sys.byteorder == "little" else "be",
        "count": int(fields["frame_index"].shape[0]) if "frame_index" in fields else -1,
        "fields": field_headers,
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if len(header_bytes) > HEADER_SIZE:
        raise ValueError(
            f"Packed header is {len(header_bytes)} bytes, exceeds {HEADER_SIZE}"
        )
    padded_header = header_bytes + (b"\x00" * (HEADER_SIZE - len(header_bytes)))
    return topic.encode("utf-8") + padded_header + b"".join(payload_parts)


def json_safe_payload(fields: dict[str, np.ndarray]) -> dict[str, object]:
    return {name: np.asarray(value).tolist() for name, value in fields.items()}


def normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).reshape(-1, 4)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    return (quat / np.maximum(norm, 1e-8)).reshape(-1, 4)


def axis_angle_to_quat_wxyz(axis_angle: np.ndarray) -> np.ndarray:
    quat_xyzw = R.from_rotvec(np.asarray(axis_angle, dtype=np.float32).reshape(-1, 3)).as_quat()
    return normalize_quat_wxyz(quat_xyzw[:, [3, 0, 1, 2]].astype(np.float32))


def quat_wxyz_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    quat = normalize_quat_wxyz(quat)
    return R.from_quat(quat[:, [1, 2, 3, 0]]).as_rotvec().astype(np.float32)


def smpl_root_ytoz_up(root_quat_y_up: np.ndarray) -> np.ndarray:
    base_rot = axis_angle_to_quat_wxyz(np.asarray([[np.pi / 2.0, 0.0, 0.0]], dtype=np.float32))
    root_quat_y_up = normalize_quat_wxyz(root_quat_y_up)
    return normalize_quat_wxyz(
        quat_mul(np.broadcast_to(base_rot, root_quat_y_up.shape), root_quat_y_up)
    )


def remove_smpl_base_rot(root_quat_wxyz: np.ndarray) -> np.ndarray:
    base_rot = quat_conjugate(
        np.asarray([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32)
    )
    root_quat_wxyz = normalize_quat_wxyz(root_quat_wxyz)
    return normalize_quat_wxyz(
        quat_mul(root_quat_wxyz, np.broadcast_to(base_rot, root_quat_wxyz.shape))
    )


def load_human_joints_info(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    resolved_path = Path(path).expanduser()
    cache_key = str(resolved_path.resolve()) if resolved_path.exists() else str(resolved_path)
    cached = _HUMAN_JOINTS_INFO_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if not resolved_path.exists():
        raise FileNotFoundError(f"SMPL human joints info not found: {resolved_path}")

    import torch

    info = torch.load(str(resolved_path), map_location="cpu")
    joints = np.asarray(info["J"].detach().cpu().numpy(), dtype=np.float32)
    parents = np.asarray(info["parents_list"], dtype=np.int64)
    if joints.shape != (55, 3):
        raise ValueError(f"Expected human joint rest pose shape (55, 3), got {joints.shape}")
    if parents.shape != (55,):
        raise ValueError(f"Expected 55 human joint parents, got {parents.shape}")

    _HUMAN_JOINTS_INFO_CACHE[cache_key] = (joints, parents)
    return joints, parents


def compute_human_joints_np(
    body_pose: np.ndarray,
    global_orient: np.ndarray,
    human_joints_info_path: str | Path,
) -> np.ndarray:
    rest_joints, parents = load_human_joints_info(human_joints_info_path)
    body_pose = np.asarray(body_pose, dtype=np.float32).reshape(-1, 63)
    global_orient = np.asarray(global_orient, dtype=np.float32).reshape(body_pose.shape[0], 3)
    other_pose = np.zeros((body_pose.shape[0], 99), dtype=np.float32)
    full_pose = np.concatenate([global_orient, body_pose, other_pose], axis=-1)
    rot_mats = R.from_rotvec(full_pose.reshape(-1, 3)).as_matrix().astype(np.float32)
    rot_mats = rot_mats.reshape(body_pose.shape[0], 55, 3, 3)

    rel_joints = np.broadcast_to(rest_joints, (body_pose.shape[0],) + rest_joints.shape).copy()
    rel_joints[:, 1:, :] -= rel_joints[:, parents[1:], :]

    global_rots = np.zeros_like(rot_mats)
    posed_joints = np.zeros((body_pose.shape[0], 55, 3), dtype=np.float32)
    for joint_idx, parent_idx in enumerate(parents):
        if parent_idx < 0:
            global_rots[:, joint_idx] = rot_mats[:, joint_idx]
            posed_joints[:, joint_idx] = rel_joints[:, joint_idx]
        else:
            global_rots[:, joint_idx] = global_rots[:, parent_idx] @ rot_mats[:, joint_idx]
            posed_joints[:, joint_idx] = (
                posed_joints[:, parent_idx]
                + np.einsum("bij,bj->bi", global_rots[:, parent_idx], rel_joints[:, joint_idx])
            )

    output_joint_index = np.concatenate([np.arange(22), np.asarray([39, 54])])
    return posed_joints[:, output_joint_index].astype(np.float32, copy=False)


def official_smpl_frame_from_pose(
    smpl_pose: np.ndarray,
    root_quat_y_up_wxyz: np.ndarray,
    *,
    human_joints_info_path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Match SONIC's process_smpl_joints output for policy observation fields."""
    smpl_pose = np.asarray(smpl_pose, dtype=np.float32).reshape(SMPL_NUM_POSE_JOINTS, 3)
    root_quat_z_up = smpl_root_ytoz_up(root_quat_y_up_wxyz.reshape(1, 4))
    root_axis_angle_z_up = quat_wxyz_to_axis_angle(root_quat_z_up)
    joints = compute_human_joints_np(
        smpl_pose.reshape(1, -1),
        root_axis_angle_z_up,
        human_joints_info_path,
    )[0]
    body_quat_w = remove_smpl_base_rot(root_quat_z_up)[0]
    root_quat = np.broadcast_to(body_quat_w.reshape(1, 4), joints.shape[:-1] + (4,))
    smpl_joints_local = quat_rotate_inverse_numpy(root_quat, joints)
    return smpl_joints_local.astype(np.float32, copy=False), body_quat_w.astype(np.float32)


def pose_dict_to_arrays(
    body_pose_dict: dict[str, list[Any]],
    joint_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    if len(joint_names) < SMPL_NUM_JOINTS:
        raise ValueError(
            f"Expected at least {SMPL_NUM_JOINTS} XRobot body joints, got {len(joint_names)}"
        )

    pos = np.zeros((SMPL_NUM_JOINTS, 3), dtype=np.float32)
    quat_wxyz = np.zeros((SMPL_NUM_JOINTS, 4), dtype=np.float32)
    for idx, name in enumerate(joint_names[:SMPL_NUM_JOINTS]):
        if name not in body_pose_dict:
            raise KeyError(f"Missing XRobot body joint '{name}'")
        raw_pos, raw_quat = body_pose_dict[name]
        pos[idx] = np.asarray(raw_pos, dtype=np.float32).reshape(3)
        quat_wxyz[idx] = np.asarray(raw_quat, dtype=np.float32).reshape(4)
    return pos, normalize_quat_wxyz(quat_wxyz)


def local_axis_angle_from_global_quats(global_quat_wxyz: np.ndarray) -> np.ndarray:
    global_quat_wxyz = normalize_quat_wxyz(global_quat_wxyz)
    global_rot = R.from_quat(global_quat_wxyz[:, [1, 2, 3, 0]])
    local_axis_angle = np.zeros((SMPL_NUM_JOINTS, 3), dtype=np.float32)
    for idx, parent_idx in enumerate(SMPL_PARENT_INDICES):
        if parent_idx < 0:
            local_rot = global_rot[idx]
        else:
            local_rot = global_rot[parent_idx].inv() * global_rot[idx]
        local_axis_angle[idx] = local_rot.as_rotvec().astype(np.float32)
    return local_axis_angle


def apply_local_yaw_offset(
    local_axis_angle: np.ndarray,
    joint_names: Sequence[str],
    target_joint_names: Sequence[str],
    yaw_offset_deg: float,
) -> np.ndarray:
    if abs(float(yaw_offset_deg)) < 1e-6:
        return np.asarray(local_axis_angle, dtype=np.float32)

    out = np.asarray(local_axis_angle, dtype=np.float32).copy()
    name_to_idx = {str(name): idx for idx, name in enumerate(joint_names[:SMPL_NUM_JOINTS])}
    offset_rot = R.from_euler("z", float(yaw_offset_deg), degrees=True)
    for joint_name in target_joint_names:
        joint_idx = name_to_idx.get(str(joint_name))
        if joint_idx is None or joint_idx >= out.shape[0]:
            continue
        local_rot = R.from_rotvec(out[joint_idx])
        out[joint_idx] = (local_rot * offset_rot).as_rotvec().astype(np.float32)
    return out


def root_local_positions(body_pos_w: np.ndarray, root_quat_wxyz: np.ndarray) -> np.ndarray:
    root_pos = body_pos_w[0:1]
    rel = body_pos_w - root_pos
    root_quat = np.broadcast_to(root_quat_wxyz.reshape(1, 4), rel.shape[:-1] + (4,))
    return quat_rotate_inverse_numpy(root_quat, rel).astype(np.float32, copy=False)


def decompose_rotation_aa(
    axis_angle: np.ndarray,
    twist_axis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Decompose axis-angle rotations into twist and swing quaternions.

    Returns quaternions in wxyz order, matching the official SONIC wrist mapping
    call sites.
    """
    axis_angle = np.asarray(axis_angle, dtype=np.float32).reshape(-1, 3)
    twist_axis = np.asarray(twist_axis, dtype=np.float32).reshape(3)
    twist_axis = twist_axis / max(float(np.linalg.norm(twist_axis)), 1e-8)

    quat_xyzw = R.from_rotvec(axis_angle).as_quat()
    quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]].astype(np.float32)
    q_vec = quat_wxyz[:, 1:]
    projected = np.sum(q_vec * twist_axis.reshape(1, 3), axis=1, keepdims=True)
    twist_vec = projected * twist_axis.reshape(1, 3)
    twist = np.concatenate([quat_wxyz[:, 0:1], twist_vec], axis=1)
    twist = normalize_quat_wxyz(twist)
    swing = quat_mul(quat_wxyz, quat_conjugate(twist)).astype(np.float32, copy=False)
    return twist.astype(np.float32, copy=False), swing


def g1_wrist_joint_pos_from_smpl_pose(
    smpl_pose: np.ndarray,
    *,
    joint_names: Sequence[str] = ISAAC_G1_JOINT_NAMES,
    mode: str = "official",
) -> np.ndarray:
    """Map SMPL elbow/wrist rotations to G1 wrist joints.

    This is an equivalent port of the wrist mapping used by SONIC's official
    PICO manager. Only the six wrist joints are filled; all other joints are 0.
    """
    body_pose = np.asarray(smpl_pose, dtype=np.float32).reshape(-1, SMPL_NUM_POSE_JOINTS, 3)
    if mode == "zero":
        joint_pos = np.zeros((body_pose.shape[0], len(joint_names)), dtype=np.float32)
        return joint_pos[0] if np.asarray(smpl_pose).ndim == 2 else joint_pos
    if mode != "official":
        raise ValueError(f"Unsupported wrist joint mapping mode: {mode}")

    smpl_l_elbow = body_pose[:, 17]
    smpl_l_wrist = body_pose[:, 19]
    smpl_r_elbow = body_pose[:, 18]
    smpl_r_wrist = body_pose[:, 20]

    _, l_elbow_swing = decompose_rotation_aa(smpl_l_elbow, np.array([0.0, 1.0, 0.0]))
    _, r_elbow_swing = decompose_rotation_aa(smpl_r_elbow, np.array([0.0, 1.0, 0.0]))

    l_elbow_swing_euler = R.from_quat(l_elbow_swing[:, [1, 2, 3, 0]]).as_euler(
        "XYZ", degrees=False
    )
    r_elbow_swing_euler = R.from_quat(r_elbow_swing[:, [1, 2, 3, 0]]).as_euler(
        "XYZ", degrees=False
    )
    l_wrist_euler = R.from_rotvec(smpl_l_wrist).as_euler("XYZ", degrees=False)
    r_wrist_euler = R.from_rotvec(smpl_r_wrist).as_euler("XYZ", degrees=False)

    values = {
        "left_wrist_roll_joint": l_elbow_swing_euler[:, 0] + l_wrist_euler[:, 0],
        "left_wrist_pitch_joint": l_wrist_euler[:, 1],
        "left_wrist_yaw_joint": l_elbow_swing_euler[:, 2] + l_wrist_euler[:, 2],
        "right_wrist_roll_joint": -(r_elbow_swing_euler[:, 0] + r_wrist_euler[:, 0]),
        "right_wrist_pitch_joint": -r_wrist_euler[:, 1],
        "right_wrist_yaw_joint": r_elbow_swing_euler[:, 2] + r_wrist_euler[:, 2],
    }

    joint_pos = np.zeros((body_pose.shape[0], len(joint_names)), dtype=np.float32)
    name_to_idx = {name: idx for idx, name in enumerate(joint_names)}
    for name, value in values.items():
        if name in name_to_idx:
            joint_pos[:, name_to_idx[name]] = value.astype(np.float32, copy=False)
    return joint_pos[0] if np.asarray(smpl_pose).ndim == 2 else joint_pos


def build_smpl_frame(
    body_pose_dict: dict[str, list[Any]],
    joint_names: Sequence[str],
    *,
    g1_joint_names: Sequence[str] = ISAAC_G1_JOINT_NAMES,
    waist_yaw_offset_deg: float = 0.0,
    wrist_mapping_mode: str = "official",
) -> dict[str, np.ndarray]:
    body_pos_w, body_quat_wxyz = pose_dict_to_arrays(body_pose_dict, joint_names)
    local_axis_angle = local_axis_angle_from_global_quats(body_quat_wxyz)
    local_axis_angle = apply_local_yaw_offset(
        local_axis_angle,
        joint_names,
        SMPL_WAIST_JOINT_NAMES,
        waist_yaw_offset_deg,
    )
    smpl_pose = local_axis_angle[1 : 1 + SMPL_NUM_POSE_JOINTS]
    smpl_joints = root_local_positions(body_pos_w, body_quat_wxyz[0])
    joint_pos = g1_wrist_joint_pos_from_smpl_pose(
        smpl_pose,
        joint_names=tuple(g1_joint_names),
        mode=wrist_mapping_mode,
    )
    return {
        "smpl_pose": smpl_pose.astype(np.float32, copy=False),
        "smpl_joints": smpl_joints.astype(np.float32, copy=False),
        "body_quat_w": body_quat_wxyz[0].astype(np.float32, copy=False),
        "joint_pos": np.asarray(joint_pos, dtype=np.float32),
    }


def smpl_pose_from_xrobot_raw(body_poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute SONIC-style SMPL local pose from raw XRobot SDK body poses.

    ``body_poses`` is expected in the raw XRobot layout
    ``[x, y, z, qx, qy, qz, qw]``. This follows SONIC's official
    ``compute_from_body_poses`` path for the local rotations.
    """
    body_poses = np.asarray(body_poses, dtype=np.float32)
    if body_poses.shape[0] < SMPL_NUM_JOINTS or body_poses.shape[1] < 7:
        raise ValueError(f"Expected raw XRobot body poses [>=24, >=7], got {body_poses.shape}")

    global_quat_wxyz = normalize_quat_wxyz(body_poses[:SMPL_NUM_JOINTS, [6, 3, 4, 5]])
    global_rots = R.from_quat(global_quat_wxyz[:, [1, 2, 3, 0]])
    global_rots = global_rots * R.from_euler("y", 180, degrees=True)

    local_axis_angle = np.zeros((SMPL_NUM_JOINTS, 3), dtype=np.float32)
    for idx, parent_idx in enumerate(SMPL_PARENT_INDICES):
        if parent_idx < 0:
            local_rot = global_rots[idx]
        else:
            local_rot = global_rots[parent_idx].inv() * global_rots[idx]
        local_axis_angle[idx] = local_rot.as_rotvec().astype(np.float32)

    root_quat_xyzw = global_rots[0].as_quat()
    root_quat_wxyz = root_quat_xyzw[[3, 0, 1, 2]].astype(np.float32)
    return local_axis_angle[1 : 1 + SMPL_NUM_POSE_JOINTS], root_quat_wxyz


def root_local_positions_from_xrobot_raw(
    body_poses: np.ndarray,
    root_quat_wxyz: np.ndarray,
) -> np.ndarray:
    body_poses = np.asarray(body_poses, dtype=np.float32)
    if body_poses.shape[0] < SMPL_NUM_JOINTS or body_poses.shape[1] < 3:
        raise ValueError(f"Expected raw XRobot body poses [>=24, >=3], got {body_poses.shape}")

    body_pos_w = body_poses[:SMPL_NUM_JOINTS, :3].astype(np.float32, copy=False)
    return root_local_positions(body_pos_w, root_quat_wxyz)


def build_smpl_frame_from_xrobot_raw(
    body_poses: np.ndarray,
    body_pose_dict_for_joints: dict[str, list[Any]],
    joint_names: Sequence[str],
    *,
    g1_joint_names: Sequence[str] = ISAAC_G1_JOINT_NAMES,
    waist_yaw_offset_deg: float = 0.0,
    wrist_mapping_mode: str = "official",
    human_joints_info_path: str | Path | None = DEFAULT_HUMAN_JOINTS_INFO_PATH,
) -> dict[str, np.ndarray]:
    smpl_pose, root_quat_wxyz = smpl_pose_from_xrobot_raw(body_poses)
    if abs(float(waist_yaw_offset_deg)) >= 1e-6:
        local_axis_angle = np.zeros((SMPL_NUM_JOINTS, 3), dtype=np.float32)
        local_axis_angle[1 : 1 + SMPL_NUM_POSE_JOINTS] = smpl_pose
        local_axis_angle = apply_local_yaw_offset(
            local_axis_angle,
            joint_names,
            SMPL_WAIST_JOINT_NAMES,
            waist_yaw_offset_deg,
        )
        smpl_pose = local_axis_angle[1 : 1 + SMPL_NUM_POSE_JOINTS]

    body_quat_w = root_quat_wxyz.astype(np.float32, copy=False)
    if human_joints_info_path:
        try:
            smpl_joints, body_quat_w = official_smpl_frame_from_pose(
                smpl_pose,
                root_quat_wxyz,
                human_joints_info_path=human_joints_info_path,
            )
        except (FileNotFoundError, KeyError, ValueError, ImportError, ModuleNotFoundError):
            smpl_joints = root_local_positions_from_xrobot_raw(body_poses, root_quat_wxyz)
    else:
        try:
            smpl_joints = root_local_positions_from_xrobot_raw(body_poses, root_quat_wxyz)
        except ValueError:
            body_pos_w, _body_quat_wxyz = pose_dict_to_arrays(body_pose_dict_for_joints, joint_names)
            smpl_joints = root_local_positions(body_pos_w, root_quat_wxyz)
    joint_pos = g1_wrist_joint_pos_from_smpl_pose(
        smpl_pose,
        joint_names=tuple(g1_joint_names),
        mode=wrist_mapping_mode,
    )
    return {
        "smpl_pose": smpl_pose.astype(np.float32, copy=False),
        "smpl_joints": smpl_joints.astype(np.float32, copy=False),
        "body_quat_w": body_quat_w.astype(np.float32, copy=False),
        "joint_pos": np.asarray(joint_pos, dtype=np.float32),
    }
