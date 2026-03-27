from __future__ import annotations

from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]

G1_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

G1_BODY_NAMES: tuple[str, ...] = (
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "left_toe_link",
    "pelvis_contour_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "right_toe_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "head_link",
    "head_mocap",
    "imu_in_torso",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "left_rubber_hand",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
    "right_rubber_hand",
)

G1_QPOS_ROOT_SIZE = 7
G1_QPOS_SIZE = G1_QPOS_ROOT_SIZE + len(G1_JOINT_NAMES)
G1_PUBLISH_HZ = 50.0

SMPLX_T_NS_KEY = "smplx_t_ns"
PUBLISH_T_NS_KEY = "publish_t_ns"
SEQ_KEY = "seq"
JOINT_NAMES_KEY = "joint_names"
BODY_NAMES_KEY = "body_names"
JOINT_POS_KEY = "joint_pos"
BODY_POS_W_KEY = "body_pos_w"
BODY_QUAT_W_KEY = "body_quat_w"

ROOT_POS_SLICE = slice(0, 3)
ROOT_QUAT_SLICE = slice(3, 7)
JOINT_POS_SLICE = slice(7, 7 + len(G1_JOINT_NAMES))

G1_MJCF_PATH = (
    PROJECT_ROOT / "sim2real" / "teleop" / "GMR" / "assets" / "unitree_g1" / "g1_mocap_29dof.xml"
)


def normalize_name_list(values: Sequence[object] | None) -> list[str] | None:
    if values is None:
        return None
    return [str(value) for value in values]


def validate_name_order(
    expected_names: Sequence[str],
    actual_names: Sequence[object] | None,
    *,
    label: str,
) -> bool:
    normalized_actual = normalize_name_list(actual_names)
    if normalized_actual is None:
        print(f"[teleop] missing {label} in payload")
        return False

    expected_list = list(expected_names)
    if normalized_actual == expected_list:
        return True

    if sorted(normalized_actual) == sorted(expected_list):
        print(
            "[teleop] "
            f"{label} order mismatch; expected canonical order from sim2real.utils.robot_defs"
        )
    else:
        missing = [name for name in expected_list if name not in normalized_actual]
        extra = [name for name in normalized_actual if name not in expected_list]
        print(f"[teleop] {label} mismatch; missing={missing} extra={extra}")
    return False
