from __future__ import annotations

from sim2real.config.robots.base import RobotCfg


def reflected_inertia_from_two_stage_planetary(
    rotor_inertia: tuple[float, float, float],
    gear_ratio: tuple[float, float, float],
) -> float:
    assert gear_ratio[0] == 1
    r1 = rotor_inertia[0] * (gear_ratio[1] * gear_ratio[2]) ** 2
    r2 = rotor_inertia[1] * gear_ratio[2] ** 2
    r3 = rotor_inertia[2]
    return r1 + r2 + r3


ROTOR_INERTIAS_5020 = (0.139e-4, 0.017e-4, 0.169e-4)
GEARS_5020 = (1, 1 + (46 / 18), 1 + (56 / 16))

ROTOR_INERTIAS_7520_14 = (0.489e-4, 0.098e-4, 0.533e-4)
GEARS_7520_14 = (1, 4.5, 1 + (48 / 22))

ROTOR_INERTIAS_7520_22 = (0.489e-4, 0.109e-4, 0.738e-4)
GEARS_7520_22 = (1, 4.5, 5)

ROTOR_INERTIAS_4010 = (0.068e-4, 0.0, 0.0)
GEARS_4010 = (1, 5, 5)

ROTOR_INERTIAS_5010 = (0.084e-4, 0.015e-4, 0.068e-4)
GEARS_5010 = (1, 4, 4)

ARMATURE_5020 = reflected_inertia_from_two_stage_planetary(ROTOR_INERTIAS_5020, GEARS_5020)
ARMATURE_7520_14 = reflected_inertia_from_two_stage_planetary(ROTOR_INERTIAS_7520_14, GEARS_7520_14)
ARMATURE_7520_22 = reflected_inertia_from_two_stage_planetary(ROTOR_INERTIAS_7520_22, GEARS_7520_22)
ARMATURE_4010 = reflected_inertia_from_two_stage_planetary(ROTOR_INERTIAS_4010, GEARS_4010)
ARMATURE_5010 = reflected_inertia_from_two_stage_planetary(ROTOR_INERTIAS_5010, GEARS_5010)

DEFAULT_JOINT_FRICTIONLOSS = 0.01


def _g1_mode_15_joint_armature() -> dict[str, float]:
    armature: dict[str, float] = {}
    for side in ("left", "right"):
        armature[f"{side}_hip_pitch_joint"] = ARMATURE_7520_22
        armature[f"{side}_hip_roll_joint"] = ARMATURE_7520_22
        armature[f"{side}_hip_yaw_joint"] = ARMATURE_7520_14
        armature[f"{side}_knee_joint"] = ARMATURE_7520_22
        armature[f"{side}_ankle_pitch_joint"] = 2.0 * ARMATURE_5020
        armature[f"{side}_ankle_roll_joint"] = 2.0 * ARMATURE_5020
        armature[f"{side}_shoulder_pitch_joint"] = ARMATURE_5020
        armature[f"{side}_shoulder_roll_joint"] = ARMATURE_5020
        armature[f"{side}_shoulder_yaw_joint"] = ARMATURE_5020
        armature[f"{side}_elbow_joint"] = ARMATURE_5020
        armature[f"{side}_wrist_roll_joint"] = ARMATURE_5020
        armature[f"{side}_wrist_pitch_joint"] = ARMATURE_5010
        armature[f"{side}_wrist_yaw_joint"] = ARMATURE_5010
    armature["waist_yaw_joint"] = ARMATURE_7520_14
    armature["waist_roll_joint"] = 2.0 * ARMATURE_5020
    armature["waist_pitch_joint"] = 2.0 * ARMATURE_5020
    return armature


def _g1_joint_frictionloss(joint_armature: dict[str, float]) -> dict[str, float]:
    return {joint_name: DEFAULT_JOINT_FRICTIONLOSS for joint_name in joint_armature}


G1_MODE_15_JOINT_ARMATURE = _g1_mode_15_joint_armature()
G1_MODE_15_JOINT_FRICTIONLOSS = _g1_joint_frictionloss(G1_MODE_15_JOINT_ARMATURE)


G1_CFG = RobotCfg(
    name="g1",
    joint_names=(
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
    ),
    body_names=(
        "pelvis",
        "left_hip_pitch_link",
        "left_hip_roll_link",
        "left_hip_yaw_link",
        "left_knee_link",
        "left_ankle_pitch_link",
        "left_ankle_roll_link",
        "left_toe_link",
        # "pelvis_contour_link",
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
        # "head_link",
        # "head_mocap",
        # "imu_in_torso",
        "left_shoulder_pitch_link",
        "left_shoulder_roll_link",
        "left_shoulder_yaw_link",
        "left_elbow_link",
        "left_wrist_roll_link",
        "left_wrist_pitch_link",
        "left_wrist_yaw_link",
        # "left_rubber_hand",
        "right_shoulder_pitch_link",
        "right_shoulder_roll_link",
        "right_shoulder_yaw_link",
        "right_elbow_link",
        "right_wrist_roll_link",
        "right_wrist_pitch_link",
        "right_wrist_yaw_link",
        # "right_rubber_hand",
    ),
    joint_pos_lower_limit={
        "left_hip_pitch_joint": -2.5307,
        "right_hip_pitch_joint": -2.5307,
        "waist_yaw_joint": -2.6180,
        "waist_pitch_joint": -0.52,
        "waist_roll_joint": -0.52,
        "left_hip_roll_joint": -0.5236,
        "right_hip_roll_joint": -2.9671,
        "left_hip_yaw_joint": -2.7576,
        "right_hip_yaw_joint": -2.7576,
        "left_knee_joint": -0.0873,
        "right_knee_joint": -0.0873,
        "left_shoulder_pitch_joint": -3.0892,
        "right_shoulder_pitch_joint": -3.0892,
        "left_ankle_pitch_joint": -0.8727,
        "right_ankle_pitch_joint": -0.8727,
        "left_shoulder_roll_joint": -1.5882,
        "right_shoulder_roll_joint": -2.2515,
        "left_ankle_roll_joint": -0.2618,
        "right_ankle_roll_joint": -0.2618,
        "left_shoulder_yaw_joint": -2.6180,
        "right_shoulder_yaw_joint": -2.6180,
        "left_elbow_joint": -1.0472,
        "right_elbow_joint": -1.0472,
        "left_wrist_roll_joint": -1.9722,
        "right_wrist_roll_joint": -1.9722,
        "left_wrist_pitch_joint": -1.6144,
        "right_wrist_pitch_joint": -1.6144,
        "left_wrist_yaw_joint": -1.6144,
        "right_wrist_yaw_joint": -1.6144,
    },
    joint_pos_upper_limit={
        "left_hip_pitch_joint": 2.8798,
        "right_hip_pitch_joint": 2.8798,
        "waist_yaw_joint": 2.6180,
        "waist_pitch_joint": 0.52,
        "waist_roll_joint": 0.52,
        "left_hip_roll_joint": 2.9671,
        "right_hip_roll_joint": 0.5236,
        "left_hip_yaw_joint": 2.7576,
        "right_hip_yaw_joint": 2.7576,
        "left_knee_joint": 2.8798,
        "right_knee_joint": 2.8798,
        "left_shoulder_pitch_joint": 2.6704,
        "right_shoulder_pitch_joint": 2.6704,
        "left_ankle_pitch_joint": 0.5236,
        "right_ankle_pitch_joint": 0.5236,
        "left_shoulder_roll_joint": 2.2515,
        "right_shoulder_roll_joint": 1.5882,
        "left_ankle_roll_joint": 0.2618,
        "right_ankle_roll_joint": 0.2618,
        "left_shoulder_yaw_joint": 2.6180,
        "right_shoulder_yaw_joint": 2.6180,
        "left_elbow_joint": 2.0944,
        "right_elbow_joint": 2.0944,
        "left_wrist_roll_joint": 1.9722,
        "right_wrist_roll_joint": 1.9722,
        "left_wrist_pitch_joint": 1.6144,
        "right_wrist_pitch_joint": 1.6144,
        "left_wrist_yaw_joint": 1.6144,
        "right_wrist_yaw_joint": 1.6144,
    },
    joint_velocity_limit={
        "left_hip_pitch_joint": 32.0,
        "right_hip_pitch_joint": 32.0,
        "waist_yaw_joint": 32.0,
        "left_hip_roll_joint": 32.0,
        "right_hip_roll_joint": 32.0,
        "left_hip_yaw_joint": 32.0,
        "right_hip_yaw_joint": 32.0,
        "left_knee_joint": 20.0,
        "right_knee_joint": 20.0,
        "left_shoulder_pitch_joint": 37.0,
        "right_shoulder_pitch_joint": 37.0,
        "left_ankle_pitch_joint": 37.0,
        "right_ankle_pitch_joint": 37.0,
        "left_shoulder_roll_joint": 37.0,
        "right_shoulder_roll_joint": 37.0,
        "left_ankle_roll_joint": 37.0,
        "right_ankle_roll_joint": 37.0,
        "left_shoulder_yaw_joint": 37.0,
        "right_shoulder_yaw_joint": 37.0,
        "left_elbow_joint": 37.0,
        "right_elbow_joint": 37.0,
        "left_wrist_roll_joint": 37.0,
        "right_wrist_roll_joint": 37.0,
        "left_wrist_pitch_joint": 22.0,
        "right_wrist_pitch_joint": 22.0,
        "left_wrist_yaw_joint": 22.0,
        "right_wrist_yaw_joint": 22.0,
    },
    joint_effort_limit={
        "left_hip_pitch_joint": 88.0,
        "left_hip_roll_joint": 88.0,
        "left_hip_yaw_joint": 88.0,
        "left_knee_joint": 139.0,
        "left_ankle_pitch_joint": 50.0,
        "left_ankle_roll_joint": 50.0,
        "right_hip_pitch_joint": 88.0,
        "right_hip_roll_joint": 88.0,
        "right_hip_yaw_joint": 88.0,
        "right_knee_joint": 139.0,
        "right_ankle_pitch_joint": 50.0,
        "right_ankle_roll_joint": 50.0,
        "waist_yaw_joint": 88.0,
        "waist_roll_joint": 50.0,
        "waist_pitch_joint": 50.0,
        "left_shoulder_pitch_joint": 25.0,
        "left_shoulder_roll_joint": 25.0,
        "left_shoulder_yaw_joint": 25.0,
        "left_elbow_joint": 25.0,
        "left_wrist_roll_joint": 25.0,
        "left_wrist_pitch_joint": 5.0,
        "left_wrist_yaw_joint": 5.0,
        "right_shoulder_pitch_joint": 25.0,
        "right_shoulder_roll_joint": 25.0,
        "right_shoulder_yaw_joint": 25.0,
        "right_elbow_joint": 25.0,
        "right_wrist_roll_joint": 25.0,
        "right_wrist_pitch_joint": 5.0,
        "right_wrist_yaw_joint": 5.0,
    },
    joint_armature=G1_MODE_15_JOINT_ARMATURE,
    joint_frictionloss=G1_MODE_15_JOINT_FRICTIONLOSS,
    mjcf_path="hf://elijahgalahad/g1_xmls@main/g1-mode_13_15.xml",
    default_qpos=(
        0.0,
        0.0,
        0.8,
        1.0,
        0.0,
        0.0,
        0.0,
        -0.2,
        0.0,
        0.0,
        0.4,
        -0.2,
        0.0,
        -0.2,
        0.0,
        0.0,
        0.4,
        -0.2,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.4,
        0.0,
        1.2,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.4,
        0.0,
        1.2,
        0.0,
        0.0,
        0.0,
    ),
    publish_hz=50.0,
    domain_id=0,
    interface="eth0",
    mocap_ip="localhost",
    viewer_track_body_names=("pelvis",),
    elastic_band_attach_body_names=("torso_link", "base_link"),
)
