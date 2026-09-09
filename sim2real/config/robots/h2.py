from __future__ import annotations

import xml.etree.ElementTree as ET

from sim2real.config.robots.base import RobotCfg, resolve_asset_reference


H2_MJCF_PATH = resolve_asset_reference(
    "hf://elijahgalahad/h2_model@beb532e8717b99816b93baace0c649a599538715/h2.xml"
)

DEFAULT_JOINT_FRICTIONLOSS = 0.01

ROTOR_INERTIAS_5020 = (0.139e-4, 0.017e-4, 0.169e-4)
GEARS_5020 = (1, 1 + (46 / 18), 1 + (56 / 16))
ROTOR_INERTIAS_7520_14 = (0.489e-4, 0.098e-4, 0.533e-4)
GEARS_7520_14 = (1, 4.5, 1 + (48 / 22))
ROTOR_INERTIAS_7520_22 = (0.489e-4, 0.109e-4, 0.738e-4)
GEARS_7520_22 = (1, 4.5, 5)
ROTOR_INERTIAS_4010 = (0.068e-4, 0.0, 0.0)
GEARS_4010 = (1, 5, 5)


def _reflected_inertia(rotor_inertias: tuple[float, float, float], gears: tuple[float, float, float]) -> float:
    return (
        rotor_inertias[0] * (gears[1] * gears[2]) ** 2
        + rotor_inertias[1] * gears[2] ** 2
        + rotor_inertias[2]
    )


ARMATURE_5020 = _reflected_inertia(ROTOR_INERTIAS_5020, GEARS_5020)
ARMATURE_7520_14 = _reflected_inertia(ROTOR_INERTIAS_7520_14, GEARS_7520_14)
ARMATURE_7520_22 = _reflected_inertia(ROTOR_INERTIAS_7520_22, GEARS_7520_22)
ARMATURE_4010 = _reflected_inertia(ROTOR_INERTIAS_4010, GEARS_4010)


def _parse_h2_xml() -> tuple[tuple[str, ...], tuple[str, ...], dict[str, tuple[float, float]]]:
    tree = ET.parse(H2_MJCF_PATH)
    root = tree.getroot()
    body_names: list[str] = []
    joint_names: list[str] = []
    joint_ranges: dict[str, tuple[float, float]] = {}
    for elem in root.iter():
        if elem.tag == "body":
            name = elem.attrib.get("name")
            if name and name not in body_names:
                body_names.append(name)
        elif elem.tag == "joint":
            name = elem.attrib.get("name")
            if not name:
                continue
            joint_names.append(name)
            range_text = elem.attrib.get("range")
            if range_text is None:
                raise ValueError(f"Joint {name} is missing range in {H2_MJCF_PATH}")
            lower_text, upper_text = range_text.split()
            joint_ranges[name] = (float(lower_text), float(upper_text))
    return tuple(joint_names), tuple(body_names), joint_ranges


H2_JOINT_NAMES, H2_BODY_NAMES, H2_JOINT_RANGES = _parse_h2_xml()


def _value_by_joint_name(joint_name: str, *, leg: float, ankle_roll: float, ankle_pitch: float, waist_yaw: float,
                         waist_roll_pitch: float, shoulder_pitch: float, upper_arm: float, wrist_pitch_yaw: float,
                         head: float) -> float:
    if "_hip_pitch_joint" in joint_name or "_hip_roll_joint" in joint_name or "_knee_joint" in joint_name:
        return leg
    if "_hip_yaw_joint" in joint_name:
        return waist_yaw
    if "_ankle_roll_joint" in joint_name:
        return ankle_roll
    if "_ankle_pitch_joint" in joint_name:
        return ankle_pitch
    if joint_name == "waist_yaw_joint":
        return waist_yaw
    if joint_name in {"waist_roll_joint", "waist_pitch_joint"}:
        return waist_roll_pitch
    if "_shoulder_pitch_joint" in joint_name:
        return shoulder_pitch
    if any(token in joint_name for token in ("_shoulder_roll_joint", "_shoulder_yaw_joint", "_elbow_joint", "_wrist_roll_joint")):
        return upper_arm
    if any(token in joint_name for token in ("_wrist_pitch_joint", "_wrist_yaw_joint")):
        return wrist_pitch_yaw
    if joint_name in {"head_pitch_joint", "head_yaw_joint"}:
        return head
    raise KeyError(f"Unhandled H2 joint {joint_name}")


def _joint_map(*, leg: float, ankle_roll: float, ankle_pitch: float, waist_yaw: float, waist_roll_pitch: float,
               shoulder_pitch: float, upper_arm: float, wrist_pitch_yaw: float, head: float) -> dict[str, float]:
    return {
        joint_name: _value_by_joint_name(
            joint_name,
            leg=leg,
            ankle_roll=ankle_roll,
            ankle_pitch=ankle_pitch,
            waist_yaw=waist_yaw,
            waist_roll_pitch=waist_roll_pitch,
            shoulder_pitch=shoulder_pitch,
            upper_arm=upper_arm,
            wrist_pitch_yaw=wrist_pitch_yaw,
            head=head,
        )
        for joint_name in H2_JOINT_NAMES
    }


def _default_joint_pos(joint_name: str) -> float:
    if "_hip_pitch_joint" in joint_name:
        return -0.1
    if "_knee_joint" in joint_name:
        return 0.3
    if "_ankle_pitch_joint" in joint_name:
        return -0.2
    if "_shoulder_pitch_joint" in joint_name:
        return 0.35
    if joint_name == "left_shoulder_roll_joint":
        return 0.18
    if joint_name == "right_shoulder_roll_joint":
        return -0.18
    if "_elbow_joint" in joint_name:
        return 0.87
    return 0.0


SAFE_JOINT_KP = {
    ".*_hip_(pitch|roll)_joint": 150.0,
    ".*_hip_yaw_joint": 100.0,
    ".*_knee_joint": 150.0,
    ".*_ankle_.*_joint": 60.0,
    "waist_yaw_joint": 100.0,
    "waist_(roll|pitch)_joint": 120.0,
    ".*_shoulder_pitch_joint": 30.0,
    ".*_(shoulder_roll|shoulder_yaw|elbow|wrist_roll)_joint": 20.0,
    ".*_(wrist_pitch|wrist_yaw)_joint": 10.0,
    "head_.*": 10.0,
}
SAFE_JOINT_KD = {
    ".*_hip_(pitch|roll)_joint": 10.0,
    ".*_hip_yaw_joint": 5.0,
    ".*_knee_joint": 10.0,
    ".*_ankle_.*_joint": 3.0,
    "waist_yaw_joint": 5.0,
    "waist_(roll|pitch)_joint": 8.0,
    ".*_shoulder_pitch_joint": 1.5,
    ".*_(shoulder_roll|shoulder_yaw|elbow|wrist_roll)_joint": 1.0,
    ".*_(wrist_pitch|wrist_yaw)_joint": 0.5,
    "head_.*": 0.5,
}


H2_CFG = RobotCfg(
    name="h2",
    joint_names=H2_JOINT_NAMES,
    body_names=H2_BODY_NAMES,
    joint_pos_lower_limit={joint_name: joint_range[0] for joint_name, joint_range in H2_JOINT_RANGES.items()},
    joint_pos_upper_limit={joint_name: joint_range[1] for joint_name, joint_range in H2_JOINT_RANGES.items()},
    joint_velocity_limit=_joint_map(
        leg=32.0,
        ankle_roll=37.0,
        ankle_pitch=37.0,
        waist_yaw=32.0,
        waist_roll_pitch=32.0,
        shoulder_pitch=37.0,
        upper_arm=37.0,
        wrist_pitch_yaw=22.0,
        head=22.0,
    ),
    joint_effort_limit=_joint_map(
        leg=360.0,
        ankle_roll=19.0,
        ankle_pitch=66.88,
        waist_yaw=120.0,
        waist_roll_pitch=180.0,
        shoulder_pitch=120.0,
        upper_arm=54.0,
        wrist_pitch_yaw=25.0,
        head=50.0,
    ),
    safe_joint_kp=SAFE_JOINT_KP,
    safe_joint_kd=SAFE_JOINT_KD,
    joint_armature=_joint_map(
        leg=ARMATURE_7520_22,
        ankle_roll=ARMATURE_5020,
        ankle_pitch=2.0 * ARMATURE_5020,
        waist_yaw=ARMATURE_7520_14,
        waist_roll_pitch=2.0 * ARMATURE_5020,
        shoulder_pitch=ARMATURE_5020,
        upper_arm=ARMATURE_5020,
        wrist_pitch_yaw=ARMATURE_4010,
        head=ARMATURE_5020,
    ),
    joint_frictionloss={joint_name: DEFAULT_JOINT_FRICTIONLOSS for joint_name in H2_JOINT_NAMES},
    mjcf_path=H2_MJCF_PATH,
    default_qpos=(
        0.0,
        0.0,
        1.2,
        1.0,
        0.0,
        0.0,
        0.0,
        *(_default_joint_pos(joint_name) for joint_name in H2_JOINT_NAMES),
    ),
    root_joint_names=("floating_base_joint",),
    viewer_track_body_names=("pelvis", "torso_link"),
    elastic_band_attach_body_names=("torso_link", "pelvis"),
)


__all__ = ["H2_CFG", "H2_BODY_NAMES", "H2_JOINT_NAMES", "H2_MJCF_PATH"]
