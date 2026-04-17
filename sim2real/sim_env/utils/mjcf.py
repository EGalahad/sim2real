from __future__ import annotations

import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

import mujoco
import numpy as np
from loguru import logger

from sim2real.config.robots.base import RobotCfg


VIEWER_VISUAL_XML = """\
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.1 0.1 0.1" specular="0.9 0.9 0.9"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="-140" elevation="-20"/>
  </visual>
"""

VIEWER_ASSET_XML = """\
    <texture type="skybox" builtin="flat" rgb1="0 0 0" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
"""

VIEWER_WORLDBODY_XML = """\
    <light pos="1 0 3.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
"""


def _inject_floor_scene_xml(xml_text: str) -> str:
    if "<visual>" not in xml_text:
        insertion_point = xml_text.find("<asset>")
        if insertion_point < 0:
            raise ValueError("Expected <asset> block in MJCF")
        xml_text = xml_text[:insertion_point] + VIEWER_VISUAL_XML + xml_text[insertion_point:]

    asset_close = xml_text.find("</asset>")
    if asset_close < 0:
        raise ValueError("Expected </asset> block in MJCF")
    xml_text = xml_text[:asset_close] + VIEWER_ASSET_XML + xml_text[asset_close:]

    worldbody_close = xml_text.find("</worldbody>")
    if worldbody_close < 0:
        raise ValueError("Expected </worldbody> block in MJCF")
    return xml_text[:worldbody_close] + VIEWER_WORLDBODY_XML + xml_text[worldbody_close:]


@contextmanager
def _temp_scene_with_floor(mjcf_path: Path) -> Path:
    source_path = Path(mjcf_path).expanduser()
    if not source_path.is_absolute():
        source_path = source_path.absolute()
    xml_text = source_path.read_text(encoding="utf-8")
    viewer_xml = _inject_floor_scene_xml(xml_text)

    temp_path: Path | None = None
    staging_dir: Path | None = None
    try:
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".xml",
                prefix=".sim_scene_",
                dir=source_path.parent,
                delete=False,
                encoding="utf-8",
            ) as tmp:
                tmp.write(viewer_xml)
                temp_path = Path(tmp.name)
        except OSError:
            staging_dir = Path(tempfile.mkdtemp(prefix=".sim_scene_"))
            for child in source_path.parent.iterdir():
                if child.name == source_path.name:
                    continue
                os.symlink(child, staging_dir / child.name, target_is_directory=child.is_dir())
            temp_path = staging_dir / source_path.name
            temp_path.write_text(viewer_xml, encoding="utf-8")

        yield temp_path
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass
        if staging_dir is not None:
            shutil.rmtree(staging_dir, ignore_errors=True)


def _add_joint_motor_actuator(
    spec: mujoco.MjSpec,
    *,
    joint_name: str,
    effort_limit: float,
    armature: float = 0.0,
    frictionloss: float = 0.0,
    gear: float = 1.0,
) -> None:
    actuator = spec.add_actuator(name=joint_name, target=joint_name)
    actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
    actuator.dyntype = mujoco.mjtDyn.mjDYN_NONE
    actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
    actuator.biastype = mujoco.mjtBias.mjBIAS_NONE
    actuator.gear[0] = float(gear)
    actuator.forcelimited = True
    actuator.forcerange[:] = np.array([-effort_limit, effort_limit], dtype=np.float64)
    actuator.ctrllimited = True
    actuator.ctrlrange[:] = np.array([-effort_limit, effort_limit], dtype=np.float64)
    spec.joint(joint_name).armature = float(armature)
    spec.joint(joint_name).frictionloss = float(frictionloss)


def ensure_joint_motor_actuators(
    spec: mujoco.MjSpec,
    robot_cfg: RobotCfg,
) -> list[str]:
    joint_names_in_spec = {joint.name for joint in spec.joints}
    actuator_names_in_spec = {actuator.name for actuator in spec.actuators}
    added_joint_names: list[str] = []

    for joint_name in robot_cfg.joint_names:
        if joint_name not in joint_names_in_spec:
            continue
        if joint_name in actuator_names_in_spec:
            continue

        effort_limit = robot_cfg.joint_effort_limit.get(joint_name)
        if effort_limit is None:
            raise KeyError(
                f"Missing joint_effort_limit for joint {joint_name!r}; "
                "cannot auto-create MuJoCo motor actuator."
            )
        armature = float(robot_cfg.joint_armature.get(joint_name, 0.0))
        frictionloss = float(robot_cfg.joint_frictionloss.get(joint_name, 0.0))

        _add_joint_motor_actuator(
            spec,
            joint_name=joint_name,
            effort_limit=float(effort_limit),
            armature=armature,
            frictionloss=frictionloss,
        )
        actuator_names_in_spec.add(joint_name)
        added_joint_names.append(joint_name)

    return added_joint_names


def load_sim_model(
    robot_cfg: RobotCfg,
    *,
    ground_rgb: tuple[float, float, float] = (0.2, 0.3, 0.4),
) -> mujoco.MjModel:
    mjcf_path = robot_cfg.resolve_mjcf_path()
    if ground_rgb != (0.2, 0.3, 0.4):
        logger.warning("load_sim_model currently ignores non-default ground_rgb={}", ground_rgb)
    with _temp_scene_with_floor(mjcf_path) as scene_mjcf_path:
        spec = mujoco.MjSpec.from_file(str(scene_mjcf_path))
        added_joint_names = ensure_joint_motor_actuators(spec, robot_cfg)
        if added_joint_names:
            logger.info(
                "Added {} missing motor actuators to sim model: {}",
                len(added_joint_names),
                ", ".join(added_joint_names),
            )
        return spec.compile()
