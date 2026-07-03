from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from mjhub import AssetReference, resolve_asset_reference
except ImportError:
    from mjhub import MjcfReference as AssetReference
    from mjhub import resolve_mjcf_reference as resolve_asset_reference
from sim2real.utils.common import PORTS, UNITREE_LEGGED_CONST


PROJECT_ROOT = Path(__file__).resolve().parents[3]

SMPLX_T_NS_KEY = "smplx_t_ns"
PICO_RECV_TIME_NS_KEY = "pico_recv_time_ns"
PUBLISH_T_NS_KEY = "publish_t_ns"
SEQ_KEY = "seq"
MOTION_FIRST_FRAME_KEY = "motion_first_frame"
JOINT_NAMES_KEY = "joint_names"
BODY_NAMES_KEY = "body_names"
JOINT_POS_KEY = "joint_pos"
JOINT_VEL_KEY = "joint_vel"
BODY_POS_W_KEY = "body_pos_w"
BODY_LIN_VEL_W_KEY = "body_lin_vel_w"
BODY_QUAT_W_KEY = "body_quat_w"
BODY_ANG_VEL_W_KEY = "body_ang_vel_w"
XROBOT_BODY_NAMES_KEY = "xrobot_body_names"
XROBOT_BODY_POS_W_KEY = "xrobot_body_pos_w"
XROBOT_BODY_QUAT_W_KEY = "xrobot_body_quat_w"

UNITREE_GO_DDS_ROBOT_NAMES = frozenset({"h1", "go2"})
UNITREE_HG_DDS_ROBOT_NAMES = frozenset(
    {"g1", "g1_real", "h1_2", "h1_2_21dof", "h1_2_27dof"}
)


@dataclass(frozen=True)
class RobotCfg:
    name: str
    joint_names: tuple[str, ...]
    body_names: tuple[str, ...]
    joint_pos_lower_limit: Mapping[str, float]
    joint_pos_upper_limit: Mapping[str, float]
    joint_velocity_limit: Mapping[str, float]
    joint_effort_limit: Mapping[str, float]
    joint_armature: Mapping[str, float] = field(default_factory=dict)
    joint_frictionloss: Mapping[str, float] = field(default_factory=dict)
    mjcf_path: AssetReference | None = None
    default_qpos: tuple[float, ...] = ()
    qpos_root_size: int = 7
    publish_hz: float = 50.0
    domain_id: int = 0
    interface: str | None = "eth0"
    mocap_ip: str = "localhost"
    low_state_port: int = PORTS["low_state"]
    low_state_bind_addr: str = "*"
    low_state_host: str = "127.0.0.1"
    low_cmd_port: int = PORTS["low_cmd"]
    low_cmd_bind_addr: str = "*"
    low_cmd_host: str = "127.0.0.1"
    unitree_legged_const: Mapping[str, int | float] = field(
        default_factory=lambda: dict(UNITREE_LEGGED_CONST)
    )
    root_joint_names: tuple[str, ...] = ("floating_base_joint", "pelvis_root")
    viewer_track_body_names: tuple[str, ...] = ("pelvis",)
    elastic_band_attach_body_names: tuple[str, ...] = ("torso_link", "base_link")

    @property
    def qpos_size(self) -> int:
        return self.qpos_root_size + len(self.joint_names)

    @property
    def root_pos_slice(self) -> slice:
        return slice(0, 3)

    @property
    def root_quat_slice(self) -> slice:
        return slice(3, 7)

    @property
    def joint_pos_slice(self) -> slice:
        return slice(self.qpos_root_size, self.qpos_size)

    def resolve_mjcf_path(self) -> Path:
        if self.mjcf_path is None:
            raise ValueError(f"Robot '{self.name}' does not define mjcf_path")
        return resolve_asset_reference(self.mjcf_path)


def normalize_name_list(values: Sequence[object] | None) -> list[str] | None:
    if values is None:
        return None
    return [str(value) for value in values]


def normalize_robot_name(name: object) -> str:
    return str(name).strip().lower().replace("-", "_")


def get_unitree_dds_family(name: object) -> str:
    normalized_name = normalize_robot_name(name)
    if normalized_name in UNITREE_GO_DDS_ROBOT_NAMES:
        return "go"
    if normalized_name in UNITREE_HG_DDS_ROBOT_NAMES:
        return "hg"
    raise NotImplementedError(
        f"Robot name {name!r} is not supported for Unitree DDS integration."
    )


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
            f"{label} order mismatch; expected canonical order from RobotCfg"
        )
    else:
        missing = [name for name in expected_list if name not in normalized_actual]
        extra = [name for name in normalized_actual if name not in expected_list]
        print(f"[teleop] {label} mismatch; missing={missing} extra={extra}")
    return False


def _coerce_mjcf_model(model_or_path: Any) -> Any:
    import mujoco

    if hasattr(model_or_path, "nbody") and hasattr(model_or_path, "njnt"):
        return model_or_path
    return mujoco.MjModel.from_xml_path(str(Path(model_or_path).expanduser().absolute()))


def resolve_mjcf_root_body_name(model_or_path: Any) -> str:
    import mujoco

    model = _coerce_mjcf_model(model_or_path)
    candidate_body_ids = [
        body_id
        for body_id in range(1, int(model.nbody))
        if int(model.body_parentid[body_id]) == 0
    ]
    if not candidate_body_ids:
        raise ValueError("Could not resolve any non-world root body from MJCF")

    articulated_root_body_ids = [
        body_id
        for body_id in candidate_body_ids
        if any(int(model.jnt_bodyid[joint_id]) == body_id for joint_id in range(int(model.njnt)))
    ]
    root_body_id = (
        articulated_root_body_ids[0]
        if articulated_root_body_ids
        else candidate_body_ids[0]
    )
    root_body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(root_body_id))
    if not root_body_name:
        raise ValueError(f"Resolved MJCF root body id {root_body_id} has no name")
    return str(root_body_name)


def resolve_mjcf_joint_names(model_or_path: Any) -> tuple[str, ...]:
    import mujoco

    model = _coerce_mjcf_model(model_or_path)
    qpos_ordered_joint_names: list[tuple[int, str]] = []
    scalar_joint_types = {
        int(mujoco.mjtJoint.mjJNT_HINGE),
        int(mujoco.mjtJoint.mjJNT_SLIDE),
    }

    for joint_id in range(int(model.njnt)):
        joint_type = int(model.jnt_type[joint_id])
        if joint_type not in scalar_joint_types:
            continue
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(joint_id))
        if not joint_name:
            raise ValueError(f"Resolved MJCF joint id {joint_id} has no name")
        qpos_ordered_joint_names.append((int(model.jnt_qposadr[joint_id]), str(joint_name)))

    qpos_ordered_joint_names.sort(key=lambda item: item[0])
    return tuple(name for _, name in qpos_ordered_joint_names)
