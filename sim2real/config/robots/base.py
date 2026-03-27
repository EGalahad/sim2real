from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from sim2real.utils.common import PORTS, UNITREE_LEGGED_CONST


PROJECT_ROOT = Path(__file__).resolve().parents[3]

SMPLX_T_NS_KEY = "smplx_t_ns"
PUBLISH_T_NS_KEY = "publish_t_ns"
SEQ_KEY = "seq"
JOINT_NAMES_KEY = "joint_names"
BODY_NAMES_KEY = "body_names"
JOINT_POS_KEY = "joint_pos"
BODY_POS_W_KEY = "body_pos_w"
BODY_QUAT_W_KEY = "body_quat_w"


@dataclass(frozen=True)
class RobotCfg:
    name: str
    robot_type: str
    joint_names: tuple[str, ...]
    body_names: tuple[str, ...]
    joint_pos_lower_limit: Mapping[str, float]
    joint_pos_upper_limit: Mapping[str, float]
    joint_velocity_limit: Mapping[str, float]
    joint_effort_limit: Mapping[str, float]
    mjcf_path: Path | None = None
    sim_mjcf_path: Path | None = None
    default_qpos: tuple[float, ...] = ()
    qpos_root_size: int = 7
    publish_hz: float = 50.0
    domain_id: int = 0
    interface: str | None = "eth0"
    mocap_ip: str = "localhost"
    use_joystick: bool = False
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
            f"{label} order mismatch; expected canonical order from RobotCfg"
        )
    else:
        missing = [name for name in expected_list if name not in normalized_actual]
        extra = [name for name in normalized_actual if name not in expected_list]
        print(f"[teleop] {label} mismatch; missing={missing} extra={extra}")
    return False
