import numpy as np
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, Optional

import tyro
from loguru import logger

from sim2real.rl_policy.base_policy import BasePolicyArgs, BasePolicy
from sim2real.rl_policy.controllers.keyboard import KeyboardController
from sim2real.rl_policy.controllers.unitree_joystick import UnitreeJoystickController
np.set_printoptions(precision=3, suppress=True, linewidth=1000)


DEFAULT_MOTION_ZMQ_CONNECT = "tcp://127.0.0.1:28701"
DEFAULT_SMPL_MOTION_ZMQ_CONNECT = "tcp://127.0.0.1:28702"
SMPL_ENCODER_OBSERVATION = "sonic_smpl_official_encoder_input"


def _policy_uses_smpl_motion(policy_config: dict[str, Any]) -> bool:
    observation_cfg = policy_config.get("observation", {})
    if not isinstance(observation_cfg, dict):
        return False

    for obs_items in observation_cfg.values():
        if not isinstance(obs_items, dict):
            continue
        for obs_name, obs_config in obs_items.items():
            target = obs_name
            if isinstance(obs_config, dict):
                target = str(obs_config.get("_target_", obs_name))
            if target.split(".")[-1] == SMPL_ENCODER_OBSERVATION:
                return True
    return False


def _default_motion_connect(motion_backend: str) -> str:
    if motion_backend == "smpl_zmq":
        return DEFAULT_SMPL_MOTION_ZMQ_CONNECT
    return DEFAULT_MOTION_ZMQ_CONNECT


def _resolve_motion_zmq_connect(
    *,
    motion_backend: str,
    requested_connect: Optional[str],
    configured_connect: Optional[str],
) -> str:
    connect = requested_connect or configured_connect or _default_motion_connect(motion_backend)
    if motion_backend == "smpl_zmq" and connect == DEFAULT_MOTION_ZMQ_CONNECT:
        logger.warning(
            "motion_backend=smpl_zmq received the normal motion endpoint "
            f"{DEFAULT_MOTION_ZMQ_CONNECT}; using SMPL endpoint "
            f"{DEFAULT_SMPL_MOTION_ZMQ_CONNECT} instead."
        )
        return DEFAULT_SMPL_MOTION_ZMQ_CONNECT
    return connect


def _apply_runtime_motion_config(
    policy_config,
    motion_backend: Optional[str],
    motion_path: Optional[str],
    motion_zmq_connect: Optional[str],
    motion_zmq_hwm: int,
    motion_dt_s: float,
    motion_tolerance_s: float,
):
    policy_config = deepcopy(policy_config)
    motion_cfg = policy_config.setdefault("motion", {})
    configured_backend = str(motion_cfg.get("motion_backend", "npz")).lower().strip()
    resolved_backend = (
        str(motion_backend).lower().strip()
        if motion_backend is not None
        else configured_backend
    )

    if _policy_uses_smpl_motion(policy_config) and resolved_backend == "zmq":
        logger.warning(
            "Policy uses SONIC SMPL encoder input; switching motion_backend=zmq "
            "to motion_backend=smpl_zmq."
        )
        resolved_backend = "smpl_zmq"

    motion_cfg["motion_backend"] = resolved_backend
    if resolved_backend == "npz" and motion_path is not None:
        motion_cfg["motion_path"] = motion_path
        motion_cfg["motion_dt_s"] = motion_dt_s
    if resolved_backend in {"zmq", "smpl_zmq"}:
        motion_cfg["motion_zmq_connect"] = _resolve_motion_zmq_connect(
            motion_backend=resolved_backend,
            requested_connect=motion_zmq_connect,
            configured_connect=motion_cfg.get("motion_zmq_connect"),
        )
        motion_cfg["motion_zmq_hwm"] = motion_zmq_hwm
        motion_cfg["motion_dt_s"] = motion_dt_s
        motion_cfg["motion_tolerance_s"] = motion_tolerance_s

    return policy_config


class Tracking(BasePolicy):
    args: "TrackingArgs"
    def prepare_policy_config(self, policy_config):
        policy_config = super().prepare_policy_config(policy_config)
        policy_config = _apply_runtime_motion_config(
            policy_config=policy_config,
            motion_backend=self.args.motion_backend,
            motion_path=self.args.motion_path,
            motion_zmq_connect=self.args.motion_zmq_connect,
            motion_zmq_hwm=self.args.motion_zmq_hwm,
            motion_dt_s=1 / self.args.rl_rate,
            motion_tolerance_s=self.args.motion_tolerance_s,
        )
        motion_cfg = policy_config.get("motion", {})
        resolved_backend = str(motion_cfg.get("motion_backend", ""))
        if resolved_backend in {"zmq", "smpl_zmq"}:
            logger.info(
                f"Using runtime motion_backend={resolved_backend} "
                f"connect={motion_cfg.get('motion_zmq_connect')}"
            )
        return policy_config

    def toggle_paused(self, *, source: str) -> None:
        paused = not bool(self.state_dict.get("paused", False))
        self.state_dict["paused"] = paused
        logger.info(f"Paused state toggled to {paused} via {source}")

    def update(self):
        self.state_processor.update(self.state_dict)
        for obs_group in self.observations.values():
            for obs_func in obs_group.funcs.values():
                obs_func.update(self.state_dict)

    def process_controllers(self) -> None:
        super().process_controllers()

        extra_keys = self.controller.get_extra_keys()
        if not extra_keys:
            return

        if isinstance(self.controller, KeyboardController) and "space" in extra_keys:
            self.toggle_paused(source="keyboard:space")
        elif isinstance(self.controller, UnitreeJoystickController) and "B" in extra_keys:
            self.toggle_paused(source="joystick:B")


@dataclass
class TrackingArgs(BasePolicyArgs):
    """Robot."""

    motion_backend: Optional[Literal["npz", "zmq", "smpl_zmq"]] = None
    motion_path: Optional[str] = None
    motion_zmq_connect: Optional[str] = None
    motion_zmq_hwm: int = 1
    motion_tolerance_s: float = 0.04


if __name__ == "__main__":
    args = tyro.cli(TrackingArgs)
    policy = Tracking(args=args)
    policy.run()
