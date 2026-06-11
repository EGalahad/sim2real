import numpy as np
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, Optional

import tyro
from loguru import logger

from sim2real.rl_policy.base_policy import BasePolicyArgs, BasePolicy
from sim2real.rl_policy.controllers.keyboard import KeyboardController
from sim2real.rl_policy.controllers.unitree_joystick import UnitreeJoystickController
np.set_printoptions(precision=3, suppress=True, linewidth=1000)


def _apply_runtime_motion_config(
    policy_config,
    motion_backend: str,
    motion_path: Optional[str],
    motion_zmq_connect: str,
    motion_zmq_hwm: int,
    motion_dt_s: float,
    motion_tolerance_s: float,
):
    policy_config = deepcopy(policy_config)
    motion_cfg = policy_config.setdefault("motion", {})
    motion_cfg["motion_backend"] = motion_backend
    if motion_backend == "npz" and motion_path is not None:
        motion_cfg["motion_path"] = motion_path
    if motion_backend == "zmq":
        motion_cfg["motion_zmq_connect"] = motion_zmq_connect
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
        if self.args.motion_backend == "zmq":
            logger.info(
                "Using runtime motion_backend=zmq "
                f"connect={self.args.motion_zmq_connect}"
            )
        return policy_config

    def toggle_paused(self, *, source: str) -> None:
        paused = not bool(self.state_dict.get("paused", False))
        self.state_dict["paused"] = paused
        logger.info(f"Paused state toggled to {paused} via {source}")

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

    motion_backend: Literal["npz", "zmq"] = "npz"
    motion_path: Optional[str] = None
    motion_zmq_connect: str = "tcp://127.0.0.1:28701"
    motion_zmq_hwm: int = 1
    motion_tolerance_s: float = 0.04


if __name__ == "__main__":
    args = tyro.cli(TrackingArgs)
    policy = Tracking(args=args)
    policy.run()
