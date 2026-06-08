from __future__ import annotations

from sim2real.rl_policy.control_mode import ControlMode
from sim2real.rl_policy.controllers.base import ControllerBase


class PassiveController(ControllerBase):
    name = "passive"

    def get_control_mode(self) -> ControlMode | None:
        return None
