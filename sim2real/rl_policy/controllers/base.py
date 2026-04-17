from __future__ import annotations

from abc import ABC, abstractmethod

from sim2real.rl_policy.control_mode import ControlMode


class ControllerBase(ABC):
    name = "controller"

    @abstractmethod
    def get_control_mode(self) -> ControlMode | None:
        """Return a newly requested control mode, or None if unchanged."""

    def get_extra_keys(self) -> list[str]:
        """Return one-shot, edge-triggered extra keys since the last call."""
        return []

    def close(self) -> None:
        """Release runtime resources such as listener threads."""
        return None
