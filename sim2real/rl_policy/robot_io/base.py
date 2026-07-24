from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RobotState:
    """Robot state in the layout consumed by the shared policy runtime."""

    qpos: np.ndarray
    qvel: np.ndarray
    joint_torque: np.ndarray
    tick: int


class RobotIO(ABC):
    """Minimal hardware/transport boundary used by the policy runtime."""

    @abstractmethod
    def read_state(self) -> RobotState | None:
        """Return the latest state, or ``None`` before one is available."""

    @abstractmethod
    def write_command(
        self,
        q_target: np.ndarray,
        dq_target: np.ndarray,
        tau_ff: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
    ) -> None:
        """Send one joint-space command."""

    def close(self) -> None:
        """Release backend resources, if any."""
