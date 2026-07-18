from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np

from .common import sort_names_by_preferred_order
from .base import Observation
from .motion import motion_obs
from sim2real.utils.math import quat_rotate_inverse_numpy


def _quat_to_roll_pitch(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float32).reshape(4)
    qw, qx, qy, qz = quat
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1.0:
        pitch = np.copysign(np.pi / 2.0, sinp)
    else:
        pitch = np.arcsin(sinp)

    return np.asarray([roll, pitch], dtype=np.float32)


class twist2_input(motion_obs, namespace="twist2"):
    """TWIST2 student-future ONNX input.

    Matches upstream TWIST2 deployment/training layout:
    current `[mimic_obs(35), proprio(92)]`, then 10 frames of that same
    127-dim observation history, then one 35-dim future mimic observation.
    """

    def __init__(
        self,
        joint_names: Sequence[str],
        history_len: int = 10,
        current_step: int = 0,
        future_step: int = 0,
        ang_vel_scale: float = 0.25,
        dof_pos_scale: float = 1.0,
        dof_vel_scale: float = 0.05,
        ankle_joint_names: Sequence[str] = (
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
        ),
        **kwargs,
    ):
        env = kwargs["env"]
        root_body_name = str(env.policy_config.get("motion", {}).get("root_body_name", "pelvis"))
        self.joint_names = sort_names_by_preferred_order(
            joint_names,
            env.joint_names_simulation,
        )
        super().__init__(
            future_steps=None,
            joint_names=self.joint_names,
            body_names=[root_body_name],
            root_body_name=root_body_name,
            anchor_body_name=root_body_name,
            joint_order="given",
            body_order="given",
            **kwargs,
        )
        self.joint_ids = [
            self.state_processor.joint_names.index(joint_name)
            for joint_name in self.joint_names
        ]
        self.history_len = int(history_len)
        self.current_step = int(current_step)
        self.future_step = int(future_step)
        self.ang_vel_scale = float(ang_vel_scale)
        self.dof_pos_scale = float(dof_pos_scale)
        self.dof_vel_scale = float(dof_vel_scale)
        self.default_joint_pos = self.env.default_dof_angles[self.joint_ids].astype(np.float32)
        self.ankle_indices = [
            self.joint_names.index(joint_name)
            for joint_name in ankle_joint_names
            if joint_name in self.joint_names
        ]
        self.history = np.zeros((self.history_len, 127), dtype=np.float32)
        self.input = np.zeros((1, 1432), dtype=np.float32)
        self._input_parts = {
            "current_motion": np.zeros(35, dtype=np.float32),
            "proprioception": np.zeros(92, dtype=np.float32),
            "observation_history": np.zeros(self.history_len * 127, dtype=np.float32),
            "future_motion": np.zeros(35, dtype=np.float32),
        }

    def _set_input_parts(
        self,
        current_motion: np.ndarray,
        proprioception: np.ndarray,
        future_motion: np.ndarray,
    ) -> None:
        self._input_parts = {
            "current_motion": current_motion.astype(np.float32, copy=True),
            "proprioception": proprioception.astype(np.float32, copy=True),
            "observation_history": self.history.reshape(-1).astype(np.float32, copy=True),
            "future_motion": future_motion.astype(np.float32, copy=True),
        }
        self.input[0, :] = np.concatenate(list(self._input_parts.values()))

    def _motion_mimic_obs(self, step_index: int) -> np.ndarray:
        root_pos = self.ref_root_pos_future_w[0, step_index]
        root_quat = self.ref_root_quat_future_w[0, step_index]
        root_lin_vel_w = self.ref_root_lin_vel_future_w[0, step_index]
        root_ang_vel_w = self.ref_root_ang_vel_future_w[0, step_index]
        joint_pos = self.ref_joint_pos_future[0, step_index]

        root_lin_vel_local = quat_rotate_inverse_numpy(
            root_quat.reshape(1, 4),
            root_lin_vel_w.reshape(1, 3),
        )[0]
        root_ang_vel_local = quat_rotate_inverse_numpy(
            root_quat.reshape(1, 4),
            root_ang_vel_w.reshape(1, 3),
        )[0]
        roll_pitch = _quat_to_roll_pitch(root_quat)

        return np.concatenate(
            [
                root_lin_vel_local[:2],
                root_pos[2:3],
                roll_pitch,
                root_ang_vel_local[2:3],
                joint_pos,
            ]
        ).astype(np.float32)

    def _robot_proprio_obs(self, data: Dict[str, Any]) -> np.ndarray:
        joint_pos = self.state_processor.joint_pos[self.joint_ids]
        joint_vel = self.state_processor.joint_vel[self.joint_ids].copy()
        if self.ankle_indices:
            joint_vel[self.ankle_indices] = 0.0
        previous_action = np.asarray(
            data.get("action", np.zeros(self.env.num_actions, dtype=np.float32)),
            dtype=np.float32,
        )
        return np.concatenate(
            [
                self.state_processor.root_ang_vel_b * self.ang_vel_scale,
                _quat_to_roll_pitch(self.state_processor.root_quat_w),
                (joint_pos - self.default_joint_pos) * self.dof_pos_scale,
                joint_vel * self.dof_vel_scale,
                previous_action,
            ]
        ).astype(np.float32)

    def reset(self) -> None:
        super().reset()
        current_step_index = self._step_to_motion_index(self.current_step)
        future_step_index = self._step_to_motion_index(self.future_step)
        current_motion = self._motion_mimic_obs(current_step_index)
        proprioception = self._robot_proprio_obs(
            {"action": np.zeros(self.env.num_actions, dtype=np.float32)}
        )
        current = np.concatenate([current_motion, proprioception]).astype(np.float32)
        self.history[:] = current
        self._set_input_parts(
            current_motion,
            proprioception,
            self._motion_mimic_obs(future_step_index),
        )

    def _step_to_motion_index(self, step: int) -> int:
        return self._motion_step_index(step)

    def update(self, data: Dict[str, Any]) -> None:
        super().update(data)
        current_step_index = self._step_to_motion_index(self.current_step)
        future_step_index = self._step_to_motion_index(self.future_step)

        current_motion = self._motion_mimic_obs(current_step_index)
        proprioception = self._robot_proprio_obs(data)
        current = np.concatenate([current_motion, proprioception]).astype(np.float32)
        future = self._motion_mimic_obs(future_step_index)

        self._set_input_parts(current_motion, proprioception, future)
        self.history[:-1] = self.history[1:]
        self.history[-1] = current

    def compute(self) -> np.ndarray:
        return self.input


class twist2_component_input(Observation, namespace="twist2"):
    """One semantic TWIST2 input group without flat-vector slicing."""

    def __init__(self, component: str, **kwargs: Any) -> None:
        if component not in {
            "current_motion",
            "proprioception",
            "observation_history",
            "future_motion",
        }:
            raise ValueError(f"Unsupported TWIST2 component: {component!r}")
        super().__init__(env=kwargs["env"])
        self.component = component
        cache_name = "_twist2_semantic_observation"
        self._owns_core = not hasattr(self.env, cache_name)
        if self._owns_core:
            setattr(self.env, cache_name, twist2_input(**kwargs))
        self._core = getattr(self.env, cache_name)

    def reset(self) -> None:
        if self._owns_core:
            self._core.reset()

    def update(self, data: Dict[str, Any]) -> None:
        if self._owns_core:
            self._core.update(data)

    def compute(self) -> np.ndarray:
        return self._core._input_parts[self.component][None, :]
