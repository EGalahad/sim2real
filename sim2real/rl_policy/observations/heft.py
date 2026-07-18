from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np

from sim2real.rl_policy.observations.motion import motion_obs
from sim2real.rl_policy.observations.base import Observation
from sim2real.utils.math import matrix_from_quat, quat_conjugate, quat_mul, quat_rotate_inverse_numpy


class heft_policy_obs(motion_obs, namespace="heft"):
    """HEFT G1 PMG policy input from the motion_tracking sim2real branch.

    Mirrors upstream ``TrackingPolicyRaw._build_obs_modules()`` order:
    boot indicator, command, target joints/root/gravity, robot histories, and
    previous raw actions. This adaptation is G1-only.
    """

    OBS_DIM = 1729
    INCLUDE_COMPLIANCE_FLAG = False
    COMPLIANCE_FLAG = False
    COMPLIANCE_FLAG_THRESHOLD = 10.0

    def __init__(
        self,
        future_steps: Sequence[int],
        root_angvel_history_steps: Sequence[int],
        projected_gravity_history_steps: Sequence[int],
        joint_pos_history_steps: Sequence[int],
        joint_vel_history_steps: Sequence[int],
        prev_action_steps: int,
        joint_names: Sequence[str] | None = None,
        root_body_name: str = "pelvis",
        boot_indicator_max: int = 25,
        **kwargs: Any,
    ) -> None:
        resolved_joint_names = (
            list(joint_names) if joint_names is not None else list(kwargs["env"].policy_joint_names)
        )
        super().__init__(
            future_steps=future_steps,
            joint_names=resolved_joint_names,
            body_names=[root_body_name],
            root_body_name=root_body_name,
            anchor_body_name=root_body_name,
            joint_order="given",
            body_order="given",
            **kwargs,
        )
        self.future_steps = np.asarray([int(step) for step in self.future_steps], dtype=np.int64)
        if self.future_steps.ndim != 1 or self.future_steps.size == 0:
            raise ValueError("future_steps must be a non-empty 1D sequence")
        if int(self.future_steps[0]) != 0:
            raise ValueError(f"future_steps[0] must be 0, got {self.future_steps.tolist()}")

        seen_negative = False
        for step in self.future_steps[1:].tolist():
            if int(step) < 0:
                seen_negative = True
            elif seen_negative:
                raise ValueError(
                    "future_steps format must be [0, ...positive/non-negative, ...negative], "
                    f"got {self.future_steps.tolist()}"
                )

        self.root_angvel_history_steps = [int(step) for step in root_angvel_history_steps]
        self.projected_gravity_history_steps = [int(step) for step in projected_gravity_history_steps]
        self.joint_pos_history_steps = [int(step) for step in joint_pos_history_steps]
        self.joint_vel_history_steps = [int(step) for step in joint_vel_history_steps]
        self.prev_action_steps = int(prev_action_steps)
        if self.prev_action_steps <= 0:
            raise ValueError("prev_action_steps must be positive")

        self.boot_indicator_max = int(boot_indicator_max)
        if self.boot_indicator_max <= 0:
            raise ValueError(f"boot_indicator_max must be positive, got {self.boot_indicator_max}")
        self._boot_indicator_value = self.boot_indicator_max

        if len(self.joint_names) != 29:
            raise ValueError(f"HEFT G1 tracking expects 29 joints, got {len(self.joint_names)}")

        self._state_joint_indices = [
            self.state_processor.joint_names.index(name) for name in self.joint_names
        ]

        self._root_angvel_history = np.zeros(
            (max(self.root_angvel_history_steps) + 1, 3),
            dtype=np.float32,
        )
        self._projected_gravity_history = np.zeros(
            (max(self.projected_gravity_history_steps) + 1, 3),
            dtype=np.float32,
        )
        self._joint_pos_history = np.zeros(
            (max(self.joint_pos_history_steps) + 1, len(self.joint_names)),
            dtype=np.float32,
        )
        self._joint_vel_history = np.zeros(
            (max(self.joint_vel_history_steps) + 1, len(self.joint_names)),
            dtype=np.float32,
        )
        self._prev_actions = np.zeros(
            (self.prev_action_steps, len(self.joint_names)),
            dtype=np.float32,
        )
        self._obs = np.zeros((1, self.OBS_DIM), dtype=np.float32)

    def reset(self) -> None:
        super().reset()
        self._boot_indicator_value = self.boot_indicator_max
        self._root_angvel_history[:] = 0.0
        self._projected_gravity_history[:] = 0.0
        self._joint_pos_history[:] = 0.0
        self._joint_vel_history[:] = 0.0
        self._prev_actions[:] = 0.0
        self._obs[:] = 0.0

    def _current_joint_pos(self) -> np.ndarray:
        return np.asarray(
            self.state_processor.joint_pos[self._state_joint_indices],
            dtype=np.float32,
        )

    def _current_joint_vel(self) -> np.ndarray:
        return np.asarray(
            self.state_processor.joint_vel[self._state_joint_indices],
            dtype=np.float32,
        )

    def _current_projected_gravity(self) -> np.ndarray:
        gravity = quat_rotate_inverse_numpy(
            np.asarray(self.state_processor.root_quat_w, dtype=np.float32).reshape(1, 4),
            np.asarray([[0.0, 0.0, -1.0]], dtype=np.float32),
        )[0]
        return (gravity / (np.linalg.norm(gravity) + 1.0e-8)).astype(np.float32)

    @staticmethod
    def _append_history(history: np.ndarray, value: np.ndarray) -> None:
        history[:] = np.roll(history, 1, axis=0)
        history[0] = value

    def _compliance_obs(self) -> np.ndarray:
        value = 1.0 if self.COMPLIANCE_FLAG else 0.0
        threshold = float(self.COMPLIANCE_FLAG_THRESHOLD)
        kp = threshold / 0.05
        return np.asarray([value, value * threshold, value * kp], dtype=np.float32)

    def update(self, data: Dict[str, Any]) -> None:
        super().update(data)
        self._boot_indicator_value = max(self._boot_indicator_value - 1, 0)
        self._append_history(
            self._root_angvel_history,
            np.asarray(self.state_processor.root_ang_vel_b, dtype=np.float32),
        )
        self._append_history(self._projected_gravity_history, self._current_projected_gravity())
        self._append_history(self._joint_pos_history, self._current_joint_pos())
        self._append_history(self._joint_vel_history, self._current_joint_vel())

        prev_action = np.asarray(
            data.get("action", np.zeros(len(self.joint_names), dtype=np.float32)),
            dtype=np.float32,
        ).reshape(-1)
        if prev_action.shape[0] != len(self.joint_names):
            raise ValueError(
                f"Previous action dim mismatch: expected {len(self.joint_names)}, "
                f"got {prev_action.shape[0]}"
            )
        self._prev_actions[:] = np.roll(self._prev_actions, 1, axis=0)
        self._prev_actions[0, :] = prev_action

        self._obs[0, :] = self._build_obs()

    def _build_obs_parts(self) -> Dict[str, np.ndarray]:
        ref_joint_pos = np.asarray(self._select(self.ref_joint_pos_future)[0], dtype=np.float32)
        ref_root_pos_w = np.asarray(self._select(self.ref_root_pos_future_w)[0], dtype=np.float32)
        ref_root_quat_w = np.asarray(self._select(self.ref_root_quat_future_w)[0], dtype=np.float32)

        pos_diff_w = ref_root_pos_w[1:] - ref_root_pos_w[0:1]
        base_ref_quat = np.broadcast_to(ref_root_quat_w[0:1], (pos_diff_w.shape[0], 4))
        pos_diff_b = quat_rotate_inverse_numpy(base_ref_quat, pos_diff_w)

        robot_root_quat_w = np.asarray(self.state_processor.root_quat_w, dtype=np.float32)
        robot_root_quat_w = np.broadcast_to(robot_root_quat_w.reshape(1, 4), ref_root_quat_w.shape)
        rel_quat = quat_mul(quat_conjugate(robot_root_quat_w), ref_root_quat_w)
        rel_rot = matrix_from_quat(rel_quat)
        rot6d = rel_rot[:, :, :2].transpose(0, 2, 1).reshape(-1).astype(np.float32)
        command = np.concatenate([pos_diff_b.reshape(-1), rot6d], axis=0)

        cur_joint_pos = self._current_joint_pos().reshape(1, -1)
        target_joint = np.concatenate(
            [
                ref_joint_pos.reshape(-1),
                (ref_joint_pos - cur_joint_pos).reshape(-1),
            ],
            axis=0,
        ).astype(np.float32)

        down = np.broadcast_to(
            np.asarray([[0.0, 0.0, -1.0]], dtype=np.float32),
            (ref_root_quat_w.shape[0], 3),
        )
        target_projected_gravity = quat_rotate_inverse_numpy(ref_root_quat_w, down).reshape(-1)

        parts = {
            "context": np.asarray(
                [self._boot_indicator_value / self.boot_indicator_max],
                dtype=np.float32,
            ),
            "motion_command": command.astype(np.float32),
        }
        if self.INCLUDE_COMPLIANCE_FLAG:
            parts["compliance_context"] = self._compliance_obs()
        parts.update(
            {
            "target_motion": np.concatenate(
                [
                    target_joint,
                    ref_root_pos_w[:, 2].astype(np.float32),
                    target_projected_gravity.astype(np.float32),
                ],
                axis=0,
            ).astype(np.float32),
            "proprioception": np.concatenate(
                [
                    self._root_angvel_history[self.root_angvel_history_steps].reshape(-1),
                    self._projected_gravity_history[
                        self.projected_gravity_history_steps
                    ].reshape(-1),
                    self._joint_pos_history[self.joint_pos_history_steps].reshape(-1),
                    self._joint_vel_history[self.joint_vel_history_steps].reshape(-1),
                    self._prev_actions.reshape(-1),
                ],
                axis=0,
            ).astype(np.float32),
            }
        )
        return parts

    def _build_obs(self) -> np.ndarray:
        obs = np.concatenate(
            list(self._build_obs_parts().values()),
            axis=0,
        ).astype(np.float32)
        if obs.shape[0] != self.OBS_DIM:
            raise ValueError(f"HEFT G1 tracking obs dim mismatch: {obs.shape[0]} != {self.OBS_DIM}")
        return obs

    def compute(self) -> np.ndarray:
        return self._obs


class heft_compliance_policy_obs(heft_policy_obs, namespace="heft"):
    """HEFT G1 Compliance policy input with compliance flag forced off."""

    OBS_DIM = 1590
    INCLUDE_COMPLIANCE_FLAG = True
    COMPLIANCE_FLAG = False


class heft_component_obs(Observation, namespace="heft"):
    """One semantic HEFT input group without relying on flat-vector offsets."""

    def __init__(
        self,
        component: str,
        policy_variant: str = "pmg",
        **kwargs: Any,
    ) -> None:
        if component not in {
            "context",
            "motion_command",
            "compliance_context",
            "target_motion",
            "proprioception",
        }:
            raise ValueError(f"Unsupported HEFT component: {component!r}")
        if policy_variant not in {"pmg", "compliance"}:
            raise ValueError(f"Unsupported HEFT policy variant: {policy_variant!r}")
        super().__init__(env=kwargs["env"])
        self.component = component
        cache_name = f"_heft_{policy_variant}_semantic_observation"
        self._owns_core = not hasattr(self.env, cache_name)
        if self._owns_core:
            core_class = (
                heft_compliance_policy_obs
                if policy_variant == "compliance"
                else heft_policy_obs
            )
            setattr(self.env, cache_name, core_class(**kwargs))
        self._core = getattr(self.env, cache_name)

    def reset(self) -> None:
        if self._owns_core:
            self._core.reset()

    def update(self, data: Dict[str, Any]) -> None:
        if self._owns_core:
            self._core.update(data)

    def compute(self) -> np.ndarray:
        parts = self._core._build_obs_parts()
        if self.component not in parts:
            raise ValueError(
                f"HEFT component {self.component!r} is not present in this policy variant"
            )
        return parts[self.component][None, :]
