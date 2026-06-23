from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np

from sim2real.rl_policy.observations.base import Observation
from sim2real.rl_policy.utils.motion import MotionData
from sim2real.utils.math import matrix_from_quat, quat_conjugate, quat_mul, quat_rotate_inverse_numpy


class axell_g1tracking_policy_obs(Observation):
    """Axellwppr/motion_tracking G1TRACKING-03-16_14-15_0315.2 policy input.

    Mirrors upstream ``TrackingPolicyRaw._build_obs_modules()`` order:
    boot, command, compliance flag, target joints/root/gravity, robot histories,
    and previous raw actions.
    """

    OBS_DIM = 1590

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
        compliance_flag: bool | None = None,
        compliance_flag_value: float = 0.0,
        compliance_flag_threshold: float = 10.0,
        target_root_z_offset: float = 0.035,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.future_steps = np.asarray([int(step) for step in future_steps], dtype=np.int64)
        if self.future_steps.ndim != 1 or self.future_steps.size == 0:
            raise ValueError("future_steps must be a non-empty 1D sequence")
        if int(self.future_steps[0]) != 0:
            raise ValueError(f"future_steps[0] must be 0, got {self.future_steps.tolist()}")

        self.root_angvel_history_steps = [int(step) for step in root_angvel_history_steps]
        self.projected_gravity_history_steps = [int(step) for step in projected_gravity_history_steps]
        self.joint_pos_history_steps = [int(step) for step in joint_pos_history_steps]
        self.joint_vel_history_steps = [int(step) for step in joint_vel_history_steps]
        self.prev_action_steps = int(prev_action_steps)
        if self.prev_action_steps <= 0:
            raise ValueError("prev_action_steps must be positive")

        self.joint_names = list(joint_names or self.env.policy_joint_names)
        self.root_body_name = str(root_body_name)
        self.target_root_z_offset = float(target_root_z_offset)
        self.compliance_flag_value = (
            1.0 if bool(compliance_flag) else 0.0
        ) if compliance_flag is not None else float(compliance_flag_value)
        self.compliance_flag_threshold = float(compliance_flag_threshold)
        self.compliance_kp = self.compliance_flag_threshold / 0.05

        if len(self.joint_names) != 29:
            raise ValueError(f"Axell G1 tracking expects 29 joints, got {len(self.joint_names)}")

        self._state_joint_indices = [
            self.state_processor.joint_names.index(name) for name in self.joint_names
        ]
        self._cached_motion_layout: tuple[tuple[str, ...], tuple[str, ...]] | None = None

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
        self._cached_motion_layout = None
        self._fill_histories_from_current()
        self._prev_actions[:] = 0.0
        self._obs[:] = 0.0

    def _fill_histories_from_current(self) -> None:
        root_angvel = np.asarray(self.state_processor.root_ang_vel_b, dtype=np.float32)
        gravity = self._current_projected_gravity()
        joint_pos = self._current_joint_pos()
        joint_vel = self._current_joint_vel()
        self._root_angvel_history[:] = root_angvel.reshape(1, 3)
        self._projected_gravity_history[:] = gravity.reshape(1, 3)
        self._joint_pos_history[:] = joint_pos.reshape(1, -1)
        self._joint_vel_history[:] = joint_vel.reshape(1, -1)

    def _refresh_motion_indices(self) -> None:
        joint_names = tuple(self.state_processor.motion_joint_names)
        body_names = tuple(self.state_processor.motion_body_names)
        layout = (joint_names, body_names)
        if self._cached_motion_layout == layout:
            return

        missing_joints = [name for name in self.joint_names if name not in joint_names]
        if missing_joints:
            raise ValueError(f"Motion source missing Axell policy joints: {missing_joints}")
        if self.root_body_name not in body_names:
            raise ValueError(f"Motion source missing root body {self.root_body_name!r}")

        self._motion_joint_indices = [joint_names.index(name) for name in self.joint_names]
        self._root_body_idx = body_names.index(self.root_body_name)
        self._cached_motion_layout = layout

    def _motion_slice(self) -> MotionData:
        self._refresh_motion_indices()
        backend = getattr(self.state_processor, "motion_backend", "")
        if backend == "npz":
            return self.state_processor.motion_dataset.get_slice(
                self.state_processor.motion_ids,
                self.state_processor.motion_t,
                self.future_steps,
            )

        motion_data: MotionData = self.state_processor.motion_data
        available_steps = [int(step) for step in self.state_processor.motion_future_steps.tolist()]
        step_indices = []
        for step in [int(step) for step in self.future_steps.tolist()]:
            if step not in available_steps:
                raise ValueError(
                    f"Axell obs requested future step {step}, "
                    f"but motion source provides {available_steps}"
                )
            step_indices.append(available_steps.index(step))

        return MotionData(
            motion_id=np.take(motion_data.motion_id, step_indices, axis=1),
            step=np.take(motion_data.step, step_indices, axis=1),
            timestamps_ns=np.take(motion_data.timestamps_ns, step_indices, axis=1),
            joint_pos=np.take(motion_data.joint_pos, step_indices, axis=1),
            joint_vel=np.take(motion_data.joint_vel, step_indices, axis=1),
            body_pos_w=np.take(motion_data.body_pos_w, step_indices, axis=1),
            body_lin_vel_w=np.take(motion_data.body_lin_vel_w, step_indices, axis=1),
            body_quat_w=np.take(motion_data.body_quat_w, step_indices, axis=1),
            body_ang_vel_w=np.take(motion_data.body_ang_vel_w, step_indices, axis=1),
        )

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

    def update(self, data: Dict[str, Any]) -> None:
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

    def _build_obs(self) -> np.ndarray:
        motion_data = self._motion_slice()
        ref_joint_pos = np.asarray(
            motion_data.joint_pos[0][:, self._motion_joint_indices],
            dtype=np.float32,
        )
        ref_root_pos_w = np.asarray(
            motion_data.body_pos_w[0, :, self._root_body_idx],
            dtype=np.float32,
        )
        ref_root_quat_w = np.asarray(
            motion_data.body_quat_w[0, :, self._root_body_idx],
            dtype=np.float32,
        )

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

        target_root_z = (ref_root_pos_w[:, 2] + self.target_root_z_offset).astype(np.float32)
        down = np.broadcast_to(
            np.asarray([[0.0, 0.0, -1.0]], dtype=np.float32),
            (ref_root_quat_w.shape[0], 3),
        )
        target_projected_gravity = quat_rotate_inverse_numpy(ref_root_quat_w, down).reshape(-1)

        v = self.compliance_flag_value
        compliance = np.asarray(
            [v, v * self.compliance_flag_threshold, v * self.compliance_kp],
            dtype=np.float32,
        )

        obs = np.concatenate(
            [
                np.asarray([0.0], dtype=np.float32),
                command,
                compliance,
                target_joint,
                target_root_z,
                target_projected_gravity.astype(np.float32),
                self._root_angvel_history[self.root_angvel_history_steps].reshape(-1),
                self._projected_gravity_history[self.projected_gravity_history_steps].reshape(-1),
                self._joint_pos_history[self.joint_pos_history_steps].reshape(-1),
                self._joint_vel_history[self.joint_vel_history_steps].reshape(-1),
                self._prev_actions.reshape(-1),
            ],
            axis=0,
        ).astype(np.float32)
        if obs.shape[0] != self.OBS_DIM:
            raise ValueError(f"Axell G1 tracking obs dim mismatch: {obs.shape[0]} != {self.OBS_DIM}")
        return obs

    def compute(self) -> np.ndarray:
        return self._obs
