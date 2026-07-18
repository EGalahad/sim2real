from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

from sim2real.rl_policy.observations.motion import motion_obs
from sim2real.rl_policy.observations.base import Observation
from sim2real.utils.math import (
    matrix_from_quat,
    quat_conjugate,
    quat_from_yaw,
    quat_mul,
    quat_rotate_inverse_numpy,
)


def _quat_to_yaw(q_wxyz: np.ndarray) -> np.ndarray:
    w = q_wxyz[..., 0]
    x = q_wxyz[..., 1]
    y = q_wxyz[..., 2]
    z = q_wxyz[..., 3]
    yaw = np.arctan2(2.0 * (x * y + w * z), 1.0 - 2.0 * (y * y + z * z))
    return _wrap_pi(yaw)


def _wrap_pi(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


def _rotate_xy(xy: np.ndarray, yaw: float) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float32)
    c, s = np.cos(yaw), np.sin(yaw)
    return np.asarray(
        [c * xy[0] - s * xy[1], s * xy[0] + c * xy[1]],
        dtype=np.float32,
    )


def _rotate_vector_z(vector: np.ndarray, yaw: float) -> np.ndarray:
    rotated = np.asarray(vector, dtype=np.float32).copy()
    rotated[:2] = _rotate_xy(rotated[:2], yaw)
    return rotated


def _quat_to_rotvec(q_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(q_wxyz, dtype=np.float64).reshape(4).copy()
    quat /= max(float(np.linalg.norm(quat)), 1.0e-12)
    if quat[0] < 0.0:
        quat *= -1.0

    vec = quat[1:]
    sin_half = float(np.linalg.norm(vec))
    if sin_half < 1.0e-12:
        return np.zeros(3, dtype=np.float32)

    angle = 2.0 * np.arctan2(sin_half, np.clip(quat[0], -1.0, 1.0))
    return (vec / sin_half * angle).astype(np.float32)


def _batch_base_to_navi(base_to_world: np.ndarray, eps: float = 1.0e-8) -> np.ndarray:
    """Match Humanoid-GPT's yaw-aligned gravity-view frame construction."""

    x_proj = np.asarray(base_to_world[..., :3, 0], dtype=np.float32)
    z_axis = np.zeros_like(x_proj)
    z_axis[..., 2] = 1.0

    y_axis = np.cross(z_axis, x_proj)
    y_norm = np.linalg.norm(y_axis, axis=-1, keepdims=True)
    fallback_y = np.zeros_like(y_axis)
    fallback_y[..., 1] = 1.0
    y_axis = np.where(y_norm > eps, y_axis / np.clip(y_norm, eps, None), fallback_y)
    x_axis = np.cross(y_axis, z_axis)
    return np.stack((x_axis, y_axis, z_axis), axis=-1).astype(np.float32)


class humanoid_gpt_pns_obs(motion_obs, namespace="humanoid_gpt"):
    """Humanoid-GPT PNS non-privileged 136D observation.

    This mirrors GalaxyGeneralRobotics/Humanoid-GPT
    ``G1TrackInferFn.get_nn_state()`` for the released
    ``pns_wo_priv216.onnx`` policy.
    """

    OBS_DIM = 136

    def __init__(
        self,
        joint_names: Sequence[str] | None = None,
        root_body_name: str = "pelvis",
        reference_cvel_source: str = "motion",
        reference_dt_s: float = 0.02,
        reference_mjcf_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        resolved_joint_names = (
            list(joint_names) if joint_names is not None else list(kwargs["env"].policy_joint_names)
        )
        super().__init__(
            future_steps=None,
            joint_names=resolved_joint_names,
            body_names=[root_body_name],
            root_body_name=root_body_name,
            anchor_body_name=root_body_name,
            joint_order="given",
            body_order="given",
            **kwargs,
        )
        self.reference_cvel_source = str(reference_cvel_source).lower().strip()
        if self.reference_cvel_source not in {"motion", "mujoco"}:
            raise ValueError(
                "Humanoid-GPT reference_cvel_source must be 'motion' or 'mujoco', "
                f"got {reference_cvel_source!r}"
            )
        self.reference_dt_s = float(reference_dt_s)
        if self.reference_dt_s <= 0.0:
            raise ValueError(f"reference_dt_s must be positive, got {reference_dt_s}")
        self.reference_mjcf_path = reference_mjcf_path
        self._obs = np.zeros((1, self.OBS_DIM), dtype=np.float32)
        self._obs_parts = {
            "proprioception": np.zeros(93, dtype=np.float32),
            "motion_command": np.zeros(43, dtype=np.float32),
        }
        self._robot_init_yaw: float | None = None
        self._robot_init_xy = np.zeros(2, dtype=np.float32)
        self._ref_init_yaw: float | None = None
        self._ref_init_xy: np.ndarray | None = None
        self._reference_model = None
        self._reference_data = None
        self._reference_root_body_id: int | None = None
        self._reference_joint_qpos_indices: np.ndarray | None = None
        self._reference_joint_dof_indices: np.ndarray | None = None

        self._state_joint_indices = [
            self.state_processor.joint_names.index(name) for name in self.joint_names
        ]
        self._default_joint_pos = np.asarray(
            [
                self.env.default_dof_angles[self.env.joint_names_simulation.index(name)]
                for name in self.joint_names
            ],
            dtype=np.float32,
        )

        if len(self.joint_names) != 29:
            raise ValueError(
                "Humanoid-GPT PNS expects 29 policy joints, "
                f"got {len(self.joint_names)}"
            )

    def reset(self) -> None:
        super().reset()
        self._obs[:] = 0.0
        for part in self._obs_parts.values():
            part[:] = 0.0
        self._robot_init_yaw = None
        self._robot_init_xy[:] = 0.0
        self._ref_init_yaw = None
        self._ref_init_xy = None

    def _resolve_reference_mjcf_path(self) -> Path:
        if self.reference_mjcf_path is not None:
            path = Path(self.reference_mjcf_path).expanduser()
            if path.is_absolute():
                return path
            cwd_path = Path.cwd() / path
            if cwd_path.is_file():
                return cwd_path
            project_path = Path(__file__).resolve().parents[3] / path
            if project_path.is_file():
                return project_path
            return cwd_path

        robot_cfg = getattr(self.env, "robot_cfg", None)
        if robot_cfg is None:
            raise ValueError(
                "Humanoid-GPT reference_cvel_source='mujoco' requires env.robot_cfg "
                "or an explicit reference_mjcf_path"
            )
        return robot_cfg.resolve_mjcf_path()

    def _ensure_reference_model(self) -> None:
        if self._reference_model is not None:
            return

        import mujoco

        mjcf_path = self._resolve_reference_mjcf_path()
        if not mjcf_path.is_file():
            raise FileNotFoundError(f"Humanoid-GPT reference MJCF not found: {mjcf_path}")

        model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        data = mujoco.MjData(model)
        root_body_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            self.root_body_name,
        )
        if root_body_id < 0:
            raise ValueError(
                f"Humanoid-GPT reference MJCF is missing body {self.root_body_name!r}"
            )

        joint_qpos_indices: list[int] = []
        joint_dof_indices: list[int] = []
        for joint_name in self.joint_names:
            joint_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )
            if joint_id < 0:
                raise ValueError(
                    f"Humanoid-GPT reference MJCF is missing joint {joint_name!r}"
                )
            joint_qpos_indices.append(int(model.jnt_qposadr[joint_id]))
            joint_dof_indices.append(int(model.jnt_dofadr[joint_id]))

        self._reference_model = model
        self._reference_data = data
        self._reference_root_body_id = int(root_body_id)
        self._reference_joint_qpos_indices = np.asarray(joint_qpos_indices, dtype=np.int64)
        self._reference_joint_dof_indices = np.asarray(joint_dof_indices, dtype=np.int64)

    def _current_next_indices(self) -> tuple[int, int]:
        curr_idx = self._motion_step_index(0)
        next_idx = self._motion_step_index(1) if 1 in self.available_future_steps else curr_idx
        return curr_idx, next_idx

    def _ensure_root_alignment(
        self,
        ref_root_pos: np.ndarray,
        ref_root_quat: np.ndarray,
        robot_root_quat: np.ndarray,
    ) -> None:
        if self._robot_init_yaw is None:
            self._robot_init_yaw = float(_quat_to_yaw(robot_root_quat.reshape(1, 4))[0])
            self._robot_init_xy[:] = 0.0

        if self._ref_init_yaw is None:
            self._ref_init_yaw = float(_quat_to_yaw(ref_root_quat.reshape(1, 4))[0])
            self._ref_init_xy = np.asarray(ref_root_pos[:2], dtype=np.float32).copy()

    def _rebias_root_pose(
        self,
        ref_root_pos: np.ndarray,
        ref_root_quat: np.ndarray,
        robot_root_quat: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        ref_root_pos = np.asarray(ref_root_pos, dtype=np.float32).copy()
        ref_root_quat = np.asarray(ref_root_quat, dtype=np.float32).copy()
        self._ensure_root_alignment(ref_root_pos, ref_root_quat, robot_root_quat)

        assert self._robot_init_yaw is not None
        assert self._ref_init_yaw is not None
        assert self._ref_init_xy is not None

        yaw_offset = float(_wrap_pi(np.asarray(self._robot_init_yaw - self._ref_init_yaw)))
        ref_yaw = float(_quat_to_yaw(ref_root_quat.reshape(1, 4))[0])
        yaw_delta_from_init = float(_wrap_pi(np.asarray(ref_yaw - self._ref_init_yaw)))
        target_yaw = self._robot_init_yaw + yaw_delta_from_init
        yaw_delta = float(_wrap_pi(np.asarray(target_yaw - ref_yaw)))

        ref_root_pos[:2] = self._robot_init_xy + _rotate_xy(
            ref_root_pos[:2] - self._ref_init_xy,
            yaw_offset,
        )
        yaw_delta_quat = quat_from_yaw(np.asarray([yaw_delta], dtype=np.float32))
        ref_root_quat = quat_mul(yaw_delta_quat, ref_root_quat.reshape(1, 4))[0]
        ref_root_quat = ref_root_quat / np.linalg.norm(ref_root_quat)
        return ref_root_pos, ref_root_quat.astype(np.float32), yaw_offset

    def _reference_root_cvel_gv_from_mujoco(
        self,
        curr_root_pos: np.ndarray,
        curr_root_quat: np.ndarray,
        curr_joint_pos: np.ndarray,
        next_root_pos: np.ndarray,
        next_root_quat: np.ndarray,
        next_joint_pos: np.ndarray,
    ) -> np.ndarray:
        self._ensure_reference_model()

        import mujoco

        assert self._reference_model is not None
        assert self._reference_data is not None
        assert self._reference_root_body_id is not None
        assert self._reference_joint_qpos_indices is not None
        assert self._reference_joint_dof_indices is not None

        model = self._reference_model
        data = self._reference_data
        qpos = np.zeros(model.nq, dtype=np.float64)
        qvel = np.zeros(model.nv, dtype=np.float64)

        curr_root_pos = np.asarray(curr_root_pos, dtype=np.float64).reshape(3)
        curr_root_quat = np.asarray(curr_root_quat, dtype=np.float64).reshape(4)
        curr_root_quat /= max(float(np.linalg.norm(curr_root_quat)), 1.0e-12)
        curr_joint_pos = np.asarray(curr_joint_pos, dtype=np.float64).reshape(-1)

        next_root_pos = np.asarray(next_root_pos, dtype=np.float64).reshape(3)
        next_root_quat = np.asarray(next_root_quat, dtype=np.float64).reshape(4)
        next_root_quat /= max(float(np.linalg.norm(next_root_quat)), 1.0e-12)
        next_joint_pos = np.asarray(next_joint_pos, dtype=np.float64).reshape(-1)

        qpos[:3] = next_root_pos
        qpos[3:7] = next_root_quat
        qpos[self._reference_joint_qpos_indices] = next_joint_pos

        inv_dt = 1.0 / self.reference_dt_s
        qvel[:3] = (next_root_pos - curr_root_pos) * inv_dt
        q_delta = quat_mul(
            quat_conjugate(curr_root_quat.reshape(1, 4)),
            next_root_quat.reshape(1, 4),
        )[0]
        omega_body = _quat_to_rotvec(q_delta) * inv_dt
        qvel[3:6] = matrix_from_quat(curr_root_quat.reshape(1, 4))[0] @ omega_body
        qvel[self._reference_joint_dof_indices] = (
            next_joint_pos - curr_joint_pos
        ) * inv_dt

        data.qpos[:] = qpos
        data.qvel[:] = qvel
        mujoco.mj_forward(model, data)

        next_root_rot_w = matrix_from_quat(next_root_quat.reshape(1, 4))[0]
        next_gv_to_world = _batch_base_to_navi(next_root_rot_w.reshape(1, 3, 3))[0]
        root_cvel = np.asarray(data.cvel[self._reference_root_body_id], dtype=np.float32)
        return np.concatenate(
            [
                next_gv_to_world.T @ root_cvel[:3],
                next_gv_to_world.T @ root_cvel[3:],
            ],
            axis=0,
        ).astype(np.float32)

    def update(self, data: Dict[str, Any]) -> None:
        super().update(data)
        curr_idx, next_idx = self._current_next_indices()

        robot_root_quat = np.asarray(self.state_processor.root_quat_w, dtype=np.float32)
        robot_root_pos = np.asarray(self.state_processor.root_pos_w, dtype=np.float32)
        robot_joint_pos = np.asarray(
            self.state_processor.joint_pos[self._state_joint_indices],
            dtype=np.float32,
        )
        robot_joint_vel = np.asarray(
            self.state_processor.joint_vel[self._state_joint_indices],
            dtype=np.float32,
        )
        robot_gyro = np.asarray(self.state_processor.root_ang_vel_b, dtype=np.float32)
        robot_gravity = quat_rotate_inverse_numpy(
            robot_root_quat.reshape(1, 4),
            np.asarray([[0.0, 0.0, -1.0]], dtype=np.float32),
        )[0]

        last_action = np.asarray(
            data.get("action", np.zeros(len(self.joint_names), dtype=np.float32)),
            dtype=np.float32,
        ).reshape(-1)
        if last_action.size != len(self.joint_names):
            raise ValueError(
                "Humanoid-GPT previous action size mismatch: "
                f"expected {len(self.joint_names)}, got {last_action.size}"
            )

        curr_root_pos = np.asarray(self.ref_root_pos_future_w[0, curr_idx], dtype=np.float32)
        curr_root_quat = np.asarray(self.ref_root_quat_future_w[0, curr_idx], dtype=np.float32)
        next_root_pos = np.asarray(self.ref_root_pos_future_w[0, next_idx], dtype=np.float32)
        next_root_quat = np.asarray(self.ref_root_quat_future_w[0, next_idx], dtype=np.float32)
        curr_root_pos, curr_root_quat, yaw_offset = self._rebias_root_pose(
            curr_root_pos,
            curr_root_quat,
            robot_root_quat,
        )
        next_root_pos, next_root_quat, _ = self._rebias_root_pose(
            next_root_pos,
            next_root_quat,
            robot_root_quat,
        )
        next_joint_pos = np.asarray(self.ref_joint_pos_future[0, next_idx], dtype=np.float32)
        curr_joint_pos = np.asarray(self.ref_joint_pos_future[0, curr_idx], dtype=np.float32)

        next_root_rot_w = matrix_from_quat(next_root_quat.reshape(1, 4))[0]
        next_gv_to_world = _batch_base_to_navi(next_root_rot_w.reshape(1, 3, 3))[0]
        next_root_rot_gv = next_gv_to_world.T @ next_root_rot_w
        next_ref_gravity = -next_root_rot_gv.T[:, 2]

        if self.reference_cvel_source == "mujoco":
            next_root_cvel_gv = self._reference_root_cvel_gv_from_mujoco(
                curr_root_pos,
                curr_root_quat,
                curr_joint_pos,
                next_root_pos,
                next_root_quat,
                next_joint_pos,
            )
        else:
            next_root_cvel_w = np.concatenate(
                [
                    _rotate_vector_z(
                        self.ref_root_ang_vel_future_w[0, next_idx],
                        yaw_offset,
                    ),
                    _rotate_vector_z(
                        self.ref_root_lin_vel_future_w[0, next_idx],
                        yaw_offset,
                    ),
                ],
                axis=0,
            )
            next_root_cvel_gv = np.concatenate(
                [
                    next_gv_to_world.T @ next_root_cvel_w[:3],
                    next_gv_to_world.T @ next_root_cvel_w[3:],
                ],
                axis=0,
            ).astype(np.float32)

        yaw_d = _wrap_pi(
            _quat_to_yaw(curr_root_quat.reshape(1, 4))[0]
            - _quat_to_yaw(robot_root_quat.reshape(1, 4))[0]
        )
        xy_d = curr_root_pos[:2] - robot_root_pos[:2]
        yaw_curr = _quat_to_yaw(robot_root_quat.reshape(1, 4))[0]
        c, s = np.cos(-yaw_curr), np.sin(-yaw_curr)
        xy_d = np.asarray(
            [c * xy_d[0] - s * xy_d[1], s * xy_d[0] + c * xy_d[1]],
            dtype=np.float32,
        )

        self._obs_parts = {
            "proprioception": np.concatenate(
                [
                robot_gyro,
                robot_gravity,
                robot_joint_pos - self._default_joint_pos,
                robot_joint_vel,
                last_action,
                ],
                axis=0,
            ).astype(np.float32),
            "motion_command": np.concatenate(
                [
                next_joint_pos - self._default_joint_pos,
                np.asarray([next_root_pos[2]], dtype=np.float32),
                next_ref_gravity.astype(np.float32),
                next_root_cvel_gv,
                np.asarray([np.cos(yaw_d), np.sin(yaw_d)], dtype=np.float32),
                xy_d,
                ],
                axis=0,
            ).astype(np.float32),
        }
        obs = np.concatenate(list(self._obs_parts.values()), axis=0)

        if obs.size != self.OBS_DIM:
            raise ValueError(f"Humanoid-GPT obs dim mismatch: {obs.size} != {self.OBS_DIM}")
        self._obs[0, :] = obs

    def compute(self) -> np.ndarray:
        return self._obs


class humanoid_gpt_pns_component_obs(Observation, namespace="humanoid_gpt"):
    """One semantic Humanoid-GPT PNS input group."""

    def __init__(self, component: str, **kwargs: Any) -> None:
        if component not in {"proprioception", "motion_command"}:
            raise ValueError(f"Unsupported Humanoid-GPT component: {component!r}")
        super().__init__(env=kwargs["env"])
        self.component = component
        cache_name = "_humanoid_gpt_pns_semantic_observation"
        self._owns_core = not hasattr(self.env, cache_name)
        if self._owns_core:
            setattr(self.env, cache_name, humanoid_gpt_pns_obs(**kwargs))
        self._core = getattr(self.env, cache_name)

    def reset(self) -> None:
        if self._owns_core:
            self._core.reset()

    def update(self, data: Dict[str, Any]) -> None:
        if self._owns_core:
            self._core.update(data)

    def compute(self) -> np.ndarray:
        return self._core._obs_parts[self.component][None, :]
