from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence

import mujoco
import numpy as np

from sim2real.rl_policy.observations.base import Observation
from sim2real.rl_policy.observations.motion import motion_obs


_GRAVITY_UNIT_W = np.asarray([0.0, 0.0, -1.0], dtype=np.float32)


def _resolve_xml_path(env: Any, xml_path: str) -> Path:
    path = Path(str(xml_path)).expanduser()
    candidates: list[Path]
    if path.is_absolute():
        candidates = [path]
    else:
        candidates = [Path.cwd() / path]
        policy_config = getattr(getattr(env, "args", None), "policy_config", None)
        if policy_config is not None:
            candidates.append(Path(policy_config).expanduser().resolve().parent / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    candidate_text = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Teleopit XML path {xml_path!r} not found; tried: {candidate_text}")


def _quat_inv_np(q: np.ndarray) -> np.ndarray:
    inv = np.asarray(q, dtype=np.float32).copy()
    inv[..., 1:] = -inv[..., 1:]
    return inv


def _quat_mul_np(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    q1 = np.asarray(q1, dtype=np.float32)
    q2 = np.asarray(q2, dtype=np.float32)
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    ).astype(np.float32)


def _quat_rotate_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    v_quat = np.zeros((*v.shape[:-1], 4), dtype=np.float32)
    v_quat[..., 1:4] = v
    result = _quat_mul_np(_quat_mul_np(q, v_quat), _quat_inv_np(q))
    return result[..., 1:4]


def _yaw_quat_np(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32).reshape(4)
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    out = np.asarray([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)], dtype=np.float32)
    out /= max(float(np.linalg.norm(out)), 1e-8)
    return out


def _quat_to_rot6d_np(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32).reshape(4)
    w, x, y, z = q[0], q[1], q[2], q[3]
    return np.asarray(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
        ],
        dtype=np.float32,
    )


def _compute_fixed_yaw_alignment_quat(robot_quat_wxyz: np.ndarray, motion_quat_wxyz: np.ndarray) -> np.ndarray:
    delta = _quat_mul_np(
        np.asarray(robot_quat_wxyz, dtype=np.float32),
        _quat_inv_np(np.asarray(motion_quat_wxyz, dtype=np.float32)),
    )
    return _yaw_quat_np(delta)


def _rotate_motion_qpos_by_yaw(
    motion_qpos: np.ndarray,
    yaw_offset_quat_wxyz: np.ndarray,
    pivot_pos_w: np.ndarray,
) -> np.ndarray:
    base_pos = np.asarray(motion_qpos[0:3], dtype=np.float32)
    base_quat = np.asarray(motion_qpos[3:7], dtype=np.float32)
    yaw_offset = np.asarray(yaw_offset_quat_wxyz, dtype=np.float32).reshape(4)
    pivot = np.asarray(pivot_pos_w, dtype=np.float32).reshape(3)
    motion_qpos[0:3] = (_quat_rotate_np(yaw_offset, base_pos - pivot) + pivot).astype(motion_qpos.dtype)
    motion_qpos[3:7] = _quat_mul_np(yaw_offset, base_quat).astype(motion_qpos.dtype)
    return motion_qpos


@dataclass
class _TeleopitVelCmdCache:
    env: Any
    xml_path: str
    anchor_body_name: str
    joint_names: list[str]
    default_dof_pos: np.ndarray
    history_length: int
    policy_hz: float
    align_reference_yaw: bool
    obs_dim: int = 166

    def __post_init__(self) -> None:
        self.xml_path = str(_resolve_xml_path(self.env, self.xml_path))
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)
        self.anchor_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            self.anchor_body_name,
        )
        if self.anchor_body_id < 0:
            raise ValueError(f"Teleopit anchor body {self.anchor_body_name!r} not found in {self.xml_path}")

        if len(self.joint_names) != 29:
            raise ValueError(f"Teleopit G1 policy expects 29 joints, got {len(self.joint_names)}")
        self.state_joint_indices = [
            self.env.state_processor.joint_names.index(name)
            for name in self.joint_names
        ]
        self.current_obs = np.zeros((1, self.obs_dim), dtype=np.float32)
        self.obs_history = np.zeros((1, self.history_length, self.obs_dim), dtype=np.float32)
        self.history_initialized = False
        self.fixed_reference_yaw_quat: np.ndarray | None = None
        self.fixed_reference_pivot_pos_w: np.ndarray | None = None
        self.fixed_reference_xy_offset_w: np.ndarray | None = None
        self.last_reference_qpos: np.ndarray | None = None

    def reset(self) -> None:
        self.current_obs[:] = 0.0
        self.obs_history[:] = 0.0
        self.history_initialized = False
        self.fixed_reference_yaw_quat = None
        self.fixed_reference_pivot_pos_w = None
        self.fixed_reference_xy_offset_w = None
        self.last_reference_qpos = None

    def _run_fk(self, base_pos: np.ndarray, base_quat: np.ndarray, joint_pos: np.ndarray) -> None:
        self.data.qpos[:] = 0.0
        self.data.qpos[0:3] = np.asarray(base_pos, dtype=np.float64).reshape(3)
        quat = np.asarray(base_quat, dtype=np.float64).reshape(4)
        quat = quat / max(float(np.linalg.norm(quat)), 1.0e-8)
        self.data.qpos[3:7] = quat
        n = min(len(joint_pos), self.model.nq - 7)
        self.data.qpos[7 : 7 + n] = np.asarray(joint_pos, dtype=np.float64)[:n]
        mujoco.mj_kinematics(self.model, self.data)

    def _anchor_pos_quat(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)
        self._run_fk(qpos[0:3], qpos[3:7], qpos[7 : 7 + len(self.joint_names)])
        return (
            np.asarray(self.data.xpos[self.anchor_body_id], dtype=np.float32).copy(),
            np.asarray(self.data.xquat[self.anchor_body_id], dtype=np.float32).copy(),
        )

    def _current_motion_qpos(self, motion_view: motion_obs) -> np.ndarray:
        root_pos = np.asarray(motion_view.ref_root_pos_w[0], dtype=np.float32)
        root_quat = np.asarray(motion_view.ref_root_quat_w[0], dtype=np.float32)
        joint_pos = np.asarray(motion_view.ref_joint_pos[0], dtype=np.float32)
        qpos = np.concatenate([root_pos, root_quat, joint_pos], axis=0).astype(np.float32)
        if not self.align_reference_yaw:
            return qpos

        robot_quat = np.asarray(self.env.state_processor.root_quat_w, dtype=np.float32)
        if self.fixed_reference_yaw_quat is None:
            self.fixed_reference_yaw_quat = _compute_fixed_yaw_alignment_quat(robot_quat, qpos[3:7])
            self.fixed_reference_pivot_pos_w = np.asarray(qpos[0:3], dtype=np.float32).copy()
        qpos = qpos.copy()
        _rotate_motion_qpos_by_yaw(
            qpos,
            self.fixed_reference_yaw_quat,
            self.fixed_reference_pivot_pos_w,
        )
        if self.fixed_reference_xy_offset_w is not None:
            qpos[0:2] = (
                np.asarray(qpos[0:2], dtype=np.float32)
                + np.asarray(self.fixed_reference_xy_offset_w, dtype=np.float32).reshape(2)
            ).astype(qpos.dtype)
        return qpos

    def _compute_motion_joint_vel(self, motion_qpos: np.ndarray) -> np.ndarray:
        if self.last_reference_qpos is None:
            return np.zeros(len(self.joint_names), dtype=np.float32)
        return np.asarray(
            (motion_qpos[7 : 7 + len(self.joint_names)]
            - self.last_reference_qpos[7 : 7 + len(self.joint_names)])
            * np.float32(self.policy_hz),
            dtype=np.float32,
        )

    def _compute_anchor_velocities(self, motion_qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cur_anchor_pos, cur_anchor_quat = self._anchor_pos_quat(motion_qpos)
        if self.last_reference_qpos is None:
            return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

        prev_anchor_pos, prev_anchor_quat = self._anchor_pos_quat(self.last_reference_qpos)
        dt = np.float32(1.0 / self.policy_hz)
        lin_vel = np.asarray((cur_anchor_pos - prev_anchor_pos) / dt, dtype=np.float32)

        q_delta = _quat_mul_np(cur_anchor_quat, _quat_inv_np(prev_anchor_quat))
        if q_delta[0] < 0:
            q_delta = -q_delta
        w_clamped = float(np.clip(q_delta[0], -1.0, 1.0))
        half_angle = np.float32(np.arccos(w_clamped))
        sin_half = np.float32(np.sin(half_angle))
        if sin_half > 1.0e-6:
            axis = q_delta[1:4] / sin_half
            ang_vel = np.asarray(axis * 2.0 * half_angle / dt, dtype=np.float32)
        else:
            ang_vel = np.zeros(3, dtype=np.float32)

        if not np.all(np.isfinite(lin_vel)):
            lin_vel = np.zeros(3, dtype=np.float32)
        if not np.all(np.isfinite(ang_vel)):
            ang_vel = np.zeros(3, dtype=np.float32)
        return lin_vel, ang_vel

    def update(self, data: Dict[str, Any], motion_view: motion_obs) -> None:
        state = self.env.state_processor
        robot_joint_pos = np.asarray(state.joint_pos[self.state_joint_indices], dtype=np.float32)
        robot_joint_vel = np.asarray(state.joint_vel[self.state_joint_indices], dtype=np.float32)
        robot_quat = np.asarray(state.root_quat_w, dtype=np.float32).reshape(4)
        robot_ang_vel = np.asarray(state.root_ang_vel_b, dtype=np.float32).reshape(3)
        robot_base_pos = np.asarray(state.root_pos_w, dtype=np.float32).reshape(3)

        motion_qpos = self._current_motion_qpos(motion_view)
        motion_joint_pos = motion_qpos[7 : 7 + len(self.joint_names)]
        motion_joint_vel = self._compute_motion_joint_vel(motion_qpos)
        anchor_lin_vel_w, anchor_ang_vel_w = self._compute_anchor_velocities(motion_qpos)

        last_action = np.asarray(
            data.get("action", np.zeros(len(self.joint_names), dtype=np.float32)),
            dtype=np.float32,
        ).reshape(-1)
        if last_action.shape[0] != len(self.joint_names):
            raise ValueError(
                f"Teleopit last_action length mismatch: expected {len(self.joint_names)}, got {last_action.shape[0]}"
            )

        self._run_fk(np.zeros(3, dtype=np.float32), robot_quat, robot_joint_pos)
        robot_anchor_quat = np.asarray(self.data.xquat[self.anchor_body_id], dtype=np.float32).copy()

        _, motion_anchor_quat = self._anchor_pos_quat(motion_qpos)

        rel_quat = _quat_mul_np(_quat_inv_np(robot_anchor_quat), motion_anchor_quat)
        motion_anchor_ori_b = _quat_to_rot6d_np(rel_quat)
        joint_pos_rel = robot_joint_pos - self.default_dof_pos

        base_obs = np.concatenate(
            [
                motion_joint_pos,
                motion_joint_vel,
                motion_anchor_ori_b,
                robot_ang_vel,
                joint_pos_rel,
                robot_joint_vel,
                last_action,
            ],
            dtype=np.float32,
        )
        if base_obs.shape[0] != 154:
            raise ValueError(f"Teleopit base obs dim mismatch: {base_obs.shape[0]} != 154")

        projected_gravity = _quat_rotate_np(_quat_inv_np(robot_quat), _GRAVITY_UNIT_W)
        self._run_fk(robot_base_pos, robot_quat, robot_joint_pos)
        robot_anchor_quat_with_pos = np.asarray(self.data.xquat[self.anchor_body_id], dtype=np.float32).copy()
        robot_inv = _quat_inv_np(robot_anchor_quat_with_pos)
        ref_base_lin_vel_b = _quat_rotate_np(robot_inv, anchor_lin_vel_w)
        ref_base_ang_vel_b = _quat_rotate_np(robot_inv, anchor_ang_vel_w)
        ref_projected_gravity_b = _quat_rotate_np(_quat_inv_np(motion_anchor_quat), _GRAVITY_UNIT_W)

        obs = np.concatenate(
            [
                base_obs,
                projected_gravity,
                ref_base_lin_vel_b,
                ref_base_ang_vel_b,
                ref_projected_gravity_b,
            ],
            dtype=np.float32,
        )
        if obs.shape[0] != self.obs_dim:
            raise ValueError(f"Teleopit obs dim mismatch: {obs.shape[0]} != {self.obs_dim}")
        if not np.all(np.isfinite(obs)):
            obs = np.where(np.isfinite(obs), obs, np.float32(0.0))

        self.current_obs[0, :] = obs
        if not self.history_initialized:
            self.obs_history[0, :, :] = obs.reshape(1, -1)
            self.history_initialized = True
        else:
            self.obs_history[0, :-1, :] = self.obs_history[0, 1:, :]
            self.obs_history[0, -1, :] = obs
        self.last_reference_qpos = motion_qpos.copy()


def _get_cache(
    env: Any,
    *,
    xml_path: str,
    anchor_body_name: str,
    joint_names: Sequence[str],
    default_dof_pos: Sequence[float],
    history_length: int,
    policy_hz: float,
    align_reference_yaw: bool,
) -> _TeleopitVelCmdCache:
    cache = getattr(env, "_teleopit_velcmd_cache", None)
    if cache is None:
        cache = _TeleopitVelCmdCache(
            env=env,
            xml_path=str(xml_path),
            anchor_body_name=str(anchor_body_name),
            joint_names=[str(name) for name in joint_names],
            default_dof_pos=np.asarray(default_dof_pos, dtype=np.float32),
            history_length=int(history_length),
            policy_hz=float(policy_hz),
            align_reference_yaw=bool(align_reference_yaw),
        )
        setattr(env, "_teleopit_velcmd_cache", cache)
    return cache


class teleopit_velcmd_obs(motion_obs, namespace="teleopit"):
    def __init__(
        self,
        xml_path: str,
        anchor_body_name: str = "torso_link",
        joint_names: Sequence[str] | None = None,
        default_dof_pos: Sequence[float] | None = None,
        history_length: int = 10,
        policy_hz: float = 50.0,
        align_reference_yaw: bool = True,
        **kwargs: Any,
    ) -> None:
        env = kwargs["env"]
        resolved_joint_names = (
            list(joint_names) if joint_names is not None else list(env.policy_joint_names)
        )
        super().__init__(
            future_steps=[0],
            joint_names=resolved_joint_names,
            body_names=["pelvis"],
            root_body_name="pelvis",
            anchor_body_name="pelvis",
            joint_order="given",
            body_order="given",
            **kwargs,
        )
        resolved_default_dof_pos = default_dof_pos if default_dof_pos is not None else self.env.default_dof_angles
        self.cache = _get_cache(
            self.env,
            xml_path=xml_path,
            anchor_body_name=anchor_body_name,
            joint_names=list(resolved_joint_names),
            default_dof_pos=list(resolved_default_dof_pos),
            history_length=history_length,
            policy_hz=policy_hz,
            align_reference_yaw=align_reference_yaw,
        )

    def reset(self) -> None:
        super().reset()
        self.cache.reset()

    def update(self, data: Dict[str, Any]) -> None:
        super().update(data)
        self.cache.update(data, self)

    def compute(self) -> np.ndarray:
        return self.cache.current_obs


class teleopit_velcmd_obs_history(Observation, namespace="teleopit"):
    def __init__(
        self,
        xml_path: str,
        anchor_body_name: str = "torso_link",
        joint_names: Sequence[str] | None = None,
        default_dof_pos: Sequence[float] | None = None,
        history_length: int = 10,
        policy_hz: float = 50.0,
        align_reference_yaw: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        resolved_joint_names = joint_names if joint_names is not None else self.env.policy_joint_names
        resolved_default_dof_pos = default_dof_pos if default_dof_pos is not None else self.env.default_dof_angles
        self.cache = _get_cache(
            self.env,
            xml_path=xml_path,
            anchor_body_name=anchor_body_name,
            joint_names=list(resolved_joint_names),
            default_dof_pos=list(resolved_default_dof_pos),
            history_length=history_length,
            policy_hz=policy_hz,
            align_reference_yaw=align_reference_yaw,
        )

    def compute(self) -> np.ndarray:
        return self.cache.obs_history
