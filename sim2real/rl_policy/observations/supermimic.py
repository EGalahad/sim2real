"""Raw inputs of a SuperMimic tracking ONNX (humanoid_s0 scripts/supermimic/export_onnx.py).

The ONNX builds the goal geometry itself; these terms only hand over
    proprio            history of [joint pos - default, joint vel, root angular velocity (body), projected gravity,
                       previous action], oldest first
    ref_joint          reference joint pos and vel per future step
    ref_anchor         reference anchor world position and wxyz quaternion per future step
    robot_anchor_quat  robot anchor world quaternion from forward kinematics, yaw-aligned to the motion at reset
Motion step 0 is the frame the next action should reach, so the policy's "current" reference frame is step -1.
"""
from __future__ import annotations

from typing import Any, Dict

import mujoco
import numpy as np

from sim2real.rl_policy.observations.base import Observation
from sim2real.rl_policy.observations.motion import motion_joint_obs, motion_obs
from sim2real.utils.math import projected_yaw_quat, quat_conjugate, quat_mul, quat_rotate_inverse_numpy


class proprio_history(Observation, namespace="supermimic"):
    def __init__(self, history_steps: int = 10, **kwargs):
        super().__init__(**kwargs)
        names = list(self.state_processor.joint_names)
        self.joint_ids = np.asarray([names.index(n) for n in self.env.policy_joint_names], dtype=int)
        self.default = np.asarray(self.env.default_dof_angles, dtype=np.float32)[
            [list(self.env.joint_names_simulation).index(n) for n in self.env.policy_joint_names]]
        self.n_act = len(self.joint_ids)
        self.hist = np.zeros((history_steps, 2 * self.n_act + 6 + self.n_act), dtype=np.float32)

    def _frame(self, prev_action: np.ndarray) -> np.ndarray:
        sp = self.state_processor
        gravity = quat_rotate_inverse_numpy(sp.root_quat_w[None], np.array([[0.0, 0.0, -1.0]]))[0]
        return np.concatenate([sp.joint_pos[self.joint_ids] - self.default, sp.joint_vel[self.joint_ids],
                               sp.root_ang_vel_b, gravity, prev_action]).astype(np.float32)

    def reset(self):
        self.hist[:] = self._frame(np.zeros(self.n_act, dtype=np.float32))

    def update(self, data: Dict[str, Any]) -> None:
        self.hist = np.roll(self.hist, -1, axis=0)
        self.hist[-1] = self._frame(np.asarray(data["action"], dtype=np.float32))

    def compute(self) -> np.ndarray:
        return self.hist.reshape(-1)


class _policy_order_joint_obs(motion_joint_obs):
    """Reference joints in ``policy_joint_names`` order, the order the ONNX and the action use."""

    def __init__(self, **kwargs):
        kwargs.setdefault("joint_order", "policy")
        super().__init__(**kwargs)


class ref_joint(_policy_order_joint_obs, namespace="supermimic"):
    def compute(self) -> np.ndarray:
        jp, jv = self._select(self.ref_joint_pos_future), self._select(self.ref_joint_vel_future)
        return np.concatenate([jp, jv], -1).reshape(-1)


class ref_anchor(motion_obs, namespace="supermimic"):
    def compute(self) -> np.ndarray:
        md = self.env.motion_data
        pos = self._select(md.body_pos_w[:, :, self._anchor_body_idx])
        quat = self._select(md.body_quat_w[:, :, self._anchor_body_idx])
        return np.concatenate([pos, quat], -1).reshape(-1)


class robot_anchor_quat(motion_obs, namespace="supermimic"):
    """Robot anchor-body orientation by forward kinematics of the measured root orientation and joint angles,
    expressed in the motion's world frame: the yaw offset between robot and motion root is fixed at reset."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = mujoco.MjModel.from_xml_path(str(self.env.robot_cfg.resolve_mjcf_path()))
        self.data = mujoco.MjData(self.model)
        self.body = self.model.body(self.anchor_body_name).id
        free = [i for i in range(self.model.njnt) if self.model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE][0]
        self.root_adr = int(self.model.jnt_qposadr[free])
        self.joint_adr = np.asarray([self.model.jnt_qposadr[self.model.joint(n).id] for n in self.state_processor.joint_names])
        self.offset = np.array([1.0, 0.0, 0.0, 0.0])

    def _fk(self) -> np.ndarray:
        sp = self.state_processor
        self.data.qpos[self.root_adr + 3:self.root_adr + 7] = sp.root_quat_w
        self.data.qpos[self.joint_adr] = sp.joint_pos
        mujoco.mj_kinematics(self.model, self.data)
        return self.data.xquat[self.body].copy()

    def reset(self):
        super().reset()
        ref = self.env.motion_data.body_quat_w[0, self._motion_step_index(-1), self._root_body_idx]
        self.offset = quat_mul(projected_yaw_quat(ref),
                               quat_conjugate(projected_yaw_quat(self.state_processor.root_quat_w)))

    def compute(self) -> np.ndarray:
        return quat_mul(self.offset, self._fk()).astype(np.float32).reshape(-1)


class ref_joint_pos(_policy_order_joint_obs, namespace="supermimic"):
    def compute(self) -> np.ndarray:
        return self._select(self.ref_joint_pos_future).reshape(-1)


class residual_proprio_history(_policy_order_joint_obs, namespace="supermimic"):
    """proprio_history of a residual-action policy (NablaMimic --residual-action): the setpoint is the next reference
    frame's joint pos + scale * a, so the previous action is handed over as a = output - (ref - default) / scale,
    with ref the reference at motion step -1 after the step (the frame that action aimed at); 0 after a reset."""

    def __init__(self, history_steps: int = 10, **kwargs):
        super().__init__(future_steps=[-1], **kwargs)
        if list(self.selected_joint_names) != list(self.env.policy_joint_names):
            raise ValueError("residual_proprio_history needs the reference joints in policy_joint_names order")
        names = list(self.state_processor.joint_names)
        self.joint_ids = np.asarray([names.index(n) for n in self.env.policy_joint_names], dtype=int)
        self.default = np.asarray(self.env.default_dof_angles, dtype=np.float32)[
            [list(self.env.joint_names_simulation).index(n) for n in self.env.policy_joint_names]]
        self.n_act = len(self.joint_ids)
        self.hist = np.zeros((history_steps, 3 * self.n_act + 6), dtype=np.float32)

    def _frame(self, prev_action: np.ndarray) -> np.ndarray:
        sp = self.state_processor
        gravity = quat_rotate_inverse_numpy(sp.root_quat_w[None], np.array([[0.0, 0.0, -1.0]]))[0]
        return np.concatenate([sp.joint_pos[self.joint_ids] - self.default, sp.joint_vel[self.joint_ids],
                               sp.root_ang_vel_b, gravity, prev_action]).astype(np.float32)

    def reset(self):
        super().reset()
        self.hist[:] = self._frame(np.zeros(self.n_act, dtype=np.float32))

    def update(self, data: Dict[str, Any]) -> None:
        super().update(data)
        ref = self._select(self.ref_joint_pos_future).reshape(-1)
        a = np.asarray(data["action"], dtype=np.float32) - (ref - self.default) / np.asarray(self.env.action_scale)
        self.hist = np.roll(self.hist, -1, axis=0)
        self.hist[-1] = self._frame(a.astype(np.float32))

    def compute(self) -> np.ndarray:
        return self.hist.reshape(-1)


class ref_root(motion_obs, namespace="supermimic"):
    """Reference root-body world position and wxyz quaternion per future step."""

    def compute(self) -> np.ndarray:
        md = self.env.motion_data
        pos = self._select(md.body_pos_w[:, :, self._root_body_idx])
        quat = self._select(md.body_quat_w[:, :, self._root_body_idx])
        return np.concatenate([pos, quat], -1).reshape(-1)


class root_quat_history(robot_anchor_quat, namespace="supermimic"):
    """History of [robot root quaternion in the motion frame, reference root quaternion at motion step -1], oldest
    first (BlinkMimic observes the reference-relative root orientation at every history step)."""

    def __init__(self, history_steps: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.hist = np.zeros((history_steps, 8), dtype=np.float32)
        self.cur = self._motion_step_index(-1)

    def _frame(self) -> np.ndarray:
        ref = self.env.motion_data.body_quat_w[0, self.cur, self._root_body_idx]
        return np.concatenate([quat_mul(self.offset, self.state_processor.root_quat_w), ref]).astype(np.float32)

    def reset(self):
        super().reset()
        self.hist[:] = self._frame()

    def update(self, data: Dict[str, Any]) -> None:
        super().update(data)
        self.hist = np.roll(self.hist, -1, axis=0)
        self.hist[-1] = self._frame()

    def compute(self) -> np.ndarray:
        return self.hist.reshape(-1)


class ref_bodies(motion_obs, namespace="supermimic"):
    """Reference world position and wxyz quaternion of ``body_names`` (in the given order) per future step, step-major
    (PartyMimic --interface sparse5 keypoints)."""

    def __init__(self, **kwargs):
        kwargs.setdefault("body_order", "given")
        super().__init__(**kwargs)

    def compute(self) -> np.ndarray:
        pos = self._select(self.ref_body_pos_future_w)
        quat = self._select(self.ref_body_quat_future_w)
        return np.concatenate([pos, quat], -1).reshape(-1).astype(np.float32)
