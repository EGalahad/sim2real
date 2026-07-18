from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from sim2real.rl_policy.observations.base import Observation
from sim2real.rl_policy.observations.common import sort_names_by_preferred_order
from sim2real.rl_policy.utils.motion import MotionData
from sim2real.utils.strings import resolve_matching_names


class motion_obs(Observation, namespace="motion"):
    def __init__(
        self,
        future_steps: Optional[Union[Sequence[int], int]] = None,
        joint_names: Optional[Union[Sequence[str], str]] = None,
        body_names: Optional[Union[Sequence[str], str]] = None,
        root_body_name: Optional[str] = None,
        anchor_body_name: Optional[str] = None,
        joint_order: str = "simulation",
        body_order: str = "simulation",
        **kwargs,
    ):
        super().__init__(**kwargs)
        selected_future_steps = future_steps
        motion_cfg = dict(getattr(self.state_processor, "motion_config", {}) or {})

        motion_future_steps = getattr(self.state_processor, "motion_future_steps", None)
        if motion_future_steps is None or len(motion_future_steps) == 0:
            motion_future_steps = motion_cfg.get("future_steps")
        joint_names = motion_cfg.get("joint_names") if joint_names is None else joint_names
        body_names = motion_cfg.get("body_names") if body_names is None else body_names
        if motion_future_steps is None or joint_names is None or body_names is None:
            raise ValueError("motion observations require future_steps, joint_names, and body_names")

        self.motion_future_steps = np.asarray(motion_future_steps, dtype=int)
        if self.motion_future_steps.ndim != 1:
            raise ValueError(f"motion.future_steps must be 1D, got shape={self.motion_future_steps.shape}")
        self.available_future_steps = [int(step) for step in self.motion_future_steps.tolist()]
        if 0 not in self.available_future_steps:
            raise ValueError("motion.future_steps must include 0 to compute current observation")
        self.obs_current_step_index = int(self.available_future_steps.index(0))
        self.future_step_indices, self.future_steps = self._resolve_future_steps(selected_future_steps)
        self.selected_future_steps = self.future_steps
        self.n_future_steps = len(self.future_steps)
        self.n_selected_future_steps = self.n_future_steps
        self.joint_names = self._resolve_motion_names(
            joint_names,
            source_names=self.state_processor.motion_joint_names,
            preferred_names=self._preferred_joint_order(joint_order),
            order=joint_order,
            kind="joint",
        )
        self.body_names = self._resolve_motion_names(
            body_names,
            source_names=self.state_processor.motion_body_names,
            preferred_names=self._preferred_body_order(body_order),
            order=body_order,
            kind="body",
        )
        self.root_body_name = str(root_body_name or motion_cfg.get("root_body_name", "pelvis"))
        self.anchor_body_name = str(anchor_body_name or motion_cfg.get("anchor_body_name", "torso_link"))
        self.n_bodies = len(self.body_names)
        self._cached_motion_layout: Optional[Tuple[Tuple[str, ...], Tuple[str, ...]]] = None

    def _preferred_joint_order(self, order: str) -> Sequence[str]:
        if order == "simulation":
            return self.env.joint_names_simulation
        if order == "policy":
            return self.env.policy_joint_names
        return ()

    def _preferred_body_order(self, order: str) -> Sequence[str]:
        if order == "simulation":
            return self.env.body_names_simulation
        return ()

    def _resolve_motion_names(
        self,
        names: Union[Sequence[str], str],
        *,
        source_names: Sequence[str],
        preferred_names: Sequence[str],
        order: str,
        kind: str,
    ) -> list[str]:
        if isinstance(names, str):
            _, matched_names = resolve_matching_names(
                names,
                source_names,
                preserve_order=True,
            )
        else:
            matched_names = [str(name) for name in names]

        if order in {"given", "source"}:
            ordered_names = list(matched_names)
        elif order in {"simulation", "policy"}:
            ordered_names = sort_names_by_preferred_order(matched_names, preferred_names)
        else:
            raise ValueError(f"Unsupported {kind} order: {order!r}")

        missing = [name for name in ordered_names if name not in source_names]
        if missing:
            raise ValueError(f"Motion source missing {kind} names: {missing}")
        return ordered_names

    def _motion_step_index(self, step: int) -> int:
        step = int(step)
        if step not in self.available_future_steps:
            raise ValueError(
                f"motion step {step} not in motion.future_steps={self.available_future_steps}"
            )
        return int(self.available_future_steps.index(step))

    def _resolve_future_steps(
        self,
        future_steps: Optional[Union[Sequence[int], int]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        if future_steps is None:
            requested_future_steps = self.available_future_steps
        elif isinstance(future_steps, (int, np.integer)):
            requested_future_steps = [int(future_steps)]
        else:
            requested_future_steps = [int(step) for step in future_steps]

        if not requested_future_steps:
            raise ValueError("future_steps must select at least one step")

        future_step_indices = []
        for step in requested_future_steps:
            if step not in self.available_future_steps:
                raise ValueError(
                    f"future step {step} not in motion.future_steps={self.available_future_steps}"
                )
            future_step_indices.append(self.available_future_steps.index(step))

        return (
            np.asarray(future_step_indices, dtype=int),
            np.asarray(requested_future_steps, dtype=int),
        )

    def _select(self, x: np.ndarray) -> np.ndarray:
        return np.take(x, self.future_step_indices, axis=1)

    def reset(self):
        # State processor reset handles motion timing; this only refreshes cached views.
        self._assign_motion_views()

    def update(self, data: Dict[str, Any]) -> None:
        self._assign_motion_views()

    def _refresh_motion_indices(self) -> None:
        joint_names = tuple(self.state_processor.motion_joint_names)
        body_names = tuple(self.state_processor.motion_body_names)
        layout = (joint_names, body_names)
        if self._cached_motion_layout == layout:
            return
        if not joint_names or not body_names:
            raise ValueError("Motion source names are not ready")

        self._joint_indices = [joint_names.index(name) for name in self.joint_names]
        self._body_indices = [body_names.index(name) for name in self.body_names]
        self._root_body_idx = body_names.index(self.root_body_name)
        self._anchor_body_idx = body_names.index(self.anchor_body_name)
        self._cached_motion_layout = layout

    def _assign_motion_views(self):
        motion_data: MotionData | None = self.state_processor.motion_data
        if motion_data is None:
            raise ValueError("Motion source data is not ready")
        self._refresh_motion_indices()

        self.ref_joint_pos_future = motion_data.joint_pos[:, :, self._joint_indices]
        self.ref_joint_vel_future = motion_data.joint_vel[:, :, self._joint_indices]
        self.ref_body_pos_future_w = motion_data.body_pos_w[:, :, self._body_indices]
        self.ref_body_lin_vel_future_w = motion_data.body_lin_vel_w[:, :, self._body_indices]
        self.ref_body_quat_future_w = motion_data.body_quat_w[:, :, self._body_indices]
        self.ref_body_ang_vel_future_w = motion_data.body_ang_vel_w[:, :, self._body_indices]

        self.ref_joint_pos = motion_data.joint_pos[
            :, self.obs_current_step_index, self._joint_indices
        ]
        self.ref_joint_vel = motion_data.joint_vel[
            :, self.obs_current_step_index, self._joint_indices
        ]

        self.ref_root_pos_future_w = motion_data.body_pos_w[:, :, self._root_body_idx, :]
        self.ref_root_lin_vel_future_w = motion_data.body_lin_vel_w[:, :, self._root_body_idx, :]
        self.ref_root_quat_future_w = motion_data.body_quat_w[:, :, self._root_body_idx, :]
        self.ref_root_ang_vel_future_w = motion_data.body_ang_vel_w[:, :, self._root_body_idx, :]

        self.ref_root_pos_w = motion_data.body_pos_w[
            :, self.obs_current_step_index, self._root_body_idx, :
        ]
        self.ref_root_lin_vel_w = motion_data.body_lin_vel_w[
            :, self.obs_current_step_index, self._root_body_idx, :
        ]
        self.ref_root_quat_w = motion_data.body_quat_w[
            :, self.obs_current_step_index, self._root_body_idx, :
        ]
        self.ref_root_ang_vel_w = motion_data.body_ang_vel_w[
            :, self.obs_current_step_index, self._root_body_idx, :
        ]

        self.ref_anchor_pos_w = motion_data.body_pos_w[
            :, self.obs_current_step_index, self._anchor_body_idx, :
        ]
        self.ref_anchor_lin_vel_w = motion_data.body_lin_vel_w[
            :, self.obs_current_step_index, self._anchor_body_idx, :
        ]
        self.ref_anchor_quat_w = motion_data.body_quat_w[
            :, self.obs_current_step_index, self._anchor_body_idx, :
        ]
        self.ref_anchor_ang_vel_w = motion_data.body_ang_vel_w[
            :, self.obs_current_step_index, self._anchor_body_idx, :
        ]


class motion_body_obs(motion_obs, namespace="motion"):
    def __init__(
        self,
        body_names: Optional[Union[Sequence[str], str]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if body_names is None:
            body_names = self.body_names

        body_indices, matched_body_names = resolve_matching_names(
            body_names,
            self.body_names,
        )
        if not matched_body_names:
            raise ValueError("No tracking body matched for observation.")

        self.body_indices_tracking = np.asarray(body_indices, dtype=int)
        self.selected_body_names = matched_body_names
        self.n_selected_bodies = len(self.body_indices_tracking)

    def _select(self, x: np.ndarray) -> np.ndarray:
        x = super()._select(x)
        return np.take(x, self.body_indices_tracking, axis=2)


class motion_joint_obs(motion_obs, namespace="motion"):
    def __init__(
        self,
        joint_names: Optional[Union[Sequence[str], str]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if joint_names is None:
            joint_names = self.joint_names

        joint_indices, matched_joint_names = resolve_matching_names(
            joint_names,
            self.joint_names,
        )
        if not matched_joint_names:
            raise ValueError("No tracking joint matched for observation.")

        self.joint_indices_tracking = np.asarray(joint_indices, dtype=int)
        self.selected_joint_names = matched_joint_names
        self.n_selected_joints = len(self.joint_indices_tracking)

    def _select(self, x: np.ndarray) -> np.ndarray:
        x = super()._select(x)
        return np.take(x, self.joint_indices_tracking, axis=2)
