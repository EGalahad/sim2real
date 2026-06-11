from __future__ import annotations

from pathlib import Path
from typing import Any, List

import numpy as np
import torch

from any4hdmi import BaseDataset as Any4HDMIBaseDataset, load_any4hdmi_dataset
from sim2real.config.robots.base import RobotCfg
from sim2real.utils.strings import resolve_matching_names

_MOTION_FIELD_NAMES = (
    "motion_id",
    "step",
    "body_pos_w",
    "body_lin_vel_w",
    "body_quat_w",
    "body_ang_vel_w",
    "joint_pos",
    "joint_vel",
)


def _normalize_quat_batch(quat_wxyz: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(quat_wxyz, axis=-1, keepdims=True)
    denom = np.clip(denom, eps, None)
    return quat_wxyz / denom


def _quat_slerp_batch(
    q0_wxyz: np.ndarray,
    q1_wxyz: np.ndarray,
    alpha,
    *,
    normalize_inputs: bool = True,
    eps: float = 1e-12,
) -> np.ndarray:
    q0 = np.asarray(q0_wxyz)
    q1 = np.asarray(q1_wxyz)
    if normalize_inputs:
        q0 = _normalize_quat_batch(q0, eps=eps)
        q1 = _normalize_quat_batch(q1, eps=eps)

    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    flip_mask = dot < 0.0
    q1 = np.where(flip_mask, -q1, q1)
    dot = np.where(flip_mask, -dot, dot)
    dot = np.clip(dot, -1.0, 1.0)

    alpha_arr = np.asarray(alpha, dtype=q0.dtype)
    while alpha_arr.ndim < dot.ndim:
        alpha_arr = np.expand_dims(alpha_arr, axis=-1)

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * alpha_arr

    safe_denom = np.where(sin_theta_0 > eps, sin_theta_0, 1.0)
    s0 = np.sin(theta_0 - theta) / safe_denom
    s1 = np.sin(theta) / safe_denom
    slerp_out = s0 * q0 + s1 * q1

    nlerp_out = (1.0 - alpha_arr) * q0 + alpha_arr * q1
    out = np.where(dot > 0.9995, nlerp_out, slerp_out)
    return _normalize_quat_batch(out, eps=eps)


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _reorder_joint_field(
    field: np.ndarray,
    *,
    src_joint_indices: np.ndarray,
    dest_joint_indices: np.ndarray,
    joint_dim: int,
) -> np.ndarray:
    reordered = np.zeros((*field.shape[:-1], joint_dim), dtype=field.dtype)
    reordered[..., dest_joint_indices] = field[..., src_joint_indices]
    return reordered


def _motion_data_to_numpy(
    motion_data: Any,
    *,
    src_joint_indices: np.ndarray,
    dest_joint_indices: np.ndarray,
    joint_dim: int,
) -> "MotionData":
    result: dict[str, np.ndarray] = {}
    for field_name in _MOTION_FIELD_NAMES:
        if hasattr(motion_data, field_name):
            result[field_name] = _to_numpy(getattr(motion_data, field_name))

    result["joint_pos"] = _reorder_joint_field(
        result["joint_pos"],
        src_joint_indices=src_joint_indices,
        dest_joint_indices=dest_joint_indices,
        joint_dim=joint_dim,
    )
    result["joint_vel"] = _reorder_joint_field(
        result["joint_vel"],
        src_joint_indices=src_joint_indices,
        dest_joint_indices=dest_joint_indices,
        joint_dim=joint_dim,
    )
    return MotionData(**result)


class MotionData:
    """Container for motion data arrays."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if key != "batch_size":
                setattr(self, key, value)

    def __getitem__(self, idx):
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                result[key] = value[idx]
        return MotionData(**result)


class MotionDataset:
    """Numpy adapter over any4hdmi full-motion datasets."""

    def __init__(
        self,
        *,
        dataset: Any4HDMIBaseDataset,
        body_names: List[str],
        joint_names: List[str],
        src_joint_indices: List[int],
        dest_joint_indices: List[int],
        motion_ids: List[int] | np.ndarray | None = None,
    ):
        self._dataset = dataset
        self.body_names = list(body_names)
        self.joint_names = list(joint_names)
        self._src_joint_indices = np.asarray(src_joint_indices, dtype=np.int64)
        self._dest_joint_indices = np.asarray(dest_joint_indices, dtype=np.int64)
        if motion_ids is None:
            self._motion_ids = np.arange(int(dataset.num_motions), dtype=np.int64)
        else:
            self._motion_ids = np.asarray(motion_ids, dtype=np.int64)
        self._source_starts = _to_numpy(dataset.starts).astype(np.int64, copy=True)
        self._source_ends = _to_numpy(dataset.ends).astype(np.int64, copy=True)
        self._storage = self._build_numpy_storage(dataset)
        self._refresh_selected_bounds()

    def _build_numpy_storage(self, dataset: Any4HDMIBaseDataset) -> dict[str, np.ndarray]:
        storage: dict[str, np.ndarray] = {}
        for field_name in _MOTION_FIELD_NAMES:
            if hasattr(dataset.data, field_name):
                storage[field_name] = _to_numpy(getattr(dataset.data, field_name)).copy()

        storage["joint_pos"] = _reorder_joint_field(
            storage["joint_pos"],
            src_joint_indices=self._src_joint_indices,
            dest_joint_indices=self._dest_joint_indices,
            joint_dim=len(self.joint_names),
        )
        storage["joint_vel"] = _reorder_joint_field(
            storage["joint_vel"],
            src_joint_indices=self._src_joint_indices,
            dest_joint_indices=self._dest_joint_indices,
            joint_dim=len(self.joint_names),
        )
        return storage

    def _refresh_selected_bounds(self) -> None:
        selected_starts = self._source_starts[self._motion_ids]
        selected_ends = self._source_ends[self._motion_ids]
        self.starts = np.zeros_like(selected_starts)
        if self.starts.size > 1:
            self.starts[1:] = np.cumsum(selected_ends[:-1] - selected_starts[:-1])
        self.ends = self.starts + (selected_ends - selected_starts)
        self.lengths = self.ends - self.starts

    def select_motions(self, motion_ids: List[int] | np.ndarray) -> "MotionDataset":
        selected = object.__new__(MotionDataset)
        selected._dataset = self._dataset
        selected.body_names = self.body_names
        selected.joint_names = self.joint_names
        selected._src_joint_indices = self._src_joint_indices
        selected._dest_joint_indices = self._dest_joint_indices
        selected._source_starts = self._source_starts
        selected._source_ends = self._source_ends
        selected._storage = self._storage
        selected._motion_ids = self._motion_ids[np.asarray(motion_ids, dtype=np.int64)]
        selected._refresh_selected_bounds()
        return selected

    @classmethod
    def create_from_path(
        cls,
        root_path: str,
        robot_cfg: RobotCfg,
        target_fps: int = 50,
    ) -> "MotionDataset":
        import sim2real

        base_dir = Path(sim2real.__file__).parent.parent
        dataset = load_any4hdmi_dataset(
            root_path=root_path,
            target_fps=target_fps,
            base_dir=base_dir,
            num_envs=1,
            full_motion=True,
        )

        canonical_joint_names = list(robot_cfg.joint_names)
        source_joint_names = list(dataset.joint_names)

        shared_joint_names = [name for name in source_joint_names if name in canonical_joint_names]
        src_joint_indices = [source_joint_names.index(name) for name in shared_joint_names]
        dest_joint_indices = [canonical_joint_names.index(name) for name in shared_joint_names]

        extra_joint_names = [name for name in source_joint_names if name not in canonical_joint_names]
        src_joint_indices.extend(source_joint_names.index(name) for name in extra_joint_names)
        dest_joint_indices.extend(len(canonical_joint_names) + idx for idx in range(len(extra_joint_names)))
        joint_names = canonical_joint_names + extra_joint_names

        return cls(
            dataset=dataset,
            body_names=list(dataset.body_names),
            joint_names=joint_names,
            src_joint_indices=src_joint_indices,
            dest_joint_indices=dest_joint_indices,
        )

    @property
    def num_motions(self) -> int:
        return int(self.starts.shape[0])

    @property
    def num_steps(self) -> int:
        return int(self.lengths.sum())

    def get_slice(self, motion_ids: np.ndarray, starts: np.ndarray, steps: np.ndarray) -> MotionData:
        motion_ids_arr = np.asarray(motion_ids, dtype=np.int64).reshape(-1)
        starts_arr = np.asarray(starts, dtype=np.int64).reshape(-1)
        steps_arr = np.asarray(steps, dtype=np.int64).reshape(-1)
        if starts_arr.shape[0] != motion_ids_arr.shape[0]:
            raise ValueError(
                "starts must have the same length as motion_ids, got "
                f"{starts_arr.shape[0]} and {motion_ids_arr.shape[0]}"
            )

        source_motion_ids = self._motion_ids[motion_ids_arr]
        source_starts = self._source_starts[source_motion_ids]
        source_ends = self._source_ends[source_motion_ids]
        idx = (source_starts + starts_arr)[:, None] + steps_arr[None, :]
        idx = np.minimum(idx, (source_ends - 1)[:, None])
        idx = np.maximum(idx, source_starts[:, None])

        return MotionData(
            **{
                field_name: field[idx]
                for field_name, field in self._storage.items()
            }
        )

    def find_joints(self, joint_names: List[str], preserve_order: bool = False) -> List[int]:
        return resolve_matching_names(joint_names, self.joint_names, preserve_order)

    def find_bodies(self, body_names: List[str], preserve_order: bool = False) -> List[int]:
        return resolve_matching_names(body_names, self.body_names, preserve_order)


def motion_dataset_first_motion(dataset: MotionDataset) -> MotionDataset:
    if dataset.num_motions <= 0:
        raise ValueError("Cannot extract the first motion from an empty MotionDataset")
    return dataset.select_motions([0])
