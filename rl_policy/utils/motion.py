import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Union
from scipy.spatial.transform import Rotation as sRot, Slerp
from tqdm import tqdm
from utils.strings import resolve_matching_names

def lerp(ts_target, ts_source, x):
    """Linear interpolation for arrays"""
    return np.stack([np.interp(ts_target, ts_source, x[:, i]) for i in range(x.shape[1])], axis=-1)

def slerp(ts_target, ts_source, quat):
    """Spherical linear interpolation for quaternions"""
    # time dim: 0
    # batch dim: 1:-1
    # quat dim: -1
    batch_shape = quat.shape[1:-1]
    quat_dim = quat.shape[-1]

    steps_target = ts_target.shape[0]
    steps_source = ts_source.shape[0]

    quat = quat.reshape(steps_source, -1, quat_dim)

    batch_size = int(np.prod(batch_shape, initial=1))
    out = np.empty((steps_target, batch_size, quat_dim))
    for i in range(batch_size):
        s = Slerp(ts_source, sRot.from_quat(quat[:, i, [1, 2, 3, 0]]))  # quat first to quat last
        out[:, i, :] = s(ts_target).as_quat()[..., [3, 0, 1, 2]]  # quat last to quat first
    out = out.reshape(steps_target, *batch_shape, quat_dim)
    return out

def interpolate(motion: Dict[str, np.ndarray], source_fps: int, target_fps: int) -> Dict[str, np.ndarray]:
    """Interpolate motion data to target fps"""
    if source_fps != target_fps:
        in_keys = ["body_pos_w", "body_lin_vel_w", "body_quat_w", "body_ang_vel_w", "joint_pos", "joint_vel"]
        if not all(key in in_keys for key in motion.keys()):
            raise NotImplementedError(f"interpolation is not fully implemented for some keys")
        
        T = motion["joint_pos"].shape[0]
        end_t = T / source_fps
        ts_source = np.arange(0, end_t, 1 / source_fps)
        ts_target = np.arange(0, end_t, 1 / target_fps)
        if ts_target[-1] > ts_source[-1]:
            ts_target = ts_target[:-1]
            
        motion["body_pos_w"] = lerp(ts_target, ts_source, motion["body_pos_w"].reshape(T, -1)).reshape(len(ts_target), -1, 3)
        motion["body_lin_vel_w"] = lerp(ts_target, ts_source, motion["body_lin_vel_w"].reshape(T, -1)).reshape(len(ts_target), -1, 3)
        motion["body_quat_w"] = slerp(ts_target, ts_source, motion["body_quat_w"])
        motion["body_ang_vel_w"] = lerp(ts_target, ts_source, motion["body_ang_vel_w"].reshape(T, -1)).reshape(len(ts_target), -1, 3)
        motion["joint_pos"] = lerp(ts_target, ts_source, motion["joint_pos"])
        motion["joint_vel"] = lerp(ts_target, ts_source, motion["joint_vel"])
    return motion

class MotionData:
    """Container for motion data arrays"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if key != "batch_size":
                setattr(self, key, value)
                
    def __getitem__(self, idx):
        """Support array indexing"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                result[key] = value[idx]
        return MotionData(**result)

class MotionDataset:
    """Dataset for motion data with numpy arrays"""
    def __init__(
        self,
        body_names: List[str],
        joint_names: List[str],
        starts: List[int],
        ends: List[int],
        data: MotionData,
    ):
        self.body_names = body_names
        self.joint_names = joint_names
        self.starts = np.array(starts)
        self.ends = np.array(ends)
        self.lengths = self.ends - self.starts
        self.data = data

    @classmethod
    def create_from_path(cls, root_path: str, target_fps: int = 50):
        """Create dataset from motion files"""
        root_path = Path(root_path)
        meta_path = root_path / "meta.json"
        with open(meta_path, "r") as f:
            meta = json.load(f)
        
        motion_paths = list(sorted(Path(root_path).rglob("*.npz")))
        if not motion_paths:
            raise RuntimeError(f"No motions found in {root_path}")
        print(f"Found {len(motion_paths)} motion files under {root_path}")

        motions = []
        total_length = 0
        for motion_path in tqdm(motion_paths):
            motion = dict(np.load(motion_path))
            motion = interpolate(motion, source_fps=meta["fps"], target_fps=target_fps)
            total_length += motion["body_pos_w"].shape[0]
            motions.append(motion)
            
        # Process joint names and indices
        unitree_joint_names = [
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
            "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
            "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
            "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
            "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
            "left_wrist_yaw_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
            "right_wrist_pitch_joint", "right_wrist_yaw_joint",
        ]
        
        share_joint_names = [name for name in meta["joint_names"] if name in unitree_joint_names]
        src_joint_indices = [meta["joint_names"].index(name) for name in share_joint_names]
        dest_joint_indices = [unitree_joint_names.index(name) for name in share_joint_names]

        more_joint_names = [name for name in meta["joint_names"] if name not in unitree_joint_names]
        src_more_joint_indices = [meta["joint_names"].index(name) for name in more_joint_names]
        dest_more_joint_indices = [len(unitree_joint_names) + i for i in range(len(more_joint_names))]

        joint_names = unitree_joint_names + more_joint_names
        src_joint_indices = src_joint_indices + src_more_joint_indices
        dest_joint_indices = dest_joint_indices + dest_more_joint_indices

        # Process joint data
        for motion in motions:
            joint_pos = np.zeros((motion["joint_pos"].shape[0], len(joint_names)))
            joint_vel = np.zeros((motion["joint_vel"].shape[0], len(joint_names)))
            joint_pos[:, dest_joint_indices] = motion["joint_pos"][:, src_joint_indices]
            joint_vel[:, dest_joint_indices] = motion["joint_vel"][:, src_joint_indices]
            motion["joint_pos"] = joint_pos
            motion["joint_vel"] = joint_vel

        # Initialize arrays
        step = np.empty(total_length, dtype=int)
        motion_id = np.empty(total_length, dtype=int)
        body_pos_w = np.empty((total_length, len(meta["body_names"]), 3))
        body_lin_vel_w = np.empty((total_length, len(meta["body_names"]), 3))
        body_quat_w = np.empty((total_length, len(meta["body_names"]), 4))
        body_ang_vel_w = np.empty((total_length, len(meta["body_names"]), 3))
        joint_pos = np.empty((total_length, len(joint_names)))
        joint_vel = np.empty((total_length, len(joint_names)))
    
        start_idx = 0
        starts = []
        ends = []

        # Fill arrays
        for i, motion in enumerate(motions):
            motion_length = motion["body_pos_w"].shape[0]
            step[start_idx:start_idx + motion_length] = np.arange(motion_length)
            motion_id[start_idx:start_idx + motion_length] = i
            
            body_pos_w[start_idx:start_idx + motion_length] = motion["body_pos_w"]
            body_lin_vel_w[start_idx:start_idx + motion_length] = motion["body_lin_vel_w"]
            body_quat_w[start_idx:start_idx + motion_length] = motion["body_quat_w"]
            body_ang_vel_w[start_idx:start_idx + motion_length] = motion["body_ang_vel_w"]
            joint_pos[start_idx:start_idx + motion_length] = motion["joint_pos"]
            joint_vel[start_idx:start_idx + motion_length] = motion["joint_vel"]
            
            starts.append(start_idx)
            start_idx += motion_length
            ends.append(start_idx)
        
        data = MotionData(
            motion_id=motion_id,
            step=step,
            body_pos_w=body_pos_w,
            body_lin_vel_w=body_lin_vel_w,
            body_quat_w=body_quat_w,
            body_ang_vel_w=body_ang_vel_w,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
        )

        return cls(
            body_names=meta["body_names"],
            joint_names=joint_names,
            starts=starts,
            ends=ends,
            data=data,
        )

    @property
    def num_motions(self):
        return len(self.starts)
    
    @property
    def num_steps(self):
        return len(self.data.step)

    def get_slice(self, motion_ids: np.ndarray, starts: np.ndarray, steps: Union[int, np.ndarray] = 1) -> MotionData:
        """Get a slice of motion data"""
        if isinstance(steps, int):
            steps = np.arange(steps)
        idx = (self.starts[motion_ids] + starts).reshape(-1, 1) + steps.reshape(1, -1)
        idx = np.clip(idx, None, self.ends[motion_ids].reshape(-1, 1) - 1)
        return self.data[idx]  # shape: [len(motion_ids), len(steps), ...]

    def find_joints(self, joint_names: List[str], preserve_order: bool = False) -> List[int]:
        """Find joint indices by names"""
        return resolve_matching_names(joint_names, self.joint_names, preserve_order)

    def find_bodies(self, body_names: List[str], preserve_order: bool = False) -> List[int]:
        """Find body indices by names"""
        return resolve_matching_names(body_names, self.body_names, preserve_order)
