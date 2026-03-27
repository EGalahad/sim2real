import numpy as np
import json
import mujoco
from pathlib import Path
from typing import List, Dict
from scipy.spatial.transform import Rotation as sRot, Slerp
from tqdm import tqdm
from sim2real.config.robots.base import RobotCfg
from sim2real.utils.strings import resolve_matching_names


ANY4HDMI_MANIFEST_NAME = "manifest.json"

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
            raise NotImplementedError("interpolation is not fully implemented for some keys")
        
        T = motion["joint_pos"].shape[0]
        ts_source = np.linspace(0, T, T)
        ts_target = np.linspace(0, T, int(T / source_fps * target_fps))
            
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
    def create_from_path(
        cls,
        root_path: str,
        robot_cfg: RobotCfg,
        target_fps: int = 50,
    ):
        """Create dataset from motion files"""
        root = Path(root_path)
        any4hdmi_root = _find_any4hdmi_root(root)
        if any4hdmi_root is not None:
            dataset_root, manifest, motion_paths = _resolve_any4hdmi_motion_paths(root)
            print(f"Matched {len(motion_paths)} qpos motions under {dataset_root}")
            meta, motions = _load_any4hdmi_motions(
                dataset_root=dataset_root,
                manifest=manifest,
                motion_paths=motion_paths,
                target_fps=target_fps,
            )
        else:
            if root.is_file() and root.suffix == ".npz":
                motion_paths = [root]
            else:
                motion_paths = list(root.rglob("motion.npz"))
            if not motion_paths:
                raise RuntimeError(f"No motions found in {root_path}")
            motion_paths = [motion_path.parent for motion_path in motion_paths]

            print(f"Matched {len(motion_paths)} motions under {root_path}")

            metas = []
            for path in motion_paths:
                meta_path = path / "meta.json"
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                    meta.pop("length", None)
                    metas.append(meta)

            for i, meta in enumerate(metas[1:], 1):
                if meta != metas[0]:
                    raise ValueError(
                        f"meta.json in {motion_paths[i]} differs from {motion_paths[0]}"
                    )
            meta = metas[0]

            motion_paths = [path / "motion.npz" for path in motion_paths]

            motions = []
            for motion_path in tqdm(motion_paths):
                motion = dict(np.load(motion_path))
                motion = interpolate(motion, source_fps=meta["fps"], target_fps=target_fps)
                motions.append(motion)

        total_length = sum(int(motion["body_pos_w"].shape[0]) for motion in motions)
            
        # Process joint names and indices
        canonical_joint_names = list(robot_cfg.joint_names)
        
        share_joint_names = [name for name in meta["joint_names"] if name in canonical_joint_names]
        src_joint_indices = [meta["joint_names"].index(name) for name in share_joint_names]
        dest_joint_indices = [canonical_joint_names.index(name) for name in share_joint_names]

        more_joint_names = [name for name in meta["joint_names"] if name not in canonical_joint_names]
        src_more_joint_indices = [meta["joint_names"].index(name) for name in more_joint_names]
        dest_more_joint_indices = [len(canonical_joint_names) + i for i in range(len(more_joint_names))]

        joint_names = canonical_joint_names + more_joint_names
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

    def get_slice(self, motion_ids: np.ndarray, starts: np.ndarray, steps: np.ndarray) -> MotionData:
        """Get a slice of motion data"""
        idx = (self.starts[motion_ids] + starts).reshape(-1, 1) + steps.reshape(1, -1)
        min_step = self.starts[motion_ids].reshape(-1, 1)
        max_step = self.ends[motion_ids].reshape(-1, 1) - 1
        idx = np.clip(idx, min_step, max_step)
        return self.data[idx]  # shape: [len(motion_ids), len(steps), ...]

    def find_joints(self, joint_names: List[str], preserve_order: bool = False) -> List[int]:
        """Find joint indices by names"""
        return resolve_matching_names(joint_names, self.joint_names, preserve_order)

    def find_bodies(self, body_names: List[str], preserve_order: bool = False) -> List[int]:
        """Find body indices by names"""
        return resolve_matching_names(body_names, self.body_names, preserve_order)


def _find_any4hdmi_root(path: Path) -> Path | None:
    current = path if path.is_dir() else path.parent
    for candidate in (current, *current.parents):
        if (candidate / ANY4HDMI_MANIFEST_NAME).is_file():
            return candidate
    return None


def _resolve_any4hdmi_motion_paths(path: Path) -> tuple[Path, dict, list[Path]]:
    dataset_root = _find_any4hdmi_root(path)
    if dataset_root is None:
        raise RuntimeError(f"Could not find {ANY4HDMI_MANIFEST_NAME} above {path}")
    manifest = json.loads((dataset_root / ANY4HDMI_MANIFEST_NAME).read_text())
    motions_root = dataset_root / manifest.get("motions_subdir", "motions")

    if path.is_file():
        if path.suffix != ".npz":
            raise ValueError(f"Expected .npz motion file, got {path}")
        motion_paths = [path.resolve()]
    else:
        scan_root = motions_root if path == dataset_root else path
        motion_paths = sorted(motion_path.resolve() for motion_path in scan_root.rglob("*.npz"))
    if not motion_paths:
        raise RuntimeError(f"No qpos motions found under {dataset_root}")
    return dataset_root, manifest, motion_paths


def _resolve_any4hdmi_mjcf_path(dataset_root: Path, manifest: dict) -> Path:
    mjcf_ref = manifest.get("mjcf")
    if isinstance(mjcf_ref, dict):
        if mjcf_ref.get("kind") != "huggingface":
            raise ValueError(f"Unsupported any4hdmi mjcf kind: {mjcf_ref.get('kind')!r}")
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ImportError(
                "Resolving any4hdmi Hugging Face MJCF references requires "
                "`huggingface-hub` to be installed."
            ) from exc

        repo_id = str(mjcf_ref["repo_id"])
        repo_path = str(mjcf_ref["path"])
        revision = str(mjcf_ref.get("revision", "main"))
        snapshot_root = Path(snapshot_download(repo_id=repo_id, revision=revision))
        mjcf_path = snapshot_root / repo_path
    elif mjcf_ref is not None:
        mjcf_path = (dataset_root / Path(mjcf_ref)).resolve()
    else:
        mjcf_path_raw = manifest.get("mjcf_path")
        if mjcf_path_raw is None:
            raise KeyError(f"{ANY4HDMI_MANIFEST_NAME} is missing mjcf or mjcf_path")
        mjcf_path = Path(mjcf_path_raw).expanduser().resolve()
    if not mjcf_path.is_file():
        raise FileNotFoundError(f"MJCF not found: {mjcf_path}")
    return mjcf_path


def _compute_qvel(model: mujoco.MjModel, qpos: np.ndarray, fps: float) -> np.ndarray:
    qpos = np.asarray(qpos, dtype=np.float64)
    qvel = np.zeros((qpos.shape[0], model.nv), dtype=np.float32)
    if qpos.shape[0] <= 1:
        return qvel
    dt = 1.0 / fps
    work = np.zeros(model.nv, dtype=np.float64)
    for frame_idx in range(qpos.shape[0] - 1):
        mujoco.mj_differentiatePos(
            model, work, dt, qpos[frame_idx], qpos[frame_idx + 1]
        )
        qvel[frame_idx] = np.asarray(work, dtype=np.float32)
    qvel[-1] = qvel[-2]
    return qvel


def _body_names_from_model(model: mujoco.MjModel) -> list[str]:
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        for body_id in range(model.nbody)
    ]


def _hinge_joint_info(model: mujoco.MjModel) -> tuple[list[str], np.ndarray, np.ndarray]:
    joint_names: list[str] = []
    joint_qpos_addrs: list[int] = []
    joint_dof_addrs: list[int] = []
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        joint_names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id))
        joint_qpos_addrs.append(int(model.jnt_qposadr[joint_id]))
        joint_dof_addrs.append(int(model.jnt_dofadr[joint_id]))
    return (
        joint_names,
        np.asarray(joint_qpos_addrs, dtype=np.int32),
        np.asarray(joint_dof_addrs, dtype=np.int32),
    )


def _run_fk(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos: np.ndarray,
    qvel: np.ndarray,
    joint_qpos_addrs: np.ndarray,
    joint_dof_addrs: np.ndarray,
) -> Dict[str, np.ndarray]:
    frames = int(qpos.shape[0])
    body_pos_w = np.zeros((frames, model.nbody, 3), dtype=np.float32)
    body_lin_vel_w = np.zeros((frames, model.nbody, 3), dtype=np.float32)
    body_quat_w = np.zeros((frames, model.nbody, 4), dtype=np.float32)
    body_ang_vel_w = np.zeros((frames, model.nbody, 3), dtype=np.float32)
    joint_pos = np.zeros((frames, joint_qpos_addrs.shape[0]), dtype=np.float32)
    joint_vel = np.zeros((frames, joint_dof_addrs.shape[0]), dtype=np.float32)

    for frame_idx in range(frames):
        data.qpos[:] = qpos[frame_idx]
        data.qvel[:] = qvel[frame_idx]
        mujoco.mj_forward(model, data)
        body_pos_w[frame_idx] = np.asarray(data.xpos, dtype=np.float32)
        body_lin_vel_w[frame_idx] = np.asarray(data.cvel[:, 3:6], dtype=np.float32)
        body_quat_w[frame_idx] = np.asarray(data.xquat, dtype=np.float32)
        body_ang_vel_w[frame_idx] = np.asarray(data.cvel[:, 0:3], dtype=np.float32)
        joint_pos[frame_idx] = np.asarray(data.qpos[joint_qpos_addrs], dtype=np.float32)
        joint_vel[frame_idx] = np.asarray(data.qvel[joint_dof_addrs], dtype=np.float32)

    return {
        "body_pos_w": body_pos_w,
        "body_lin_vel_w": body_lin_vel_w,
        "body_quat_w": body_quat_w,
        "body_ang_vel_w": body_ang_vel_w,
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
    }


def _load_any4hdmi_motions(
    *,
    dataset_root: Path,
    manifest: dict,
    motion_paths: list[Path],
    target_fps: int,
) -> tuple[dict, list[Dict[str, np.ndarray]]]:
    mjcf_path = _resolve_any4hdmi_mjcf_path(dataset_root, manifest)
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    data = mujoco.MjData(model)
    body_names = _body_names_from_model(model)
    joint_names, joint_qpos_addrs, joint_dof_addrs = _hinge_joint_info(model)

    source_fps = float(manifest.get("fps", 0.0))
    if source_fps <= 0.0:
        timestep = float(manifest.get("timestep", 0.0))
        if timestep <= 0.0:
            raise ValueError("any4hdmi manifest must contain fps or timestep")
        source_fps = 1.0 / timestep

    motions: list[Dict[str, np.ndarray]] = []
    for motion_path in tqdm(motion_paths):
        payload = np.load(motion_path, allow_pickle=False)
        qpos = np.asarray(payload["qpos"], dtype=np.float32)
        qvel = _compute_qvel(model, qpos, source_fps)
        motion = _run_fk(
            model,
            data,
            qpos,
            qvel,
            joint_qpos_addrs=joint_qpos_addrs,
            joint_dof_addrs=joint_dof_addrs,
        )
        motion = interpolate(motion, source_fps=int(round(source_fps)), target_fps=target_fps)
        motions.append(motion)

    meta = {
        "body_names": body_names,
        "joint_names": joint_names,
        "fps": int(target_fps),
    }
    return meta, motions
