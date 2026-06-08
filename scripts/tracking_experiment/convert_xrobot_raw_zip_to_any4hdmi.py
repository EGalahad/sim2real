from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np

from any4hdmi.core.format import MOTION_DTYPE, MOTIONS_SUBDIR, save_motion, write_manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert G1 xrobot_raw npz files from a zip archive to any4hdmi qpos format."
    )
    parser.add_argument("raw_zip", help="Path to raw.zip containing xrobot_raw_*.npz files.")
    parser.add_argument(
        "--out-dir",
        default="../any4hdmi/output/xrobot_raw_20260524",
        help="Output any4hdmi dataset root.",
    )
    parser.add_argument(
        "--reference-manifest",
        default="../any4hdmi/output/sonic/manifest.json",
        help="Reference any4hdmi manifest used for mjcf and qpos_names.",
    )
    parser.add_argument(
        "--root-rot-order",
        choices=["xyzw", "wxyz"],
        default="xyzw",
        help="Quaternion order in raw root_rot. raw.zip from XRobot uses xyzw.",
    )
    parser.add_argument("--max-files", type=int, default=None, help="Convert only the first N raw files.")
    return parser.parse_args()


def _load_reference_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    if "qpos_names" not in manifest or "mjcf" not in manifest:
        raise ValueError(f"Reference manifest is missing qpos_names or mjcf: {path}")
    return manifest


def _raw_quat_to_wxyz(root_rot: np.ndarray, order: str) -> np.ndarray:
    quat = np.asarray(root_rot, dtype=MOTION_DTYPE)
    if order == "xyzw":
        quat = quat[:, [3, 0, 1, 2]]
    norms = np.linalg.norm(quat, axis=-1, keepdims=True)
    return quat / np.clip(norms, 1e-12, None)


def _build_qpos(raw: np.lib.npyio.NpzFile, qpos_names: list[str], root_rot_order: str) -> np.ndarray:
    root_pos = np.asarray(raw["root_pos"], dtype=MOTION_DTYPE)
    root_quat_wxyz = _raw_quat_to_wxyz(raw["root_rot"], root_rot_order)
    dof_pos = np.asarray(raw["dof_pos"], dtype=MOTION_DTYPE)
    joint_names = [str(name) for name in raw["joint_names"].tolist()]

    expected_joint_names = qpos_names[7:]
    if joint_names != expected_joint_names:
        raise ValueError(
            "raw joint_names do not match reference qpos_names[7:]. "
            f"raw={joint_names}, reference={expected_joint_names}"
        )
    if root_pos.shape[0] != dof_pos.shape[0] or root_pos.shape[0] != root_quat_wxyz.shape[0]:
        raise ValueError(
            "raw frame count mismatch: "
            f"root_pos={root_pos.shape}, root_rot={root_quat_wxyz.shape}, dof_pos={dof_pos.shape}"
        )

    qpos = np.zeros((root_pos.shape[0], len(qpos_names)), dtype=MOTION_DTYPE)
    qpos[:, 0:3] = root_pos
    qpos[:, 3:7] = root_quat_wxyz
    qpos[:, 7:] = dof_pos
    return qpos


def main() -> None:
    args = _parse_args()
    raw_zip = Path(args.raw_zip).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    reference_manifest_path = Path(args.reference_manifest).expanduser().resolve()
    reference_manifest = _load_reference_manifest(reference_manifest_path)
    qpos_names = [str(name) for name in reference_manifest["qpos_names"]]

    with zipfile.ZipFile(raw_zip) as archive:
        raw_names = sorted(name for name in archive.namelist() if name.endswith(".npz"))
        if args.max_files is not None:
            raw_names = raw_names[: int(args.max_files)]
        if not raw_names:
            raise RuntimeError(f"No .npz files found in {raw_zip}")

        total_frames = 0
        fps_values: list[float] = []
        for raw_name in raw_names:
            with archive.open(raw_name) as f:
                raw = np.load(f, allow_pickle=True)
                qpos = _build_qpos(raw, qpos_names, args.root_rot_order)
                fps_values.append(float(np.asarray(raw["fps"]).reshape(())))

            rel_path = Path(raw_name)
            if rel_path.parts and rel_path.parts[0] == "raw":
                rel_path = Path(*rel_path.parts[1:])
            out_path = out_dir / MOTIONS_SUBDIR / rel_path
            save_motion(out_path, qpos)
            total_frames += int(qpos.shape[0])

    fps = float(np.median(fps_values))
    write_manifest(
        out_dir,
        dataset_name="xrobot_raw_20260524",
        mjcf=reference_manifest["mjcf"],
        timestep=1.0 / fps,
        qpos_names=qpos_names,
        num_motions=len(raw_names),
        total_hours=total_frames / fps / 3600.0,
        source={
            "raw_zip": str(raw_zip),
            "reference_manifest": str(reference_manifest_path),
            "root_rot_order": args.root_rot_order,
            "fps_values": fps_values,
        },
    )
    print(f"converted {len(raw_names)} motions, {total_frames} frames -> {out_dir}")


if __name__ == "__main__":
    main()
