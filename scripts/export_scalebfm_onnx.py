#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import yaml

from sim2real.config.robots import get_robot_cfg


INPUT_NAMES = [
    "root_quat_buffer",
    "base_ang_vel_buffer",
    "dof_pos_buffer",
    "dof_vel_buffer",
    "actions_buffer",
    "target_body_pos_future_to_robot_base",
    "target_body_rot_future_to_robot_base",
    "control_mode",
    "future_time_offsets",
]
OUTPUT_NAMES = ["tgt_dof_pos", "action"]


def _repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_humanoid_transformer_module(scalebfm_root: Path):
    module_path = (
        scalebfm_root
        / "ScaleTrack"
        / "source"
        / "my_rsl_rl"
        / "my_rsl_rl"
        / "networks"
        / "humanoid_transformer.py"
    )
    if not module_path.is_file():
        raise FileNotFoundError(f"ScaleBFM humanoid_transformer.py not found: {module_path}")
    spec = importlib.util.spec_from_file_location("scalebfm_humanoid_transformer", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    shape = vec.shape
    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)
    xyz = quat[:, 1:]
    t = xyz.cross(vec, dim=-1) * 2
    return (vec + quat[:, 0:1] * t + xyz.cross(t, dim=-1)).view(shape)


def quat_apply_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    shape = vec.shape
    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)
    xyz = quat[:, 1:]
    t = xyz.cross(vec, dim=-1) * 2
    return (vec - quat[:, 0:1] * t + xyz.cross(t, dim=-1)).view(shape)


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    if q1.shape != q2.shape:
        raise ValueError(f"Expected input quaternion shape mismatch: {q1.shape} != {q2.shape}.")
    shape = q1.shape
    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    return torch.stack([w, x, y, z], dim=-1).view(shape)


def quat_mul_inverse_left(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    q1_inv = q1.clone()
    q1_inv[..., 1:] = -q1_inv[..., 1:]
    return quat_mul(q1_inv, q2)


def quat_mul_inverse_right(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    q2_inv = q2.clone()
    q2_inv[..., 1:] = -q2_inv[..., 1:]
    return quat_mul(q1, q2_inv)


def parse_xml(xml_file: Path, device: torch.device) -> tuple[list[str], list[str], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    body_names: list[str] = []
    parent_indices: list[int] = []
    local_translation: list[np.ndarray] = []
    local_rotation: list[np.ndarray] = []
    joint_axis: list[torch.Tensor] = []
    joint_names: list[str] = []

    tree = ET.parse(xml_file)
    xml_doc_root = tree.getroot()
    xml_world_body = xml_doc_root.find("worldbody")
    if xml_world_body is None:
        raise ValueError(f"worldbody not found in {xml_file}")
    xml_body_root = xml_world_body.find("body")
    if xml_body_root is None:
        raise ValueError(f"root body not found in {xml_file}")

    def _add_xml_body(xml_node, parent_index: int, body_index: int) -> int:
        body_name = xml_node.attrib.get("name")
        pos = np.fromstring(xml_node.attrib.get("pos", "0 0 0"), dtype=float, sep=" ")
        rot = np.fromstring(xml_node.attrib.get("quat", "1 0 0 0"), dtype=float, sep=" ")

        if body_index != 0:
            curr_joints = xml_node.findall("joint")
            if len(curr_joints) != 1:
                raise ValueError(f"Expected exactly one joint under body {body_name}, got {len(curr_joints)}")
            axis = torch.from_numpy(np.fromstring(curr_joints[0].attrib.get("axis"), dtype=float, sep=" "))
            local_rotation.append(rot)
            local_translation.append(pos)
            joint_axis.append(axis)
            joint_names.append(curr_joints[0].attrib.get("name"))

        body_names.append(body_name)
        parent_indices.append(parent_index)

        curr_index = body_index
        body_index += 1
        for child in xml_node.findall("body"):
            body_index = _add_xml_body(child, curr_index, body_index)
        return body_index

    _add_xml_body(xml_body_root, -1, 0)
    return (
        body_names,
        joint_names,
        torch.tensor(parent_indices, dtype=torch.long, device=device),
        torch.stack(joint_axis, dim=0).float().to(device),
        torch.tensor(np.array(local_translation), dtype=torch.float32, device=device),
        torch.tensor(np.array(local_rotation), dtype=torch.float32, device=device),
    )


def build_mode_mappings(
    mode_table: torch.Tensor,
    feature_dims_per_link: Sequence[int],
    *,
    with_time: bool,
) -> torch.Tensor:
    mode_table = mode_table.float()
    num_modes = mode_table.shape[0]
    mappings = []
    for dim in feature_dims_per_link:
        mappings.append(
            (torch.ones(num_modes, mode_table.shape[1], int(dim), device=mode_table.device) * mode_table.unsqueeze(-1))
            .reshape(num_modes, -1)
        )
    if with_time:
        mappings.append(torch.ones(num_modes, 1, device=mode_table.device))
    return torch.cat(mappings, dim=-1)


class _PolicyModules:
    def __init__(self, actor: nn.Module, actor_task_embedder: nn.Module) -> None:
        self.actor = actor
        self.actor_task_embedder = actor_task_embedder


class HumanoidTransformerPolicyWrapperWithMode(nn.Module):
    def __init__(
        self,
        policy: _PolicyModules,
        mode_mappings: torch.Tensor,
        mode_vectors: torch.Tensor,
        default_dof_pos: torch.Tensor,
        action_scale: torch.Tensor,
        context_len: int,
        future_len: int,
        local_translation: torch.Tensor,
        local_rotation: torch.Tensor,
        parent_indices: torch.Tensor,
        joint_axis: torch.Tensor,
        selected_link_indices: torch.Tensor,
        lab_to_xml_joint_indices: torch.Tensor,
    ) -> None:
        super().__init__()
        self.task_embedder = policy.actor_task_embedder
        self.prop_embedder = policy.actor.prop_projection
        self.action_embedder = policy.actor.action_projection
        self.transformer_blocks = policy.actor.transformer_blocks
        self.final_norm = policy.actor.final_norm
        self.projection_head = policy.actor.projection_head
        self.register_buffer("empty_embedding", policy.actor.empty_embedding)

        attn_mask = torch.zeros(2 * context_len, 2 * context_len, dtype=torch.bool, device=mode_vectors.device)
        row_idx = torch.arange(2 * context_len - 1, device=mode_vectors.device)
        col_idx = torch.full((2 * context_len - 1,), 2 * context_len - 1, device=mode_vectors.device)
        attn_mask[row_idx, col_idx] = True
        self.register_buffer("self_attn_mask", attn_mask)

        self.register_buffer("mode_mappings", mode_mappings)
        self.register_buffer("mode_vectors", mode_vectors)
        self.register_buffer(
            "gravity_vec",
            torch.tensor([[0, 0, -1]], dtype=torch.float32, device=mode_vectors.device)
            .unsqueeze(1)
            .expand(-1, context_len, -1),
        )
        self.register_buffer("default_dof_pos", default_dof_pos)
        self.register_buffer("action_scale", action_scale)
        self.register_buffer("selected_link_indices", selected_link_indices)

        tan_vec = torch.zeros(1, future_len, selected_link_indices.shape[-1], 3, dtype=torch.float32, device=mode_vectors.device)
        norm_vec = torch.zeros(1, future_len, selected_link_indices.shape[-1], 3, dtype=torch.float32, device=mode_vectors.device)
        tan_vec[..., 0] = 1
        norm_vec[..., -1] = 1
        self.register_buffer("tan_vec", tan_vec)
        self.register_buffer("norm_vec", norm_vec)

        self.register_buffer("local_translation", local_translation)
        self.register_buffer("local_rotation", local_rotation)
        self.register_buffer("parent_indices", parent_indices)
        self.register_buffer("joint_axis", joint_axis)
        self.register_buffer("lab_to_xml_joint_indices", lab_to_xml_joint_indices)

    def forward(
        self,
        root_quat_buffer: torch.Tensor,
        base_ang_vel_buffer: torch.Tensor,
        dof_pos_buffer: torch.Tensor,
        dof_vel_buffer: torch.Tensor,
        actions_buffer: torch.Tensor,
        target_body_pos_future_to_robot_base: torch.Tensor,
        target_body_rot_future_to_robot_base: torch.Tensor,
        control_mode: torch.Tensor,
        future_time_offsets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projected_gravity_buffer = quat_apply_inverse(root_quat_buffer, self.gravity_vec)
        dof_pos_rel_buffer = dof_pos_buffer - self.default_dof_pos
        prop_obs = torch.cat(
            [
                projected_gravity_buffer,
                base_ang_vel_buffer,
                dof_pos_rel_buffer,
                dof_vel_buffer * 0.05,
            ],
            dim=-1,
        )
        prop_token = self.prop_embedder(prop_obs)
        action_token = self.action_embedder(actions_buffer)

        x = torch.empty(prop_token.shape[0], 2 * prop_token.shape[1], prop_token.shape[2], dtype=prop_token.dtype, device=prop_token.device)
        x[:, ::2] = prop_token
        x[:, 1:-1:2] = action_token[:, 1:]
        x[:, 2 * prop_token.shape[1] - 1] = self.empty_embedding

        dof_pos = dof_pos_buffer[:, -1]
        half_angles = dof_pos[:, self.lab_to_xml_joint_indices].unsqueeze(-1) / 2
        joint_rot = torch.cat([torch.cos(half_angles), self.joint_axis.unsqueeze(0) * torch.sin(half_angles)], dim=-1)

        batch_size = joint_rot.shape[0]
        body_pos = torch.zeros(batch_size, len(self.parent_indices), 3, dtype=torch.float32, device=joint_rot.device)
        body_quat = torch.zeros(batch_size, len(self.parent_indices), 4, dtype=torch.float32, device=joint_rot.device)
        body_quat[..., 0] = 1

        for j in range(1, len(self.parent_indices)):
            j_rot = joint_rot[:, j - 1]
            local_trans = self.local_translation[j - 1 : j]
            local_rot = self.local_rotation[j - 1 : j]
            parent_idx = self.parent_indices[j : j + 1]
            parent_pos = body_pos[:, parent_idx].squeeze(1)
            parent_rot = body_quat[:, parent_idx].squeeze(1)
            curr_pos = parent_pos + quat_apply(parent_rot, local_trans)
            curr_rot = quat_mul(parent_rot, quat_mul(local_rot.expand_as(j_rot), j_rot))
            body_pos[:, j] = curr_pos
            body_quat[:, j] = curr_rot

        body_pos_to_robot_base = body_pos[:, self.selected_link_indices]
        body_quat_to_robot_base = body_quat[:, self.selected_link_indices]

        target_body_pos_future_rel_to_robot_base = target_body_pos_future_to_robot_base - body_pos_to_robot_base[:, None, :, :]
        target_body_rot_future_to_robot_base_tan_norm = torch.cat(
            [
                quat_apply(target_body_rot_future_to_robot_base, self.tan_vec),
                quat_apply(target_body_rot_future_to_robot_base, self.norm_vec),
            ],
            dim=-1,
        )
        target_body_rot_future_rel_to_robot_base = quat_mul_inverse_right(
            target_body_rot_future_to_robot_base,
            body_quat_to_robot_base[:, None].expand(-1, target_body_rot_future_to_robot_base.shape[1], -1, -1),
        )
        target_body_rot_future_rel_to_robot_base_tan_norm = torch.cat(
            [
                quat_apply(target_body_rot_future_rel_to_robot_base, self.tan_vec),
                quat_apply(target_body_rot_future_rel_to_robot_base, self.norm_vec),
            ],
            dim=-1,
        )

        task_obs = torch.cat(
            [
                target_body_pos_future_to_robot_base.flatten(2, 3),
                target_body_pos_future_rel_to_robot_base.flatten(2, 3),
                target_body_rot_future_to_robot_base_tan_norm.flatten(2, 3),
                target_body_rot_future_rel_to_robot_base_tan_norm.flatten(2, 3),
                future_time_offsets.to(dtype=target_body_pos_future_to_robot_base.dtype),
            ],
            dim=-1,
        )

        mapping = self.mode_mappings[control_mode]
        task_obs_masked = task_obs * mapping.unsqueeze(1)
        mode_vec = self.mode_vectors[control_mode]
        task_input = torch.cat([task_obs_masked, mode_vec.unsqueeze(1).expand(-1, task_obs_masked.shape[1], -1)], dim=-1)
        task_tokens = self.task_embedder(task_input)

        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, task_tokens, self_attn_mask=self.self_attn_mask)
        x = self.final_norm(x)
        action = self.projection_head(x[:, -1, :])
        return action * self.action_scale + self.default_dof_pos, action


@dataclass
class ExportSpec:
    checkpoint: Path
    metadata: Path
    mode_table: Path
    output_dir: Path
    scalebfm_root: Path
    xml: Path
    opset: int


def _load_policy_modules(metadata: dict[str, Any], checkpoint: Path, scalebfm_root: Path) -> _PolicyModules:
    module = _load_humanoid_transformer_module(scalebfm_root)
    HumanoidTransformer = module.HumanoidTransformer
    TaskEmbedder = module.TaskEmbedder

    arch = metadata["policy_architecture"]
    actor = HumanoidTransformer(
        prop_obs_dim=int(arch["prop_obs_dim"]),
        action_dim=int(arch["action_dim"]),
        output_dim=int(arch["output_dim"]),
        embed_dim=int(arch["embedding_dim"]),
        num_heads=int(arch["num_heads"]),
        ff_dim=int(arch["ff_dim"]),
        num_layers=int(arch["num_layers"]),
    )
    actor_task_embedder = TaskEmbedder(
        task_obs_dim=int(arch["task_obs_dim"]),
        embedding_dim=int(arch["embedding_dim"]),
        reduced_task_dim=arch.get("reduced_task_dim"),
        hidden_dims=arch.get("task_embedder_hidden_dims") or [],
    )

    checkpoint_obj = torch.load(checkpoint, map_location="cpu")
    state_dict = checkpoint_obj["model_state_dict"] if "model_state_dict" in checkpoint_obj else checkpoint_obj
    actor_state = {key[len("actor.") :]: value for key, value in state_dict.items() if key.startswith("actor.")}
    task_state = {
        key[len("actor_task_embedder.") :]: value
        for key, value in state_dict.items()
        if key.startswith("actor_task_embedder.")
    }
    actor.load_state_dict(actor_state, strict=True)
    actor_task_embedder.load_state_dict(task_state, strict=True)
    actor.eval()
    actor_task_embedder.eval()
    return _PolicyModules(actor=actor, actor_task_embedder=actor_task_embedder)


def _build_wrapper(spec: ExportSpec, metadata: dict[str, Any]) -> HumanoidTransformerPolicyWrapperWithMode:
    device = torch.device("cpu")
    policy = _load_policy_modules(metadata, spec.checkpoint, spec.scalebfm_root)
    mode_table = torch.load(spec.mode_table, map_location=device).float()
    mode_mappings = build_mode_mappings(
        mode_table,
        metadata["mode_feature_dims"],
        with_time=bool(metadata["mode_mapping_with_time"]),
    )
    default_dof_pos = torch.tensor(metadata["default_dof_pos"], dtype=torch.float32, device=device)
    action_scale = torch.tensor(metadata["action_scale"], dtype=torch.float32, device=device)

    body_names, xml_joint_names, parent_indices, joint_axis, local_translation, local_rotation = parse_xml(spec.xml, device)
    local_rotation = local_rotation / local_rotation.norm(dim=-1, keepdim=True)
    selected_link_indices = torch.tensor(
        [body_names.index(name) for name in metadata["selected_body_names"]],
        dtype=torch.long,
        device=device,
    )
    lab_to_xml_joint_indices = torch.tensor(
        [metadata["joint_names"].index(name) for name in xml_joint_names],
        dtype=torch.long,
        device=device,
    )

    wrapper = HumanoidTransformerPolicyWrapperWithMode(
        policy,
        mode_mappings,
        mode_table,
        default_dof_pos,
        action_scale,
        int(metadata["history_buffer_size"]),
        len(metadata["future_idx"]),
        local_translation,
        local_rotation,
        parent_indices,
        joint_axis,
        selected_link_indices,
        lab_to_xml_joint_indices,
    )
    wrapper.eval()
    return wrapper


def _random_unit_quat(shape: Sequence[int]) -> torch.Tensor:
    quat = torch.randn(*shape, dtype=torch.float32)
    quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
    return quat


def _example_inputs(metadata: dict[str, Any]) -> tuple[torch.Tensor, ...]:
    context = int(metadata["history_buffer_size"])
    future = len(metadata["future_idx"])
    num_joints = len(metadata["joint_names"])
    num_bodies = len(metadata["selected_body_names"])
    return (
        _random_unit_quat((1, context, 4)),
        torch.randn(1, context, 3, dtype=torch.float32) * 0.1,
        torch.tensor(metadata["default_dof_pos"], dtype=torch.float32).reshape(1, 1, num_joints).repeat(1, context, 1),
        torch.randn(1, context, num_joints, dtype=torch.float32) * 0.1,
        torch.zeros(1, context, num_joints, dtype=torch.float32),
        torch.randn(1, future, num_bodies, 3, dtype=torch.float32) * 0.1,
        _random_unit_quat((1, future, num_bodies, 4)),
        torch.tensor([7], dtype=torch.long),
        torch.tensor(metadata["future_idx"], dtype=torch.long).reshape(1, future, 1),
    )


def _write_onnx_meta(onnx_path: Path) -> None:
    meta_path = onnx_path.with_suffix(".json")
    meta_path.write_text(
        json.dumps({"in_keys": INPUT_NAMES, "out_keys": OUTPUT_NAMES}, indent=2) + "\n",
        encoding="utf-8",
    )


def _check_onnx(onnx_path: Path, wrapper: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_inputs = {name: tensor.detach().cpu().numpy() for name, tensor in zip(INPUT_NAMES, inputs)}
    with torch.inference_mode():
        torch_outputs = wrapper(*inputs)
    ort_outputs = session.run(None, ort_inputs)
    for name, torch_value, ort_value in zip(OUTPUT_NAMES, torch_outputs, ort_outputs):
        np.testing.assert_allclose(torch_value.detach().cpu().numpy(), ort_value, rtol=1.0e-4, atol=1.0e-5, err_msg=name)


def _ordered_dict(names: Sequence[str], values: Sequence[Any]) -> dict[str, Any]:
    return {name: float(value) for name, value in zip(names, values)}


def _policy_yaml(metadata: dict[str, Any]) -> dict[str, Any]:
    robot_cfg = get_robot_cfg("g1")
    common = {
        "joint_names": metadata["joint_names"],
        "action_names": metadata["action_names"],
        "body_names": metadata["body_names"],
        "selected_body_names": metadata["selected_body_names"],
        "future_idx": metadata["future_idx"],
        "history_buffer_size": metadata["history_buffer_size"],
        "control_mode": 7,
        "reference_forcing": True,
    }
    observations = {
        name: {
            name: {
                "_target_": f"scalebfm.scalebfm_{name}",
                **common,
            }
        }
        for name in INPUT_NAMES
    }
    return {
        "model_path": "policy.onnx",
        "observation": observations,
        "joint_names_simulation": list(robot_cfg.joint_names),
        "body_names_simulation": list(robot_cfg.body_names),
        "policy_joint_names": metadata["action_names"],
        "default_joint_pos": _ordered_dict(metadata["action_names"], metadata["default_dof_pos"]),
        "joint_kp": _ordered_dict(metadata["action_names"], metadata["stiffness"]),
        "joint_kd": _ordered_dict(metadata["action_names"], metadata["damping"]),
        "action_scale": _ordered_dict(metadata["action_names"], metadata["action_scale"]),
        "clip_actions": 10.0,
        "motion": {
            "motion_backend": "npz",
            "future_steps": metadata["future_idx"],
            "root_body_name": "pelvis",
            "motion_dt_s": 0.02,
        },
    }


def _write_policy_files(output_dir: Path, metadata: dict[str, Any], source_checkpoint: Path) -> None:
    yaml_path = output_dir / "policy.yaml"
    yaml_path.write_text(yaml.safe_dump(_policy_yaml(metadata), sort_keys=False), encoding="utf-8")
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# ScaleBFM sim2real artifact",
                "",
                f"Source checkpoint: `{source_checkpoint}`",
                "",
                "This directory contains a complete ONNX export of the ScaleBFM actor wrapper.",
                "The sim2real runtime consumes the raw `action` output and applies the YAML action scale/default pose.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def export_one(spec: ExportSpec) -> Path:
    metadata = json.loads(spec.metadata.read_text(encoding="utf-8"))
    wrapper = _build_wrapper(spec, metadata)
    inputs = _example_inputs(metadata)

    spec.output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = spec.output_dir / "policy.onnx"
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            inputs,
            str(onnx_path),
            input_names=INPUT_NAMES,
            output_names=OUTPUT_NAMES,
            opset_version=spec.opset,
            do_constant_folding=True,
            dynamo=False,
        )
    _write_onnx_meta(onnx_path)
    _write_policy_files(spec.output_dir, metadata, spec.checkpoint)
    _check_onnx(onnx_path, wrapper, inputs)
    return onnx_path


def main() -> int:
    repo_root = _repo_root_from_script()
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--mode-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scalebfm-root", type=Path, default=repo_root / "ScaleBFM")
    parser.add_argument(
        "--xml",
        type=Path,
        default=repo_root / "ScaleBFM" / "ScaleTrack" / "source" / "scaletrack" / "scaletrack" / "assets" / "robots" / "g1_29dof" / "g1_29dof.xml",
    )
    parser.add_argument("--opset", type=int, default=19)
    args = parser.parse_args()

    spec = ExportSpec(
        checkpoint=args.checkpoint.resolve(),
        metadata=args.metadata.resolve(),
        mode_table=args.mode_table.resolve(),
        output_dir=args.output_dir.resolve(),
        scalebfm_root=args.scalebfm_root.resolve(),
        xml=args.xml.resolve(),
        opset=args.opset,
    )
    onnx_path = export_one(spec)
    print(f"Exported and verified {onnx_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
