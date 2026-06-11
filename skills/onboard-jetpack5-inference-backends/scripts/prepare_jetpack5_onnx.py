#!/usr/bin/env python3
"""Prepare sim2real ONNX policies for JetPack 5 ORT GPU or TensorRT 8.5."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def _unique_name(model: onnx.ModelProto, base: str) -> str:
    existing = {init.name for init in model.graph.initializer}
    for node in model.graph.node:
        existing.update(node.input)
        existing.update(node.output)
    candidate = base
    suffix = 0
    while candidate in existing:
        suffix += 1
        candidate = f"{base}_{suffix}"
    return candidate


def _constant_tensors(model: onnx.ModelProto) -> dict[str, onnx.TensorProto]:
    constants = {init.name: init for init in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name == "value":
                constants[node.output[0]] = attr.t
    return constants


def _set_jetpack5_versions(model: onnx.ModelProto, mode: str) -> None:
    if model.ir_version > 9:
        model.ir_version = 9
    max_opset = 13 if mode == "tensorrt" else 19
    for opset in model.opset_import:
        if opset.domain == "" and opset.version > max_opset:
            opset.version = max_opset


def _patch_reduce_axes(node: onnx.NodeProto, constants: dict[str, onnx.TensorProto]) -> bool:
    if not node.op_type.startswith("Reduce") or len(node.input) < 2:
        return False
    axes_tensor = constants.get(node.input[1])
    if axes_tensor is None:
        return False
    axes = numpy_helper.to_array(axes_tensor).astype(np.int64).tolist()
    del node.input[1]
    del node.attribute[:]
    node.attribute.add(name="axes", ints=axes)
    return True


def _expand_layer_norm(model: onnx.ModelProto, node: onnx.NodeProto) -> list[onnx.NodeProto] | None:
    if len(node.input) < 2:
        return None
    axis = -1
    epsilon = 1e-5
    for attr in node.attribute:
        if attr.name == "axis":
            axis = int(attr.i)
        elif attr.name == "epsilon":
            epsilon = float(attr.f)
    if axis != -1:
        return None

    x = node.input[0]
    scale = node.input[1]
    bias = node.input[2] if len(node.input) >= 3 and node.input[2] else None
    out = node.output[0]
    prefix = _unique_name(model, f"{node.name or out}_ln")
    mean = f"{prefix}_mean"
    centered = f"{prefix}_centered"
    squared = f"{prefix}_squared"
    var = f"{prefix}_var"
    var_eps = f"{prefix}_var_eps"
    std = f"{prefix}_std"
    normalized = f"{prefix}_normalized"
    scaled = f"{prefix}_scaled"
    eps_name = f"{prefix}_eps"
    pow_name = f"{prefix}_pow"

    model.graph.initializer.extend(
        [
            numpy_helper.from_array(np.asarray(epsilon, dtype=np.float32), eps_name),
            numpy_helper.from_array(np.asarray(2.0, dtype=np.float32), pow_name),
        ]
    )

    nodes = [
        helper.make_node("ReduceMean", [x], [mean], name=f"{prefix}_mean_node", axes=[-1], keepdims=1),
        helper.make_node("Sub", [x, mean], [centered], name=f"{prefix}_center_node"),
        helper.make_node("Pow", [centered, pow_name], [squared], name=f"{prefix}_pow_node"),
        helper.make_node("ReduceMean", [squared], [var], name=f"{prefix}_var_node", axes=[-1], keepdims=1),
        helper.make_node("Add", [var, eps_name], [var_eps], name=f"{prefix}_eps_node"),
        helper.make_node("Sqrt", [var_eps], [std], name=f"{prefix}_sqrt_node"),
        helper.make_node("Div", [centered, std], [normalized], name=f"{prefix}_div_node"),
        helper.make_node("Mul", [normalized, scale], [scaled if bias else out], name=f"{prefix}_scale_node"),
    ]
    if bias:
        nodes.append(helper.make_node("Add", [scaled, bias], [out], name=f"{prefix}_bias_node"))
    return nodes


def _expand_mish(model: onnx.ModelProto, node: onnx.NodeProto) -> list[onnx.NodeProto]:
    x = node.input[0]
    out = node.output[0]
    prefix = _unique_name(model, f"{node.name or out}_mish")
    one = f"{prefix}_one"
    exp = f"{prefix}_exp"
    exp_plus_one = f"{prefix}_exp_plus_one"
    softplus = f"{prefix}_softplus"
    tanh = f"{prefix}_tanh"
    model.graph.initializer.append(numpy_helper.from_array(np.asarray(1.0, dtype=np.float32), one))
    return [
        helper.make_node("Exp", [x], [exp], name=f"{prefix}_exp_node"),
        helper.make_node("Add", [exp, one], [exp_plus_one], name=f"{prefix}_add_node"),
        helper.make_node("Log", [exp_plus_one], [softplus], name=f"{prefix}_log_node"),
        helper.make_node("Tanh", [softplus], [tanh], name=f"{prefix}_tanh_node"),
        helper.make_node("Mul", [x, tanh], [out], name=f"{prefix}_mul_node"),
    ]


def prepare_model(input_path: Path, output_path: Path, mode: str) -> None:
    model = onnx.load(input_path)
    _set_jetpack5_versions(model, mode)

    if mode == "tensorrt":
        constants = _constant_tensors(model)
        new_nodes: list[onnx.NodeProto] = []
        for node in model.graph.node:
            if _patch_reduce_axes(node, constants):
                new_nodes.append(node)
                continue
            if node.op_type == "LayerNormalization":
                expanded = _expand_layer_norm(model, node)
                if expanded is not None:
                    new_nodes.extend(expanded)
                    continue
            if node.op_type == "Mish":
                new_nodes.extend(_expand_mish(model, node))
                continue
            new_nodes.append(node)

        del model.graph.node[:]
        model.graph.node.extend(new_nodes)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_path)
    onnx.checker.check_model(output_path)


def copy_sidecars(input_path: Path, output_path: Path) -> None:
    for suffix in (".json", ".yaml", ".yml"):
        src = input_path.with_suffix(suffix)
        if src.exists():
            shutil.copy2(src, output_path.with_suffix(suffix))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("ort-gpu", "tensorrt"), required=True)
    parser.add_argument("--copy-sidecars", action="store_true")
    args = parser.parse_args()

    prepare_model(args.input, args.output, args.mode)
    if args.copy_sidecars:
        copy_sidecars(args.input, args.output)


if __name__ == "__main__":
    main()
