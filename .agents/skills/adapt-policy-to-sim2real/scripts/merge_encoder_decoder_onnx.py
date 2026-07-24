#!/usr/bin/env python3
"""Build one semantic-input SONIC ONNX with action and token outputs.

The script supports both source artifact shapes used by SONIC:

1. Merge separate encoder and decoder graphs. Selected release encoders are
   bound directly to one semantic mode input. Universal low-latency encoders
   are specialized with ``--mode`` and inactive mode fields are zero-filled.
2. Rewrite an already merged flat-input graph with semantic inputs, optionally
   exposing an internal encoder tensor as the token output in the same pass.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, compose, helper, numpy_helper


UNIVERSAL_MODE_LAYOUTS = {
    "g1": {"mode_id": 0.0, "input_name": "g1_input", "input_dim": 640},
    "smpl": {"mode_id": 2.0, "input_name": "smpl_input", "input_dim": 336},
}
UNIVERSAL_ENCODER_FIELD_DIMS = {
    "mode": 4,
    "g1": 640,
    "teleop": 267,
    "smpl": 336,
}


def _shape_dims(value_info: onnx.ValueInfoProto) -> list[int | str]:
    return [
        dim.dim_value if dim.HasField("dim_value") else dim.dim_param
        for dim in value_info.type.tensor_type.shape.dim
    ]


def _last_static_dim(value_info: onnx.ValueInfoProto, label: str) -> int:
    dims = _shape_dims(value_info)
    if not dims or not isinstance(dims[-1], int) or dims[-1] <= 0:
        raise ValueError(f"{label} must have a static last dimension, got {dims}")
    return int(dims[-1])


def _value_info(name: str, shape: list[int | str]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def _const_i64(name: str, values: list[int]) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(values, dtype=np.int64), name=name)


def _parse_input_spec(value: str) -> tuple[str, int]:
    try:
        name, dim_text = value.rsplit("=", 1)
        dim = int(dim_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("input specs must use NAME=DIM") from exc
    if not name or dim <= 0:
        raise argparse.ArgumentTypeError("input name must be non-empty and DIM positive")
    return name, dim


def _validate_unique_names(names: list[str]) -> None:
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"ONNX input/output names must be unique, got {duplicates}")


def merge_encoder_decoder(
    *,
    encoder_path: Path,
    decoder_path: Path,
    output_path: Path,
    encoder_input_name: str | None,
    proprioception_input_name: str,
    decoder_extra_input_name: str,
    output_name: str,
    token_output_name: str,
    proprio_dim: int | None,
    decoder_extra_dim: int,
    token_dim: int,
    mode: str | None,
    batched: bool,
) -> None:
    encoder = compose.add_prefix(onnx.load(encoder_path), "encoder/")
    decoder = compose.add_prefix(onnx.load(decoder_path), "decoder/")

    if len(encoder.graph.input) != 1 or len(encoder.graph.output) != 1:
        raise ValueError("Expected encoder to have exactly one input and one output")
    if len(decoder.graph.input) != 1 or len(decoder.graph.output) != 1:
        raise ValueError("Expected decoder to have exactly one input and one output")

    encoder_input = encoder.graph.input[0]
    encoder_output = encoder.graph.output[0]
    decoder_input = decoder.graph.input[0]
    decoder_output = decoder.graph.output[0]
    encoder_dim = _last_static_dim(encoder_input, "encoder input")
    actual_token_dim = _last_static_dim(encoder_output, "encoder output")
    decoder_dim = _last_static_dim(decoder_input, "decoder input")
    action_dim = _last_static_dim(decoder_output, "decoder output")
    if actual_token_dim != token_dim:
        raise ValueError(f"Encoder token dim is {actual_token_dim}, expected {token_dim}")

    if proprio_dim is None:
        proprio_dim = decoder_dim - actual_token_dim - decoder_extra_dim
    if proprio_dim <= 0 or actual_token_dim + decoder_extra_dim + proprio_dim != decoder_dim:
        raise ValueError(
            "Invalid decoder dimensions: "
            f"decoder={decoder_dim}, token={actual_token_dim}, "
            f"decoder_extra={decoder_extra_dim}, proprioception={proprio_dim}"
        )

    universal_dim = sum(UNIVERSAL_ENCODER_FIELD_DIMS.values())
    is_universal = encoder_dim == universal_dim
    if is_universal:
        if batched:
            raise ValueError("Universal low-latency mode currently supports unbatched export only")
        if mode is None:
            raise ValueError(f"Universal {encoder_dim}D encoder requires --mode")
        mode_spec = UNIVERSAL_MODE_LAYOUTS[mode]
        semantic_input_name = encoder_input_name or str(mode_spec["input_name"])
        semantic_input_dim = int(mode_spec["input_dim"])
    else:
        if mode is not None:
            expected_dim = int(UNIVERSAL_MODE_LAYOUTS[mode]["input_dim"])
            if encoder_dim != expected_dim:
                raise ValueError(
                    f"Selected {mode} encoder has dim {encoder_dim}, expected {expected_dim}; "
                    "omit --mode for non-low-latency layouts such as release SMPL"
                )
        semantic_input_name = encoder_input_name
        if semantic_input_name is None:
            raise ValueError("Selected encoder requires --encoder-input-name")
        semantic_input_dim = encoder_dim

    graph_input_names = [semantic_input_name, proprioception_input_name]
    if decoder_extra_dim:
        graph_input_names.append(decoder_extra_input_name)
    _validate_unique_names([*graph_input_names, output_name, token_output_name])

    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = [
        *encoder.graph.initializer,
        *decoder.graph.initializer,
    ]
    axes_name = "wrapper_batch_axis"
    working_mode_input = semantic_input_name
    working_proprioception = proprioception_input_name
    working_decoder_extra = decoder_extra_input_name
    if not batched:
        initializers.append(_const_i64(axes_name, [0]))
        working_mode_input = "wrapper_mode_input_batched"
        working_proprioception = "wrapper_proprioception_batched"
        nodes.extend(
            [
                helper.make_node(
                    "Unsqueeze",
                    [semantic_input_name, axes_name],
                    [working_mode_input],
                    name="wrapper_unsqueeze_mode_input",
                ),
                helper.make_node(
                    "Unsqueeze",
                    [proprioception_input_name, axes_name],
                    [working_proprioception],
                    name="wrapper_unsqueeze_proprioception",
                ),
            ]
        )
        if decoder_extra_dim:
            working_decoder_extra = "wrapper_decoder_extra_batched"
            nodes.append(
                helper.make_node(
                    "Unsqueeze",
                    [decoder_extra_input_name, axes_name],
                    [working_decoder_extra],
                    name="wrapper_unsqueeze_decoder_extra",
                )
            )

    if is_universal:
        assert mode is not None
        mode_spec = UNIVERSAL_MODE_LAYOUTS[mode]
        mode_selector_name = "wrapper_mode_selector"
        initializers.append(
            numpy_helper.from_array(
                np.asarray([[mode_spec["mode_id"], 0.0, 0.0, 0.0]], dtype=np.float32),
                mode_selector_name,
            )
        )
        encoder_parts = [mode_selector_name]
        for field in ("g1", "teleop", "smpl"):
            if field == mode:
                encoder_parts.append(working_mode_input)
            else:
                zero_name = f"wrapper_zero_{field}"
                initializers.append(
                    numpy_helper.from_array(
                        np.zeros((1, UNIVERSAL_ENCODER_FIELD_DIMS[field]), dtype=np.float32),
                        zero_name,
                    )
                )
                encoder_parts.append(zero_name)
        nodes.append(
            helper.make_node(
                "Concat",
                encoder_parts,
                [encoder_input.name],
                axis=1,
                name="wrapper_build_universal_encoder_input",
            )
        )
    else:
        nodes.append(
            helper.make_node(
                "Identity",
                [working_mode_input],
                [encoder_input.name],
                name="wrapper_bind_encoder_input",
            )
        )

    nodes.extend(encoder.graph.node)
    decoder_parts = [encoder_output.name]
    if decoder_extra_dim:
        decoder_parts.append(working_decoder_extra)
    decoder_parts.append(working_proprioception)
    nodes.append(
        helper.make_node(
            "Concat",
            decoder_parts,
            [decoder_input.name],
            axis=1,
            name="wrapper_build_decoder_input",
        )
    )
    nodes.extend(decoder.graph.node)

    if batched:
        nodes.extend(
            [
                helper.make_node(
                    "Identity", [decoder_output.name], [output_name], name="wrapper_action"
                ),
                helper.make_node(
                    "Identity",
                    [encoder_output.name],
                    [token_output_name],
                    name="wrapper_token",
                ),
            ]
        )
        input_prefix: list[int | str] = ["batch"]
        output_prefix: list[int | str] = ["batch"]
    else:
        nodes.extend(
            [
                helper.make_node(
                    "Squeeze",
                    [decoder_output.name, axes_name],
                    [output_name],
                    name="wrapper_squeeze_action",
                ),
                helper.make_node(
                    "Squeeze",
                    [encoder_output.name, axes_name],
                    [token_output_name],
                    name="wrapper_squeeze_token",
                ),
            ]
        )
        input_prefix = []
        output_prefix = []

    graph_inputs = [
        _value_info(semantic_input_name, [*input_prefix, semantic_input_dim]),
        _value_info(proprioception_input_name, [*input_prefix, proprio_dim]),
    ]
    if decoder_extra_dim:
        graph_inputs.append(
            _value_info(decoder_extra_input_name, [*input_prefix, decoder_extra_dim])
        )
    graph = helper.make_graph(
        nodes,
        f"merged_sonic_{mode or 'selected'}_policy",
        graph_inputs,
        [
            _value_info(output_name, [*output_prefix, action_dim]),
            _value_info(token_output_name, [*output_prefix, actual_token_dim]),
        ],
        initializer=initializers,
        value_info=[*encoder.graph.value_info, *decoder.graph.value_info],
    )
    model = helper.make_model(graph, producer_name="merge_sonic_encoder_decoder_onnx")
    model.ir_version = max(encoder.ir_version, decoder.ir_version)
    del model.opset_import[:]
    opsets: dict[str, int] = {}
    for opset in [*encoder.opset_import, *decoder.opset_import]:
        opsets[opset.domain] = max(opsets.get(opset.domain, 0), int(opset.version))
    for domain, version in sorted(opsets.items()):
        model.opset_import.append(helper.make_opsetid(domain, version))
    onnx.checker.check_model(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_path)
    print(
        f"saved inputs={[(value.name, _shape_dims(value)) for value in graph_inputs]} "
        f"outputs={[(value.name, _shape_dims(value)) for value in graph.output]} "
        f"to {output_path}"
    )


def rewrite_flat_model(
    *,
    source_path: Path,
    output_path: Path,
    source_input_name: str,
    input_specs: list[tuple[str, int]],
    source_output_name: str | None,
    output_name: str,
    token_source_tensor: str | None,
    token_output_name: str,
    token_dim: int,
) -> None:
    model = onnx.load(source_path)
    matching_inputs = [value for value in model.graph.input if value.name == source_input_name]
    if len(matching_inputs) != 1:
        raise ValueError(
            f"Expected one ONNX input named {source_input_name!r}, found {len(matching_inputs)}"
        )
    names = [*[name for name, _ in input_specs], output_name]
    if token_source_tensor is not None:
        names.append(token_output_name)
    _validate_unique_names(names)
    source_input = matching_inputs[0]
    source_shape = _shape_dims(source_input)
    if not source_shape or not isinstance(source_shape[-1], int):
        raise ValueError(f"Source input must have a static final dimension, got {source_shape}")
    requested_dim = sum(dim for _, dim in input_specs)
    if requested_dim != source_shape[-1]:
        raise ValueError(f"Semantic dimensions sum to {requested_dim}, expected {source_shape[-1]}")

    tensor_type = source_input.type.tensor_type
    semantic_inputs = [
        helper.make_tensor_value_info(
            name,
            tensor_type.elem_type,
            [*source_shape[:-1], dim],
        )
        for name, dim in input_specs
    ]
    retained_inputs = [value for value in model.graph.input if value.name != source_input_name]
    del model.graph.input[:]
    model.graph.input.extend([*semantic_inputs, *retained_inputs])
    model.graph.node.insert(
        0,
        helper.make_node(
            "Concat",
            [name for name, _ in input_specs],
            [source_input_name],
            axis=len(source_shape) - 1,
            name=f"assemble_{source_input_name}",
        ),
    )

    if source_output_name is not None and source_output_name != output_name:
        matching_outputs = [
            (index, value)
            for index, value in enumerate(model.graph.output)
            if value.name == source_output_name
        ]
        if len(matching_outputs) != 1:
            raise ValueError(
                f"Expected one output named {source_output_name!r}, found {len(matching_outputs)}"
            )
        output_index, source_output = matching_outputs[0]
        aliased_output = deepcopy(source_output)
        aliased_output.name = output_name
        del model.graph.output[output_index]
        model.graph.output.insert(output_index, aliased_output)
        model.graph.node.append(
            helper.make_node(
                "Identity",
                [source_output_name],
                [output_name],
                name=f"alias_{source_output_name}_as_{output_name}",
            )
        )

    if token_source_tensor is not None:
        tensor_names = {
            name
            for node in model.graph.node
            for name in (*node.input, *node.output)
            if name
        }
        if token_source_tensor not in tensor_names:
            raise ValueError(f"Tensor {token_source_tensor!r} is absent from {source_path}")
        token_shape_name = "sonic_token_output_shape"
        model.graph.initializer.append(_const_i64(token_shape_name, [token_dim]))
        model.graph.node.append(
            helper.make_node(
                "Reshape",
                [token_source_tensor, token_shape_name],
                [token_output_name],
                name="expose_sonic_token",
            )
        )
        model.graph.output.append(_value_info(token_output_name, [token_dim]))

    onnx.checker.check_model(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_path)
    print(
        f"saved inputs={[(value.name, _shape_dims(value)) for value in model.graph.input]} "
        f"outputs={[(value.name, _shape_dims(value)) for value in model.graph.output]} "
        f"to {output_path}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--encoder", type=Path, help="Encoder ONNX to merge with --decoder.")
    source.add_argument("--flat-source", type=Path, help="Already merged flat-input ONNX.")
    parser.add_argument("--decoder", type=Path, help="Decoder ONNX used with --encoder.")
    parser.add_argument("--output", type=Path, required=True)

    parser.add_argument("--mode", choices=sorted(UNIVERSAL_MODE_LAYOUTS))
    parser.add_argument("--encoder-input-name")
    parser.add_argument("--proprioception-input-name", default="proprioception")
    parser.add_argument("--decoder-extra-input-name", default="decoder_extra")
    parser.add_argument("--proprio-dim", type=int)
    parser.add_argument("--decoder-extra-dim", type=int, default=0)
    parser.add_argument("--batched", action="store_true")

    parser.add_argument("--source-input-name")
    parser.add_argument("--source-output-name")
    parser.add_argument(
        "--input",
        dest="input_specs",
        action="append",
        type=_parse_input_spec,
        metavar="NAME=DIM",
    )
    parser.add_argument("--token-source-tensor")
    parser.add_argument("--output-name", default="action")
    parser.add_argument("--token-output-name", default="token")
    parser.add_argument("--token-dim", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.encoder is not None:
        if args.decoder is None:
            raise ValueError("--encoder requires --decoder")
        if args.input_specs or args.source_input_name or args.token_source_tensor:
            raise ValueError("Flat-model options cannot be used with --encoder")
        merge_encoder_decoder(
            encoder_path=args.encoder,
            decoder_path=args.decoder,
            output_path=args.output,
            encoder_input_name=args.encoder_input_name,
            proprioception_input_name=args.proprioception_input_name,
            decoder_extra_input_name=args.decoder_extra_input_name,
            output_name=args.output_name,
            token_output_name=args.token_output_name,
            proprio_dim=args.proprio_dim,
            decoder_extra_dim=args.decoder_extra_dim,
            token_dim=args.token_dim,
            mode=args.mode,
            batched=args.batched,
        )
        return

    if args.decoder is not None or args.mode is not None or args.batched:
        raise ValueError("Merge-only options cannot be used with --flat-source")
    if args.source_input_name is None or not args.input_specs:
        raise ValueError("--flat-source requires --source-input-name and at least one --input")
    rewrite_flat_model(
        source_path=args.flat_source,
        output_path=args.output,
        source_input_name=args.source_input_name,
        input_specs=args.input_specs,
        source_output_name=args.source_output_name,
        output_name=args.output_name,
        token_source_tensor=args.token_source_tensor,
        token_output_name=args.token_output_name,
        token_dim=args.token_dim,
    )


if __name__ == "__main__":
    main()
