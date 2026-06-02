from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, compose, helper, numpy_helper


INT64_MAX = np.iinfo(np.int64).max


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge a SONIC encoder ONNX and decoder ONNX into one standalone ONNX. "
            "The merged input layout is [encoder_input | decoder_extra | proprioception]."
        )
    )
    parser.add_argument("--encoder", required=True, help="Encoder ONNX path.")
    parser.add_argument("--decoder", required=True, help="Decoder ONNX path.")
    parser.add_argument("--output", required=True, help="Merged ONNX output path.")
    parser.add_argument("--input-name", default="obs_dict", help="Merged model input name.")
    parser.add_argument("--output-name", default="action", help="Merged model output name.")
    parser.add_argument(
        "--proprio-dim",
        type=int,
        default=None,
        help="Decoder proprioception dimension. Defaults to decoder_input_dim - token_dim - decoder_extra_dim.",
    )
    parser.add_argument(
        "--decoder-extra-dim",
        type=int,
        default=0,
        help="Extra decoder tokenizer dimensions placed between encoder_input and proprioception.",
    )
    parser.add_argument(
        "--batched",
        action="store_true",
        help="Keep a [B, D] input and [B, A] output. Default exports unbatched [D] -> [A].",
    )
    return parser.parse_args()


def _shape_dims(value_info) -> list[int | str]:
    return [
        dim.dim_value if dim.dim_value else dim.dim_param
        for dim in value_info.type.tensor_type.shape.dim
    ]


def _last_static_dim(value_info, label: str) -> int:
    dims = _shape_dims(value_info)
    if not dims or not isinstance(dims[-1], int) or dims[-1] <= 0:
        raise ValueError(f"{label} must have a static last dimension, got {dims}")
    return int(dims[-1])


def _const_i64(name: str, values: list[int]) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(values, dtype=np.int64), name=name)


def _slice_node(
    *,
    input_name: str,
    output_name: str,
    start: int,
    end: int,
    rank: int,
    name: str,
) -> tuple[onnx.NodeProto, list[onnx.TensorProto]]:
    axis = 1 if rank == 2 else 0
    prefix = name.replace("/", "_")
    initializers = [
        _const_i64(f"{prefix}_starts", [start]),
        _const_i64(f"{prefix}_ends", [end]),
        _const_i64(f"{prefix}_axes", [axis]),
        _const_i64(f"{prefix}_steps", [1]),
    ]
    node = helper.make_node(
        "Slice",
        [input_name, *[init.name for init in initializers]],
        [output_name],
        name=name,
    )
    return node, initializers


def main() -> None:
    args = _parse_args()
    encoder = compose.add_prefix(onnx.load(args.encoder), "encoder/")
    decoder = compose.add_prefix(onnx.load(args.decoder), "decoder/")

    if len(encoder.graph.input) != 1 or len(encoder.graph.output) != 1:
        raise ValueError("Expected encoder to have exactly one input and one output")
    if len(decoder.graph.input) != 1 or len(decoder.graph.output) != 1:
        raise ValueError("Expected decoder to have exactly one input and one output")

    encoder_input = encoder.graph.input[0]
    encoder_output = encoder.graph.output[0]
    decoder_input = decoder.graph.input[0]
    decoder_output = decoder.graph.output[0]

    encoder_dim = _last_static_dim(encoder_input, "encoder input")
    token_dim = _last_static_dim(encoder_output, "encoder output")
    decoder_dim = _last_static_dim(decoder_input, "decoder input")
    action_dim = _last_static_dim(decoder_output, "decoder output")
    decoder_extra_dim = int(args.decoder_extra_dim)
    proprio_dim = args.proprio_dim
    if proprio_dim is None:
        proprio_dim = decoder_dim - token_dim - decoder_extra_dim
    if proprio_dim < 0 or token_dim + decoder_extra_dim + proprio_dim != decoder_dim:
        raise ValueError(
            "Invalid decoder dimensions: "
            f"decoder_dim={decoder_dim}, token_dim={token_dim}, "
            f"decoder_extra_dim={decoder_extra_dim}, proprio_dim={proprio_dim}"
        )

    merged_dim = encoder_dim + decoder_extra_dim + proprio_dim
    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []
    input_rank = 2 if args.batched else 1
    working_input = args.input_name

    graph_input_shape: list[int | str] = ["batch", merged_dim] if args.batched else [merged_dim]
    graph_output_shape: list[int | str] = ["batch", action_dim] if args.batched else [action_dim]

    if not args.batched:
        unsqueeze_axes = _const_i64("merged_unsqueeze_axes", [0])
        initializers.append(unsqueeze_axes)
        working_input = "merged_batched_input"
        nodes.append(
            helper.make_node(
                "Unsqueeze",
                [args.input_name, unsqueeze_axes.name],
                [working_input],
                name="merged_unsqueeze_input",
            )
        )
        input_rank = 2

    enc_slice, enc_inits = _slice_node(
        input_name=working_input,
        output_name=encoder_input.name,
        start=0,
        end=encoder_dim,
        rank=input_rank,
        name="merged_slice_encoder_input",
    )
    nodes.append(enc_slice)
    initializers.extend(enc_inits)

    decoder_concat_inputs = [encoder_output.name]
    if decoder_extra_dim:
        extra_slice, extra_inits = _slice_node(
            input_name=working_input,
            output_name="merged_decoder_extra",
            start=encoder_dim,
            end=encoder_dim + decoder_extra_dim,
            rank=input_rank,
            name="merged_slice_decoder_extra",
        )
        nodes.append(extra_slice)
        initializers.extend(extra_inits)
        decoder_concat_inputs.append("merged_decoder_extra")

    proprio_slice, proprio_inits = _slice_node(
        input_name=working_input,
        output_name="merged_proprioception",
        start=encoder_dim + decoder_extra_dim,
        end=merged_dim,
        rank=input_rank,
        name="merged_slice_proprioception",
    )
    nodes.append(proprio_slice)
    initializers.extend(proprio_inits)
    decoder_concat_inputs.append("merged_proprioception")

    nodes.extend(encoder.graph.node)
    nodes.append(
        helper.make_node(
            "Concat",
            decoder_concat_inputs,
            [decoder_input.name],
            name="merged_build_decoder_input",
            axis=1,
        )
    )
    nodes.extend(decoder.graph.node)

    graph_output_name = decoder_output.name
    if not args.batched:
        squeeze_axes = _const_i64("merged_squeeze_axes", [0])
        initializers.append(squeeze_axes)
        graph_output_name = args.output_name
        nodes.append(
            helper.make_node(
                "Squeeze",
                [decoder_output.name, squeeze_axes.name],
                [graph_output_name],
                name="merged_squeeze_output",
            )
        )

    graph = helper.make_graph(
        nodes,
        "merged_sonic_encoder_decoder",
        [
            helper.make_tensor_value_info(
                args.input_name,
                TensorProto.FLOAT,
                graph_input_shape,
            )
        ],
        [
            helper.make_tensor_value_info(
                graph_output_name,
                TensorProto.FLOAT,
                graph_output_shape,
            )
        ],
        initializer=[
            *initializers,
            *encoder.graph.initializer,
            *decoder.graph.initializer,
        ],
        value_info=[
            *encoder.graph.value_info,
            *decoder.graph.value_info,
        ],
    )
    model = helper.make_model(graph, producer_name="merge_sonic_encoder_decoder_onnx")
    model.ir_version = max(encoder.ir_version, decoder.ir_version)
    del model.opset_import[:]
    opsets = {}
    for opset in [*encoder.opset_import, *decoder.opset_import]:
        opsets[opset.domain] = max(opsets.get(opset.domain, 0), int(opset.version))
    for domain, version in sorted(opsets.items()):
        model.opset_import.append(helper.make_opsetid(domain, version))
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(
        f"merged encoder_dim={encoder_dim}, token_dim={token_dim}, "
        f"decoder_extra_dim={decoder_extra_dim}, proprio_dim={proprio_dim} -> {output_path}"
    )


if __name__ == "__main__":
    main()
