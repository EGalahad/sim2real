#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Sequence

import numpy as np
import onnx
from onnx import checker, version_converter


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert an exported ONNX to an older ONNX Runtime-compatible "
            "opset/IR pair. The default target is useful for Jetson/G1 "
            "onnxruntime-gpu==1.16.0, which supports IR <= 9."
        )
    )
    parser.add_argument("input", help="Source ONNX path.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output ONNX path. Defaults to <input-stem><suffix>.onnx.",
    )
    parser.add_argument(
        "--suffix",
        default="-ort116",
        help="Suffix used when --output is omitted.",
    )
    parser.add_argument(
        "--target-opset",
        type=int,
        default=19,
        help="Target ai.onnx opset version.",
    )
    parser.add_argument(
        "--target-ir",
        type=int,
        default=9,
        help="Target ONNX IR version.",
    )
    parser.add_argument(
        "--compare-runs",
        type=int,
        default=100,
        help=(
            "Number of random CPU ONNX Runtime comparisons against the source model. "
            "Set to 0 to skip. Requires onnxruntime."
        ),
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1.0e-6,
        help="Absolute tolerance for random-output comparison.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1.0e-5,
        help="Relative tolerance for random-output comparison.",
    )
    parser.add_argument(
        "--copy-yaml",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy adjacent .yaml to the converted ONNX stem when present.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser.parse_args(argv)


def _output_path(input_path: Path, output: str | None, suffix: str) -> Path:
    if output:
        return Path(output).expanduser()
    return input_path.with_name(f"{input_path.stem}{suffix}{input_path.suffix}")


def _opset_summary(model: onnx.ModelProto) -> list[tuple[str, int]]:
    return [(opset.domain or "ai.onnx", int(opset.version)) for opset in model.opset_import]


def _convert_model(
    input_path: Path,
    output_path: Path,
    *,
    target_opset: int,
    target_ir: int,
    overwrite: bool,
) -> onnx.ModelProto:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists, pass --overwrite to replace: {output_path}")

    model = onnx.load(input_path)
    converted = version_converter.convert_version(model, target_opset)
    converted.ir_version = int(target_ir)
    checker.check_model(converted)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(converted, output_path)
    return converted


def _tensor_shape(input_meta) -> tuple[int, ...]:
    shape = []
    for dim in input_meta.shape:
        if isinstance(dim, int) and dim > 0:
            shape.append(int(dim))
        else:
            shape.append(1)
    return tuple(shape)


def _random_inputs(session, rng: np.random.Generator) -> dict[str, np.ndarray]:
    inputs: dict[str, np.ndarray] = {}
    for item in session.get_inputs():
        if item.type != "tensor(float)":
            raise TypeError(
                f"Random comparison only supports tensor(float) inputs, got {item.name}: {item.type}"
            )
        inputs[item.name] = rng.standard_normal(_tensor_shape(item)).astype(np.float32)
    return inputs


def _compare_outputs(
    source_path: Path,
    converted_path: Path,
    *,
    runs: int,
    atol: float,
    rtol: float,
) -> tuple[float, float]:
    if runs <= 0:
        return 0.0, 0.0

    import onnxruntime as ort

    source = ort.InferenceSession(str(source_path), providers=["CPUExecutionProvider"])
    converted = ort.InferenceSession(str(converted_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    max_abs = 0.0
    max_rel = 0.0
    for _ in range(int(runs)):
        inputs = _random_inputs(source, rng)
        source_outputs = source.run(None, inputs)
        converted_outputs = converted.run(None, inputs)
        if len(source_outputs) != len(converted_outputs):
            raise ValueError(
                f"Output count mismatch: {len(source_outputs)} vs {len(converted_outputs)}"
            )
        for expected, actual in zip(source_outputs, converted_outputs, strict=True):
            diff = np.abs(np.asarray(expected) - np.asarray(actual))
            max_abs = max(max_abs, float(diff.max(initial=0.0)))
            max_rel = max(
                max_rel,
                float((diff / np.maximum(np.abs(expected), 1.0e-6)).max(initial=0.0)),
            )

    if max_abs > atol and max_rel > rtol:
        raise RuntimeError(
            f"Converted model differs from source: max_abs={max_abs}, max_rel={max_rel}, "
            f"atol={atol}, rtol={rtol}"
        )
    return max_abs, max_rel


def _copy_yaml(input_path: Path, output_path: Path, *, overwrite: bool) -> Path | None:
    source_yaml = input_path.with_suffix(".yaml")
    if not source_yaml.exists():
        return None

    output_yaml = output_path.with_suffix(".yaml")
    if output_yaml.exists() and not overwrite:
        raise FileExistsError(f"YAML output exists, pass --overwrite to replace: {output_yaml}")
    shutil.copy2(source_yaml, output_yaml)
    return output_yaml


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    input_path = Path(args.input).expanduser()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    output_path = _output_path(input_path, args.output, args.suffix)

    source_model = onnx.load(input_path)
    print(f"source={input_path}")
    print(f"source_ir={source_model.ir_version}")
    print(f"source_opsets={_opset_summary(source_model)}")

    converted = _convert_model(
        input_path,
        output_path,
        target_opset=args.target_opset,
        target_ir=args.target_ir,
        overwrite=bool(args.overwrite),
    )
    yaml_path = (
        _copy_yaml(input_path, output_path, overwrite=bool(args.overwrite))
        if args.copy_yaml
        else None
    )
    max_abs, max_rel = _compare_outputs(
        input_path,
        output_path,
        runs=int(args.compare_runs),
        atol=float(args.atol),
        rtol=float(args.rtol),
    )

    print(f"output={output_path}")
    print(f"output_size={output_path.stat().st_size}")
    if yaml_path is not None:
        print(f"yaml={yaml_path}")
    print(f"target_ir={converted.ir_version}")
    print(f"target_opsets={_opset_summary(converted)}")
    if args.compare_runs:
        print(f"compare_runs={args.compare_runs}")
        print(f"max_abs={max_abs}")
        print(f"max_rel={max_rel}")


if __name__ == "__main__":
    main()
