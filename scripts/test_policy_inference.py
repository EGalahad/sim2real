#!/usr/bin/env python3
"""
Benchmark exported policy inference with a unified backend switch.

TensorRT runtime options are configured in
`sim2real.rl_policy.inference.tensorrt_module` and can be overridden with:
  - HDMI_TRT_FP16=0|1
  - HDMI_TRT_WORKSPACE=<bytes>
  - HDMI_TRT_FORCE_REBUILD=0|1
"""

import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import tyro

from sim2real.rl_policy.inference import build_inference_module

InferenceBackend = Literal["onnx-gpu", "onnx-cpu", "tensorrt"]


def build_runtime_module(
    model_path: str,
    inference_backend: InferenceBackend,
):
    runtime_module = build_inference_module(model_path, inference_backend)
    runtime_label: str = inference_backend
    if inference_backend == "tensorrt":
        runtime_label = (
            "tensorrt"
            f"[fp16={runtime_module.use_fp16}, "
            f"workspace={runtime_module.workspace_size}]"
        )
    return runtime_module, runtime_label


class PolicyInferenceTest:
    def __init__(self, model_path: str, inference_backend: InferenceBackend):
        self.model_path = model_path
        self.inference_backend = inference_backend
        self.setup_policy(model_path, inference_backend)
        self.setup_mock_data()

    def setup_policy(self, model_path: str, inference_backend: InferenceBackend):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        self.runtime_module, runtime_label = build_runtime_module(
            model_path,
            inference_backend=inference_backend,
        )
        print(f"Loading {runtime_label} model from: {model_path}")

        def policy(input_dict):
            output_dict = self.runtime_module(input_dict)
            action = np.asarray(output_dict["action"], dtype=np.float32)
            carry = {
                k[1]: v
                for k, v in output_dict.items()
                if isinstance(k, tuple) and len(k) == 2 and k[0] == "next"
            }
            return action, carry

        self.policy = policy

        print(f"Model input keys: {self.runtime_module.in_keys}")
        print(f"Model output keys: {self.runtime_module.out_keys}")

    def setup_mock_data(self):
        print("Setting up mock input data...")
        input_specs = list(zip(self.runtime_module.in_keys, self.runtime_module.input_shapes))

        print("Input specifications:")
        for name, shape in input_specs:
            print(f"  {name}: {shape}")

        self.mock_input = {}
        for key, shape in input_specs:
            if "adapt_hx" in str(key):
                self.mock_input[key] = np.zeros(shape, dtype=np.float32)
            elif "action" in str(key):
                self.mock_input[key] = np.random.randn(*shape).astype(np.float32) * 0.1
            elif "is_init" in str(key):
                self.mock_input[key] = np.zeros(shape, dtype=bool)
            else:
                self.mock_input[key] = np.random.randn(*shape).astype(np.float32)

        print(f"Created mock input with keys: {list(self.mock_input.keys())}")
        for key, value in self.mock_input.items():
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")

    def run_single_inference(self) -> Tuple[float, Optional[np.ndarray]]:
        start_time = time.perf_counter()

        try:
            action, carry = self.policy(self.mock_input)
            for key, value in carry.items():
                if key in self.mock_input:
                    self.mock_input[key] = value
        except Exception as exc:
            print(f"Inference error: {exc}")
            return 0.0, None

        return time.perf_counter() - start_time, action

    def benchmark(self, num_warmup: int = 10, num_runs: int = 1000):
        print(f"\nStarting benchmark: {num_warmup} warmup + {num_runs} test runs")

        print("Warming up...")
        for _ in range(num_warmup):
            self.run_single_inference()

        print("Running benchmark...")
        times = []

        for i in range(num_runs):
            inference_time, action = self.run_single_inference()
            if action is None:
                continue
            times.append(inference_time)

            if (i + 1) % 100 == 0 and len(times) >= 100:
                avg_time = statistics.mean(times[-100:])
                print(f"  Progress: {i+1}/{num_runs}, Recent avg: {avg_time*1000:.2f}ms")

        self.print_statistics(times)

    def print_statistics(self, times):
        if not times:
            print("\nNo successful inference runs to report.")
            return

        times_ms = [t * 1000 for t in times]

        print("\n" + "=" * 50)
        print("POLICY INFERENCE BENCHMARK RESULTS")
        print("=" * 50)
        print(f"Backend: {self.inference_backend}")
        print(f"Model: {self.model_path}")
        print(f"Number of runs: {len(times)}")
        print(f"Mean time: {statistics.mean(times_ms):.3f} ms")
        print(f"Median time: {statistics.median(times_ms):.3f} ms")
        print(f"Min time: {min(times_ms):.3f} ms")
        print(f"Max time: {max(times_ms):.3f} ms")
        if len(times_ms) > 1:
            print(f"Std deviation: {statistics.stdev(times_ms):.3f} ms")
        else:
            print("Std deviation: N/A (need at least 2 runs)")

        times_sorted = sorted(times_ms)
        p50 = times_sorted[len(times_sorted) // 2]
        p95 = times_sorted[int(len(times_sorted) * 0.95)]
        p99 = times_sorted[int(len(times_sorted) * 0.99)]

        print(f"50th percentile: {p50:.3f} ms")
        print(f"95th percentile: {p95:.3f} ms")
        print(f"99th percentile: {p99:.3f} ms")
        print(f"Average frequency: {1.0 / statistics.mean(times):.1f} Hz")
        print("=" * 50)


@dataclass
class Args:
    policy_config: str
    warmup: int = 50
    runs: int = 1000
    single: bool = False
    inference_backend: InferenceBackend = "onnx-cpu"


def main(args: Args) -> int:
    try:
        model_path = args.policy_config.replace(".yaml", ".onnx")
        tester = PolicyInferenceTest(
            model_path=model_path,
            inference_backend=args.inference_backend,
        )

        if args.single:
            print("Running single inference test...")
            inference_time, action = tester.run_single_inference()
            if action is None:
                return 1
            print(f"Inference time: {inference_time*1000:.3f} ms")
            print(f"Action shape: {action.shape}")
            print(f"Action sample: {action[:5]}")
        else:
            tester.benchmark(num_warmup=args.warmup, num_runs=args.runs)
    except Exception as exc:
        print(f"Error: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main(tyro.cli(Args)))
