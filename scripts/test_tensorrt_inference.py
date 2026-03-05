#!/usr/bin/env python3
"""
Benchmark TensorRT engine latency for exported ONNX policies.

Usage example:
    uv run scripts/test_tensorrt_inference.py \
        --policy_config checkpoints/lafan/policy-z7txtq03-4000.yaml \
        --warmup 50 --runs 1000 --fp16

The script builds (or loads cached) TensorRT plan next to the ONNX file and
uses metadata json to map input/output keys, mirroring scripts/test_onnx_inference.py.
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.append(".")
from rl_policy.utils.tensorrt_module import TensorRTModule


class TensorRTInferenceTest:
    def __init__(self, model_path: str, use_fp16: bool, rebuild: bool, workspace: int):
        self.model_path = model_path
        self.setup_policy(model_path, use_fp16, rebuild, workspace)
        self.setup_mock_data()

    def setup_policy(self, model_path: str, use_fp16: bool, rebuild: bool, workspace: int):
        print(f"Loading TensorRT engine from: {model_path}")

        if not Path(model_path).exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        json_path = model_path.replace(".onnx", ".json")
        if not Path(json_path).exists():
            raise FileNotFoundError(f"JSON metadata not found: {json_path}")

        self.trt_module = TensorRTModule(
            model_path,
            workspace_size=workspace,
            use_fp16=use_fp16,
            force_rebuild=rebuild,
        )

        def policy(input_dict):
            output_dict = self.trt_module(input_dict)
            action = output_dict["action"].squeeze(0)
            carry = {k[1]: v for k, v in output_dict.items() if isinstance(k, tuple) and k[0] == "next"}
            return action, carry

        self.policy = policy

        print(f"Model input keys: {self.trt_module.in_keys}")
        print(f"Model output keys: {self.trt_module.out_keys}")

    def setup_mock_data(self):
        print("Setting up mock input data...")

        input_specs = list(zip(self.trt_module.in_keys, self.trt_module.input_shapes))

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

    def run_single_inference(self) -> tuple:
        start_time = time.perf_counter()

        try:
            action, carry = self.policy(self.mock_input)
            for key, value in carry.items():
                if key in self.mock_input:
                    self.mock_input[key] = value
        except Exception as e:
            print(f"Inference error: {e}")
            return 0.0, None

        end_time = time.perf_counter()
        inference_time = end_time - start_time
        return inference_time, action

    def benchmark(self, num_warmup: int = 10, num_runs: int = 1000):
        print(f"\nStarting benchmark: {num_warmup} warmup + {num_runs} test runs")

        print("Warming up...")
        for _ in range(num_warmup):
            self.run_single_inference()

        print("Running benchmark...")
        times = []

        for i in range(num_runs):
            inference_time, _ = self.run_single_inference()
            times.append(inference_time)

            if (i + 1) % 100 == 0:
                avg_time = statistics.mean(times[-100:])
                print(f"  Progress: {i+1}/{num_runs}, Recent avg: {avg_time*1000:.2f}ms")

        self.print_statistics(times)

    def print_statistics(self, times):
        times_ms = [t * 1000 for t in times]

        print("\n" + "=" * 50)
        print("TENSORRT INFERENCE BENCHMARK RESULTS")
        print("=" * 50)
        print(f"Model: {self.model_path}")
        print(f"Number of runs: {len(times)}")
        print(f"Mean time: {statistics.mean(times_ms):.3f} ms")
        print(f"Median time: {statistics.median(times_ms):.3f} ms")
        print(f"Min time: {min(times_ms):.3f} ms")
        print(f"Max time: {max(times_ms):.3f} ms")
        print(f"Std deviation: {statistics.stdev(times_ms):.3f} ms")

        times_sorted = sorted(times_ms)
        p50 = times_sorted[len(times_sorted) // 2]
        p95 = times_sorted[int(len(times_sorted) * 0.95)]
        p99 = times_sorted[int(len(times_sorted) * 0.99)]

        print(f"50th percentile: {p50:.3f} ms")
        print(f"95th percentile: {p95:.3f} ms")
        print(f"99th percentile: {p99:.3f} ms")

        hz = 1.0 / statistics.mean(times)
        print(f"Average frequency: {hz:.1f} Hz")
        print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="TensorRT Inference Performance Test")
    parser.add_argument(
        "--policy_config",
        type=str,
        required=True,
        help="Path to policy config file (ONNX + JSON must exist).",
    )
    parser.add_argument("--warmup", type=int, default=50, help="Number of warmup runs")
    parser.add_argument("--runs", type=int, default=1000, help="Number of test runs")
    parser.add_argument("--single", action="store_true", help="Run single inference test only")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 builder flag")
    parser.add_argument(
        "--workspace",
        type=int,
        default=1 << 30,
        help="Workspace size in bytes (default 1GB)",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild TensorRT engine even if cached .plan exists",
    )

    args = parser.parse_args()

    try:
        model_path = args.policy_config.replace(".yaml", ".onnx")
        tester = TensorRTInferenceTest(
            model_path,
            use_fp16=args.fp16,
            rebuild=args.rebuild,
            workspace=args.workspace,
        )

        if args.single:
            print("Running single inference test...")
            inference_time, action = tester.run_single_inference()
            print(f"Inference time: {inference_time*1000:.3f} ms")
            print(f"Action shape: {action.shape}")
            print(f"Action sample: {action[:5]}")
        else:
            tester.benchmark(num_warmup=args.warmup, num_runs=args.runs)

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
