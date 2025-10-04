#!/usr/bin/env python3
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import statistics

# 添加项目路径
sys.path.append(".")
from rl_policy.utils.onnx_module import ONNXModule


class ONNXInferenceTest:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.setup_policy(model_path)
        self.setup_mock_data()
        
    def setup_policy(self, model_path: str):
        print(f"Loading ONNX model from: {model_path}")
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        
        json_path = model_path.replace(".onnx", ".json")
        if not Path(json_path).exists():
            raise FileNotFoundError(f"JSON metadata not found: {json_path}")
        
        self.onnx_module = ONNXModule(model_path)
        
        def policy(input_dict):
            output_dict = self.onnx_module(input_dict)
            action = output_dict["action"].squeeze(0)
            carry = {k[1]: v for k, v in output_dict.items() if k[0] == "next"}
            return action, carry
        
        self.policy = policy
        
        print(f"Model input keys: {self.onnx_module.in_keys}")
        print(f"Model output keys: {self.onnx_module.out_keys}")

    def setup_mock_data(self):
        print("Setting up mock input data...")
        
        session = self.onnx_module.ort_session
        input_specs = [(inp.name, inp.shape) for inp in session.get_inputs()]
        
        print("Input specifications:")
        for name, shape in input_specs:
            print(f"  {name}: {shape}")
        
        self.mock_input = {}
        for i, (input_name, input_shape) in enumerate(input_specs):
            actual_shape = []
            for dim in input_shape:
                if isinstance(dim, str) or dim == -1:
                    actual_shape.append(1)  # batch size = 1
                else:
                    actual_shape.append(dim)
            
            input_key = self.onnx_module.in_keys[i]
            
            if "adapt_hx" in str(input_key):
                self.mock_input[input_key] = np.zeros(actual_shape, dtype=np.float32)
            elif "action" in str(input_key):
                self.mock_input[input_key] = np.random.randn(*actual_shape).astype(np.float32) * 0.1
            elif "is_init" in str(input_key):
                self.mock_input[input_key] = np.zeros(actual_shape, dtype=bool)
            else:
                self.mock_input[input_key] = np.random.randn(*actual_shape).astype(np.float32)
        
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
        for i in range(num_warmup):
            self.run_single_inference()
        
        print("Running benchmark...")
        times = []
        
        for i in range(num_runs):
            inference_time, action = self.run_single_inference()
            times.append(inference_time)
            
            if (i + 1) % 100 == 0:
                avg_time = statistics.mean(times[-100:])
                print(f"  Progress: {i+1}/{num_runs}, Recent avg: {avg_time*1000:.2f}ms")
        
        self.print_statistics(times)

    def print_statistics(self, times):
        times_ms = [t * 1000 for t in times]  # 转换为毫秒
        
        print("\n" + "="*50)
        print("ONNX INFERENCE BENCHMARK RESULTS")
        print("="*50)
        print(f"Model: {self.model_path}")
        print(f"Number of runs: {len(times)}")
        print(f"Mean time: {statistics.mean(times_ms):.3f} ms")
        print(f"Median time: {statistics.median(times_ms):.3f} ms")
        print(f"Min time: {min(times_ms):.3f} ms")
        print(f"Max time: {max(times_ms):.3f} ms")
        print(f"Std deviation: {statistics.stdev(times_ms):.3f} ms")
        
        times_sorted = sorted(times_ms)
        p50 = times_sorted[len(times_sorted)//2]
        p95 = times_sorted[int(len(times_sorted)*0.95)]
        p99 = times_sorted[int(len(times_sorted)*0.99)]
        
        print(f"50th percentile: {p50:.3f} ms")
        print(f"95th percentile: {p95:.3f} ms") 
        print(f"99th percentile: {p99:.3f} ms")
        
        hz = 1.0 / statistics.mean(times)
        print(f"Average frequency: {hz:.1f} Hz")
        
        print("="*50)


def main():
    parser = argparse.ArgumentParser(description="ONNX Inference Performance Test")
    parser.add_argument(
        "--policy_config", 
        type=str,
        help="Path to policy config file"
    )
    parser.add_argument(
        "--warmup", 
        type=int, 
        default=50,
        help="Number of warmup runs"
    )
    parser.add_argument(
        "--runs", 
        type=int, 
        default=1000,
        help="Number of test runs"
    )
    parser.add_argument(
        "--single", 
        action="store_true",
        help="Run single inference test only"
    )
    
    args = parser.parse_args()
    
    try:
        model_path = args.policy_config.replace(".yaml", ".onnx")
        tester = ONNXInferenceTest(model_path)
        
        if args.single:
            print("Running single inference test...")
            inference_time, action = tester.run_single_inference()
            print(f"Inference time: {inference_time*1000:.3f} ms")
            print(f"Action shape: {action.shape}")
            print(f"Action sample: {action[:5]}")  # 显示前5个元素
        else:
            tester.benchmark(num_warmup=args.warmup, num_runs=args.runs)
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 