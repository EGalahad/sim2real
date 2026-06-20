from typing import Literal

from sim2real.rl_policy.inference.onnx_module import ONNXModule, Timer as Timer

InferenceBackend = Literal["onnx-gpu", "onnx-cpu", "tensorrt"]
__all__ = ["InferenceBackend", "ONNXModule", "Timer", "build_inference_module"]


def build_inference_module(onnx_path: str, inference_backend: InferenceBackend):
    if inference_backend in {"onnx-cpu", "onnx-gpu"}:
        provider = "gpu" if inference_backend == "onnx-gpu" else "cpu"
        return ONNXModule(onnx_path, providers=provider)
    if inference_backend == "tensorrt":
        from sim2real.rl_policy.inference.tensorrt_module import TensorRTModule

        return TensorRTModule(onnx_path)
    raise ValueError(f"Unsupported inference backend: {inference_backend}")
