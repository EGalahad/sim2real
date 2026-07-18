import json
import os
import time
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np

from sim2real.rl_policy.inference.cuda_runtime import preload_cuda_runtime_libraries

preload_cuda_runtime_libraries()

import onnxruntime as ort  # type: ignore[import-untyped]  # noqa: E402


def _normalize_input_name(name: str):
    key = name
    if key.endswith("_orig"):
        key = key[: -len("_orig")]
    if key.startswith("next_"):
        key = key[len("next_") :]
    return key


def _normalize_output_name(name: str):
    key = name
    if key.endswith("_orig"):
        key = key[: -len("_orig")]
    if key.startswith("next_"):
        return ("next", key[len("next_") :])
    return key


def _concrete_shape(shape) -> Tuple[int, ...]:
    return tuple(1 if isinstance(dim, str) or dim == -1 else int(dim) for dim in shape)


def _squeeze_leading_batch_dim(value: np.ndarray, expected_shape: Tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value)
    if tuple(array.shape) == expected_shape:
        return array
    if array.ndim > 0 and array.shape[0] == 1:
        squeezed = np.squeeze(array, axis=0)
        if tuple(squeezed.shape) == expected_shape:
            return squeezed
    raise ValueError(f"Expected input shape {expected_shape}, got {array.shape}")


def _squeeze_output_value(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim > 0 and array.shape[0] == 1:
        return np.squeeze(array, axis=0)
    return array

class ONNXModule:
    
    def __init__(self, path: str, providers=None):
        """
        providers: str, either "cpu" or "gpu"
        """
        if not isinstance(providers, str):
            raise TypeError(f"Unsupported providers type: {type(providers)}")

        norm = providers.lower().strip()
        if norm == "cpu":
            requested = ["CPUExecutionProvider"]
        elif norm in {"gpu", "cuda"}:
            requested = ["CUDAExecutionProvider"]
        else:
            raise ValueError(f"Unsupported provider: {providers}. Use 'cpu' or 'gpu'.")

        available = set(ort.get_available_providers())

        if requested[0] not in available:
            raise RuntimeError(
                f"Requested provider {requested[0]} is not available. available={available}"
            )

        session_options = None
        eval_threads = int(os.environ.get("SIM2REAL_ORT_NUM_THREADS", "0"))
        if eval_threads > 0:
            session_options = ort.SessionOptions()
            session_options.intra_op_num_threads = eval_threads
            session_options.inter_op_num_threads = 1
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self.ort_session = ort.InferenceSession(
            path,
            sess_options=session_options,
            providers=requested,
        )
        active_providers = self.ort_session.get_providers()
        if requested[0] not in active_providers:
            raise RuntimeError(
                f"Requested provider {requested[0]} did not become active. "
                f"active_providers={active_providers}. This usually means the process "
                "cannot access the requested accelerator."
            )
        meta_path = Path(path.replace(".onnx", ".json"))
        if meta_path.exists():
            with open(meta_path, "r") as f:
                self.meta = json.load(f)
            self.in_keys = [k if isinstance(k, str) else tuple(k) for k in self.meta["in_keys"]]
            self.out_keys = [k if isinstance(k, str) else tuple(k) for k in self.meta["out_keys"]]
        else:
            self.meta = {}
            self.in_keys = [
                _normalize_input_name(inp.name) for inp in self.ort_session.get_inputs()
            ]
            self.out_keys = [
                _normalize_output_name(out.name) for out in self.ort_session.get_outputs()
            ]
        self.input_shapes = [
            _concrete_shape(inp.shape) for inp in self.ort_session.get_inputs()
        ]

    @staticmethod
    def _get_input_value(
        input_dict: Dict[Union[str, Tuple[str, str]], np.ndarray],
        key,
        ort_input_name: str,
    ):
        if key in input_dict:
            return input_dict[key]
        if ort_input_name in input_dict:
            return input_dict[ort_input_name]
        normalized = _normalize_input_name(ort_input_name)
        if normalized in input_dict:
            return input_dict[normalized]
        if isinstance(key, tuple) and len(key) == 2 and key[0] == "next" and key[1] in input_dict:
            return input_dict[key[1]]
        if isinstance(key, str) and ("next", key) in input_dict:
            return input_dict[("next", key)]
        raise KeyError(
            f'Missing ONNX input for binding "{ort_input_name}" (expected key "{key}")'
        )
    
    def __call__(
        self, input_dict: Dict[Union[str, Tuple[str, str]], np.ndarray]
    ) -> Dict[Union[str, Tuple[str, str]], np.ndarray]:
        args = {}
        for inp, key, expected_shape in zip(
            self.ort_session.get_inputs(),
            self.in_keys,
            self.input_shapes,
        ):
            args[inp.name] = _squeeze_leading_batch_dim(
                self._get_input_value(input_dict, key, inp.name),
                expected_shape,
            )
        outputs = self.ort_session.run(None, args)
        outputs = {
            k: _squeeze_output_value(v) for k, v in zip(self.out_keys, outputs)
        }
        return outputs

class Timer:
    def __init__(self, perf_dict: Dict[str, float], name: str):
        self.perf_dict = perf_dict
        self.name = name

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        elapsed_time = time.perf_counter() - self.start_time
        if self.name not in self.perf_dict:
            self.perf_dict[self.name] = 0
        self.perf_dict[self.name] += elapsed_time
