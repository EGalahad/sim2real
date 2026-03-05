import json
import numpy as np
import onnxruntime as ort
import time
from typing import Dict
from pathlib import Path


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

class ONNXModule:
    
    def __init__(self, path: str, providers=None):
        """
        providers: list or str. Examples: "cpu", "cuda", ["CUDAExecutionProvider", "CPUExecutionProvider"]
        """
        available = set(ort.get_available_providers())
        if providers is None:
            requested = ["CPUExecutionProvider"]
        elif isinstance(providers, str):
            norm = providers.lower()
            if norm == "cpu":
                requested = ["CPUExecutionProvider"]
            elif norm == "cuda":
                requested = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                requested = [providers]
        # keep only available providers to avoid ort crash; error if none left
        providers = [p for p in requested if p in available]
        if not providers:
            raise RuntimeError(
                f"Requested providers not available. Requested={requested}, available={available}"
            )

        self.ort_session = ort.InferenceSession(path, providers=providers)
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

    @staticmethod
    def _get_input_value(input_dict: Dict[str, np.ndarray], key, ort_input_name: str):
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
    
    def __call__(self, input_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        args = {}
        for inp, key in zip(self.ort_session.get_inputs(), self.in_keys):
            args[inp.name] = self._get_input_value(input_dict, key, inp.name)
        outputs = self.ort_session.run(None, args)
        outputs = {k: v for k, v in zip(self.out_keys, outputs)}
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
