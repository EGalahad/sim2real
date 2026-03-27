import json
import ctypes
import site
import numpy as np
import time
from typing import Dict
from pathlib import Path


def _preload_cuda_runtime_libraries() -> None:
    """Preload packaged CUDA/cuDNN shared libraries so ORT can resolve them."""
    lib_relpaths = [
        ("cuda_runtime", "lib", "libcudart.so.12"),
        ("cuda_nvrtc", "lib", "libnvrtc.so.12"),
        ("cuda_nvrtc", "lib", "libnvrtc-builtins.so.12.8"),
        ("nvjitlink", "lib", "libnvJitLink.so.12"),
        ("curand", "lib", "libcurand.so.10"),
        ("cublas", "lib", "libcublas.so.12"),
        ("cublas", "lib", "libcublasLt.so.12"),
        ("cudnn", "lib", "libcudnn.so.9"),
        ("cufft", "lib", "libcufft.so.11"),
        ("cusolver", "lib", "libcusolver.so.11"),
        ("cusparse", "lib", "libcusparse.so.12"),
        ("cusparselt", "lib", "libcusparseLt.so.0"),
    ]

    search_roots = []
    try:
        search_roots.extend(site.getsitepackages())
    except Exception:
        pass
    try:
        user_site = site.getusersitepackages()
        if isinstance(user_site, str):
            search_roots.append(user_site)
        else:
            search_roots.extend(user_site)
    except Exception:
        pass

    loaded: set[str] = set()
    for root in search_roots:
        nvidia_root = Path(root) / "nvidia"
        for rel in lib_relpaths:
            lib_path = nvidia_root.joinpath(*rel)
            if not lib_path.exists():
                continue
            lib_key = str(lib_path)
            if lib_key in loaded:
                continue
            ctypes.CDLL(lib_key, mode=ctypes.RTLD_GLOBAL)
            loaded.add(lib_key)


_preload_cuda_runtime_libraries()

import onnxruntime as ort


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

        self.ort_session = ort.InferenceSession(path, providers=requested)
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
            args[inp.name] = self._get_input_value(input_dict, key, inp.name).squeeze(0)
        # print(args.keys())
        # print(args["policy"].shape)
        # print(args["command"].shape)
        # breakpoint()
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
