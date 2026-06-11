import json
import os
import platform
import sys
import importlib.util
import importlib
import ctypes
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
from sim2real.rl_policy.inference.cuda_runtime import load_cudart_library

# TensorRT relies on the CUDA runtime. The cuda-python package exposes the
# low-level cudart API that the official TensorRT quick-start guide uses.
# https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/quick-start-guide.html
def _load_tensorrt_from_init(init_path: Path):
    spec = importlib.util.spec_from_file_location(
        "tensorrt",
        init_path,
        submodule_search_locations=[str(init_path.parent)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to create TensorRT spec from {init_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["tensorrt"] = module
    spec.loader.exec_module(module)
    return module


def _import_tensorrt():
    prefer_system = (
        platform.machine() == "aarch64"
        and Path("/etc/nv_tegra_release").exists()
        and os.environ.get("HDMI_PREFER_SYSTEM_TENSORRT", "1") != "0"
    )
    if prefer_system:
        for base in (
            Path("/usr/lib/python3/dist-packages"),
            Path("/usr/lib/python3.10/dist-packages"),
        ):
            init_path = base / "tensorrt" / "__init__.py"
            if init_path.is_file():
                return _load_tensorrt_from_init(init_path)

    try:
        import tensorrt as trt_mod  # type: ignore[import-untyped]
        return trt_mod
    except ImportError as exc:  # pragma: no cover - optional dependency
        print("TensorRTModule requires the `tensorrt` package.")
        return None


trt = _import_tensorrt()


TRT_FP16 = False
TRT_WORKSPACE = 1 << 30
TRT_FORCE_REBUILD = False


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value for {name}: {value}")


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    return int(value.strip())


def get_tensorrt_runtime_config() -> Dict[str, Union[bool, int]]:
    return {
        "use_fp16": _env_bool("HDMI_TRT_FP16", TRT_FP16),
        "workspace_size": _env_int("HDMI_TRT_WORKSPACE", TRT_WORKSPACE),
        "force_rebuild": _env_bool("HDMI_TRT_FORCE_REBUILD", TRT_FORCE_REBUILD),
    }


class _CudaMemcpyKind:
    cudaMemcpyHostToDevice = 1
    cudaMemcpyDeviceToHost = 2


class _CudaErrorEnum(int):
    _name: str

    def __new__(cls, value: int, name: str = ""):
        obj = int.__new__(cls, value)
        obj._name = name
        return obj

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<cudaError_t.{self._name}: {int(self)}>"


class _CudaErrorType:
    cudaSuccess = _CudaErrorEnum(0, "cudaSuccess")
    cudaErrorInsufficientDriver = _CudaErrorEnum(35, "cudaErrorInsufficientDriver")
    cudaErrorNoDevice = _CudaErrorEnum(100, "cudaErrorNoDevice")


class _CtypesCudart:
    cudaMemcpyKind = _CudaMemcpyKind
    cudaError_t = _CudaErrorType

    def __init__(self):
        self.lib = load_cudart_library()

        self.lib.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self.lib.cudaGetDeviceCount.restype = ctypes.c_int

        self.lib.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.cudaStreamCreate.restype = ctypes.c_int

        self.lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.lib.cudaMalloc.restype = ctypes.c_int

        self.lib.cudaFree.argtypes = [ctypes.c_void_p]
        self.lib.cudaFree.restype = ctypes.c_int

        self.lib.cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.lib.cudaMemcpyAsync.restype = ctypes.c_int

        self.lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        self.lib.cudaStreamSynchronize.restype = ctypes.c_int

        self.lib.cudaEventCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.cudaEventCreate.restype = ctypes.c_int

        self.lib.cudaEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.lib.cudaEventRecord.restype = ctypes.c_int

        self.lib.cudaEventSynchronize.argtypes = [ctypes.c_void_p]
        self.lib.cudaEventSynchronize.restype = ctypes.c_int

        self.lib.cudaEventElapsedTime.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.lib.cudaEventElapsedTime.restype = ctypes.c_int

    @staticmethod
    def _wrap_error(code: int) -> _CudaErrorEnum:
        if code == 0:
            return _CudaErrorType.cudaSuccess
        if code == 35:
            return _CudaErrorType.cudaErrorInsufficientDriver
        if code == 100:
            return _CudaErrorType.cudaErrorNoDevice
        return _CudaErrorEnum(code, f"cudaError{code}")

    def cudaGetDeviceCount(self):
        count = ctypes.c_int()
        code = self.lib.cudaGetDeviceCount(ctypes.byref(count))
        return self._wrap_error(code), (count.value if code == 0 else None)

    def cudaStreamCreate(self):
        stream = ctypes.c_void_p()
        code = self.lib.cudaStreamCreate(ctypes.byref(stream))
        return self._wrap_error(code), (stream.value if code == 0 else None)

    def cudaMalloc(self, size: int):
        ptr = ctypes.c_void_p()
        code = self.lib.cudaMalloc(ctypes.byref(ptr), int(size))
        return self._wrap_error(code), (ptr.value if code == 0 else None)

    def cudaFree(self, ptr: int):
        return self._wrap_error(self.lib.cudaFree(ctypes.c_void_p(ptr)))

    def cudaMemcpyAsync(self, dst: int, src: int, size: int, kind: int, stream: int):
        return self._wrap_error(
            self.lib.cudaMemcpyAsync(
                ctypes.c_void_p(dst),
                ctypes.c_void_p(src),
                int(size),
                int(kind),
                ctypes.c_void_p(stream),
            )
        )

    def cudaStreamSynchronize(self, stream: int):
        return self._wrap_error(
            self.lib.cudaStreamSynchronize(ctypes.c_void_p(stream))
        )

    def cudaEventCreate(self):
        event = ctypes.c_void_p()
        code = self.lib.cudaEventCreate(ctypes.byref(event))
        return self._wrap_error(code), (event.value if code == 0 else None)

    def cudaEventRecord(self, event: int, stream: int):
        return self._wrap_error(
            self.lib.cudaEventRecord(ctypes.c_void_p(event), ctypes.c_void_p(stream))
        )

    def cudaEventSynchronize(self, event: int):
        return self._wrap_error(
            self.lib.cudaEventSynchronize(ctypes.c_void_p(event))
        )

    def cudaEventElapsedTime(self, start: int, end: int):
        elapsed_ms = ctypes.c_float()
        code = self.lib.cudaEventElapsedTime(
            ctypes.byref(elapsed_ms),
            ctypes.c_void_p(start),
            ctypes.c_void_p(end),
        )
        return self._wrap_error(code), elapsed_ms.value


def _import_cudart():
    # cuda-python wheels on some Python versions expose cudart via cuda.cudart;
    # on others only via cuda.bindings.runtime. Try both, then verify they can
    # actually enumerate a device. If not, fall back to system libcudart.
    for importer in (
        lambda: importlib.import_module("cuda.cudart"),
        lambda: importlib.import_module("cuda.bindings.runtime"),
    ):
        try:
            candidate = importer()
        except Exception:
            continue

        try:
            err, _count = candidate.cudaGetDeviceCount()
            if int(err) == 0:
                return candidate
        except Exception:
            pass

    return _CtypesCudart()


cudart = _import_cudart()


def _ensure_cuda_runtime_ready() -> None:
    err, count = cudart.cudaGetDeviceCount()
    if int(err) == 0 and (count or 0) > 0:
        return

    lib_name = getattr(getattr(cudart, "lib", None), "_name", "<cuda-python>")
    raise RuntimeError(
        "TensorRT requires an accessible CUDA device, but cudaGetDeviceCount "
        f"returned {err} while using {lib_name}. "
        "Confirm the NVIDIA driver is installed, the process has GPU access, "
        "and the CUDA runtime version matches the driver."
    )

# Optional ONNX dependency for lightweight graph fixes.
try:  # pragma: no cover - optional
    import onnx
except Exception:  # pragma: no cover
    onnx = None  # type: ignore

Key = Union[str, Tuple[str, str]]


def _lookup_input(input_dict: Dict[Key, np.ndarray], key: Key) -> np.ndarray:
    if key in input_dict:
        return input_dict[key]
    if isinstance(key, tuple) and len(key) == 2 and key[0] == "next" and key[1] in input_dict:
        return input_dict[key[1]]
    if isinstance(key, str) and ("next", key) in input_dict:
        return input_dict[("next", key)]
    raise KeyError(f"Missing policy input key: {key}")


def _prepare_input_value(
    input_dict: Dict[Key, np.ndarray],
    key: Key,
    expected_shape: Tuple[int, ...],
) -> np.ndarray:
    value = np.asarray(_lookup_input(input_dict, key))
    if tuple(value.shape) == expected_shape:
        return np.ascontiguousarray(value)

    if value.ndim > 0 and value.shape[0] == 1:
        squeezed = np.squeeze(value, axis=0)
        if tuple(squeezed.shape) == expected_shape:
            return np.ascontiguousarray(squeezed)

    raise ValueError(
        f"TensorRT input shape mismatch for {key}: "
        f"expected {expected_shape}, got {value.shape}"
    )


def _squeeze_output_value(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim > 0 and array.shape[0] == 1:
        return np.squeeze(array, axis=0)
    return array


def _cuda_stream_handle(stream) -> int:
    return int(stream)


def _normalize_input_name(name: str) -> str:
    key = name
    if key.endswith("_orig"):
        key = key[: -len("_orig")]
    if key.startswith("next_"):
        key = key[len("next_") :]
    return key


def _normalize_output_name(name: str) -> Key:
    key = name
    if key.endswith("_orig"):
        key = key[: -len("_orig")]
    if key.startswith("next_"):
        return ("next", key[len("next_") :])
    return key


def _normalize_keys(raw_keys: Sequence[Union[str, Sequence[str]]]) -> List[Key]:
    normalized: List[Key] = []
    for key in raw_keys:
        if isinstance(key, str):
            normalized.append(key)
        else:
            normalized.append(cast(Key, tuple(key)))
    return normalized


def _infer_meta_from_onnx(onnx_path: str) -> Tuple[List[Key], List[Key], List[Tuple[int, ...]]]:
    import onnxruntime as ort  # type: ignore[import-untyped]

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_keys: List[Key] = [_normalize_input_name(inp.name) for inp in session.get_inputs()]
    out_keys: List[Key] = [_normalize_output_name(out.name) for out in session.get_outputs()]

    input_shapes: List[Tuple[int, ...]] = []
    for inp in session.get_inputs():
        concrete = [
            1 if isinstance(dim, str) or dim == -1 else int(dim)
            for dim in inp.shape
        ]
        input_shapes.append(tuple(concrete))

    return in_keys, out_keys, input_shapes


class TensorRTModule:
    """Lightweight TensorRT wrapper that mirrors the ONNXModule interface."""

    def __init__(
        self,
        onnx_path: str,
        *,
        workspace_size: Optional[int] = None,
        use_fp16: Optional[bool] = None,
        force_rebuild: Optional[bool] = None,
    ):
        config = get_tensorrt_runtime_config()
        self.onnx_path = str(onnx_path)
        self.plan_path = self.onnx_path.replace(".onnx", ".plan")
        self.workspace_size = int(
            config["workspace_size"] if workspace_size is None else workspace_size
        )
        self.use_fp16 = bool(config["use_fp16"] if use_fp16 is None else use_fp16)
        self.force_rebuild = bool(
            config["force_rebuild"] if force_rebuild is None else force_rebuild
        )
        self.logger = trt.Logger(trt.Logger.WARNING)
        _ensure_cuda_runtime_ready()

        meta_path = self.onnx_path.replace(".onnx", ".json")
        if Path(meta_path).exists():
            with open(meta_path, "r") as f:
                meta = json.load(f)
            self.in_keys = _normalize_keys(meta.get("in_keys", []))
            self.out_keys = _normalize_keys(meta.get("out_keys", []))
            # meta["in_shapes"] is stored as a list of shape-sets; we take the first set
            # and replace dynamic dims (-1 / str) with 1 for explicit batch.
            raw_shapes = meta.get("in_shapes", [])
            shape_set = raw_shapes[0] if raw_shapes else []
            self.input_shapes = []
            for shape in shape_set:
                concrete = [
                    1 if isinstance(dim, str) or dim == -1 else int(dim)
                    for dim in shape
                ]
                self.input_shapes.append(tuple(concrete))
        else:
            self.in_keys, self.out_keys, self.input_shapes = _infer_meta_from_onnx(
                self.onnx_path
            )

        self.runtime = trt.Runtime(self.logger)
        self.engine = self._load_or_build_engine(self.force_rebuild)
        self.context = self.engine.create_execution_context()
        # TensorRT 10 removes the legacy binding API (num_bindings, binding_is_input, etc.)
        # and replaces it with the tensor API (num_io_tensors, get_tensor_*). Detect which
        # variant is available so we can work on both TRT <10 and TRT >=10.
        self.use_tensor_api = not hasattr(self.engine, "num_bindings")
        # Cache tensor / binding names indexed the same way we iterate over them.
        if self.use_tensor_api:
            self.binding_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        else:
            self.binding_names = [self.engine.get_binding_name(i) for i in range(self.engine.num_bindings)]

        self.stream = cudart.cudaStreamCreate()[1]

        self.input_binding_idxs: List[int] = []
        self.output_binding_idxs: List[int] = []
        self._prepare_binding_lists()
        self._set_input_binding_shapes()
        self._allocate_buffers()

    # ------------------------------------------------------------------
    # Engine creation / caching
    # ------------------------------------------------------------------
    def _load_or_build_engine(self, force_rebuild: bool):
        if Path(self.plan_path).exists() and not force_rebuild:
            with open(self.plan_path, "rb") as f:
                serialized = f.read()
            return self.runtime.deserialize_cuda_engine(serialized)

        if not Path(self.onnx_path).exists():
            raise FileNotFoundError(f"ONNX model not found: {self.onnx_path}")

        builder = trt.Builder(self.logger)
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)
        parser = trt.OnnxParser(network, self.logger)

        onnx_bytes = Path(self.onnx_path).read_bytes()
        if not parser.parse(onnx_bytes):
            err_msgs = [parser.get_error(i) for i in range(parser.num_errors)]
            patched = self._patch_for_tensorrt(self.onnx_path)
            if patched is None:
                raise RuntimeError(f"Failed to parse ONNX for TensorRT: {err_msgs}")

            network = builder.create_network(network_flags)
            parser = trt.OnnxParser(network, self.logger)
            if parser.parse(patched):
                Path(self.onnx_path + ".patched").write_bytes(patched)
            else:
                patched_errors = [parser.get_error(i) for i in range(parser.num_errors)]
                raise RuntimeError(
                    f"Failed to parse ONNX for TensorRT: {err_msgs}; "
                    f"failed to parse patched ONNX: {patched_errors}"
                )

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, self.workspace_size)
        if self.use_fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        # TensorRT 8.6+ returns a serialized network directly; fall back if older
        if hasattr(builder, "build_serialized_network"):
            plan = builder.build_serialized_network(network, config)
            if plan is None:
                raise RuntimeError("TensorRT failed to build serialized network")
            Path(self.plan_path).write_bytes(plan)
            return self.runtime.deserialize_cuda_engine(plan)

        # Legacy API
        engine = builder.build_engine(network, config)
        if engine is None:
            raise RuntimeError("TensorRT failed to build engine")
        Path(self.plan_path).write_bytes(engine.serialize())
        return engine

    def _patch_reduce_axes(self, onnx_path: str):
        """Convert ReduceMean nodes with axes input -> axes attribute when axes is constant."""
        try:
            model = onnx.load(onnx_path)
        except Exception:
            return None

        initializers = {init.name: init for init in model.graph.initializer}
        name_to_const = {}
        for node in model.graph.node:
            if node.op_type == "Constant" and "value" in {a.name for a in node.attribute}:
                for attr in node.attribute:
                    if attr.name == "value":
                        name_to_const[node.output[0]] = attr.t

        patched = False
        for node in model.graph.node:
            if node.op_type.startswith("Reduce") and len(node.input) >= 2:
                axes_name = node.input[1]
                tensor = None
                if axes_name in initializers:
                    tensor = initializers[axes_name]
                elif axes_name in name_to_const:
                    tensor = name_to_const[axes_name]
                if tensor is None:
                    continue
                axes_np = onnx.numpy_helper.to_array(tensor).tolist()
                # Remove axes input and add attribute
                node.input.pop(1)
                node.attribute.add(name="axes", ints=axes_np)
                patched = True

        if not patched:
            return None
        return model.SerializeToString()

    def _patch_for_tensorrt(self, onnx_path: str):
        """Patch common training-export ops that TensorRT 8.5 cannot import directly."""
        if onnx is None:
            return None

        try:
            model = onnx.load(onnx_path)
        except Exception:
            return None

        helper = onnx.helper
        numpy_helper = onnx.numpy_helper
        tensor_proto = onnx.TensorProto
        initializers = {init.name: init for init in model.graph.initializer}
        name_to_const = {}
        for node in model.graph.node:
            if node.op_type == "Constant":
                for attr in node.attribute:
                    if attr.name == "value":
                        name_to_const[node.output[0]] = attr.t

        def unique(base: str) -> str:
            existing = {
                value
                for node in model.graph.node
                for value in list(node.input) + list(node.output)
            }
            existing.update(init.name for init in model.graph.initializer)
            candidate = base
            suffix = 0
            while candidate in existing:
                suffix += 1
                candidate = f"{base}_{suffix}"
            return candidate

        patched = False
        new_nodes = []
        for node in model.graph.node:
            if node.op_type.startswith("Reduce") and len(node.input) >= 2:
                axes_name = node.input[1]
                tensor = initializers.get(axes_name) or name_to_const.get(axes_name)
                if tensor is not None:
                    axes = numpy_helper.to_array(tensor).astype(np.int64).tolist()
                    del node.input[1]
                    del node.attribute[:]
                    node.attribute.add(name="axes", ints=axes)
                    patched = True
                new_nodes.append(node)
                continue

            if node.op_type == "LayerNormalization":
                if len(node.input) < 2:
                    new_nodes.append(node)
                    continue
                axis = -1
                epsilon = 1e-5
                for attr in node.attribute:
                    if attr.name == "axis":
                        axis = int(attr.i)
                    elif attr.name == "epsilon":
                        epsilon = float(attr.f)
                if axis != -1:
                    new_nodes.append(node)
                    continue

                x = node.input[0]
                scale = node.input[1]
                bias = node.input[2] if len(node.input) >= 3 and node.input[2] else None
                out = node.output[0]
                prefix = unique(f"{node.name or out}_ln")
                mean = f"{prefix}_mean"
                centered = f"{prefix}_centered"
                squared = f"{prefix}_squared"
                var = f"{prefix}_var"
                var_eps = f"{prefix}_var_eps"
                std = f"{prefix}_std"
                normalized = f"{prefix}_normalized"
                scaled = f"{prefix}_scaled"
                eps_name = f"{prefix}_eps"
                pow_name = f"{prefix}_pow"
                model.graph.initializer.extend(
                    [
                        numpy_helper.from_array(np.asarray(epsilon, dtype=np.float32), eps_name),
                        numpy_helper.from_array(np.asarray(2.0, dtype=np.float32), pow_name),
                    ]
                )
                new_nodes.extend(
                    [
                        helper.make_node("ReduceMean", [x], [mean], name=f"{prefix}_mean_node", axes=[-1], keepdims=1),
                        helper.make_node("Sub", [x, mean], [centered], name=f"{prefix}_center_node"),
                        helper.make_node("Pow", [centered, pow_name], [squared], name=f"{prefix}_pow_node"),
                        helper.make_node("ReduceMean", [squared], [var], name=f"{prefix}_var_node", axes=[-1], keepdims=1),
                        helper.make_node("Add", [var, eps_name], [var_eps], name=f"{prefix}_eps_node"),
                        helper.make_node("Sqrt", [var_eps], [std], name=f"{prefix}_sqrt_node"),
                        helper.make_node("Div", [centered, std], [normalized], name=f"{prefix}_div_node"),
                        helper.make_node("Mul", [normalized, scale], [scaled if bias else out], name=f"{prefix}_scale_node"),
                    ]
                )
                if bias:
                    new_nodes.append(helper.make_node("Add", [scaled, bias], [out], name=f"{prefix}_bias_node"))
                patched = True
                continue

            if node.op_type == "Mish":
                x = node.input[0]
                out = node.output[0]
                prefix = unique(f"{node.name or out}_mish")
                one = f"{prefix}_one"
                exp = f"{prefix}_exp"
                exp_plus_one = f"{prefix}_exp_plus_one"
                softplus = f"{prefix}_softplus"
                tanh = f"{prefix}_tanh"
                model.graph.initializer.append(
                    numpy_helper.from_array(np.asarray(1.0, dtype=np.float32), one)
                )
                new_nodes.extend(
                    [
                        helper.make_node("Exp", [x], [exp], name=f"{prefix}_exp_node"),
                        helper.make_node("Add", [exp, one], [exp_plus_one], name=f"{prefix}_add_node"),
                        helper.make_node("Log", [exp_plus_one], [softplus], name=f"{prefix}_log_node"),
                        helper.make_node("Tanh", [softplus], [tanh], name=f"{prefix}_tanh_node"),
                        helper.make_node("Mul", [x, tanh], [out], name=f"{prefix}_mul_node"),
                    ]
                )
                patched = True
                continue

            new_nodes.append(node)

        if not patched:
            return None

        del model.graph.node[:]
        model.graph.node.extend(new_nodes)
        if model.ir_version > 9:
            model.ir_version = 9
        for opset in model.opset_import:
            if opset.domain == "" and opset.version > 19:
                opset.version = 19
        return model.SerializeToString()

    # ------------------------------------------------------------------
    # Binding helpers
    # ------------------------------------------------------------------
    def _prepare_binding_lists(self):
        if self.use_tensor_api:
            for idx, name in enumerate(self.binding_names):
                mode = self.engine.get_tensor_mode(name)
                if mode == trt.TensorIOMode.INPUT:
                    self.input_binding_idxs.append(idx)
                else:
                    self.output_binding_idxs.append(idx)
        else:
            for idx in range(self.engine.num_bindings):
                if self.engine.binding_is_input(idx):
                    self.input_binding_idxs.append(idx)
                else:
                    self.output_binding_idxs.append(idx)
        if len(self.input_binding_idxs) != len(self.in_keys):
            raise ValueError(
                f"Input count mismatch: engine has {len(self.input_binding_idxs)} bindings "
                f"but meta lists {len(self.in_keys)} keys."
            )
        if len(self.output_binding_idxs) != len(self.out_keys):
            raise ValueError(
                f"Output count mismatch: engine has {len(self.output_binding_idxs)} bindings "
                f"but meta lists {len(self.out_keys)} keys."
            )

    def _set_input_binding_shapes(self):
        for key, idx in zip(self.in_keys, self.input_binding_idxs):
            if idx >= len(self.input_shapes):
                continue
            shape = tuple(self.input_shapes[idx])
            if self.use_tensor_api:
                self.context.set_input_shape(self.binding_names[idx], shape)
            else:
                self.context.set_binding_shape(idx, shape)

    # ------------------------------------------------------------------
    # Buffer allocation
    # ------------------------------------------------------------------
    def _allocate_buffers(self):
        if self.use_tensor_api:
            num_entries = self.engine.num_io_tensors
        else:
            num_entries = self.engine.num_bindings
        self.bindings: List[int] = [0] * num_entries
        self.host_inputs: Dict[int, np.ndarray] = {}
        self.host_outputs: Dict[int, np.ndarray] = {}
        self.device_allocations: Dict[int, int] = {}

        for idx in range(num_entries):
            if self.use_tensor_api:
                name = self.binding_names[idx]
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                shape = tuple(self.context.get_tensor_shape(name))
            else:
                dtype = trt.nptype(self.engine.get_binding_dtype(idx))
                shape = tuple(self.context.get_binding_shape(idx))
            if any(dim == -1 for dim in shape):
                raise ValueError(
                    f"Binding {self.binding_names[idx]} has dynamic shape {shape}. "
                    "Provide concrete shapes in meta['in_shapes']."
                )

            host_mem = np.empty(shape, dtype=dtype)
            status, device_ptr = cudart.cudaMalloc(host_mem.nbytes)
            if status != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(f"cudaMalloc failed for binding {idx}")

            self.bindings[idx] = device_ptr
            self.device_allocations[idx] = device_ptr
            if self.use_tensor_api:
                self.context.set_tensor_address(self.binding_names[idx], device_ptr)
                if idx in self.input_binding_idxs:
                    self.host_inputs[idx] = host_mem
                else:
                    self.host_outputs[idx] = host_mem
            else:
                if self.engine.binding_is_input(idx):
                    self.host_inputs[idx] = host_mem
                else:
                    self.host_outputs[idx] = host_mem

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def __call__(self, input_dict: Dict[Key, np.ndarray]) -> Dict[Key, np.ndarray]:
        # Host-to-device copies
        for key, idx in zip(self.in_keys, self.input_binding_idxs):
            host_buf = self.host_inputs[idx]
            arr = _prepare_input_value(input_dict, key, host_buf.shape)
            np.copyto(host_buf, arr)
            cudart.cudaMemcpyAsync(
                self.device_allocations[idx],
                host_buf.ctypes.data,
                host_buf.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                self.stream,
            )

        stream_handle = _cuda_stream_handle(self.stream)
        if self.use_tensor_api:
            self.context.execute_async_v3(stream_handle)
        elif hasattr(self.context, "execute_async_v2"):
            self.context.execute_async_v2(self.bindings, stream_handle)
        else:
            self.context.execute_async_v3(stream_handle, self.bindings)

        # Device-to-host copies
        for idx in self.output_binding_idxs:
            host_buf = self.host_outputs[idx]
            cudart.cudaMemcpyAsync(
                host_buf.ctypes.data,
                self.device_allocations[idx],
                host_buf.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                self.stream,
            )

        cudart.cudaStreamSynchronize(self.stream)

        outputs: Dict[Key, np.ndarray] = {}
        for key, idx in zip(self.out_keys, self.output_binding_idxs):
            outputs[key] = _squeeze_output_value(self.host_outputs[idx].copy())
        return outputs


class Timer:
    """Small helper that mirrors ONNXModule.Timer for perf aggregation."""

    def __init__(self, perf_dict: Dict[str, float], name: str):
        self.perf_dict = perf_dict
        self.name = name

    def __enter__(self):
        self.start_time = cudart.cudaEventCreate()[1]
        self.end_time = cudart.cudaEventCreate()[1]
        cudart.cudaEventRecord(self.start_time, 0)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        cudart.cudaEventRecord(self.end_time, 0)
        cudart.cudaEventSynchronize(self.end_time)
        elapsed_ms = cudart.cudaEventElapsedTime(self.start_time, self.end_time)[1]
        self.perf_dict[self.name] = self.perf_dict.get(self.name, 0.0) + elapsed_ms / 1000.0
