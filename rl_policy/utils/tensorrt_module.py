import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np

# TensorRT relies on the CUDA runtime. The cuda-python package exposes the
# low-level cudart API that the official TensorRT quick-start guide uses.
# https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/quick-start-guide.html
try:
    import tensorrt as trt
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("TensorRTModule requires the `tensorrt` package.") from exc

# cuda-python wheels on some Python versions expose cudart via cuda.cudart; on
# others only via cuda.bindings.runtime. Try both to be resilient.
try:
    from cuda import cudart  # type: ignore
except ImportError:  # pragma: no cover
    try:
        from cuda.bindings import runtime as cudart  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "TensorRTModule requires `cuda-python` (cuda.cudart or cuda.bindings.runtime)."
        ) from exc

# Optional ONNX dependency for lightweight graph fixes.
try:  # pragma: no cover - optional
    import onnx
except Exception:  # pragma: no cover
    onnx = None  # type: ignore

Key = Union[str, Tuple[str, str]]


def _normalize_keys(raw_keys: Sequence[Union[str, Sequence[str]]]) -> List[Key]:
    normalized: List[Key] = []
    for key in raw_keys:
        if isinstance(key, str):
            normalized.append(key)
        else:
            normalized.append(tuple(key))
    return normalized


class TensorRTModule:
    """Lightweight TensorRT wrapper that mirrors the ONNXModule interface."""

    def __init__(
        self,
        onnx_path: str,
        *,
        workspace_size: int = 1 << 30,
        use_fp16: bool = False,
        force_rebuild: bool = False,
    ):
        self.onnx_path = str(onnx_path)
        self.plan_path = self.onnx_path.replace(".onnx", ".plan")
        self.workspace_size = workspace_size
        self.use_fp16 = use_fp16
        self.logger = trt.Logger(trt.Logger.WARNING)

        meta_path = self.onnx_path.replace(".onnx", ".json")
        with open(meta_path, "r") as f:
            meta = json.load(f)
        self.in_keys: List[Key] = _normalize_keys(meta.get("in_keys", []))
        self.out_keys: List[Key] = _normalize_keys(meta.get("out_keys", []))
        # meta["in_shapes"] is stored as a list of shape-sets; we take the first set
        # and replace dynamic dims (-1 / str) with 1 for explicit batch.
        raw_shapes = meta.get("in_shapes", [])
        shape_set = raw_shapes[0] if raw_shapes else []
        self.input_shapes: List[Tuple[int, ...]] = []
        for shape in shape_set:
            concrete = [1 if isinstance(dim, str) or dim == -1 else int(dim) for dim in shape]
            self.input_shapes.append(tuple(concrete))

        self.runtime = trt.Runtime(self.logger)
        self.engine = self._load_or_build_engine(force_rebuild)
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
            # Try a small ONNX patch to convert ReduceMean axes inputs to attributes
            # (TensorRT requires axes to be initializers). Only run if onnx is available.
            if onnx is not None and any(
                "Axis input must be an initializer" in parser.get_error(i).desc()
                for i in range(parser.num_errors)
            ):
                patched = self._patch_reduce_axes(self.onnx_path)
                if patched is not None:
                    parser = trt.OnnxParser(network, self.logger)
                    if parser.parse(patched):
                        Path(self.onnx_path + ".patched").write_bytes(patched)
                    else:
                        err_msgs = [parser.get_error(i) for i in range(parser.num_errors)]
                        raise RuntimeError(f"Failed to parse patched ONNX for TensorRT: {err_msgs}")
                else:
                    err_msgs = [parser.get_error(i) for i in range(parser.num_errors)]
                    raise RuntimeError(
                        "Failed to parse ONNX for TensorRT and could not auto-patch ReduceMean axes. "
                        "Install `onnx` to enable auto-fix. "
                        f"Errors: {err_msgs}"
                    )
            else:
                err_msgs = [parser.get_error(i) for i in range(parser.num_errors)]
                raise RuntimeError(f"Failed to parse ONNX for TensorRT: {err_msgs}")

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
                # Convert tensor to list of ints
                import numpy as _np  # local import to avoid global dep

                axes_np = onnx.numpy_helper.to_array(tensor).tolist()
                # Remove axes input and add attribute
                node.input.pop(1)
                node.attribute.add(name="axes", ints=axes_np)
                patched = True

        if not patched:
            return None
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
            if key not in input_dict:
                raise KeyError(f"Missing input key: {key}")
            arr = np.ascontiguousarray(input_dict[key])
            host_buf = self.host_inputs[idx]
            if host_buf.shape != arr.shape:
                raise ValueError(
                    f"Shape mismatch for {key}: expected {host_buf.shape}, got {arr.shape}"
                )
            np.copyto(host_buf, arr)
            cudart.cudaMemcpyAsync(
                self.device_allocations[idx],
                host_buf.ctypes.data,
                host_buf.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                self.stream,
            )

        if self.use_tensor_api:
            self.context.execute_async_v3(self.stream)
        else:
            self.context.execute_async_v3(self.stream, self.bindings)

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
            outputs[key] = self.host_outputs[idx].copy()
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
