---
name: onboard-jetpack5-inference-backends
description: Install, convert, debug, and benchmark sim2real ONNX GPU and TensorRT inference backends on onboard JetPack 5 Orin hosts such as g1-cable. Use when a task mentions JetPack 5, L4T R35, onboard Orin, onnx-gpu, CUDAExecutionProvider, TensorRT 8.5, policy ONNX conversion, or sim2real inference benchmark failures on the robot computer.
---

# Onboard JetPack 5 Inference Backends

Use this skill for sim2real policy inference work on JetPack 5 / L4T R35
onboard Orin machines.

## Workflow

1. Confirm the host is JetPack 5:
   - `cat /etc/nv_tegra_release`
   - `python3 --version`
   - `python3 -c "import tensorrt as trt; print(trt.__version__)"`
2. Keep the default project `uv` env separate. JetPack 5 GPU inference uses a
   dedicated Python 3.8 venv, normally `~/sim2real/.venv-jp5-ort`.
3. Use the Python 3.8 compatibility source branches for runtime code:
   `sim2real@py38`, `any4hdmi@py38`, and
   `mjhub @ git+https://github.com/EGalahad/mjhub.git@py38`. A backend-only
   inference benchmark may pass before full policy tracking works if these
   source branches or runtime packages are missing.
4. Do not spend time trying to force the JetPack 5 NVIDIA inference stack into
   Python 3.10 unless the user explicitly accepts unsupported source builds.
   JetPack 5 prebuilt ORT GPU and TensorRT Python bindings are Python 3.8
   artifacts; Python 3.10 is the supported route on JetPack 6.
5. For install details, read `references/install-and-convert.md`.
6. For ONNX conversion, run `scripts/prepare_jetpack5_onnx.py` from this skill.
7. Benchmark with `scripts/test_policy_inference.py` on the onboard host.
8. If deployment scripts sync to the robot, protect `.venv*/`, `.plan`, and
   converted ONNX artifacts from `rsync --delete`.

## Backend Choice

- Use `onnx-gpu` when a lower-risk GPU path is enough. It requires the NVIDIA
  Jetson ONNX Runtime GPU wheel and a model lowered to IR 9 / opset 19.
- Use direct `tensorrt` when lower median latency matters. It requires system
  TensorRT 8.5, `cuda-python`, `numpy==1.23.5`, and a TensorRT-friendly ONNX
  lowered to IR 9 / opset 13 where `LayerNormalization` and `Mish` are expanded.
- Do not install the Jetson `cp38` ORT wheel into a Python 3.10 `uv` env.
- Use `mjhub @ git+https://github.com/EGalahad/mjhub.git@py38` in the JetPack
  5 venv when sim2real imports robot configs.

## Conversion Commands

Prepare an ONNX Runtime CUDA model:

```bash
cd ~/sim2real
.venv-jp5-ort/bin/python /path/to/skill/scripts/prepare_jetpack5_onnx.py \
  --input checkpoints/.../policy.onnx \
  --output /tmp/ort-test/policy-op19.onnx \
  --mode ort-gpu \
  --copy-sidecars
```

Prepare a direct TensorRT model:

```bash
cd ~/sim2real
.venv-jp5-ort/bin/python /path/to/skill/scripts/prepare_jetpack5_onnx.py \
  --input checkpoints/.../policy.onnx \
  --output /tmp/ort-test/policy-trt.onnx \
  --mode tensorrt \
  --copy-sidecars
```

After copying sidecars, update the converted YAML if its policy path still
points to the original ONNX.

## Benchmark Commands

ONNX GPU:

```bash
cd ~/sim2real
PYTHONPATH=$PWD:~/any4hdmi/src .venv-jp5-ort/bin/python scripts/test_policy_inference.py \
  --policy_config /tmp/ort-test/policy-op19.yaml \
  --inference_backend onnx-gpu \
  --warmup 50 --runs 1000
```

TensorRT:

```bash
cd ~/sim2real
PYTHONPATH=$PWD:~/any4hdmi/src HDMI_TRT_FP16=1 HDMI_TRT_FORCE_REBUILD=0 .venv-jp5-ort/bin/python scripts/test_policy_inference.py \
  --policy_config /tmp/ort-test/policy-trt.yaml \
  --inference_backend tensorrt \
  --warmup 50 --runs 1000
```

Use `HDMI_TRT_FORCE_REBUILD=1` after changing the ONNX.

## Common Failure Mapping

- `CUDAExecutionProvider is not available`: wrong environment or CPU-only ORT.
- `Unsupported model IR version: 10`: convert to IR 9.
- Opset 20 warning/failure: lower default-domain opset to 19.
- TensorRT plugin error around `LayerNormalization` or `Mish`: convert with
  `--mode tensorrt`.
- `module 'numpy' has no attribute 'bool'`: install `numpy==1.23.5`.
- `execute_async_v3()` incompatible arguments with bindings list: TensorRT 8.5
  legacy binding runtime must call `execute_async_v2(bindings, stream)`.
- User asks whether Python 3.10 can be used on JetPack 5: answer no for the
  supported prebuilt route. Recommend Python 3.8 on JetPack 5 or JetPack 6 for
  Python 3.10.
