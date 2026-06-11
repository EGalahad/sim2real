---
title: Onboard JetPack 5 Inference Backends
slug: /reference/onboard-jetpack5-inference-backends
---

# Onboard JetPack 5 Inference Backends

This note records the working `onnx-gpu` and `tensorrt` setup for a JetPack 5
onboard Orin such as `g1-cable`.

## Platform

Use this path only for JetPack 5 / L4T R35 / Ubuntu 20.04 / aarch64. On the
tested host:

- L4T: `R35.3.1`
- CUDA: `11.4`
- TensorRT: `8.5.2.2`
- System Python: `3.8`

Do not install these Jetson wheels into the default project `uv` environment if
that environment uses Python 3.10. The Jetson ORT wheel and system TensorRT
binding used here are Python 3.8 artifacts. There is no supported JetPack 5 /
Python 3.10 prebuilt route for ONNX Runtime GPU or TensorRT; use Python 3.8 on
JetPack 5, or upgrade to JetPack 6 for a supported Python 3.10 stack.

## Shared Environment

### Code branches

Run the onboard Python 3.8 stack from the Python 3.8 compatibility branches.
The backend wheels are not the only Python 3.8 constraint: full policy tracking
also imports `sim2real`, `any4hdmi`, and `mjhub`, so the source trees must avoid
Python 3.10-only typing syntax.

For git-managed checkouts:

```bash
cd ~/sim2real
git fetch origin py38
git switch py38

cd ~/any4hdmi
git fetch origin py38
git switch py38
```

For rsync-managed robot copies, sync from the local `py38` branches rather than
from the normal development branch when the target process runs inside
`.venv-jp5-ort`.

Create a Python 3.8 venv that can see JetPack system packages:

```bash
cd ~/sim2real
python3 -m venv --system-site-packages .venv-jp5-ort
bash -lic 'proxy_on; .venv-jp5-ort/bin/python -m pip install -U pip'
```

Install common runtime packages:

```bash
cd ~/sim2real
bash -lic 'proxy_on; .venv-jp5-ort/bin/python -m pip install onnx==1.17.0 loguru pyyaml tyro cuda-python==12.3.0'
```

Install the Python 3.8-compatible `mjhub` branch when running full sim2real
tracking scripts:

```bash
bash -lic 'proxy_on; .venv-jp5-ort/bin/python -m pip install --no-deps "mjhub @ git+https://github.com/EGalahad/mjhub.git@py38"'
```

Policy tracking also needs the normal runtime imports that the minimal backend
benchmark may not touch:

```bash
cd ~/sim2real
bash -lic 'proxy_on; .venv-jp5-ort/bin/python -m pip install \
  mujoco==3.2.3 \
  pyzmq==25.1.2 \
  tqdm==4.66.6 \
  sshkeyboard==2.3.1 \
  huggingface-hub==0.36.2 \
  torch==2.4.1 \
  tensordict==0.1.2'
```

Install `viser==0.2.23` only when the Python 3.8 process needs to create a
Viser server. The current `mjviser` release uses Python 3.10 typing syntax, so
run viewer-heavy base-sim processes from the normal environment or use a
Python 3.8-compatible `mjviser` branch before moving them into
`.venv-jp5-ort`.

## ONNX GPU Backend

Install NVIDIA's Jetson ONNX Runtime GPU wheel:

```bash
mkdir -p /tmp/ort-gpu
bash -lic 'proxy_on; wget -O /tmp/ort-gpu/onnxruntime_gpu-1.16.0-cp38-cp38-linux_aarch64.whl https://nvidia.box.com/shared/static/iizg3ggrtdkqawkmebbfixo7sce6j365.whl'
.venv-jp5-ort/bin/python -m pip install /tmp/ort-gpu/onnxruntime_gpu-1.16.0-cp38-cp38-linux_aarch64.whl
```

Verify providers:

```bash
.venv-jp5-ort/bin/python -c "import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())"
```

Expected providers include:

```text
['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

JetPack 5 ORT 1.16 does not accept newer exports such as IR 10 / opset 20. Use
an IR 9 / opset 19 copy for `onnx-gpu`.

Example benchmark:

```bash
cd ~/sim2real
PYTHONPATH=$PWD:~/any4hdmi/src .venv-jp5-ort/bin/python scripts/test_policy_inference.py \
  --policy_config /tmp/ort-test/policy-op19.yaml \
  --inference_backend onnx-gpu \
  --warmup 50 --runs 1000
```

Example tracking command:

```bash
cd ~/sim2real
PYTHONPATH=$PWD:~/any4hdmi/src .venv-jp5-ort/bin/python sim2real/rl_policy/tracking.py \
  --robot g1 \
  --policy_config /tmp/ort-test/policy-sonic-release-op19.yaml \
  --motion_path ../any4hdmi/output/root_tracking_test/motions/forward_1.npz \
  --inference_backend onnx-gpu
```

## TensorRT Backend

TensorRT itself comes from JetPack system packages, for example
`/usr/lib/python3.8/dist-packages/tensorrt`. The venv sees it because it was
created with `--system-site-packages`.

TensorRT 8.5's Python binding still references `np.bool`, so use NumPy 1.23:

```bash
cd ~/sim2real
.venv-jp5-ort/bin/python -m pip install numpy==1.23.5
```

Verify TensorRT:

```bash
.venv-jp5-ort/bin/python -c "import tensorrt as trt; print(trt.__version__)"
```

Expected output:

```text
8.5.2.2
```

TensorRT 8.5 may reject exported ONNX graphs containing `LayerNormalization`
and `Mish`. Use a TensorRT-friendly copy that lowers to IR 9 / opset 13 and
expands those ops into primitive ONNX ops before building the engine.

Example benchmark:

```bash
cd ~/sim2real
PYTHONPATH=$PWD:~/any4hdmi/src HDMI_TRT_FP16=1 HDMI_TRT_FORCE_REBUILD=0 .venv-jp5-ort/bin/python scripts/test_policy_inference.py \
  --policy_config /tmp/ort-test/policy-trt.yaml \
  --inference_backend tensorrt \
  --warmup 50 --runs 1000
```

Set `HDMI_TRT_FORCE_REBUILD=1` when changing the ONNX or when the cached
`.plan` should be regenerated.

Example tracking command:

```bash
cd ~/sim2real
PYTHONPATH=$PWD:~/any4hdmi/src HDMI_TRT_FP16=1 HDMI_TRT_FORCE_REBUILD=0 .venv-jp5-ort/bin/python sim2real/rl_policy/tracking.py \
  --robot g1 \
  --policy_config /tmp/ort-test/policy-sonic-release-trt.yaml \
  --motion_path ../any4hdmi/output/root_tracking_test/motions/forward_1.npz \
  --inference_backend tensorrt
```

## Known Issues

- `pip install onnxruntime-gpu` from PyPI does not provide a JetPack 5 aarch64
  wheel. Use the NVIDIA Jetson wheel above.
- `proxy_on` is most reliable when run through `bash -lic`.
- Python 3.10 project venvs cannot load the `cp38` Jetson ORT wheel or Python
  3.8 TensorRT binding.
- The JetPack 5 package indexes do not provide a supported `cp310` ONNX Runtime
  GPU wheel, and JetPack 5 TensorRT Python bindings are built for system Python
  3.8. A Python 3.10 route means source builds or JetPack 6.
- ORT 1.16 rejects IR 10 models with `Unsupported model IR version: 10, max
  supported IR version: 9`.
- ORT 1.16 warns that opset 20 is newer than its guaranteed support. Use opset
  19 for JetPack 5.
- TensorRT 8.5 parser can reject `LayerNormalization` and `Mish` as missing
  plugins. Expand them into primitive ops.
- TensorRT 8.5 legacy binding execution should use
  `execute_async_v2(bindings, stream)`. Passing bindings to `execute_async_v3`
  causes an incompatible-arguments error.
- Full tracking on JetPack 5 Python 3.8 may expose Python 3.10-only annotations
  in project code or dependencies. Keep Python 3.8 compatibility patches on
  dedicated `py38` branches and merge normal development changes into `py38`
  when they are needed onboard.
- A minimal inference benchmark can pass before full policy tracking works.
  Missing runtime packages usually surface later as imports such as `mujoco`,
  `zmq`, `sshkeyboard`, `huggingface_hub`, `torch`, or `tensordict`.
