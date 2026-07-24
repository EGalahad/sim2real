# JetPack 5 ONNX GPU and TensorRT Install and Convert Notes

## Host Assumptions

- Host is an onboard Orin running JetPack 5 / L4T R35 / Ubuntu 20.04 / aarch64.
- System Python is 3.8.
- CUDA is 11.4.
- TensorRT is installed by JetPack, commonly at
  `/usr/lib/python3.8/dist-packages/tensorrt`, with version `8.5.2.2`.
- The default sim2real `uv` environment may be Python 3.10 and should not be
  used for these Jetson Python 3.8 artifacts.
- There is no supported JetPack 5 + Python 3.10 prebuilt ONNX Runtime GPU or
  TensorRT route. Use Python 3.8 on JetPack 5, or JetPack 6 for Python 3.10.

Check:

```bash
cat /etc/nv_tegra_release
python3 --version
python3 -c "import tensorrt as trt; print(trt.__version__)"
```

## Shared Environment

Use the Python 3.8 compatibility branches for source code as well as wheels:

```bash
cd ~/sim2real
git fetch origin py38
git switch py38

cd ~/any4hdmi
git fetch origin py38
git switch py38
```

If the robot tree is maintained by rsync rather than git, sync from the local
`py38` branches for processes that run inside `.venv-jp5-ort`.

Use a dedicated Python 3.8 venv that can import JetPack system packages:

```bash
cd ~/sim2real
python3 -m venv --system-site-packages .venv-jp5-ort
bash -lic 'proxy_on; .venv-jp5-ort/bin/python -m pip install -U pip'
bash -lic 'proxy_on; .venv-jp5-ort/bin/python -m pip install onnx==1.17.0 loguru pyyaml tyro cuda-python==12.3.0'
bash -lic 'proxy_on; .venv-jp5-ort/bin/python -m pip install --no-deps "mjhub @ git+https://github.com/EGalahad/mjhub.git@py38"'
bash -lic 'proxy_on; .venv-jp5-ort/bin/python -m pip install mujoco==3.2.3 pyzmq==25.1.2 tqdm==4.66.6 sshkeyboard==2.3.1 huggingface-hub==0.36.2 torch==2.4.1 tensordict==0.1.2'
```

Do not run `uv sync` into this venv. It can install incompatible Python,
PyTorch, ONNX Runtime, or NumPy versions for JetPack 5.

Install `viser==0.2.23` only if the Python 3.8 process creates a Viser server.
The current `mjviser` release is not Python 3.8-compatible because of Python
3.10 typing syntax, so keep viewer-heavy base-sim processes on the normal env
or use a dedicated Python 3.8-compatible `mjviser` branch.

## ONNX GPU Backend

Install the NVIDIA Jetson ONNX Runtime GPU wheel:

```bash
cd ~/sim2real
mkdir -p /tmp/ort-gpu
bash -lic 'proxy_on; wget -O /tmp/ort-gpu/onnxruntime_gpu-1.16.0-cp38-cp38-linux_aarch64.whl https://nvidia.box.com/shared/static/iizg3ggrtdkqawkmebbfixo7sce6j365.whl'
.venv-jp5-ort/bin/python -m pip install /tmp/ort-gpu/onnxruntime_gpu-1.16.0-cp38-cp38-linux_aarch64.whl
```

Verify:

```bash
cd ~/sim2real
.venv-jp5-ort/bin/python -c "import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())"
```

Expected providers:

```text
['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

If CUDA provider is missing, the wrong ORT package is installed.

## TensorRT Backend

TensorRT comes from JetPack system packages. The venv needs
`--system-site-packages` to import it. Install NumPy 1.23 for TensorRT 8.5
Python compatibility:

```bash
cd ~/sim2real
.venv-jp5-ort/bin/python -m pip install numpy==1.23.5
```

Verify:

```bash
cd ~/sim2real
.venv-jp5-ort/bin/python -c "import tensorrt as trt; print(trt.__version__)"
```

Expected:

```text
8.5.2.2
```

## Convert ONNX for JetPack 5

Use the bundled conversion helper from this skill:

```bash
cd ~/sim2real
.venv-jp5-ort/bin/python /path/to/skill/scripts/prepare_jetpack5_onnx.py \
  --input checkpoints/.../policy.onnx \
  --output /tmp/ort-test/policy-op19.onnx \
  --mode ort-gpu \
  --copy-sidecars
```

Use `--mode ort-gpu` when the target backend is ONNX Runtime CUDA. It lowers
the model to IR 9 / opset 19, which ORT 1.16 accepts.

Use `--mode tensorrt` for direct TensorRT:

```bash
cd ~/sim2real
.venv-jp5-ort/bin/python /path/to/skill/scripts/prepare_jetpack5_onnx.py \
  --input checkpoints/.../policy.onnx \
  --output /tmp/ort-test/policy-trt.onnx \
  --mode tensorrt \
  --copy-sidecars
```

TensorRT mode lowers the model to IR 9 / opset 13 so Reduce ops can use the
attribute-style `axes` format expected by TensorRT 8.5. It also expands:

- `LayerNormalization` into `ReduceMean`, `Sub`, `Pow`, `Add`, `Sqrt`, `Div`,
  `Mul`, and optional bias `Add`.
- `Mish` into `Exp`, `Add`, `Log`, `Tanh`, `Mul`.
- Reduce op axes inputs into TensorRT 8.5-compatible `axes` attributes when the
  axes are constant.

After conversion, edit or copy the deploy YAML so `policy_path` points to the
converted ONNX. Keep the original policy metadata sidecars aligned with the
converted model.

## Benchmark Commands

ONNX GPU:

```bash
cd ~/sim2real
PYTHONPATH=$PWD:~/any4hdmi/src .venv-jp5-ort/bin/python scripts/test_policy_inference.py \
  --policy_config /tmp/ort-test/policy-op19.yaml \
  --inference_backend onnx-gpu \
  --warmup 50 --runs 1000
```

Tracking with ONNX GPU:

```bash
cd ~/sim2real
PYTHONPATH=$PWD:~/any4hdmi/src .venv-jp5-ort/bin/python sim2real/rl_policy/tracking.py \
  --robot g1 \
  --policy_config /tmp/ort-test/policy-op19.yaml \
  --motion_path ../any4hdmi/output/root_tracking_test/motions/forward_1.npz \
  --inference_backend onnx-gpu
```

TensorRT:

```bash
cd ~/sim2real
PYTHONPATH=$PWD:~/any4hdmi/src HDMI_TRT_FP16=1 HDMI_TRT_FORCE_REBUILD=0 .venv-jp5-ort/bin/python scripts/test_policy_inference.py \
  --policy_config /tmp/ort-test/policy-trt.yaml \
  --inference_backend tensorrt \
  --warmup 50 --runs 1000
```

Use `HDMI_TRT_FORCE_REBUILD=1` to rebuild the `.plan` file.

Tracking with TensorRT:

```bash
cd ~/sim2real
PYTHONPATH=$PWD:~/any4hdmi/src HDMI_TRT_FP16=1 HDMI_TRT_FORCE_REBUILD=0 .venv-jp5-ort/bin/python sim2real/rl_policy/tracking.py \
  --robot g1 \
  --policy_config /tmp/ort-test/policy-trt.yaml \
  --motion_path ../any4hdmi/output/root_tracking_test/motions/forward_1.npz \
  --inference_backend tensorrt
```

## Known Failures and Fixes

- `CUDAExecutionProvider is not available`: installed CPU-only ONNX Runtime or
  used the wrong Python env.
- `Unsupported model IR version: 10, max supported IR version: 9`: convert the
  ONNX with `--mode ort-gpu` or `--mode tensorrt`.
- Opset 20 warning or failure: convert to opset 19 for JetPack 5 ORT 1.16.
- TensorRT plugin error on `LayerNormalization` or `Mish`: convert with
  `--mode tensorrt`.
- `module 'numpy' has no attribute 'bool'`: use `numpy==1.23.5` with TensorRT
  8.5 Python binding.
- `execute_async_v3()` incompatible arguments with bindings list: direct
  TensorRT legacy binding execution must use `execute_async_v2(bindings,
  stream)` on JetPack 5 / TensorRT 8.5.
- PyPI cannot find `onnxruntime-gpu`: PyPI does not have the required JetPack 5
  aarch64 wheel. Use the NVIDIA wheel URL above.
- Python 3.10 requested on JetPack 5: not supported by NVIDIA prebuilt ORT GPU
  or TensorRT bindings. Use JetPack 5 Python 3.8 or upgrade to JetPack 6.
- Full policy tracking fails after inference benchmark succeeds: install the
  runtime packages listed above and make sure `sim2real`, `any4hdmi`, and
  `mjhub` are on their `py38` branches.
- `mjhub` fails on Python 3.8 typing syntax: install the `py38` branch.
