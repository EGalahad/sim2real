---
title: Onboard JetPack 5 Inference Backends
slug: /reference/onboard-jetpack5-inference-backends
---

# Onboard JetPack 5 Inference Backends

这页记录 `g1-cable` 这类 JetPack 5 onboard Orin 上可用的 `onnx-gpu` 和
`tensorrt` 后端安装方法。

## 平台

这条路线只适用于 JetPack 5 / L4T R35 / Ubuntu 20.04 / aarch64。已验证机器：

- L4T: `R35.3.1`
- CUDA: `11.4`
- TensorRT: `8.5.2.2`
- 系统 Python: `3.8`

不要把这些 Jetson wheel 装进默认项目 `uv` 环境，尤其是 Python 3.10 的
`.venv`。这里用到的 Jetson ORT wheel 和系统 TensorRT binding 都是 Python
3.8 生态。JetPack 5 没有受支持的 Python 3.10 ONNX Runtime GPU / TensorRT
预编译路线；留在 JetPack 5 就用 Python 3.8，需要干净的 Python 3.10 路线就升级
JetPack 6。

## 共享环境

### 代码分支

onboard 的 Python 3.8 栈要配合 Python 3.8 兼容分支使用。限制不只在 wheel：
完整 policy tracking 会 import `sim2real`、`any4hdmi` 和 `mjhub`，源码里也不能有
Python 3.10-only 的 typing 写法。

如果机器人上是 git checkout：

```bash
cd ~/sim2real
git fetch origin py38
git switch py38

cd ~/any4hdmi
git fetch origin py38
git switch py38
```

如果机器人目录是 rsync 过去的，就从本地 `py38` 分支同步，而不是从普通开发分支同步。
凡是目标进程跑在 `.venv-jp5-ort` 里，都按这条规则处理。

创建一个能看到 JetPack 系统包的 Python 3.8 venv：

```bash
cd ~/sim2real
python3 -m venv --system-site-packages .venv-jp5-ort
bash -lic 'proxy_on; .venv-jp5-ort/bin/python -m pip install -U pip'
```

安装共同依赖：

```bash
cd ~/sim2real
bash -lic 'proxy_on; .venv-jp5-ort/bin/python -m pip install onnx==1.17.0 loguru pyyaml tyro cuda-python==12.3.0'
```

如果要跑完整 sim2real tracking 脚本，安装 Python 3.8 兼容的 `mjhub` 分支：

```bash
bash -lic 'proxy_on; .venv-jp5-ort/bin/python -m pip install --no-deps "mjhub @ git+https://github.com/EGalahad/mjhub.git@py38"'
```

完整 policy tracking 还需要一些 benchmark 不一定会 import 到的 runtime 依赖：

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

只有当 Python 3.8 进程需要自己创建 Viser server 时才安装 `viser==0.2.23`。
当前 `mjviser` release 使用了 Python 3.10 typing 写法，所以 viewer-heavy 的
base-sim 进程建议先用普通环境跑；如果必须放进 `.venv-jp5-ort`，需要先使用
Python 3.8 兼容的 `mjviser` 分支。

## ONNX GPU 后端

安装 NVIDIA 的 Jetson ONNX Runtime GPU wheel：

```bash
mkdir -p /tmp/ort-gpu
bash -lic 'proxy_on; wget -O /tmp/ort-gpu/onnxruntime_gpu-1.16.0-cp38-cp38-linux_aarch64.whl https://nvidia.box.com/shared/static/iizg3ggrtdkqawkmebbfixo7sce6j365.whl'
.venv-jp5-ort/bin/python -m pip install /tmp/ort-gpu/onnxruntime_gpu-1.16.0-cp38-cp38-linux_aarch64.whl
```

检查 provider：

```bash
.venv-jp5-ort/bin/python -c "import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())"
```

期望包含：

```text
['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

JetPack 5 的 ORT 1.16 不接受 IR 10 / opset 20 这类较新的导出模型。`onnx-gpu`
要使用 IR 9 / opset 19 的 ONNX 副本。

benchmark 示例：

```bash
cd ~/sim2real
PYTHONPATH=$PWD:~/any4hdmi/src .venv-jp5-ort/bin/python scripts/test_policy_inference.py \
  --policy_config /tmp/ort-test/policy-op19.yaml \
  --inference_backend onnx-gpu \
  --warmup 50 --runs 1000
```

tracking 示例：

```bash
cd ~/sim2real
PYTHONPATH=$PWD:~/any4hdmi/src .venv-jp5-ort/bin/python sim2real/rl_policy/tracking.py \
  --robot g1 \
  --policy_config /tmp/ort-test/policy-sonic-release-op19.yaml \
  --motion_path ../any4hdmi/output/root_tracking_test/motions/forward_1.npz \
  --inference_backend onnx-gpu
```

## TensorRT 后端

TensorRT 本身来自 JetPack 系统包，例如
`/usr/lib/python3.8/dist-packages/tensorrt`。因为 venv 使用
`--system-site-packages` 创建，所以能 import 系统 TensorRT。

TensorRT 8.5 的 Python binding 仍然引用 `np.bool`，所以要使用 NumPy 1.23：

```bash
cd ~/sim2real
.venv-jp5-ort/bin/python -m pip install numpy==1.23.5
```

检查 TensorRT：

```bash
.venv-jp5-ort/bin/python -c "import tensorrt as trt; print(trt.__version__)"
```

期望输出：

```text
8.5.2.2
```

TensorRT 8.5 可能无法直接 parse 包含 `LayerNormalization` 和 `Mish` 的 ONNX。
需要先生成 TensorRT-friendly 副本：降到 IR 9 / opset 13，并把这些 op 展开成基础
ONNX op。

benchmark 示例：

```bash
cd ~/sim2real
PYTHONPATH=$PWD:~/any4hdmi/src HDMI_TRT_FP16=1 HDMI_TRT_FORCE_REBUILD=0 .venv-jp5-ort/bin/python scripts/test_policy_inference.py \
  --policy_config /tmp/ort-test/policy-trt.yaml \
  --inference_backend tensorrt \
  --warmup 50 --runs 1000
```

如果 ONNX 有变化或需要重新生成 `.plan`，设置 `HDMI_TRT_FORCE_REBUILD=1`。

tracking 示例：

```bash
cd ~/sim2real
PYTHONPATH=$PWD:~/any4hdmi/src HDMI_TRT_FP16=1 HDMI_TRT_FORCE_REBUILD=0 .venv-jp5-ort/bin/python sim2real/rl_policy/tracking.py \
  --robot g1 \
  --policy_config /tmp/ort-test/policy-sonic-release-trt.yaml \
  --motion_path ../any4hdmi/output/root_tracking_test/motions/forward_1.npz \
  --inference_backend tensorrt
```

## 已知问题

- PyPI 的 `pip install onnxruntime-gpu` 没有 JetPack 5 aarch64 wheel。要用上面的
  NVIDIA Jetson wheel。
- `proxy_on` 放在 `bash -lic` 里最稳定。
- Python 3.10 项目 venv 不能加载 `cp38` Jetson ORT wheel，也不能加载 Python 3.8
  TensorRT binding。
- JetPack 5 package index 没有受支持的 `cp310` ONNX Runtime GPU wheel，JetPack 5
  TensorRT Python binding 也面向系统 Python 3.8。Python 3.10 路线基本意味着源码构建或升级
  JetPack 6。
- ORT 1.16 会拒绝 IR 10：`Unsupported model IR version: 10, max supported IR
  version: 9`。
- ORT 1.16 对 opset 20 不保证支持。JetPack 5 用 opset 19。
- TensorRT 8.5 parser 可能把 `LayerNormalization` 和 `Mish` 当成缺失 plugin。
  需要把它们展开成基础 op。
- TensorRT 8.5 legacy binding 执行应该用 `execute_async_v2(bindings, stream)`。
  把 bindings 传给 `execute_async_v3` 会报参数不兼容。
- JetPack 5 Python 3.8 跑完整 tracking 时，可能撞到项目代码或依赖里的 Python 3.10-only
  annotation。Python 3.8 兼容改动要放在专门的 `py38` 分支；普通开发分支有新改动需要 onboard
  时，再 merge 到 `py38`。
- 最小 inference benchmark 能跑，不代表完整 policy tracking 已经能跑。后者还可能缺
  `mujoco`、`zmq`、`sshkeyboard`、`huggingface_hub`、`torch`、`tensordict`
  这类 runtime import。
