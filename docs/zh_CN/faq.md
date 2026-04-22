# FAQ

## `uv sync` 在 onboard Orin 上拉 Git 依赖失败

如果 Orin 开机后的系统时间不对，`uv sync` 在从 GitHub 拉取 Git 依赖时可能失败。报错有时看起来像 TLS 问题，但根因其实是系统时间错误。

先手动校时，再重试：

```bash
sudo date -s "2026-04-17 10:00:00"  # 改成当前时间
uv sync
```

如果重启后还会反复出现，检查设备的时间同步或 RTC 配置。

## `ImportError: cannot allocate memory in static TLS block`

在 onboard Orin 上，Python 导入原生库时可能失败，并提示 `libc10.so` 或 `libGLdispatch.so.0` 无法在 static TLS block 里分配内存。

这通常是因为 PyTorch、OpenGL 这类大型原生库加载得太晚，前面已经把可用的 static thread-local storage slot 占掉了。

如果是在 `aarch64` 上遇到 `libGLdispatch.so.0` 错误，可以在启动 Python 前先 preload：

```bash
export LD_PRELOAD=/home/elijah/sim2real/venv/teleop/.venv/lib/python3.10/site-packages/torch/lib/libtorch.so:/lib/aarch64-linux-gnu/libGLdispatch.so.0:$LD_PRELOAD
```

也可以尝试把 `import torch` 移到 Python 脚本的最前面。
