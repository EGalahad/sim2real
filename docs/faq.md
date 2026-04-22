---
title: FAQ
sidebar_position: 4
---

## `uv sync` fails to fetch Git dependencies on onboard Orin

If the Orin system clock is wrong after boot, `uv sync` can fail while
fetching Git dependencies from GitHub. The error may look like a TLS issue even
though the root cause is the incorrect system time.

Set the current time manually, then retry:

```bash
sudo date -s "2026-04-17 10:00:00"  # Replace with the current time
uv sync
```

If this keeps happening after reboot, check the device time synchronization or
RTC configuration.

## `ImportError: cannot allocate memory in static TLS block`

On onboard Orin, Python can fail to import a native library and report that
`libc10.so` or `libGLdispatch.so.0` cannot allocate memory in the static TLS
block.

This usually happens when large native libraries such as PyTorch or OpenGL are
loaded after other dependencies have already consumed the available static
thread-local storage slots.

For a `libGLdispatch.so.0` error on `aarch64`, preload the library before
starting Python:

```bash
export LD_PRELOAD=/home/elijah/sim2real/venv/teleop/.venv/lib/python3.10/site-packages/torch/lib/libtorch.so:/lib/aarch64-linux-gnu/libGLdispatch.so.0:$LD_PRELOAD
```

You can also try moving `import torch` to the first import in the Python
script.
