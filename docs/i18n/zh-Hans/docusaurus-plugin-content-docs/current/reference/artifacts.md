---
title: Download Artifacts
slug: /reference/artifacts
---

# Download Artifacts

大文件不放在 git 里。部署或 onboard setup 之前，先下载共享的
[sim2real artifacts](https://drive.google.com/drive/folders/1lrPyiiy7anyG3P4wHNIQQQlydboLPd9e)
目录。

下载后，repo 根目录应该长这样：

```text
checkpoints/
third_party/
  wheels/
  prebuilt/
```

`checkpoints/` 里放导出的 policy YAML 和 ONNX。教程里的命令默认这些路径已经存在，
例如 `checkpoints/mimic-lite/32x8192-huge/policy.yaml`。

`third_party/wheels/` 里放部署用 wheel。`uv` 会通过
`find-links = ["third_party/wheels"]` 解析这些包。G1 上安装基础真机依赖：

```bash
uv sync --group g1
```

如果 onboard 环境还要安装 JetPack 6 的 ONNX Runtime GPU wheel，用：

```bash
uv sync --group g1-gpu
```

`third_party/prebuilt/` 里放非 Python wheel 的预编译包。JetPack 5 的 Pico onboard
setup 需要：

```text
third_party/prebuilt/jetpack5-aarch64/
```

不要把下载下来的 artifact 内容提交进 git。更新 artifact 时，把新文件放回 Google Drive，
本地保持上面的目录结构即可。
