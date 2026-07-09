---
title: Download Artifacts
slug: /reference/artifacts
---

# Download Artifacts

Large runtime artifacts are stored outside git. Download the shared
[sim2real artifacts](https://drive.google.com/drive/folders/1lrPyiiy7anyG3P4wHNIQQQlydboLPd9e)
folder before running deploy or onboard setup commands.

After download, the repo root should contain:

```text
checkpoints/
third_party/
  wheels/
  prebuilt/
```

`checkpoints/` contains exported policy YAML and ONNX files. Tutorial commands
assume these paths exist locally, for example
`checkpoints/mimic-lite/32x8192-huge/policy.yaml`.

`third_party/wheels/` contains deployment-only wheels that are resolved by
`uv` through `find-links = ["third_party/wheels"]`. On G1, use:

```bash
uv sync --group g1
```

Use the GPU group when the onboard environment should install the JetPack 6
ONNX Runtime GPU wheel:

```bash
uv sync --group g1-gpu
```

`third_party/prebuilt/` contains prebuilt packages that are not Python wheels.
The JetPack 5 Pico onboard setup expects:

```text
third_party/prebuilt/jetpack5-aarch64/
```

Do not commit downloaded artifact contents. Keep refreshed artifacts in the
Google Drive folder and keep local paths matching the layout above.
