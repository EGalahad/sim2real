# Local Wheelhouse

This directory contains prebuilt deployment-only wheels. The wheel files are
downloaded artifacts and are not tracked by git.

`unitree_interface-0.1.0-cp310-cp310-linux_aarch64.whl` is built from
YanjieZe's modified `unitree_sdk2` Python binding and is installed only on
Linux aarch64 environments by the `g1` dependency group marker in
`pyproject.toml`.

`onnxruntime_gpu-1.16.0-cp310-cp310-linux_aarch64.whl` is the JetPack 6
Python 3.10 ONNX Runtime GPU wheel and is installed by the `g1-gpu`
dependency group marker.

To refresh it on G1, rebuild `/home/elijah/unitree_sdk2/python_binding`, then
package the resulting `unitree_interface.so` as a wheel with the same package
name and compatible Python/platform tag.
