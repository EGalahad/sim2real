# Local Wheelhouse

This directory contains prebuilt deployment-only wheels that are not available
from a package index.

`unitree_interface-0.1.0-cp310-cp310-linux_aarch64.whl` is built from
YanjieZe's modified `unitree_sdk2` Python binding and is installed only on
Linux aarch64 environments by the `g1` dependency group marker in
`pyproject.toml`.

To refresh it on G1, rebuild `/home/elijah/unitree_sdk2/python_binding`, then
package the resulting `unitree_interface.so` as a wheel with the same package
name and compatible Python/platform tag.
