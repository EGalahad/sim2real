Place deployment-only Python wheels here before installing robot-specific extras.

This directory is tracked so `uv` can read the repository-level
`find-links = ["third_party/wheels"]` setting from `pyproject.toml` in a clean
clone. Wheel binaries are intentionally ignored and must not be committed.

For G1 deployment, download the required private or platform-specific wheels
into this directory before running:

```bash
uv sync --extra inference-cpu --extra robot-g1
```
