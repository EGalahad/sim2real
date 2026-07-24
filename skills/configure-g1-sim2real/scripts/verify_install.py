from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path


PROFILES = {
    "root-cpu": (
        "sim2real",
        "any4hdmi",
        "mujoco",
        "onnxruntime",
    ),
    "g1-cpu": (
        "sim2real",
        "any4hdmi",
        "mujoco",
        "onnxruntime",
        "cyclonedds",
        "unitree_sdk2py",
        "unitree_interface",
    ),
    "g1-gpu": (
        "sim2real",
        "any4hdmi",
        "mujoco",
        "onnxruntime",
        "cyclonedds",
        "unitree_sdk2py",
        "unitree_interface",
    ),
}


def check_imports(profile: str) -> list[str]:
    failures: list[str] = []
    for module_name in PROFILES[profile]:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            failures.append(f"import {module_name}: {exc}")
        else:
            location = getattr(module, "__file__", "<namespace>")
            print(f"PASS import {module_name}: {location}")
    return failures


def check_native_environment(profile: str) -> list[str]:
    if not profile.startswith("g1-"):
        return []

    failures: list[str] = []
    cyclone_home = os.environ.get("CYCLONEDDS_HOME")
    if not cyclone_home:
        failures.append("CYCLONEDDS_HOME is unset")
    elif not Path(cyclone_home).is_dir():
        failures.append(f"CYCLONEDDS_HOME does not exist: {cyclone_home}")
    else:
        print(f"PASS CYCLONEDDS_HOME: {cyclone_home}")
    return failures


def check_asset(asset: str | None) -> list[str]:
    if asset is None:
        return []
    try:
        from mjhub import resolve_asset_reference

        resolved = Path(resolve_asset_reference(asset))
        if not resolved.exists():
            raise FileNotFoundError(resolved)
    except Exception as exc:
        return [f"resolve asset {asset}: {exc}"]
    print(f"PASS asset: {resolved}")
    return []


def check_onnx(model_path: Path | None, profile: str) -> list[str]:
    if model_path is None:
        return []
    if not model_path.is_file():
        return [f"ONNX model does not exist: {model_path}"]

    try:
        import onnxruntime as ort

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if profile == "g1-gpu"
            else ["CPUExecutionProvider"]
        )
        session = ort.InferenceSession(str(model_path), providers=providers)
        active = session.get_providers()
        if profile == "g1-gpu" and "CUDAExecutionProvider" not in active:
            raise RuntimeError(f"CUDAExecutionProvider is not active: {active}")
    except Exception as exc:
        return [f"load ONNX {model_path}: {exc}"]
    print(f"PASS ONNX session: providers={active}")
    return []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify a sim2real installation without creating RobotIO."
    )
    parser.add_argument("--profile", choices=tuple(PROFILES), required=True)
    parser.add_argument("--asset")
    parser.add_argument("--onnx", type=Path)
    args = parser.parse_args()

    failures = [
        *check_imports(args.profile),
        *check_native_environment(args.profile),
        *check_asset(args.asset),
        *check_onnx(args.onnx, args.profile),
    ]
    if failures:
        print("\nFAIL")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)

    print("\nPASS installation verification")


if __name__ == "__main__":
    main()
