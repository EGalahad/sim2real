#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/setup/setup_xrobot_pybind.sh [--arch x86_64|aarch64]

Build and install xrobotoolkit_sdk into venv/teleop.

Expected repos:
  external/XRoboToolkit-PC-Service
  external/XRoboToolkit-PC-Service-Pybind
EOF
}

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ARCH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --arch)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --arch" >&2
        exit 1
      fi
      ARCH="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$ARCH" ]]; then
  case "$(uname -m)" in
    x86_64|amd64)
      ARCH="x86_64"
      ;;
    aarch64|arm64)
      ARCH="aarch64"
      ;;
    *)
      echo "Unsupported architecture: $(uname -m). Pass --arch explicitly." >&2
      exit 1
      ;;
  esac
fi

case "$ARCH" in
  x86_64|aarch64)
    ;;
  *)
    echo "Unsupported --arch value: $ARCH" >&2
    exit 1
    ;;
esac

SERVICE_DIR="$ROOT_DIR/external/XRoboToolkit-PC-Service"
PYBIND_DIR="$ROOT_DIR/external/XRoboToolkit-PC-Service-Pybind"
SDK_DIR="$SERVICE_DIR/RoboticsService/PXREARobotSDK"
PYTHON_BIN="$ROOT_DIR/venv/teleop/.venv/bin/python"
UV_BIN=$(command -v uv || true)

if [[ -z "$UV_BIN" && -x "$HOME/.local/bin/uv" ]]; then
  UV_BIN="$HOME/.local/bin/uv"
fi

if [[ ! -d "$SERVICE_DIR" ]]; then
  echo "Missing repo: $SERVICE_DIR" >&2
  exit 1
fi

if [[ ! -d "$PYBIND_DIR" ]]; then
  echo "Missing repo: $PYBIND_DIR" >&2
  exit 1
fi

if [[ ! -d "$SDK_DIR" ]]; then
  echo "Missing SDK directory: $SDK_DIR" >&2
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing teleop environment at $PYTHON_BIN. Run 'uv --project venv/teleop sync' first." >&2
  exit 1
fi

if [[ -z "$UV_BIN" ]]; then
  echo "Missing uv. Install it or add it to PATH." >&2
  exit 1
fi

if [[ "$ARCH" == "aarch64" ]]; then
  GRPC_DIR="$SERVICE_DIR/RoboticsService/Redistributable/linux_aarch64/grpc"
  GRPC_UPSTREAM_DIR="$SERVICE_DIR/RoboticsService/Redistributable/linux_aarch64/grpc.upstream"
  PROTOBUF_STUBS_DIR="$GRPC_DIR/include/google/protobuf/stubs"
  PROTOBUF_UPSTREAM_STUBS_DIR="$GRPC_UPSTREAM_DIR/include/google/protobuf/stubs"

  if [[ ! -d "$PROTOBUF_STUBS_DIR" && -d "$PROTOBUF_UPSTREAM_STUBS_DIR" ]]; then
    echo "[setup_xrobot_pybind] restoring aarch64 protobuf stubs from grpc.upstream"
    cp -r "$PROTOBUF_UPSTREAM_STUBS_DIR" "$PROTOBUF_STUBS_DIR"
  fi
fi

echo "[setup_xrobot_pybind] repo_root=$ROOT_DIR"
echo "[setup_xrobot_pybind] arch=$ARCH"
echo "[setup_xrobot_pybind] building PXREARobotSDK"

(
  cd "$SDK_DIR"
  bash build.sh
)

case "$ARCH" in
  x86_64)
    INCLUDE_DIR="$PYBIND_DIR/include"
    LIB_DIR="$PYBIND_DIR/lib"
    ;;
  aarch64)
    INCLUDE_DIR="$PYBIND_DIR/include/aarch64"
    LIB_DIR="$PYBIND_DIR/lib/aarch64"
    ;;
esac

mkdir -p "$INCLUDE_DIR" "$LIB_DIR"

cp "$SDK_DIR/PXREARobotSDK.h" "$INCLUDE_DIR/PXREARobotSDK.h"
rm -rf "$INCLUDE_DIR/nlohmann"
cp -r "$SDK_DIR/nlohmann" "$INCLUDE_DIR/nlohmann"
cp "$SDK_DIR/build/libPXREARobotSDK.so" "$LIB_DIR/libPXREARobotSDK.so"

echo "[setup_xrobot_pybind] copied headers and libraries"
ls -l "$INCLUDE_DIR/PXREARobotSDK.h"
ls -ld "$INCLUDE_DIR/nlohmann"
ls -l "$LIB_DIR/libPXREARobotSDK.so"

if command -v ldd >/dev/null 2>&1; then
  ldd "$LIB_DIR/libPXREARobotSDK.so" || true
fi

export pybind11_DIR
pybind11_DIR=$(
  "$UV_BIN" --project "$ROOT_DIR/venv/teleop" run python -c "import pybind11; print(pybind11.get_cmake_dir())"
)

echo "[setup_xrobot_pybind] pybind11_DIR=$pybind11_DIR"
"$UV_BIN" --project "$ROOT_DIR/venv/teleop" pip uninstall --python "$PYTHON_BIN" xrobotoolkit_sdk >/dev/null 2>&1 || true
"$UV_BIN" --project "$ROOT_DIR/venv/teleop" pip install --python "$PYTHON_BIN" -e "$PYBIND_DIR"

echo "[setup_xrobot_pybind] xrobotoolkit_sdk installed"
