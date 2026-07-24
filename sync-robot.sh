#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ROBOT="${1:-g1}"

usage() {
  printf '%s\n' \
    "Sync the local sim2real checkout to a robot." \
    "" \
    "Usage: $0 [g1]" \
    "" \
    "Environment overrides:" \
    "  G1_HOST / G1_REMOTE_HOME" \
    "  SYNC_CHECKPOINTS=0|1" \
    "  SYNC_ANY4HDMI=0|1"
}

case "${ROBOT}" in
  g1)
    ROBOT_HOST="${G1_HOST:-g1-hotspot}"
    REMOTE_HOME="${G1_REMOTE_HOME:-/home/elijah}"
    DEFAULT_SYNC_CHECKPOINTS=1
    DEFAULT_SYNC_ANY4HDMI=1
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    printf 'Unknown robot: %s\n' "${ROBOT}" >&2
    usage >&2
    exit 2
    ;;
esac

SYNC_CHECKPOINTS="${SYNC_CHECKPOINTS:-${DEFAULT_SYNC_CHECKPOINTS}}"
SYNC_ANY4HDMI="${SYNC_ANY4HDMI:-${DEFAULT_SYNC_ANY4HDMI}}"

for value_name in SYNC_CHECKPOINTS SYNC_ANY4HDMI; do
  value="${!value_name}"
  if [[ "${value}" != "0" && "${value}" != "1" ]]; then
    printf '%s must be 0 or 1, got %s\n' "${value_name}" "${value}" >&2
    exit 2
  fi
done

checkpoint_args=()
if [[ "${SYNC_CHECKPOINTS}" == "0" ]]; then
  checkpoint_args+=(--exclude='checkpoints/')
fi

robot_args=(
  --exclude='sim2real/teleop/GMR/'
  --exclude='external/XRoboToolkit-PC-Service-Pybind/tmp/***'
  --exclude='external/XRoboToolkit-PC-Service-Pybind/xrobotoolkit_sdk*.so'
  --exclude='external/XRoboToolkit-PC-Service-Pybind/lib/libPXREARobotSDK.so'
  --include='external/'
  --include='external/GMR/***'
  --include='external/smplx/***'
  --include='external/XRoboToolkit-PC-Service-Pybind/***'
  --exclude='external/**'
)

printf 'Syncing sim2real to %s (%s:%s/sim2real)\n' \
  "${ROBOT}" "${ROBOT_HOST}" "${REMOTE_HOME}"

rsync -avr --delete \
  "${checkpoint_args[@]}" \
  "${robot_args[@]}" \
  --filter='P checkpoints/***' \
  --filter='P **/.venv/***' \
  --filter='P *.plan' \
  --filter='P *.engine' \
  --exclude='outputs/' \
  --exclude='outputs-orin/' \
  --exclude='outputs_play/' \
  --exclude='wandb/' \
  --exclude='memmap_td/' \
  --exclude='.git/' \
  --exclude='.cache/' \
  --exclude='.codex/' \
  --exclude='.omx/' \
  --exclude='.pytest_cache/' \
  --exclude='.ruff_cache/' \
  --exclude='.mypy_cache/' \
  --exclude='stubs/' \
  --exclude='checkpoints/lafan-old/' \
  --exclude='checkpoints/lafan/' \
  --exclude='*.pt' \
  --exclude='*.pth' \
  --exclude='*.pyc' \
  --exclude='*.egg-info/' \
  --exclude='MUJOCO_LOG.TXT' \
  --exclude='.DS_Store' \
  --exclude='sync*.sh' \
  --exclude='.venv/' \
  --exclude='**/.venv/' \
  --exclude='*.plan' \
  --exclude='*.engine' \
  --exclude='third_party/dh116s_sdk/python/SDK_ARCH' \
  --exclude='third_party/dh116s_sdk/python/lhandprolib_python_sdk/lib/' \
  --exclude='/datasets/' \
  --exclude='docs/' \
  --exclude='__pycache__/' \
  --exclude='*.nsys-rep' \
  --exclude='real_vr.tar' \
  --exclude='robot_motion_pair.npz' \
  "${SCRIPT_DIR}/" \
  "${ROBOT_HOST}:${REMOTE_HOME}/sim2real/"

if [[ "${SYNC_ANY4HDMI}" != "1" ]]; then
  exit 0
fi

printf 'Syncing selected any4hdmi assets to %s\n' "${ROBOT_HOST}"
rsync -avr --delete \
  --exclude='.git/' \
  --exclude='.venv/' \
  --exclude='.venv*/' \
  --exclude='stubs/' \
  --exclude='.cache/' \
  --exclude='.codex/' \
  --exclude='.omx/' \
  --exclude='.pytest_cache/' \
  --exclude='.ruff_cache/' \
  --exclude='.mypy_cache/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='*.egg-info/' \
  --exclude='.DS_Store' \
  --include='output/' \
  --include='output/cartwheel/***' \
  --include='output/g1/' \
  --include='output/g1/amass_hard/***' \
  --include='output/g1/cartwheel-1/***' \
  --include='output/g1/extreme_motions/***' \
  --include='output/g1/extreme-demo/***' \
  --include='output/g1/lafan/***' \
  --include='output/g1/omni_extreme/***' \
  --include='output/g1/root_tracking_test/***' \
  --include='output/g1/high_dynamic_sequences/***' \
  --include='output/lafan/' \
  --include='output/lafan/**' \
  --include='output/sonic/' \
  --include='output/sonic/manifest.json' \
  --include='output/sonic/motions/' \
  --include='output/sonic/motions/240529/' \
  --include='output/sonic/motions/240529/macarena_001__A545.npz' \
  --include='output/sonic/motions/230509/' \
  --include='output/sonic/motions/230509/forward_lunge_R_002__A359.npz' \
  --include='output/sonic/motions/230509/squat_001__A359.npz' \
  --include='output/sonic/motions/220713/' \
  --include='output/sonic/motions/220713/walk_backward_start_001__A021.npz' \
  --include='output/sonic/motions/240327/' \
  --include='output/sonic/motions/240327/one_leg_idle_R_002__A533.npz' \
  --exclude='output/sonic/**' \
  --exclude='output/sonic*' \
  --exclude='output/**' \
  "${WORKSPACE_ROOT}/any4hdmi/" \
  "${ROBOT_HOST}:${REMOTE_HOME}/any4hdmi/"
