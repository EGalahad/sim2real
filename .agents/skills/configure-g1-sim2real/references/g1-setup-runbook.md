# G1 sim2real setup runbook

Use this reference after `scripts/inspect_host.sh`. Commands assume the remote
checkout is `~/sim2real`; adjust only after verifying the actual path.

## Contents

1. Host and network inspection
2. Code synchronization
3. CycloneDDS and shell environment
4. Root uv profiles
5. Hugging Face assets
6. ONNX Runtime validation
7. Inline G1 validation
8. PICO and DH116S isolation
9. Failure map

## 1. Host and network inspection

Run the bundled read-only inspection script:

```bash
ssh <host> 'bash -s -- "$HOME/sim2real"' \
  < .agents/skills/configure-g1-sim2real/scripts/inspect_host.sh
```

Verify:

- `/etc/os-release`, `uname -m`, Python 3.10, and uv;
- available interfaces and the interface connected to the G1 control network;
- sufficient free space for the selected dependencies and artifacts;
- whether `~/cyclonedds/install` and `third_party/wheels` exist;
- current and fresh-login values of `CYCLONEDDS_HOME`,
  `LD_LIBRARY_PATH`, `HF_HUB_OFFLINE`, and `HF_ENDPOINT`.

On `g1-ygx`, the interface validated for inline control was `enP8p1s0`.
Recheck it every time; do not turn this observation into a universal default.

If the local workstation is on the wrong Wi-Fi, inspect saved NetworkManager
profiles and switch to the operator-approved G1 network. Never embed the Wi-Fi
password in scripts or documentation.

## 2. Code synchronization

Make all persistent code changes in the local checkout, then inspect and run:

```bash
cd /path/to/local/sim2real
G1_HOST=<host> SYNC_CHECKPOINTS=0 SYNC_ANY4HDMI=0 ./sync-robot.sh g1
```

The sync must not:

- copy root, PICO, or DH116S virtual environments;
- delete remote-only checkpoints;
- transfer SONIC or other large datasets by default;
- overwrite host-local credentials or shell configuration.

Transfer explicitly requested large assets separately, with a resumable
command. Verify the destination and file size after transfer.

## 3. CycloneDDS and shell environment

The G1 Python SDK needs a native CycloneDDS installation while its Python
binding builds. Install it when inspection shows no usable installation:

```bash
mkdir -p "$HOME/src"
git clone --branch releases/0.10.x \
  https://github.com/eclipse-cyclonedds/cyclonedds.git \
  "$HOME/src/cyclonedds"
cmake -S "$HOME/src/cyclonedds" -B "$HOME/src/cyclonedds/build" \
  -DCMAKE_INSTALL_PREFIX="$HOME/cyclonedds/install" \
  -DBUILD_TESTING=OFF
cmake --build "$HOME/src/cyclonedds/build" \
  --target install -j"$(nproc)"
```

If the source directory already exists, inspect it instead of cloning over it.

Create `~/.config/sim2real/env.sh`:

```bash
mkdir -p "$HOME/.config/sim2real"
install -m 0644 /dev/stdin "$HOME/.config/sim2real/env.sh" <<'EOF'
export CYCLONEDDS_HOME="$HOME/cyclonedds/install"
export LD_LIBRARY_PATH="$CYCLONEDDS_HOME/lib:${LD_LIBRARY_PATH:-}"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE=1
EOF
```

Add this exact source line once to `~/.profile`:

```bash
[ -f "$HOME/.config/sim2real/env.sh" ] && . "$HOME/.config/sim2real/env.sh"
```

Add it to interactive shell configuration as well if desired. A standard
Ubuntu `.bashrc` returns early for noninteractive shells, so appending exports
to its end does not make them visible to `ssh host 'bash -lc ...'`.

Verify a new login shell:

```bash
ssh <host> 'bash -lc '"'"'
  source "$HOME/.config/sim2real/env.sh"
  printf "CYCLONEDDS_HOME=%s\n" "$CYCLONEDDS_HOME"
  printf "HF_HUB_OFFLINE=%s\n" "$HF_HUB_OFFLINE"
'"'"''
```

## 4. Root uv profiles

Run exactly one:

```bash
# Generic or ZMQ RobotIO
uv sync --extra inference-cpu

# G1 inline, CPU ONNX Runtime
uv sync --extra inference-cpu --extra robot-g1

# G1 inline, GPU ONNX Runtime
uv sync --extra inference-gpu --extra robot-g1
```

Prefer the CPU profile for bring-up. The root project intentionally keeps
PICO and DH116S dependencies out of this environment.

If GitHub downloads hang, establish a bounded reverse HTTP proxy tunnel:

```bash
ssh -N -R 127.0.0.1:7891:127.0.0.1:7890 <host>
```

Use `HTTP_PROXY` and `HTTPS_PROXY` for the affected install command. Avoid
inheriting `ALL_PROXY=socks5://...` unless `socksio` is installed.

## 5. Hugging Face assets

Refresh a missing asset once while online:

```bash
source "$HOME/.config/sim2real/env.sh"
unset HF_HUB_OFFLINE
HTTPS_PROXY=http://127.0.0.1:7891 \
HTTP_PROXY=http://127.0.0.1:7891 \
uv run --no-sync python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download("elijahgalahad/g1_xmls"))
PY
```

Then restore offline deployment and resolve through the consuming library:

```bash
source "$HOME/.config/sim2real/env.sh"
uv run --no-sync python - <<'PY'
from mjhub import resolve_asset_reference
print(resolve_asset_reference(
    "hf://elijahgalahad/g1_xmls@main/g1-mode_13_15.xml"
))
PY
```

Do not diagnose an asset-resolution stall by repeatedly starting a policy.

## 6. ONNX Runtime validation

Run the bundled verifier:

```bash
uv run --no-sync python \
  .agents/skills/configure-g1-sim2real/scripts/verify_install.py \
  --profile g1-cpu \
  --asset hf://elijahgalahad/g1_xmls@main/g1-mode_13_15.xml \
  --onnx checkpoints/mimic-lite/32x8192-huge/policy.onnx
```

For GPU, use `--profile g1-gpu`. The verifier must create an
`InferenceSession`; `ort.get_available_providers()` alone can report CUDA even
when `libcublasLt`, cuDNN, or another runtime library is missing.

If the skill directory has not been synced, stream the local verifier:

```bash
ssh <host> 'bash -lc '"'"'
  cd "$HOME/sim2real"
  source "$HOME/.config/sim2real/env.sh"
  uv run --no-sync python - --profile g1-cpu
'"'"'' < .agents/skills/configure-g1-sim2real/scripts/verify_install.py
```

If ORT reports `Unsupported model IR version`, prefer installing the intended
runtime version. Convert the model IR/opset only when compatibility has been
checked and output equivalence has been measured.

## 7. Inline G1 validation

Do not perform this section during an install-only request.

Before an authorized real test:

```bash
pgrep -af 'tracking.py|real_bridge.py|g1_debug_mode' || true
ip -br link
```

Use the inspected G1 interface:

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
uv run --no-sync sim2real/rl_policy/tracking.py \
  --robot g1 \
  --policy-config <policy.yaml> \
  --inference-backend onnx-cpu \
  --motion-backend zmq \
  --robot-io inline \
  --robot-interface <interface> \
  --controller keyboard
```

The Python `unitree_sdk2py` MotionSwitcher participant and the C++
`unitree_interface` participant cannot safely initialize in the same process
on the validated G1 stack. Inline startup therefore switches to debug mode in
a short-lived helper process, lets that DDS participant exit, and only then
imports `unitree_interface`.

Typical DDS interpretations:

- `eth0: does not match an available interface`: wrong interface;
- `DDS_RETCODE_BAD_PARAMETER` or `PRECONDITION_NOT_MET` on MotionSwitcher:
  `unitree_interface` initialized DDS too early;
- `Failed to create domain explicitly` after MotionSwitcher succeeds:
  the Python DDS participant is still alive when the C++ SDK starts.

## 8. PICO and DH116S isolation

Install PICO separately:

```bash
uv sync --project venv/pico
uv run --project venv/pico --no-sync python -c \
  'import xrobotoolkit_sdk, general_motion_retargeting, smplx'
```

Install DH116S separately:

```bash
bash third_party/dh116s_sdk/install.sh
uv sync --project venv/dh116s
uv run --project venv/dh116s --no-sync \
  scripts/dh116s_control.py --dry-run
```

Verify the expected `libLHandProLib.so` path before a hardware run. Do not home
or enable the hand during environment verification.

## 9. Failure map

| Symptom | Likely boundary | Action |
| --- | --- | --- |
| `Could not locate cyclonedds` | native build env | Install native CycloneDDS and export its prefix |
| HF request hangs | network/cache | Forward HTTP proxy or seed cache, then deploy offline |
| `Unsupported model IR version` | model/runtime | Align ORT or validate a converted model |
| CUDA provider listed but session fails | GPU shared libraries | Inspect loader error; use CPU for bring-up |
| DDS interface mismatch | RobotIO network | Use the inspected control interface |
| MotionSwitcher topic error | mixed DDS participants | Run mode switch in the helper process |
| `low state not ready` | robot/bridge state | Stop debugging HF; inspect low-state publisher |
| keyboard `Inappropriate ioctl` | noninteractive SSH | Use a TTY or non-keyboard controller for smoke tests |
| `libLHandProLib.so` missing | DH116S SDK layout | Run/repair the isolated SDK installer |
