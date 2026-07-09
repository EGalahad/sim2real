# G1 sim2real Setup Runbook

This runbook captures the G1 setup path validated on the `g1-deploy` JetPack 5
host. Prefer current host inspection over blind replay, but keep this order:
network and HF cache first, root env second, dataset sync third, teleop last.

## 1. Connect and Check the Host

Use `g1-deploy` when the Ethernet route to `g1-cable` is unavailable.

```bash
ssh g1-deploy 'bash -lc "
  hostname
  cat /etc/nv_tegra_release || true
  ip -br addr
  ip route
  uv --version
"'
```

If WiFi is not already connected, have the operator run:

```bash
sudo nmcli dev wifi connect deploy_5G password 88888888 ifname wlan0
sudo nmcli connection modify deploy_5G ipv4.route-metric 50 ipv6.route-metric 50
sudo nmcli connection up deploy_5G
```

Check that `wlan0` has internet egress and that robot-control interfaces remain
on their robot subnet.

## 2. Proxy and Hugging Face Cache

Direct `huggingface.co` from the robot may hang. If the local machine has an
HTTP proxy on `127.0.0.1:7890`, forward it to the robot as `127.0.0.1:7891`:

```bash
ssh -N -R 127.0.0.1:7891:127.0.0.1:7890 g1-deploy
```

In another shell, verify from the robot:

```bash
ssh g1-deploy 'bash -lc "
  HTTPS_PROXY=http://127.0.0.1:7891 \
  HTTP_PROXY=http://127.0.0.1:7891 \
  curl -I --max-time 20 https://huggingface.co/api/models/elijahgalahad/g1_xmls
"'
```

Refresh the G1 XML cache online, then default deploy shells back to offline:

```bash
ssh g1-deploy 'bash -lc "
  unset HF_HUB_OFFLINE
  export HF_ENDPOINT=https://hf-mirror.com
  export HF_HUB_DISABLE_TELEMETRY=1
  export HTTPS_PROXY=http://127.0.0.1:7891
  export HTTP_PROXY=http://127.0.0.1:7891
  python3 - <<PY
from huggingface_hub import snapshot_download
print(snapshot_download(\"elijahgalahad/g1_xmls\"))
PY
"'
```

Persist deploy defaults in `~/.profile` and `~/.bashrc` if they are absent:

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE=1
```

Verify through the actual asset resolver:

```bash
ssh g1-deploy 'bash -lc "
  cd ~/sim2real
  uv run python - <<PY
from mjhub import resolve_asset_reference
print(resolve_asset_reference(\"hf://elijahgalahad/g1_xmls@main/g1-mode_13_15.xml\"))
PY
"'
```

Expected result: a local path under
`~/.cache/huggingface/hub/models--elijahgalahad--g1_xmls/snapshots/...`.

## 3. Root Project Setup

Build CycloneDDS if `uv sync --group g1` or imports cannot locate it:

```bash
ssh g1-deploy 'bash -lc "
  mkdir -p ~/src
  if [ ! -d ~/src/cyclonedds ]; then
    git clone --branch releases/0.10.x https://github.com/eclipse-cyclonedds/cyclonedds.git ~/src/cyclonedds
  fi
  cmake -S ~/src/cyclonedds -B ~/src/cyclonedds/build \
    -DCMAKE_INSTALL_PREFIX=$HOME/cyclonedds/install \
    -DBUILD_TESTING=OFF
  cmake --build ~/src/cyclonedds/build --target install -j\$(nproc)
"'
```

Run the root sync:

```bash
ssh g1-deploy 'bash -lc "
  cd ~/sim2real
  export CYCLONEDDS_HOME=$HOME/cyclonedds/install
  export LD_LIBRARY_PATH=$CYCLONEDDS_HOME/lib:\${LD_LIBRARY_PATH:-}
  uv sync --group g1
"'
```

If these variables are needed outside the current shell, append them to
`~/.profile` and `~/.bashrc`:

```bash
export CYCLONEDDS_HOME=$HOME/cyclonedds/install
export LD_LIBRARY_PATH=$CYCLONEDDS_HOME/lib:${LD_LIBRARY_PATH:-}
```

Verify imports and a CPU ONNX smoke:

```bash
ssh g1-deploy 'bash -lc "
  cd ~/sim2real
  export CYCLONEDDS_HOME=$HOME/cyclonedds/install
  export LD_LIBRARY_PATH=$CYCLONEDDS_HOME/lib:\${LD_LIBRARY_PATH:-}
  uv run python - <<PY
for name in [\"sim2real\", \"any4hdmi\", \"mujoco\", \"onnxruntime\", \"cyclonedds\", \"unitree_sdk2py\", \"unitree_interface\"]:
    __import__(name)
    print(\"ok\", name)
PY
  uv run scripts/test_policy_inference.py \
    --policy_config checkpoints/lafan-aa/policy-ec592bb4_lafan_100style_student-5000.yaml \
    --inference_backend onnx-cpu \
    --warmup 20 --runs 100
"'
```

## 4. Dataset Placement and Sync

For root tracking tests, keep the G1-specific copy under
`any4hdmi/output/g1/root_tracking_test/`:

```bash
mkdir -p any4hdmi/output/g1/root_tracking_test
rsync -a --delete any4hdmi/output/root_tracking_test/ any4hdmi/output/g1/root_tracking_test/
```

Add a sync include in `sim2real/sync-g1.sh` near the other `any4hdmi/output`
includes:

```bash
--include='output/g1/root_tracking_test/***'
```

Sync to the robot:

```bash
G1_HOST=g1-deploy bash sim2real/sync-g1.sh
```

If the sync script excludes `sync*.sh`, copy the script itself after editing:

```bash
rsync -az sim2real/sync-g1.sh g1-deploy:/home/elijah/sim2real/sync-g1.sh
```

Verify the remote dataset:

```bash
ssh g1-deploy 'bash -lc "
  find ~/any4hdmi/output/g1/root_tracking_test -maxdepth 1 -type f -printf \"%f\n\" | sort
  du -sh ~/any4hdmi/output/g1/root_tracking_test
"'
```

## 5. Teleop uv Dependencies

Start with the normal sync:

```bash
ssh g1-deploy 'bash -lc "
  cd ~/sim2real
  uv --project venv/teleop sync
"'
```

The preferred repository shape is for teleop to install GMR, smplx, and
xrobotoolkit-sdk as editable path sources from `sim2real/external/GMR`,
`sim2real/external/smplx`, and
`sim2real/external/XRoboToolkit-PC-Service-Pybind`:

```toml
[project]
dependencies = [
  "general_motion_retargeting",
]

[tool.uv.sources]
general-motion-retargeting = { path = "../../external/GMR", editable = true }
smplx = { path = "../../external/smplx", editable = true }
xrobotoolkit-sdk = { path = "../../external/XRoboToolkit-PC-Service-Pybind", editable = true }
```

This matches the GMR README's `pip install -e .` expectation, avoids remote
GitHub fetches for both GMR and smplx, and prevents exact `uv sync` from
removing `xrobotoolkit_sdk`. A non-editable GMR wheel install can import the
Python package but miss top-level `assets/`, which breaks
`g1_mocap_29dof.xml` lookup at runtime.

If `uv` hangs on GitHub fetches for `EGalahad/GMR` or `vchoutas/smplx`, seed
local source copies on the robot and patch only the robot-side dependency files
as a temporary recovery path:

```bash
rsync -az --delete --exclude='.git' <local-GMR>/ g1-deploy:/home/elijah/src/GMR/
rsync -az --delete --exclude='.git' <local-smplx>/ g1-deploy:/home/elijah/src/smplx/
```

Remote-only dependency rewrites:

```toml
general_motion_retargeting @ file:///home/elijah/src/GMR
smplx @ file:///home/elijah/src/smplx
```

Place the first line in `~/sim2real/venv/teleop/pyproject.toml` and the second
line in `~/src/GMR/pyproject.toml`. Make backups before editing. Do not commit
these file URLs to the local repo unless the project intentionally vendors
those dependencies.

Run the sync again and verify core imports:

```bash
ssh g1-deploy 'bash -lc "
  cd ~/sim2real
  uv --project venv/teleop sync
  uv --project venv/teleop run python - <<PY
for name in [\"sim2real.teleop\", \"general_motion_retargeting\", \"smplx\", \"mjviser\", \"mjhub\", \"mujoco\", \"pybind11\", \"torch\"]:
    __import__(name)
    print(\"ok\", name)
PY
"'
```

## 6. XRoboToolkit Service and Pybind

Install the service package with sudo:

```bash
sudo apt install -y /home/elijah/sim2real/third_party/prebuilt/jetpack5-aarch64/xrobotservice/XRoboToolkit-PC-Service_1.0.0.0_arm64_ubuntu20.04.deb
```

If sudo is only available in a user-created tmux, send the command there and
wait for the user to enter the password. Do not claim the service is installed
until `/opt/apps/roboticsservice` exists.

If the robot lacks the external sources, sync them from local:

```bash
rsync -az --delete sim2real/external/XRoboToolkit-PC-Service/ g1-deploy:/home/elijah/sim2real/external/XRoboToolkit-PC-Service/
rsync -az --delete sim2real/external/XRoboToolkit-PC-Service-Pybind/ g1-deploy:/home/elijah/sim2real/external/XRoboToolkit-PC-Service-Pybind/
```

On JetPack 5, ensure the SDK uses the repo's prebuilt aarch64 gRPC tree:

```bash
ssh g1-deploy 'bash -lc "
  cd ~/sim2real
  rm -rf external/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/build
  rm -rf external/XRoboToolkit-PC-Service/RoboticsService/Redistributable/linux_aarch64/grpc
  cp -a third_party/prebuilt/jetpack5-aarch64/xrobot-grpc \
    external/XRoboToolkit-PC-Service/RoboticsService/Redistributable/linux_aarch64/grpc
  test -f external/XRoboToolkit-PC-Service/RoboticsService/Redistributable/linux_aarch64/grpc/include/google/protobuf/stubs/common.h
"'
```

Build and install the Python binding:

```bash
ssh g1-deploy 'bash -lc "
  cd ~/sim2real
  bash scripts/setup/setup_xrobot_pybind.sh --arch aarch64
  uv --project venv/teleop run python - <<PY
import xrobotoolkit_sdk
print(\"xrobot import ok\", xrobotoolkit_sdk.__file__)
PY
"'
```

Start or check the service after the `.deb` is installed:

```bash
ssh g1-deploy 'bash -lc "
  test -d /opt/apps/roboticsservice
  pgrep -af \"roboticsservice|runService|xrobot\" || true
  # Start manually only when needed:
  # bash /opt/apps/roboticsservice/runService.sh
"'
```

Compile teleop scripts:

```bash
ssh g1-deploy 'bash -lc "
  cd ~/sim2real
  uv --project venv/teleop run python -m py_compile \
    sim2real/teleop/pico_retarget_pub.py \
    sim2real/teleop/record_smplx.py \
    sim2real/teleop/benchmark_smplx_retarget.py
"'
```

## 7. Failure Map

- `LocalEntryNotFoundError`, connection stalls, or asset fetch during policy
  startup: fix HF endpoint/proxy/cache and then set `HF_HUB_OFFLINE=1`.
- `ALL_PROXY=socks5://127.0.0.1:7890` plus missing `socksio`: unset
  `ALL_PROXY/all_proxy` for `uv run` or use HTTP(S) proxy variables only.
- `uv --project venv/teleop sync` hangs on GitHub: prefer repo-local
  `external/GMR` with an editable `tool.uv.sources` path. Use `/home/elijah/src`
  `file://` dependencies only as a temporary robot-local recovery path.
- `ParseXML: Error opening file ... site-packages/general_motion_retargeting/../assets/unitree_g1/g1_mocap_29dof.xml`:
  GMR was installed as a non-editable wheel and its top-level assets were not
  installed. Fix by making GMR editable from `external/GMR` and rerunning
  `uv --project venv/teleop sync`.
- CMake path mismatch under `PXREARobotSDK/build`: remove the stale build
  directory copied from another machine.
- `runtime_version.h` missing during XRoboToolkit build: replace the SDK gRPC
  directory with `third_party/prebuilt/jetpack5-aarch64/xrobot-grpc`.
- Protobuf errors mentioning `OnShutdownDelete`, `StrongReference`, or
  `google/protobuf/stubs`: the build is mixing system and bundled protobuf
  headers or the bundled `stubs` directory is incomplete. Restore the prebuilt
  gRPC include tree and rebuild from a clean build directory.
- `low state not ready` after a policy enters the loop: robot low-state or
  bridge is not publishing. Do not keep debugging HF for this symptom.
- `Keyboard listener stopped unexpectedly: Inappropriate ioctl`: common in
  noninteractive SSH smoke tests. Re-test in an interactive terminal only if
  keyboard control is required for that specific run.

## 8. Ready State Summary

Root project is ready when the HF resolver, `uv sync --group g1`, imports, and
ONNX CPU benchmark all pass on `g1-deploy`.

Teleop Python is ready when `uv --project venv/teleop sync`, core imports,
`xrobotoolkit_sdk` import, and teleop script compilation pass.

Teleop hardware is ready only after the XRoboToolkit `.deb` is installed,
`/opt/apps/roboticsservice` exists, the service is running, and the Pico/robot
devices are connected.
