---
name: configure-g1-sim2real
description: Configure and repair the G1 sim2real robot computer environment for root policy deployment and teleop. Use when a task mentions g1-cable, g1-deploy, deploy_5G WiFi, Hugging Face or g1_xmls cache failures, sim2real root project setup, any4hdmi sync to G1, teleop uv sync, GMR or smplx GitHub hangs, XRoboToolkit, xrobotoolkit_sdk, or JetPack 5 G1 deployment bringup.
---

# Configure G1 sim2real

Use this skill for G1 onboard setup and repair on JetPack 5 / Ubuntu 20.04
hosts such as `g1-cable` or `g1-deploy`.

## Operating Rules

1. Treat the local checkout as the source of truth for code. Use
   `sim2real/sync-g1.sh` or explicit `rsync` for deployment. Do not make
   persistent source edits only on the robot unless the change is a
   host-local environment workaround.
2. Use `g1-deploy` when Ethernet is disconnected. Use `bash -lc` over SSH so
   `uv`, `.profile`, and robot shell defaults load.
3. Keep root policy deployment and teleop separate:
   - root project: `~/sim2real` plus `~/any4hdmi`, `uv sync --group g1`,
     CycloneDDS, HF asset cache, ONNX inference smoke.
   - teleop project: `~/sim2real/venv/teleop`, GMR/smplx dependencies,
     XRoboToolkit service and `xrobotoolkit_sdk`.
4. If a command needs `sudo`, provide the exact command and use the user's
   interactive sudo tmux only when available. Do not hide a blocked sudo prompt
   behind a "done" claim.

## Workflow

1. Establish network access.
   - If WiFi is not connected, ask the operator to run:
     `sudo nmcli dev wifi connect deploy_5G password 88888888 ifname wlan0`
     and then set low route metrics for `deploy_5G`.
   - If Hugging Face or GitHub hangs, create a reverse proxy tunnel from the
     local machine to the robot, normally local `127.0.0.1:7890` to remote
     `127.0.0.1:7891`.
2. Fix Hugging Face first.
   - Ensure `elijahgalahad/g1_xmls` is cached on the robot.
   - Set persistent robot shell defaults for `HF_ENDPOINT`,
     `HF_HUB_DISABLE_TELEMETRY`, and deploy-time `HF_HUB_OFFLINE=1`.
   - Verify with `mjhub.resolve_asset_reference(...)`, not only by listing the
     cache directory.
3. Configure the root project.
   - Build or locate CycloneDDS, export `CYCLONEDDS_HOME` and
     `LD_LIBRARY_PATH`, run `uv sync --group g1`, then verify imports.
   - Run the onboard ONNX CPU benchmark before claiming root deployment is
     ready.
4. Sync motion assets.
   - Put G1-targeted motion datasets under `any4hdmi/output/g1/...`.
   - Add explicit include rules to `sim2real/sync-g1.sh` for any new dataset
     path and run the sync against `G1_HOST=g1-deploy`.
   - If the sync script excludes `sync*.sh`, copy the updated sync script to
     the robot explicitly after editing it.
5. Configure teleop.
   - Run `uv --project venv/teleop sync`.
   - If remote GitHub fetches hang, seed local source copies for GMR and smplx
     on the robot and patch only the robot's teleop dependency files to
     `file:///home/elijah/src/...`.
   - Install the XRoboToolkit service `.deb` with sudo, copy/build the
     pybind project, and verify `xrobotoolkit_sdk` import in the teleop env.
6. Read `references/g1-setup-runbook.md` for concrete commands and failure
   mappings before touching a live robot.

## Completion Checks

- `ssh g1-deploy 'bash -lc "echo $HF_HUB_OFFLINE $HF_ENDPOINT; uv --version"'`
  shows the intended shell defaults and `uv`.
- `mjhub.resolve_asset_reference("hf://elijahgalahad/g1_xmls@main/g1-mode_13_15.xml")`
  returns a local cached file.
- `cd ~/sim2real && uv sync --group g1` is complete and the root env imports
  `sim2real`, `any4hdmi`, `mujoco`, `onnxruntime`, `cyclonedds`,
  `unitree_sdk2py`, and `unitree_interface`.
- `scripts/test_policy_inference.py` runs with `--inference_backend onnx-cpu`
  on the robot.
- `uv --project venv/teleop run python -c 'import xrobotoolkit_sdk'` succeeds.
- Teleop scripts at least compile with `py_compile`; real teleop still needs
  the XRoboToolkit service running and hardware state ready.

## Failure Reading

- `low state not ready` after policy startup is a robot/bridge low-state issue,
  not a Hugging Face issue.
- `Keyboard listener stopped unexpectedly: Inappropriate ioctl` during SSH
  smoke tests usually means noninteractive SSH and is not itself a deploy
  failure.
- GitHub dependency hangs during `uv sync` are better handled by local source
  seeding than by waiting indefinitely on the robot network.
- `runtime_version.h` missing or protobuf `stubs` compile failures in
  XRoboToolkit mean the SDK is using the wrong or incomplete bundled gRPC tree.
