---
name: configure-g1-sim2real
description: Install, repair, and verify sim2real on G1 robot computers. Use for root uv environments and inference extras, robot-g1 dependencies, CycloneDDS builds, G1 network interfaces, Hugging Face offline assets, ONNX Runtime compatibility, inline RobotIO or MotionSwitcher DDS failures, local-to-robot sync, isolated PICO or DH116S environments, and G1 deployment bring-up.
---

# Configure G1 sim2real

Configure the G1 host from current evidence. Do not infer its OS, network
interface, Python environment, or GPU runtime from its SSH alias.

## Operating rules

1. Treat the local checkout as the source of truth. Make source edits locally,
   then use the repository sync script. Never sync `.venv` directories.
2. Run `scripts/inspect_host.sh` before installing or repairing anything.
3. Keep these environments separate:
   - root project: policy, simulation, ZMQ RobotIO, and optional G1 inline SDK;
   - `venv/pico`: XRoboToolkit, GMR, SMPL-X, and PICO publisher;
   - `venv/dh116s`: LHandPro/DH116S control.
4. Default G1 deployment to CPU inference unless the user explicitly requests
   GPU inference. A listed CUDA provider is not proof that its shared libraries
   load.
5. Never store sudo, Wi-Fi, user, or SDK credentials in this skill or the
   repository. Let the operator enter secrets interactively.
6. Do not start a policy, switch modes, home a hand, or send motor commands
   during installation unless the user explicitly requests a hardware test.
7. Use `bash -lc` for remote commands, but explicitly source the sim2real
   environment file. Do not assume variables appended to `.bashrc` are visible
   to noninteractive shells.

## Workflow

### 1. Inspect and choose a profile

Run locally against the checkout:

```bash
ssh <host> 'bash -s -- "$HOME/sim2real"' \
  < .agents/skills/configure-g1-sim2real/scripts/inspect_host.sh
```

Record the OS, architecture, Python, uv, disk space, interfaces,
`CYCLONEDDS_HOME`, HF settings, native library paths, and existing environments.
Read `references/g1-setup-runbook.md` before modifying a live host.

Choose exactly one root profile:

- generic/ZMQ CPU: `--extra inference-cpu`
- G1 inline CPU: `--extra inference-cpu --extra robot-g1`
- G1 inline GPU: `--extra inference-gpu --extra robot-g1`

Do not install both inference extras in one environment.

### 2. Sync code without environments or large artifacts

Use the current repository sync entry point, normally:

```bash
G1_HOST=<host> SYNC_CHECKPOINTS=0 SYNC_ANY4HDMI=0 ./sync-robot.sh g1
```

Inspect the sync script before running it. Preserve remote-only checkpoints and
exclude `.venv`, caches, logs, and large motion datasets unless the user placed
them in scope. Transfer large assets separately and resumably.

### 3. Configure native and shell dependencies

For `robot-g1`, install CycloneDDS first if the required installation is
missing. Put persistent variables in `~/.config/sim2real/env.sh`, source it
from `~/.profile`, and source it explicitly in automation:

```bash
source "$HOME/.config/sim2real/env.sh"
```

Verify the variables with a fresh `bash -lc`; checking the current interactive
shell is insufficient.

### 4. Install the selected root profile

Run from `~/sim2real`:

```bash
source "$HOME/.config/sim2real/env.sh"
uv sync --extra inference-cpu --extra robot-g1
```

Replace `inference-cpu` only when the selected profile requires another
backend. Do not use the legacy `--group g1`.

If Git or package downloads hang, use a bounded HTTP(S) proxy tunnel or seed
the required source/cache from the local machine. Do not wait indefinitely.

### 5. Prepare deploy-time assets

Resolve required Hugging Face assets once while online, then run deployments
with `HF_HUB_OFFLINE=1` and `HF_HUB_DISABLE_TELEMETRY=1`. Validate with the
actual `mjhub.resolve_asset_reference()` call rather than cache directory
names.

Treat an HF stall, an ONNX model-format error, a missing CUDA library, a DDS
interface error, and missing low state as different failures. Use the failure
map in the runbook.

### 6. Install isolated peripherals only when requested

Use:

```bash
uv sync --project venv/pico
uv sync --project venv/dh116s
```

Keep XRoboToolkit/GMR native dependencies out of root. Keep LHandPro libraries
out of root. Verify PICO and DH116S with imports or dry-run modes before any
hardware action.

### 7. Verify without commanding hardware

Run the verifier inside the selected root environment:

```bash
uv run --no-sync python \
  .agents/skills/configure-g1-sim2real/scripts/verify_install.py \
  --profile g1-cpu \
  --asset hf://elijahgalahad/g1_xmls@main/g1-mode_13_15.xml
```

For a policy artifact, add `--onnx <path>`. The verifier imports dependencies,
resolves optional cached assets, and loads the ONNX session; it never creates
RobotIO or sends commands.

When the skill files have not been synced yet, stream the verifier instead of
assuming its remote path exists:

```bash
ssh <host> 'bash -lc '"'"'
  cd "$HOME/sim2real"
  source "$HOME/.config/sim2real/env.sh"
  uv run --no-sync python - --profile g1-cpu
'"'"'' < .agents/skills/configure-g1-sim2real/scripts/verify_install.py
```

Only after this passes, and only with user authorization, test inline startup
using the real robot interface. Confirm no old `tracking.py` or
`real_bridge.py` process is active first.

## Completion report

Report:

- host, OS, architecture, Python, uv, and selected install profile;
- exact sync and install commands used;
- installed, skipped, and failed components;
- environment variables visible in a fresh login shell;
- import, HF resolver, and ONNX load results;
- whether any real hardware action was intentionally not tested;
- a directly runnable next command.

Do not claim readiness from a successful `uv sync` alone.
