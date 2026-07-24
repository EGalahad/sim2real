# DH116S LHandPro SDK

This directory installs the DH116S SDK inside the sim2real checkout. The
hardware controller defaults to `third_party/dh116s_sdk/python`; it no longer
requires `~/lhandpro_project`.

## SDK source

The newest Linux SDK currently present in `Roboparty/RP_teleoperate_ygx` is
`LHandProLib-API-Linux-20260325`, added in commit `5a5407c2a6a6` on
2026-04-09. The package contains `aarch64`, `x86_64`, and `i386` builds. The
installer downloads it with the authenticated GitHub CLI because the source
repository is private.

The Feishu manual page could be opened but its content and attachments were not
available to the automation session. Therefore, `20260325` is verified as the
latest version in the ygx repository, not claimed as the latest attachment on
Feishu.

## Install

From the sim2real repository root:

```bash
./third_party/dh116s_sdk/install.sh
```

The script detects `aarch64`, `x86_64`, or `i386`, caches the untouched vendor
package under `vendor/`, and creates the importable package under `python/`.
Both generated directories are intentionally ignored by Git.

The CAN-FD Python layer also needs `python-can~=4.6.1`. The dedicated minimal
environment is defined in `venv/dh116s/pyproject.toml`:

```bash
uv sync --project venv/dh116s
```

Alternatively, let the installer install it into a selected interpreter:

```bash
./third_party/dh116s_sdk/install.sh \
  --with-python-deps \
  --python /path/to/python
```

For an SDK directory copied from Feishu or another machine, avoid GitHub access
with:

```bash
./third_party/dh116s_sdk/install.sh --source /path/to/LHandProLib-API-Linux-20260325
```

Use `--force` to replace both the cached payload and generated Python package.

Before hardware startup, configure both SocketCAN interfaces once. The
repo-local compatibility layer deliberately does not reload `gs_usb` while
connecting each hand, because doing so would disconnect the first adapter while
the second hand initializes:

```bash
sudo modprobe gs_usb
echo 'a8fa 8598' | sudo tee /sys/bus/usb/drivers/gs_usb/new_id
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000 dbitrate 5000000 fd on
sudo ip link set can0 up
sudo ip link set can1 down
sudo ip link set can1 type can bitrate 1000000 dbitrate 5000000 fd on
sudo ip link set can1 up
```

If `new_id` reports that the device is already registered, continue after
confirming that `can0` and `can1` exist.

On Jetson systems, the onboard MTT CAN controller may already own `can0`. After
binding both USB adapters, inspect `ip -details link show type can` and pass the
actual sorted interface indices with `--left-device-index` and
`--right-device-index`. Do not assume the defaults until this mapping has been
verified for the host.

## Verify without moving hardware

The installer checks Python syntax and loads the architecture-specific shared
library. Check the final import after installing `python-can`:

```bash
PYTHONPATH=third_party/dh116s_sdk/python uv run python -c \
  'from lhandprolib_python_sdk.controller import LHandProController; print(LHandProController)'
```

The sim2real transport can be tested without importing the SDK or touching CAN:

```bash
uv run scripts/dh116s_control.py --dry-run --hand-dir double
```

After staged hardware bring-up, the direct keyboard controller is available as:

```bash
uv run --project venv/dh116s --no-sync \
  scripts/dh116s_keyboard_control.py --hand-dir double
```

It homes and enters normalized grip `0.0` at startup. Press `,` to open one
step, `.` to close one step, `0` to return to zero, and `q` to disconnect.
The default step is `0.05`; change it with `--step`.

## Hardware warning

Non-dry-run startup connects, clears alarms, enables motors, and homes every
requested hand. Clear the hand workspace first. Stop disconnects the SDK but
does not command the hands open.
