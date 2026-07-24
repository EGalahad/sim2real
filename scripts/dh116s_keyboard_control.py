#!/usr/bin/env python3
"""Interactively open and close DH116S hands with comma and period keys."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
import sys
import termios
import tty

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim2real.utils.dh116s import (
    DEFAULT_ANGULAR_VELOCITY,
    DEFAULT_HOME_WAIT_TIME,
    DEFAULT_LEFT_DEVICE_INDEX,
    DEFAULT_LEFT_NODE_ID,
    DEFAULT_MAX_CURRENT,
    DEFAULT_RIGHT_DEVICE_INDEX,
    DEFAULT_RIGHT_NODE_ID,
    DEFAULT_SDK_DIR,
    DH116SHandDriver,
)


OPEN_KEY = ","
CLOSE_KEY = "."
RESET_KEY = "0"
QUIT_KEYS = {"q", "Q", "\x03"}


def next_grip(current: float, key: str, step: float) -> float:
    """Return the normalized grip after applying one keyboard command."""
    if not np.isfinite(current) or not 0.0 <= current <= 1.0:
        raise ValueError("current grip must be finite and within [0, 1]")
    if not np.isfinite(step) or not 0.0 < step <= 1.0:
        raise ValueError("step must be finite and within (0, 1]")
    if key == OPEN_KEY:
        return max(0.0, current - step)
    if key == CLOSE_KEY:
        return min(1.0, current + step)
    if key == RESET_KEY:
        return 0.0
    return current


@contextmanager
def raw_keyboard(stream):
    """Temporarily read one key at a time and always restore terminal state."""
    file_descriptor = stream.fileno()
    previous = termios.tcgetattr(file_descriptor)
    try:
        tty.setcbreak(file_descriptor)
        yield
    finally:
        termios.tcsetattr(file_descriptor, termios.TCSADRAIN, previous)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand-dir", choices=["left", "right", "double"], default="double")
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--sdk-dir", default=DEFAULT_SDK_DIR)
    parser.add_argument("--left-node-id", type=int, default=DEFAULT_LEFT_NODE_ID)
    parser.add_argument("--right-node-id", type=int, default=DEFAULT_RIGHT_NODE_ID)
    parser.add_argument("--left-device-index", type=int, default=DEFAULT_LEFT_DEVICE_INDEX)
    parser.add_argument("--right-device-index", type=int, default=DEFAULT_RIGHT_DEVICE_INDEX)
    parser.add_argument("--max-current", type=int, default=DEFAULT_MAX_CURRENT)
    parser.add_argument("--angular-velocity", type=float, default=DEFAULT_ANGULAR_VELOCITY)
    parser.add_argument("--home-wait-time", type=float, default=DEFAULT_HOME_WAIT_TIME)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not sys.stdin.isatty():
        raise RuntimeError("keyboard control requires an interactive terminal")
    # Validate before connecting or homing hardware.
    next_grip(0.0, "", args.step)

    print(
        "[DH116S keyboard] startup connects, enables, and homes the selected hand(s).",
        flush=True,
    )
    driver = DH116SHandDriver(
        hand_dir=args.hand_dir,
        sdk_dir=args.sdk_dir,
        left_node_id=args.left_node_id,
        right_node_id=args.right_node_id,
        left_device_index=args.left_device_index,
        right_device_index=args.right_device_index,
        max_current=args.max_current,
        angular_velocity=args.angular_velocity,
        home_wait_time=args.home_wait_time,
    )
    grip = 0.0
    try:
        driver.apply_grips(grip, grip)
        print(
            "[DH116S keyboard] grip=0.00 (open/zero). "
            "Keys: ',' open, '.' close, '0' reset, 'q' quit.",
            flush=True,
        )
        with raw_keyboard(sys.stdin):
            while True:
                key = sys.stdin.read(1)
                if key in QUIT_KEYS:
                    break
                updated = next_grip(grip, key, args.step)
                if updated == grip and key not in {OPEN_KEY, CLOSE_KEY, RESET_KEY}:
                    continue
                grip = updated
                driver.apply_grips(grip, grip)
                print(f"\r[DH116S keyboard] grip={grip:.2f}   ", end="", flush=True)
    finally:
        print("\n[DH116S keyboard] disconnecting without another motion command.", flush=True)
        driver.close()


if __name__ == "__main__":
    main()
