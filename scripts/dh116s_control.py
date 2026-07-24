#!/usr/bin/env python3
"""Receive normalized PICO grip commands over ZMQ and control DH116S hands."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import zmq

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim2real.utils.common import PORTS
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
    DryRunDH116SHandDriver,
    HandGripConsumer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--connect",
        default=f"tcp://127.0.0.1:{PORTS['hand_grip']}",
        help="HandGripMessage ZMQ publisher endpoint.",
    )
    parser.add_argument("--hand-dir", choices=["left", "right", "double"], default="double")
    parser.add_argument("--sdk-dir", default=DEFAULT_SDK_DIR)
    parser.add_argument("--left-node-id", type=int, default=DEFAULT_LEFT_NODE_ID)
    parser.add_argument("--right-node-id", type=int, default=DEFAULT_RIGHT_NODE_ID)
    parser.add_argument("--left-device-index", type=int, default=DEFAULT_LEFT_DEVICE_INDEX)
    parser.add_argument("--right-device-index", type=int, default=DEFAULT_RIGHT_DEVICE_INDEX)
    parser.add_argument("--max-current", type=int, default=DEFAULT_MAX_CURRENT)
    parser.add_argument("--angular-velocity", type=float, default=DEFAULT_ANGULAR_VELOCITY)
    parser.add_argument("--home-wait-time", type=float, default=DEFAULT_HOME_WAIT_TIME)
    parser.add_argument("--hwm", type=int, default=1)
    parser.add_argument("--stale-timeout-s", type=float, default=1.0)
    parser.add_argument("--status-interval-s", type=float, default=2.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Decode and map commands without importing the hand SDK or moving hardware.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stale_timeout_s <= 0 or args.status_interval_s <= 0:
        raise ValueError("stale-timeout-s and status-interval-s must be > 0")

    driver = (
        DryRunDH116SHandDriver(hand_dir=args.hand_dir)
        if args.dry_run
        else DH116SHandDriver(
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
    )
    consumer = HandGripConsumer(driver)

    socket = zmq.Context.instance().socket(zmq.SUB)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVHWM, int(args.hwm))
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.setsockopt(zmq.SUBSCRIBE, b"")
    socket.connect(args.connect)

    print(
        f"[DH116S] connect={args.connect} hand_dir={args.hand_dir} "
        f"dry_run={args.dry_run}; non-dry-run startup includes automatic homing",
        flush=True,
    )
    last_status = 0.0
    stale = False
    try:
        while True:
            try:
                raw = socket.recv(flags=zmq.DONTWAIT)
            except zmq.Again:
                raw = None

            now = time.monotonic()
            if raw is not None:
                try:
                    message = consumer.consume(raw, received_monotonic=now)
                except Exception as exc:
                    print(f"[DH116S] ignored invalid command: {exc}", flush=True)
                else:
                    if stale:
                        print("[DH116S] grip stream recovered", flush=True)
                    stale = False
                    if args.dry_run and now - last_status >= args.status_interval_s:
                        print(
                            f"[DH116S dry-run] left={message.left_grip:.3f} "
                            f"right={message.right_grip:.3f}",
                            flush=True,
                        )
                        last_status = now

            if consumer.is_stale(now_monotonic=now, timeout_s=args.stale_timeout_s):
                stale = True
                if now - last_status >= args.status_interval_s:
                    print(
                        "[DH116S] grip stream stale; holding the last commanded pose",
                        flush=True,
                    )
                    last_status = now
            time.sleep(0.005)
    except KeyboardInterrupt:
        print("[DH116S] KeyboardInterrupt, disconnecting without opening hands", flush=True)
    finally:
        socket.close(0)
        driver.close()


if __name__ == "__main__":
    main()
