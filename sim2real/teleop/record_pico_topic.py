#!/usr/bin/env python3
"""
Record the full ZMQ payload stream published by pico_retarget_pub.py.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import tyro
import zmq

from sim2real.config.robots.base import PUBLISH_T_NS_KEY, SEQ_KEY, SMPLX_T_NS_KEY


def _empty_recording() -> dict[str, np.ndarray]:
    return {
        "frames": np.empty((0,), dtype=object),
        "receive_time_ns": np.empty((0,), dtype=np.int64),
        PUBLISH_T_NS_KEY: np.empty((0,), dtype=np.int64),
        SMPLX_T_NS_KEY: np.empty((0,), dtype=np.int64),
        SEQ_KEY: np.empty((0,), dtype=np.int64),
    }


def _save_recording(
    output_path: Path,
    *,
    connect: str,
    frames: list[dict[str, Any]],
    receive_time_ns: list[int],
    publish_time_ns: list[int],
    smplx_time_ns: list[int],
    seq_list: list[int],
) -> None:
    if frames:
        arrays = {
            "frames": np.asarray(frames, dtype=object),
            "receive_time_ns": np.asarray(receive_time_ns, dtype=np.int64),
            PUBLISH_T_NS_KEY: np.asarray(publish_time_ns, dtype=np.int64),
            SMPLX_T_NS_KEY: np.asarray(smplx_time_ns, dtype=np.int64),
            SEQ_KEY: np.asarray(seq_list, dtype=np.int64),
        }
    else:
        arrays = _empty_recording()

    metadata = {
        "schema": "pico_retarget_topic_v1",
        "source": "pico_retarget_pub.py",
        "connect": str(connect),
        "frame_count": int(len(frames)),
        "recorded_at_unix_ns": int(time.time_ns()),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        metadata_json=np.array(json.dumps(metadata)),
        **arrays,
    )


@dataclass
class Args:
    """Record the full ZMQ topic from pico_retarget_pub.py into npz."""

    connect: str = "tcp://127.0.0.1:28701"
    output: Path | None = None
    hwm: int = 4096


def main(args: Args) -> None:
    output_path = args.output
    if output_path is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = Path.cwd() / f"pico_topic_{timestamp}.npz"

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVHWM, int(args.hwm))
    sock.connect(args.connect)
    sock.setsockopt(zmq.SUBSCRIBE, b"")

    frames: list[dict[str, Any]] = []
    receive_time_ns: list[int] = []
    publish_time_ns: list[int] = []
    smplx_time_ns: list[int] = []
    seq_list: list[int] = []
    invalid_frames = 0
    start_monotonic = time.monotonic()

    print(f"[record] connect={args.connect}")
    print(f"[record] output={output_path}")
    print("Recording full pico_retarget_pub topic. Press Ctrl-C to stop.")

    try:
        while True:
            raw = sock.recv_string()
            receive_t_ns = time.time_ns()
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                invalid_frames += 1
                print(f"[record] bad JSON payload: {exc}")
                continue
            if not isinstance(payload, dict):
                invalid_frames += 1
                print("[record] non-dict payload skipped")
                continue

            frames.append(payload)
            receive_time_ns.append(int(receive_t_ns))
            publish_time_ns.append(int(payload.get(PUBLISH_T_NS_KEY, 0) or 0))
            smplx_time_ns.append(int(payload.get(SMPLX_T_NS_KEY, 0) or 0))
            seq_list.append(int(payload.get(SEQ_KEY, len(frames) - 1) or 0))

            if len(frames) % 50 == 0:
                elapsed = max(1e-6, time.monotonic() - start_monotonic)
                print(
                    f"[record] frames={len(frames)} invalid={invalid_frames} "
                    f"recv_fps={len(frames) / elapsed:.2f}"
                )
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received, saving topic recording...")
    finally:
        sock.close(0)

    _save_recording(
        output_path,
        connect=args.connect,
        frames=frames,
        receive_time_ns=receive_time_ns,
        publish_time_ns=publish_time_ns,
        smplx_time_ns=smplx_time_ns,
        seq_list=seq_list,
    )

    print(f"[record] saved {len(frames)} frames to {output_path}")
    print(f"[record] invalid frames: {invalid_frames}")


if __name__ == "__main__":
    main(tyro.cli(Args))
