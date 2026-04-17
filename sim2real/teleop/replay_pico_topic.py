#!/usr/bin/env python3
"""
Replay a recorded pico_retarget_pub.py topic capture over ZMQ.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import select
import sys
import termios
import threading
import time
from pathlib import Path
from typing import Any, Literal
import tty

import numpy as np
import tyro
import zmq

from sim2real.config.robots.base import PUBLISH_T_NS_KEY, SEQ_KEY, SMPLX_T_NS_KEY


def _load_frames(input_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    with np.load(input_path, allow_pickle=True) as raw:
        metadata = {}
        if "metadata_json" in raw:
            metadata = json.loads(str(raw["metadata_json"].item()))
        if "frames" not in raw:
            raise ValueError(f"{input_path} does not contain 'frames'")
        frames = raw["frames"].tolist()
        if not isinstance(frames, list):
            raise ValueError(f"{input_path} frames payload is not a list")
        payloads = [frame for frame in frames if isinstance(frame, dict)]
        if len(payloads) != len(frames):
            raise ValueError(f"{input_path} contains non-dict frames")

        if PUBLISH_T_NS_KEY in raw:
            timestamps_ns = np.asarray(raw[PUBLISH_T_NS_KEY], dtype=np.int64)
        elif SMPLX_T_NS_KEY in raw:
            timestamps_ns = np.asarray(raw[SMPLX_T_NS_KEY], dtype=np.int64)
        elif "receive_time_ns" in raw:
            timestamps_ns = np.asarray(raw["receive_time_ns"], dtype=np.int64)
        else:
            timestamps_ns = np.zeros((len(payloads),), dtype=np.int64)

    if timestamps_ns.shape[0] != len(payloads):
        raise ValueError("Timestamp count mismatch in recording")
    return metadata, payloads, timestamps_ns


@dataclass
class Args:
    """Replay a recorded pico_retarget_pub topic capture over ZMQ."""

    input: Path
    bind: str = "tcp://*:28702"
    hwm: int = 1
    fps: float = 0.0
    speed: float = 1.0
    loop: bool = False
    timing_key: Literal["publish_t_ns", "smplx_t_ns", "receive_time_ns", "auto"] = "auto"


def _resolve_timestamps_ns(
    *,
    input_path: Path,
    timing_key: str,
    frame_count: int,
) -> np.ndarray:
    with np.load(input_path, allow_pickle=True) as raw:
        if timing_key == "auto":
            for key in (PUBLISH_T_NS_KEY, SMPLX_T_NS_KEY, "receive_time_ns"):
                if key in raw:
                    values = np.asarray(raw[key], dtype=np.int64)
                    if values.shape[0] == frame_count:
                        return values
            return np.zeros((frame_count,), dtype=np.int64)
        if timing_key not in raw:
            raise ValueError(f"{input_path} does not contain {timing_key!r}")
        values = np.asarray(raw[timing_key], dtype=np.int64)
        if values.shape[0] != frame_count:
            raise ValueError(f"{timing_key} count mismatch in {input_path}")
        return values


class SpacePauseController:
    def __init__(self) -> None:
        self.paused = False
        self._running = False
        self._thread: threading.Thread | None = None
        self._fd: int | None = None
        self._termios_attrs: list[Any] | None = None

    def start(self) -> None:
        if not sys.stdin.isatty():
            print("[replay] stdin is not a TTY; keyboard pause disabled")
            return

        self._fd = sys.stdin.fileno()
        self._termios_attrs = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self._running = True
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()
        print("[replay] keyboard control enabled: press space to pause/resume")

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=0.2)
        if self._fd is not None and self._termios_attrs is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._termios_attrs)

    def wait_if_paused(self) -> None:
        while self.paused:
            time.sleep(0.05)

    def sleep_with_pause(self, dt_s: float) -> None:
        deadline = time.perf_counter() + max(0.0, dt_s)
        while True:
            self.wait_if_paused()
            remaining = deadline - time.perf_counter()
            if remaining <= 0.0:
                return
            time.sleep(min(0.01, remaining))

    def _listen(self) -> None:
        assert self._fd is not None
        while self._running:
            ready, _, _ = select.select([self._fd], [], [], 0.1)
            if not ready:
                continue
            try:
                key = sys.stdin.read(1)
            except Exception:
                continue
            if key == " ":
                self.paused = not self.paused
                print(f"[replay] paused={self.paused}")


def main(args: Args) -> None:
    metadata, frames, auto_timestamps_ns = _load_frames(args.input)
    if not frames:
        raise RuntimeError(f"No frames found in {args.input}")

    timestamps_ns = (
        auto_timestamps_ns
        if args.timing_key == "auto"
        else _resolve_timestamps_ns(
            input_path=args.input,
            timing_key=str(args.timing_key),
            frame_count=len(frames),
        )
    )

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUB)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.SNDHWM, int(args.hwm))
    sock.setsockopt(zmq.CONFLATE, 1)
    sock.bind(args.bind)
    pause_controller = SpacePauseController()
    pause_controller.start()

    print(f"[replay] input={args.input}")
    print(f"[replay] bind={args.bind}")
    print(f"[replay] frames={len(frames)}")
    if metadata:
        print(f"[replay] schema={metadata.get('schema', 'unknown')}")

    try:
        while True:
            replay_start = time.perf_counter()
            first_timestamp_ns = int(timestamps_ns[0]) if timestamps_ns.size else 0

            for frame_idx, payload in enumerate(frames):
                pause_controller.wait_if_paused()
                payload_out = dict(payload)
                if SEQ_KEY in payload_out:
                    payload_out[SEQ_KEY] = int(frame_idx)
                sock.send_string(
                    json.dumps(payload_out, separators=(",", ":")),
                    flags=zmq.NOBLOCK,
                )

                if frame_idx + 1 >= len(frames):
                    continue

                if args.fps > 0:
                    dt_s = 1.0 / float(args.fps)
                else:
                    curr_t_ns = int(timestamps_ns[frame_idx])
                    next_t_ns = int(timestamps_ns[frame_idx + 1])
                    if curr_t_ns <= 0 or next_t_ns <= 0 or next_t_ns <= curr_t_ns:
                        dt_s = 0.0
                    else:
                        dt_s = (next_t_ns - curr_t_ns) / 1e9

                if dt_s > 0.0:
                    pause_controller.sleep_with_pause(dt_s / max(float(args.speed), 1e-6))

            elapsed = time.perf_counter() - replay_start
            print(
                f"[replay] published {len(frames)} frames in {elapsed:.3f}s "
                f"(loop={args.loop})"
            )
            if not args.loop:
                break
            if first_timestamp_ns <= 0 and args.fps <= 0:
                time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received, stopping replay...")
    finally:
        pause_controller.stop()
        sock.close(0)


if __name__ == "__main__":
    main(tyro.cli(Args))
