from __future__ import annotations

import argparse
import time

from loguru import logger


def switch_to_debug_mode(
    *, domain_id: int, interface: str, release_delay_s: float
) -> None:
    if release_delay_s < 0:
        raise ValueError("release_delay_s must be >= 0")

    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
        MotionSwitcherClient,
    )
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize

    ChannelFactoryInitialize(domain_id, interface)
    motion_switcher = MotionSwitcherClient()
    motion_switcher.SetTimeout(5.0)
    motion_switcher.Init()

    while True:
        status, result = motion_switcher.CheckMode()
        if status != 0 or result is None:
            raise RuntimeError(
                "Failed to query the G1 motion mode before inline control: "
                f"status={status} result={result!r}"
            )
        if not result.get("name"):
            logger.info("G1 inline robot is in debug mode")
            return

        logger.info("Releasing active G1 motion mode: {}", result.get("name"))
        release_status, _ = motion_switcher.ReleaseMode()
        if release_status != 0:
            raise RuntimeError(
                "Failed to release the active G1 motion mode before inline control: "
                f"status={release_status} mode={result.get('name')!r}"
            )
        if release_delay_s > 0:
            time.sleep(release_delay_s)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain-id", type=int, required=True)
    parser.add_argument("--interface", required=True)
    parser.add_argument("--release-delay-s", type=float, default=1.0)
    args = parser.parse_args()
    switch_to_debug_mode(
        domain_id=args.domain_id,
        interface=args.interface,
        release_delay_s=args.release_delay_s,
    )


if __name__ == "__main__":
    main()
