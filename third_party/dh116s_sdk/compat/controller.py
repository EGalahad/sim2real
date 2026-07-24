"""Stable controller API over the upstream LHandProLib CAN-FD example."""

from __future__ import annotations

import subprocess

from . import lhandpro_controller as _upstream


def _setup_can_interface_without_driver_reload(
    _canfd,
    ifname: str,
    nom_baudrate: int,
    dat_baudrate: int,
) -> bool:
    """Configure one SocketCAN interface without resetting the other hand."""
    details = subprocess.run(
        ["ip", "-details", "link", "show", ifname],
        capture_output=True,
        text=True,
        check=False,
    )
    first_line = details.stdout.splitlines()[0] if details.stdout else ""
    flags = first_line.partition("<")[2].partition(">")[0].split(",")
    already_ready = (
        details.returncode == 0
        and "UP" in flags
        and f"bitrate {nom_baudrate}" in details.stdout
        and f"dbitrate {dat_baudrate}" in details.stdout
    )
    if already_ready:
        return True

    commands = (
        ["ip", "link", "set", ifname, "down"],
        [
            "ip",
            "link",
            "set",
            ifname,
            "type",
            "can",
            "bitrate",
            str(nom_baudrate),
            "dbitrate",
            str(dat_baudrate),
            "fd",
            "on",
            "loopback",
            "off",
            "listen-only",
            "off",
        ],
        ["ip", "link", "set", ifname, "up"],
    )
    for command in commands:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            print(f"Failed to configure {ifname}: {result.stderr.strip()}")
            return False
    return True


# The vendor example reloads gs_usb on every connect(), which disconnects the
# first adapter while initializing the second hand. Configure only the selected
# interface so dual-hand startup does not reset its peer.
if hasattr(_upstream.CANFD, "_setup_can_interface"):
    _upstream.CANFD._setup_can_interface = _setup_can_interface_without_driver_reload


class LHandProController(_upstream.LHandProController):
    """Expose the node-aware constructor used by sim2real's DH116S driver."""

    def __init__(self, canfd_node_id: int = 1) -> None:
        node_id = int(canfd_node_id)
        if not 1 <= node_id <= 127:
            raise ValueError("canfd_node_id must be within [1, 127]")
        self.canfd_node_id = node_id
        super().__init__("CANFD")

    def connect(self, *args, **kwargs) -> bool:
        # The vendor example exposes its node ID through a module-level setting.
        # connect() is called sequentially during dual-hand startup, so setting it
        # immediately before the upstream initial_ex() call is deterministic.
        _upstream.CANFD_NODE_ID = self.canfd_node_id
        return super().connect(*args, **kwargs)

    def _canfd_receive_callback(self, msg) -> None:
        if self.lhp and self.is_connected:
            expected_id = 0x480 + self.canfd_node_id
            if msg["id"] == expected_id:
                self.lhp.set_canfd_data_decode(msg["id"], msg["data"])
