#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: sudo bash scripts/setup/setup_g1_hotspot_via_laptop.sh [options]

Create a G1 Wi-Fi hotspot whose internet default route points at the laptop.
Run this on g1-rp. Run setup_laptop_g1_gateway.sh on the laptop after the AP is up.
Other devices connected to the hotspot receive DHCP leases from G1 and are
relayed through G1 to the laptop internet gateway.
Intended flow: boot G1 in normal Wi-Fi client mode, SSH in over that network,
then run this command manually to switch the G1 Wi-Fi interface into hotspot
mode. Reconnect from the laptop through the g1-hotspot SSH host after the
laptop joins the hotspot.
This changes G1 network interfaces. Keep an Ethernet cable connected to G1
before running it so you do not lose access if Wi-Fi setup fails.
The hotspot profile is intentionally not set to autoconnect, so it will not
take over the built-in Wi-Fi interface on the next boot.

Required:
  --interface IFACE          G1 Wi-Fi interface to use for AP mode, e.g. wlan0

Options:
  --ssid SSID                Hotspot SSID (default: hdmi-deploy)
  --password PASSWORD        WPA-PSK password, 8+ chars (default: hdmi1234)
  --address CIDR             G1 AP IPv4 address (default: 10.42.7.1/24)
  --laptop-ip IP             Laptop IP on this hotspot subnet (default: 10.42.7.2)
  --dns DNS                  DNS server for G1 while using laptop egress (default: 8.8.8.8)
  --dhcp-start IP            First DHCP lease address for hotspot clients (default: subnet + 20)
  --dhcp-end IP              Last DHCP lease address for hotspot clients (default: subnet + 200)
  --dhcp-lease-time TIME     dnsmasq DHCP lease time (default: 12h)
  --connection NAME          NetworkManager connection name (default: hdmi-g1-ap-via-laptop)
  --channel CHANNEL          Wi-Fi channel (default: 6)
  --band BAND                Wi-Fi band for NetworkManager, e.g. bg/a (default: bg)
  --route-metric METRIC      Default-route metric via laptop (default: 50)
  -h, --help                 Show this help
EOF
}

AP_IFACE=""
SSID="hdmi-deploy"
PASSWORD="hdmi1234"
ADDRESS="10.42.7.1/24"
LAPTOP_IP="10.42.7.2"
DNS_SERVER="8.8.8.8"
DHCP_START=""
DHCP_END=""
DHCP_LEASE_TIME="12h"
CONNECTION_NAME="hdmi-g1-ap-via-laptop"
CHANNEL="6"
BAND="bg"
ROUTE_METRIC="50"
ORIGINAL_ARGS=("$@")

while [[ $# -gt 0 ]]; do
  case "$1" in
    --interface)
      [[ $# -ge 2 ]] || { echo "Missing value for --interface" >&2; exit 1; }
      AP_IFACE="$2"
      shift 2
      ;;
    --ssid)
      [[ $# -ge 2 ]] || { echo "Missing value for --ssid" >&2; exit 1; }
      SSID="$2"
      shift 2
      ;;
    --password)
      [[ $# -ge 2 ]] || { echo "Missing value for --password" >&2; exit 1; }
      PASSWORD="$2"
      shift 2
      ;;
    --address)
      [[ $# -ge 2 ]] || { echo "Missing value for --address" >&2; exit 1; }
      ADDRESS="$2"
      shift 2
      ;;
    --laptop-ip)
      [[ $# -ge 2 ]] || { echo "Missing value for --laptop-ip" >&2; exit 1; }
      LAPTOP_IP="$2"
      shift 2
      ;;
    --dns)
      [[ $# -ge 2 ]] || { echo "Missing value for --dns" >&2; exit 1; }
      DNS_SERVER="$2"
      shift 2
      ;;
    --dhcp-start)
      [[ $# -ge 2 ]] || { echo "Missing value for --dhcp-start" >&2; exit 1; }
      DHCP_START="$2"
      shift 2
      ;;
    --dhcp-end)
      [[ $# -ge 2 ]] || { echo "Missing value for --dhcp-end" >&2; exit 1; }
      DHCP_END="$2"
      shift 2
      ;;
    --dhcp-lease-time)
      [[ $# -ge 2 ]] || { echo "Missing value for --dhcp-lease-time" >&2; exit 1; }
      DHCP_LEASE_TIME="$2"
      shift 2
      ;;
    --connection)
      [[ $# -ge 2 ]] || { echo "Missing value for --connection" >&2; exit 1; }
      CONNECTION_NAME="$2"
      shift 2
      ;;
    --channel)
      [[ $# -ge 2 ]] || { echo "Missing value for --channel" >&2; exit 1; }
      CHANNEL="$2"
      shift 2
      ;;
    --band)
      [[ $# -ge 2 ]] || { echo "Missing value for --band" >&2; exit 1; }
      BAND="$2"
      shift 2
      ;;
    --route-metric)
      [[ $# -ge 2 ]] || { echo "Missing value for --route-metric" >&2; exit 1; }
      ROUTE_METRIC="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$AP_IFACE" ]]; then
  echo "Missing required --interface" >&2
  usage >&2
  exit 1
fi

if [[ ${#PASSWORD} -lt 8 ]]; then
  echo "Wi-Fi hotspot password must be at least 8 characters." >&2
  exit 1
fi

for cmd in nmcli ip iptables python3; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing required command: $cmd" >&2
    exit 1
  fi
done

if [[ $EUID -ne 0 ]]; then
  exec sudo --preserve-env=PATH bash "$0" "${ORIGINAL_ARGS[@]}"
fi

echo "[setup_g1_hotspot_via_laptop] WARNING: this changes G1 network interfaces."
echo "[setup_g1_hotspot_via_laptop] Keep an Ethernet cable connected to G1 before continuing."
echo "[setup_g1_hotspot_via_laptop] Current Wi-Fi SSH may disconnect when $AP_IFACE switches to AP mode."

for cmd in dnsmasq systemctl; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing required command: $cmd" >&2
    if [[ "$cmd" == "dnsmasq" ]]; then
      echo "Install dnsmasq on G1, e.g. sudo apt install dnsmasq" >&2
    fi
    exit 1
  fi
done

if ! ip link show "$AP_IFACE" >/dev/null 2>&1; then
  echo "Missing Wi-Fi interface: $AP_IFACE" >&2
  ip -br link >&2
  exit 1
fi

if command -v iw >/dev/null 2>&1; then
  if [[ ! -e "/sys/class/net/$AP_IFACE/phy80211" ]]; then
    echo "Interface $AP_IFACE is not a Wi-Fi interface." >&2
    exit 1
  fi
  phy_name=$(basename "$(readlink -f "/sys/class/net/$AP_IFACE/phy80211")")
  if ! iw phy "$phy_name" info | sed -n '/Supported interface modes:/,/Band 1:/p' | grep -q '^[[:space:]]*\* AP$'; then
    echo "Interface $AP_IFACE does not report AP mode support." >&2
    exit 1
  fi
fi

read -r HOTSPOT_SUBNET G1_IP NETMASK DEFAULT_DHCP_START DEFAULT_DHCP_END < <(python3 - "$ADDRESS" "$LAPTOP_IP" <<'PY'
import ipaddress
import sys

iface = ipaddress.ip_interface(sys.argv[1])
laptop_ip = ipaddress.ip_address(sys.argv[2])
network = iface.network
if network.version != 4:
    raise SystemExit("Only IPv4 hotspot addresses are supported")

first = int(network.network_address) + 1
last = int(network.broadcast_address) - 1
if last < first:
    raise SystemExit(f"Network {network} does not have usable host addresses")

start = max(first, int(network.network_address) + 20)
end = min(last, int(network.network_address) + 200)
excluded = {int(iface.ip), int(laptop_ip)}
while start in excluded and start <= end:
    start += 1
while end in excluded and end >= start:
    end -= 1
if start > end:
    raise SystemExit(f"Could not choose a DHCP range for {network}")

print(network, iface.ip, network.netmask, ipaddress.ip_address(start), ipaddress.ip_address(end))
PY
)

if [[ -z "$DHCP_START" ]]; then
  DHCP_START="$DEFAULT_DHCP_START"
fi
if [[ -z "$DHCP_END" ]]; then
  DHCP_END="$DEFAULT_DHCP_END"
fi

python3 - "$HOTSPOT_SUBNET" "$G1_IP" "$LAPTOP_IP" "$DHCP_START" "$DHCP_END" <<'PY'
import ipaddress
import sys

network = ipaddress.ip_network(sys.argv[1])
g1_ip, laptop_ip, dhcp_start, dhcp_end = map(ipaddress.ip_address, sys.argv[2:])
for name, ip in (
    ("G1 address", g1_ip),
    ("laptop IP", laptop_ip),
    ("DHCP start", dhcp_start),
    ("DHCP end", dhcp_end),
):
    if ip not in network:
        raise SystemExit(f"{name} {ip} is outside hotspot subnet {network}")
if int(dhcp_start) > int(dhcp_end):
    raise SystemExit(f"DHCP start {dhcp_start} is after DHCP end {dhcp_end}")
if g1_ip in (dhcp_start, dhcp_end) or laptop_ip in (dhcp_start, dhcp_end):
    raise SystemExit("DHCP range endpoint overlaps G1 or laptop IP")
if int(dhcp_start) <= int(g1_ip) <= int(dhcp_end):
    raise SystemExit("DHCP range includes the G1 AP address")
if int(dhcp_start) <= int(laptop_ip) <= int(dhcp_end):
    raise SystemExit("DHCP range includes the laptop gateway IP")
PY

echo "[setup_g1_hotspot_via_laptop] interface=$AP_IFACE"
echo "[setup_g1_hotspot_via_laptop] ssid=$SSID"
echo "[setup_g1_hotspot_via_laptop] password=$PASSWORD"
echo "[setup_g1_hotspot_via_laptop] g1_address=$ADDRESS"
echo "[setup_g1_hotspot_via_laptop] laptop_gateway=$LAPTOP_IP"
echo "[setup_g1_hotspot_via_laptop] subnet=$HOTSPOT_SUBNET"
echo "[setup_g1_hotspot_via_laptop] client_dhcp_range=$DHCP_START,$DHCP_END,$DHCP_LEASE_TIME"

if nmcli -t -f NAME con show | grep -Fxq "$CONNECTION_NAME"; then
  echo "[setup_g1_hotspot_via_laptop] updating existing connection: $CONNECTION_NAME"
else
  echo "[setup_g1_hotspot_via_laptop] creating connection: $CONNECTION_NAME"
  nmcli con add type wifi ifname "$AP_IFACE" con-name "$CONNECTION_NAME" ssid "$SSID"
fi

nmcli con modify "$CONNECTION_NAME" \
  connection.interface-name "$AP_IFACE" \
  connection.autoconnect no \
  802-11-wireless.mode ap \
  802-11-wireless.band "$BAND" \
  802-11-wireless.channel "$CHANNEL" \
  802-11-wireless.ssid "$SSID" \
  wifi-sec.key-mgmt wpa-psk \
  wifi-sec.psk "$PASSWORD" \
  ipv4.method manual \
  ipv4.addresses "$ADDRESS" \
  ipv4.gateway "$LAPTOP_IP" \
  ipv4.route-metric "$ROUTE_METRIC" \
  ipv4.dns "$DNS_SERVER" \
  ipv4.never-default no \
  ipv6.method disabled

nmcli con up "$CONNECTION_NAME"

echo "[setup_g1_hotspot_via_laptop] replacing default route via laptop"
ip route replace default via "$LAPTOP_IP" dev "$AP_IFACE" metric "$ROUTE_METRIC"

add_iptables_rule() {
  local table=$1
  local chain=$2
  shift 2
  if ! iptables -t "$table" -C "$chain" "$@" 2>/dev/null; then
    iptables -t "$table" -A "$chain" "$@"
  fi
}

echo "[setup_g1_hotspot_via_laptop] enabling relay for other hotspot clients"
sysctl -w net.ipv4.ip_forward=1 >/dev/null
sysctl -w "net.ipv4.conf.$AP_IFACE.send_redirects=0" >/dev/null || true
add_iptables_rule nat POSTROUTING -s "$HOTSPOT_SUBNET" ! -d "$HOTSPOT_SUBNET" -o "$AP_IFACE" -j MASQUERADE
add_iptables_rule filter FORWARD -i "$AP_IFACE" -o "$AP_IFACE" -s "$HOTSPOT_SUBNET" ! -d "$HOTSPOT_SUBNET" -j ACCEPT
add_iptables_rule filter FORWARD -i "$AP_IFACE" -o "$AP_IFACE" -d "$HOTSPOT_SUBNET" -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT

DISPATCHER_NAME=${CONNECTION_NAME//[^A-Za-z0-9_.-]/_}
DISPATCHER_PATH="/etc/NetworkManager/dispatcher.d/91-${DISPATCHER_NAME}-laptop-egress"
DNSMASQ_CONFIG_DIR="/etc/sim2real"
DNSMASQ_CONFIG="$DNSMASQ_CONFIG_DIR/${DISPATCHER_NAME}-dnsmasq.conf"
DNSMASQ_SERVICE="${DISPATCHER_NAME}-dnsmasq.service"

install -d -m 0755 "$DNSMASQ_CONFIG_DIR"
cat >"$DNSMASQ_CONFIG" <<EOF
# Generated by setup_g1_hotspot_via_laptop.sh.
port=0
interface=$AP_IFACE
bind-dynamic
listen-address=$G1_IP
dhcp-authoritative
dhcp-range=$DHCP_START,$DHCP_END,$NETMASK,$DHCP_LEASE_TIME
dhcp-option=option:router,$G1_IP
dhcp-option=option:dns-server,$DNS_SERVER
dhcp-option=option:netmask,$NETMASK
EOF

cat >"/etc/systemd/system/$DNSMASQ_SERVICE" <<EOF
[Unit]
Description=DHCP server for $SSID hotspot clients
After=NetworkManager.service
Wants=NetworkManager.service

[Service]
Type=simple
ExecStart=$(command -v dnsmasq) --keep-in-foreground --conf-file=$DNSMASQ_CONFIG
Restart=on-failure
RestartSec=2

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl disable "$DNSMASQ_SERVICE" >/dev/null 2>&1 || true
systemctl restart "$DNSMASQ_SERVICE"

cat >"$DISPATCHER_PATH" <<EOF
#!/usr/bin/env bash
set -euo pipefail

AP_IFACE="$AP_IFACE"
LAPTOP_IP="$LAPTOP_IP"
ROUTE_METRIC="$ROUTE_METRIC"
HOTSPOT_SUBNET="$HOTSPOT_SUBNET"

case "\${2:-}" in
  up|dhcp4-change|connectivity-change)
    ;;
  *)
    exit 0
    ;;
esac

if [[ "\${1:-}" != "\$AP_IFACE" ]]; then
  exit 0
fi

ip route replace default via "\$LAPTOP_IP" dev "\$AP_IFACE" metric "\$ROUTE_METRIC"
sysctl -w net.ipv4.ip_forward=1 >/dev/null
sysctl -w "net.ipv4.conf.\$AP_IFACE.send_redirects=0" >/dev/null || true

add_iptables_rule() {
  local table=\$1
  local chain=\$2
  shift 2
  if ! iptables -t "\$table" -C "\$chain" "\$@" 2>/dev/null; then
    iptables -t "\$table" -A "\$chain" "\$@"
  fi
}

add_iptables_rule nat POSTROUTING -s "\$HOTSPOT_SUBNET" ! -d "\$HOTSPOT_SUBNET" -o "\$AP_IFACE" -j MASQUERADE
add_iptables_rule filter FORWARD -i "\$AP_IFACE" -o "\$AP_IFACE" -s "\$HOTSPOT_SUBNET" ! -d "\$HOTSPOT_SUBNET" -j ACCEPT
add_iptables_rule filter FORWARD -i "\$AP_IFACE" -o "\$AP_IFACE" -d "\$HOTSPOT_SUBNET" -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT
EOF
chmod 0755 "$DISPATCHER_PATH"

echo "[setup_g1_hotspot_via_laptop] installed dispatcher: $DISPATCHER_PATH"
echo "[setup_g1_hotspot_via_laptop] installed DHCP config: $DNSMASQ_CONFIG"
echo "[setup_g1_hotspot_via_laptop] started DHCP service: $DNSMASQ_SERVICE"
echo "[setup_g1_hotspot_via_laptop] hotspot autoconnect is disabled; rerun this script or use nmcli con up $CONNECTION_NAME after reboot."
echo "[setup_g1_hotspot_via_laptop] AP is active; now run setup_laptop_g1_gateway.sh on the laptop."
echo "[setup_g1_hotspot_via_laptop] After the laptop joins the hotspot, reconnect with: ssh g1-hotspot"
echo "[setup_g1_hotspot_via_laptop] other devices can join $SSID and should receive DHCP leases from $DHCP_START-$DHCP_END."
nmcli -f GENERAL.STATE,GENERAL.CONNECTION,IP4.ADDRESS,IP4.GATEWAY dev show "$AP_IFACE"
echo "[setup_g1_hotspot_via_laptop] route to 8.8.8.8:"
ip route get 8.8.8.8 || true
echo "[setup_g1_hotspot_via_laptop] DHCP service state:"
systemctl --no-pager --full status "$DNSMASQ_SERVICE" | sed -n '1,12p' || true
