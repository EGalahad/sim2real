#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/setup/setup_g1_hotspot.sh [options]

Create and start a NetworkManager Wi-Fi hotspot on a G1.
This changes G1 network interfaces. Keep an Ethernet cable connected to G1
before running it so you do not lose access if Wi-Fi setup fails.

Defaults:
  SSID:       hdmi-deploy
  Password:   hdmi1234
  Interface:  wlan1
  Upstream:   wlan0
  Address:    10.42.7.1/24

Options:
  --interface IFACE      Wi-Fi interface to use for AP mode
  --upstream IFACE       Interface that hotspot clients use for internet egress
  --ssid SSID            Hotspot SSID
  --password PASSWORD    WPA-PSK password, 8+ chars
  --address CIDR         Static IPv4 address for the AP
  --connection NAME      NetworkManager connection name
  -h, --help             Show this help
EOF
}

WIFI_IFACE="wlan1"
UPSTREAM_IFACE="wlan0"
SSID="hdmi-deploy"
PASSWORD="hdmi1234"
ADDRESS="10.42.7.1/24"
CONNECTION_NAME="hdmi-deploy"
CHANNEL="6"
BAND="bg"
ROUTE_TABLE="107"
ORIGINAL_ARGS=("$@")

while [[ $# -gt 0 ]]; do
  case "$1" in
    --interface)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --interface" >&2
        exit 1
      fi
      WIFI_IFACE="$2"
      shift 2
      ;;
    --upstream)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --upstream" >&2
        exit 1
      fi
      UPSTREAM_IFACE="$2"
      shift 2
      ;;
    --ssid)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --ssid" >&2
        exit 1
      fi
      SSID="$2"
      shift 2
      ;;
    --password)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --password" >&2
        exit 1
      fi
      PASSWORD="$2"
      shift 2
      ;;
    --address)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --address" >&2
        exit 1
      fi
      ADDRESS="$2"
      shift 2
      ;;
    --connection)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --connection" >&2
        exit 1
      fi
      CONNECTION_NAME="$2"
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

echo "[setup_g1_hotspot] WARNING: this changes G1 network interfaces."
echo "[setup_g1_hotspot] Keep an Ethernet cable connected to G1 before continuing."

if ! ip link show "$WIFI_IFACE" >/dev/null 2>&1; then
  echo "Missing Wi-Fi interface: $WIFI_IFACE" >&2
  echo "Available interfaces:" >&2
  ip -br link >&2
  exit 1
fi

if ! ip link show "$UPSTREAM_IFACE" >/dev/null 2>&1; then
  echo "Missing upstream interface: $UPSTREAM_IFACE" >&2
  echo "Available interfaces:" >&2
  ip -br link >&2
  exit 1
fi

if command -v iw >/dev/null 2>&1; then
  if [[ ! -e "/sys/class/net/$WIFI_IFACE/phy80211" ]]; then
    echo "Interface $WIFI_IFACE is not a Wi-Fi interface." >&2
    exit 1
  fi
  phy_name=$(basename "$(readlink -f "/sys/class/net/$WIFI_IFACE/phy80211")")
  if ! iw phy "$phy_name" info | sed -n '/Supported interface modes:/,/Band 1:/p' | grep -q '^[[:space:]]*\* AP$'; then
    echo "Interface $WIFI_IFACE does not report AP mode support." >&2
    exit 1
  fi
fi

echo "[setup_g1_hotspot] interface=$WIFI_IFACE"
echo "[setup_g1_hotspot] upstream=$UPSTREAM_IFACE"
echo "[setup_g1_hotspot] ssid=$SSID"
echo "[setup_g1_hotspot] password=$PASSWORD"
echo "[setup_g1_hotspot] address=$ADDRESS"

if nmcli -t -f NAME con show | grep -Fxq "$CONNECTION_NAME"; then
  echo "[setup_g1_hotspot] updating existing connection: $CONNECTION_NAME"
else
  echo "[setup_g1_hotspot] creating connection: $CONNECTION_NAME"
  nmcli con add type wifi ifname "$WIFI_IFACE" con-name "$CONNECTION_NAME" ssid "$SSID"
fi

nmcli con modify "$CONNECTION_NAME" \
  connection.interface-name "$WIFI_IFACE" \
  connection.autoconnect yes \
  802-11-wireless.mode ap \
  802-11-wireless.band "$BAND" \
  802-11-wireless.channel "$CHANNEL" \
  802-11-wireless.ssid "$SSID" \
  wifi-sec.key-mgmt wpa-psk \
  wifi-sec.psk "$PASSWORD" \
  ipv4.method shared \
  ipv4.addresses "$ADDRESS" \
  ipv6.method disabled

nmcli con up "$CONNECTION_NAME"

HOTSPOT_SUBNET=$(python3 - "$ADDRESS" <<'PY'
import ipaddress
import sys

iface = ipaddress.ip_interface(sys.argv[1])
print(iface.network)
PY
)

UPSTREAM_GATEWAY=$(
  ip -4 route show default dev "$UPSTREAM_IFACE" |
    awk 'NR == 1 { for (i = 1; i <= NF; i++) if ($i == "via") { print $(i + 1); exit } }'
)

echo "[setup_g1_hotspot] enabling IPv4 forwarding"
sysctl -w net.ipv4.ip_forward=1 >/dev/null

echo "[setup_g1_hotspot] routing $HOTSPOT_SUBNET out via $UPSTREAM_IFACE"
ip route replace "$HOTSPOT_SUBNET" dev "$WIFI_IFACE" table "$ROUTE_TABLE"
if [[ -n "$UPSTREAM_GATEWAY" ]]; then
  ip route replace default via "$UPSTREAM_GATEWAY" dev "$UPSTREAM_IFACE" table "$ROUTE_TABLE"
else
  ip route replace default dev "$UPSTREAM_IFACE" table "$ROUTE_TABLE"
fi
ip rule add from "$HOTSPOT_SUBNET" table "$ROUTE_TABLE" priority "$ROUTE_TABLE" 2>/dev/null || true

add_iptables_rule() {
  local table=$1
  local chain=$2
  shift 2
  if ! iptables -t "$table" -C "$chain" "$@" 2>/dev/null; then
    iptables -t "$table" -A "$chain" "$@"
  fi
}

echo "[setup_g1_hotspot] enabling NAT from $WIFI_IFACE to $UPSTREAM_IFACE"
add_iptables_rule nat POSTROUTING -s "$HOTSPOT_SUBNET" -o "$UPSTREAM_IFACE" -j MASQUERADE
add_iptables_rule filter FORWARD -i "$WIFI_IFACE" -o "$UPSTREAM_IFACE" -s "$HOTSPOT_SUBNET" -j ACCEPT
add_iptables_rule filter FORWARD -i "$UPSTREAM_IFACE" -o "$WIFI_IFACE" -d "$HOTSPOT_SUBNET" -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT

DISPATCHER_NAME=${CONNECTION_NAME//[^A-Za-z0-9_.-]/_}
DISPATCHER_PATH="/etc/NetworkManager/dispatcher.d/90-${DISPATCHER_NAME}-forwarding"
cat >"$DISPATCHER_PATH" <<EOF
#!/usr/bin/env bash
set -euo pipefail

WIFI_IFACE="$WIFI_IFACE"
UPSTREAM_IFACE="$UPSTREAM_IFACE"
HOTSPOT_SUBNET="$HOTSPOT_SUBNET"
ROUTE_TABLE="$ROUTE_TABLE"

case "\${2:-}" in
  up|dhcp4-change|connectivity-change)
    ;;
  *)
    exit 0
    ;;
esac

if [[ "\${1:-}" != "\$WIFI_IFACE" && "\${1:-}" != "\$UPSTREAM_IFACE" ]]; then
  exit 0
fi

add_iptables_rule() {
  local table=\$1
  local chain=\$2
  shift 2
  if ! iptables -t "\$table" -C "\$chain" "\$@" 2>/dev/null; then
    iptables -t "\$table" -A "\$chain" "\$@"
  fi
}

gateway=\$(
  ip -4 route show default dev "\$UPSTREAM_IFACE" |
    awk 'NR == 1 { for (i = 1; i <= NF; i++) if (\$i == "via") { print \$(i + 1); exit } }'
)

sysctl -w net.ipv4.ip_forward=1 >/dev/null
ip route replace "\$HOTSPOT_SUBNET" dev "\$WIFI_IFACE" table "\$ROUTE_TABLE"
if [[ -n "\$gateway" ]]; then
  ip route replace default via "\$gateway" dev "\$UPSTREAM_IFACE" table "\$ROUTE_TABLE"
else
  ip route replace default dev "\$UPSTREAM_IFACE" table "\$ROUTE_TABLE"
fi
ip rule add from "\$HOTSPOT_SUBNET" table "\$ROUTE_TABLE" priority "\$ROUTE_TABLE" 2>/dev/null || true

add_iptables_rule nat POSTROUTING -s "\$HOTSPOT_SUBNET" -o "\$UPSTREAM_IFACE" -j MASQUERADE
add_iptables_rule filter FORWARD -i "\$WIFI_IFACE" -o "\$UPSTREAM_IFACE" -s "\$HOTSPOT_SUBNET" -j ACCEPT
add_iptables_rule filter FORWARD -i "\$UPSTREAM_IFACE" -o "\$WIFI_IFACE" -d "\$HOTSPOT_SUBNET" -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT
EOF
chmod 0755 "$DISPATCHER_PATH"
echo "[setup_g1_hotspot] installed dispatcher: $DISPATCHER_PATH"

echo "[setup_g1_hotspot] hotspot is active"
nmcli -f GENERAL.STATE,GENERAL.CONNECTION,IP4.ADDRESS,IP4.GATEWAY dev show "$WIFI_IFACE"
echo "[setup_g1_hotspot] hotspot clients route table:"
ip route show table "$ROUTE_TABLE"
