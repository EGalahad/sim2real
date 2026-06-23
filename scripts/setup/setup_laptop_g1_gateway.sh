#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: sudo bash scripts/setup/setup_laptop_g1_gateway.sh [options]

Connect the laptop's USB Wi-Fi adapter to the G1 hotspot and NAT G1 traffic
through the laptop's normal internet interface. Run this on the laptop.

Required:
  --wifi-interface IFACE      Laptop USB Wi-Fi interface connected to the G1 AP
  --upstream-interface IFACE  Laptop interface with internet access

Options:
  --ssid SSID                 G1 hotspot SSID (default: hdmi-deploy)
  --password PASSWORD         WPA-PSK password, 8+ chars (default: hdmi1234)
  --address CIDR              Laptop IP on G1 hotspot subnet (default: 10.42.7.2/24)
  --g1-ip IP                  G1 AP IP on this subnet (default: 10.42.7.1)
  --connection NAME           NetworkManager connection name (default: hdmi-g1-client)
  -h, --help                  Show this help
EOF
}

WIFI_IFACE=""
UPSTREAM_IFACE=""
SSID="hdmi-deploy"
PASSWORD="hdmi1234"
ADDRESS="10.42.7.2/24"
G1_IP="10.42.7.1"
CONNECTION_NAME="hdmi-g1-client"
ORIGINAL_ARGS=("$@")

while [[ $# -gt 0 ]]; do
  case "$1" in
    --wifi-interface)
      [[ $# -ge 2 ]] || { echo "Missing value for --wifi-interface" >&2; exit 1; }
      WIFI_IFACE="$2"
      shift 2
      ;;
    --upstream-interface)
      [[ $# -ge 2 ]] || { echo "Missing value for --upstream-interface" >&2; exit 1; }
      UPSTREAM_IFACE="$2"
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
    --g1-ip)
      [[ $# -ge 2 ]] || { echo "Missing value for --g1-ip" >&2; exit 1; }
      G1_IP="$2"
      shift 2
      ;;
    --connection)
      [[ $# -ge 2 ]] || { echo "Missing value for --connection" >&2; exit 1; }
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

if [[ -z "$WIFI_IFACE" || -z "$UPSTREAM_IFACE" ]]; then
  echo "Missing required --wifi-interface or --upstream-interface" >&2
  usage >&2
  exit 1
fi

if [[ ${#PASSWORD} -lt 8 ]]; then
  echo "Wi-Fi password must be at least 8 characters." >&2
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

for iface in "$WIFI_IFACE" "$UPSTREAM_IFACE"; do
  if ! ip link show "$iface" >/dev/null 2>&1; then
    echo "Missing interface: $iface" >&2
    ip -br link >&2
    exit 1
  fi
done

if [[ "$WIFI_IFACE" == "$UPSTREAM_IFACE" ]]; then
  echo "--wifi-interface and --upstream-interface must be different." >&2
  exit 1
fi

HOTSPOT_SUBNET=$(python3 - "$ADDRESS" <<'PY'
import ipaddress
import sys

print(ipaddress.ip_interface(sys.argv[1]).network)
PY
)

echo "[setup_laptop_g1_gateway] wifi_interface=$WIFI_IFACE"
echo "[setup_laptop_g1_gateway] upstream_interface=$UPSTREAM_IFACE"
echo "[setup_laptop_g1_gateway] ssid=$SSID"
echo "[setup_laptop_g1_gateway] password=$PASSWORD"
echo "[setup_laptop_g1_gateway] laptop_address=$ADDRESS"
echo "[setup_laptop_g1_gateway] g1_ip=$G1_IP"
echo "[setup_laptop_g1_gateway] subnet=$HOTSPOT_SUBNET"

if nmcli -t -f NAME con show | grep -Fxq "$CONNECTION_NAME"; then
  echo "[setup_laptop_g1_gateway] updating existing connection: $CONNECTION_NAME"
else
  echo "[setup_laptop_g1_gateway] creating connection: $CONNECTION_NAME"
  nmcli con add type wifi ifname "$WIFI_IFACE" con-name "$CONNECTION_NAME" ssid "$SSID"
fi

nmcli con modify "$CONNECTION_NAME" \
  connection.interface-name "$WIFI_IFACE" \
  connection.autoconnect yes \
  802-11-wireless.mode infrastructure \
  802-11-wireless.ssid "$SSID" \
  wifi-sec.key-mgmt wpa-psk \
  wifi-sec.psk "$PASSWORD" \
  ipv4.method manual \
  ipv4.addresses "$ADDRESS" \
  ipv4.gateway "" \
  ipv4.never-default yes \
  ipv4.ignore-auto-dns yes \
  ipv6.method disabled

nmcli con up "$CONNECTION_NAME"

echo "[setup_laptop_g1_gateway] enabling IPv4 forwarding"
sysctl -w net.ipv4.ip_forward=1 >/dev/null

add_iptables_rule() {
  local table=$1
  local chain=$2
  shift 2
  if ! iptables -t "$table" -C "$chain" "$@" 2>/dev/null; then
    iptables -t "$table" -A "$chain" "$@"
  fi
}

echo "[setup_laptop_g1_gateway] enabling NAT from $WIFI_IFACE to $UPSTREAM_IFACE"
add_iptables_rule nat POSTROUTING -s "$HOTSPOT_SUBNET" -o "$UPSTREAM_IFACE" -j MASQUERADE
add_iptables_rule filter FORWARD -i "$WIFI_IFACE" -o "$UPSTREAM_IFACE" -s "$HOTSPOT_SUBNET" -j ACCEPT
add_iptables_rule filter FORWARD -i "$UPSTREAM_IFACE" -o "$WIFI_IFACE" -d "$HOTSPOT_SUBNET" -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT

DISPATCHER_NAME=${CONNECTION_NAME//[^A-Za-z0-9_.-]/_}
DISPATCHER_PATH="/etc/NetworkManager/dispatcher.d/91-${DISPATCHER_NAME}-g1-nat"
cat >"$DISPATCHER_PATH" <<EOF
#!/usr/bin/env bash
set -euo pipefail

WIFI_IFACE="$WIFI_IFACE"
UPSTREAM_IFACE="$UPSTREAM_IFACE"
HOTSPOT_SUBNET="$HOTSPOT_SUBNET"

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

sysctl -w net.ipv4.ip_forward=1 >/dev/null
add_iptables_rule nat POSTROUTING -s "\$HOTSPOT_SUBNET" -o "\$UPSTREAM_IFACE" -j MASQUERADE
add_iptables_rule filter FORWARD -i "\$WIFI_IFACE" -o "\$UPSTREAM_IFACE" -s "\$HOTSPOT_SUBNET" -j ACCEPT
add_iptables_rule filter FORWARD -i "\$UPSTREAM_IFACE" -o "\$WIFI_IFACE" -d "\$HOTSPOT_SUBNET" -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT
EOF
chmod 0755 "$DISPATCHER_PATH"

echo "[setup_laptop_g1_gateway] installed dispatcher: $DISPATCHER_PATH"
echo "[setup_laptop_g1_gateway] connection is active; laptop should keep its own default route on $UPSTREAM_IFACE."
echo "[setup_laptop_g1_gateway] after this succeeds, connect to G1 with: ssh g1-hotspot"
nmcli -f GENERAL.STATE,GENERAL.CONNECTION,IP4.ADDRESS,IP4.GATEWAY dev show "$WIFI_IFACE"
echo "[setup_laptop_g1_gateway] laptop route to internet:"
ip route get 8.8.8.8 || true
echo "[setup_laptop_g1_gateway] route to G1:"
ip route get "$G1_IP" || true
