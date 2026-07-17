#!/usr/bin/env bash
set -euo pipefail

TARGET="can0"
BITRATE="1000000"
SELECTED_IFACE=""
SELECTED_USB=""
SELECTED_SERIAL=""
SAVE=0
LIST_ONLY=0
STATE_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/piper_vla"
STATE_FILE="$STATE_DIR/piper_can.conf"
VENDOR_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/can_activate.sh"

usage() {
  cat <<'EOF'
Usage: ./scripts/activate_can0.sh [options]

Automatically finds the Piper USB-CAN adapter and activates it as can0 at 1 Mbps.

Options:
  --list                 Show detected CAN adapters without changing anything.
  --interface NAME       Select the current Linux CAN interface (for example can1).
  --usb BUS_INFO         Select an ethtool bus-info value (for example 3-12:1.0).
  --serial SERIAL        Select a USB serial number.
  --save                 Save the selected serial/bus for later runs.
  --target NAME          Target CAN name (default: can0).
  --bitrate RATE         CAN bitrate (default: 1000000).
  -h, --help             Show this help.

With one detected CAN adapter, no selection option is needed. With multiple
adapters, select the Piper adapter once with --interface/--usb/--serial --save.
EOF
}

while (($#)); do
  case "$1" in
    --list) LIST_ONLY=1; shift ;;
    --interface) SELECTED_IFACE="${2:?missing interface}"; shift 2 ;;
    --usb) SELECTED_USB="${2:?missing USB bus-info}"; shift 2 ;;
    --serial) SELECTED_SERIAL="${2:?missing USB serial}"; shift 2 ;;
    --save) SAVE=1; shift ;;
    --target) TARGET="${2:?missing target}"; shift 2 ;;
    --bitrate) BITRATE="${2:?missing bitrate}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for command in ip ethtool awk sed udevadm; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "Required command is missing: $command" >&2
    exit 127
  fi
done
if [[ ! -x "$VENDOR_SCRIPT" ]]; then
  echo "Vendor CAN activation script is missing or not executable: $VENDOR_SCRIPT" >&2
  exit 1
fi

mapfile -t IFACES < <(ip -br link show type can | awk '{print $1}')
if ((${#IFACES[@]} == 0)); then
  echo "No Linux CAN interface detected. Connect the Piper USB-CAN adapter first." >&2
  exit 1
fi

bus_of() {
  ethtool -i "$1" 2>/dev/null | awk '$1 == "bus-info:" {print $2; exit}'
}

serial_of() {
  local sys_path
  sys_path="$(readlink -f "/sys/class/net/$1/device" 2>/dev/null || true)"
  [[ -n "$sys_path" ]] || return 0
  udevadm info --query=property --path="$sys_path" 2>/dev/null \
    | sed -n 's/^ID_SERIAL_SHORT=//p' | head -n 1
}

driver_of() {
  ethtool -i "$1" 2>/dev/null | awk '$1 == "driver:" {print $2; exit}'
}

print_adapters() {
  printf '%-12s %-12s %-18s %-20s %s\n' "INTERFACE" "STATE" "DRIVER" "USB BUS" "SERIAL"
  for iface in "${IFACES[@]}"; do
    printf '%-12s %-12s %-18s %-20s %s\n' \
      "$iface" \
      "$(ip -br link show "$iface" | awk '{print $2}')" \
      "$(driver_of "$iface")" \
      "$(bus_of "$iface")" \
      "$(serial_of "$iface")"
  done
}

if ((LIST_ONLY)); then
  print_adapters
  exit 0
fi

SAVED_SERIAL=""
SAVED_USB=""
if [[ -r "$STATE_FILE" ]]; then
  SAVED_SERIAL="$(sed -n 's/^SERIAL=//p' "$STATE_FILE" | head -n 1)"
  SAVED_USB="$(sed -n 's/^USB_BUS=//p' "$STATE_FILE" | head -n 1)"
fi

matches=()
for iface in "${IFACES[@]}"; do
  bus="$(bus_of "$iface")"
  serial="$(serial_of "$iface")"
  if [[ -n "$SELECTED_IFACE" && "$iface" == "$SELECTED_IFACE" ]] \
    || [[ -n "$SELECTED_USB" && "$bus" == "$SELECTED_USB" ]] \
    || [[ -n "$SELECTED_SERIAL" && "$serial" == "$SELECTED_SERIAL" ]] \
    || [[ -z "$SELECTED_IFACE$SELECTED_USB$SELECTED_SERIAL" && -n "$SAVED_SERIAL" && "$serial" == "$SAVED_SERIAL" ]] \
    || [[ -z "$SELECTED_IFACE$SELECTED_USB$SELECTED_SERIAL$SAVED_SERIAL" && -n "$SAVED_USB" && "$bus" == "$SAVED_USB" ]]; then
    matches+=("$iface")
  fi
done

if ((${#matches[@]} == 0)) && [[ -z "$SELECTED_IFACE$SELECTED_USB$SELECTED_SERIAL" ]] \
  && ((${#IFACES[@]} == 1)); then
  # A unique CAN adapter is safe to select even when a previously saved USB
  # bus address changed because the adapter was plugged into another port.
  matches=("${IFACES[0]}")
fi

if ((${#matches[@]} != 1)); then
  echo "Could not uniquely identify the Piper CAN adapter; no interface was changed." >&2
  print_adapters >&2
  echo >&2
  echo "Select it once, for example:" >&2
  echo "  ./scripts/activate_can0.sh --interface can1 --save" >&2
  exit 2
fi

SOURCE="${matches[0]}"
USB_BUS="$(bus_of "$SOURCE")"
SERIAL="$(serial_of "$SOURCE")"
if [[ -z "$USB_BUS" ]]; then
  echo "Could not read USB bus-info for $SOURCE; no interface was changed." >&2
  exit 1
fi

if [[ "$SOURCE" != "$TARGET" ]] && ip link show "$TARGET" >/dev/null 2>&1; then
  echo "Refusing to rename $SOURCE: target $TARGET already belongs to another interface." >&2
  print_adapters >&2
  exit 2
fi

echo "Selected Piper CAN adapter: interface=$SOURCE usb=$USB_BUS serial=${SERIAL:-unknown}"
sudo -v
bash "$VENDOR_SCRIPT" "$TARGET" "$BITRATE" "$USB_BUS"

DETAILS="$(ip -details link show "$TARGET")"
if ! grep -q "bitrate $BITRATE" <<<"$DETAILS"; then
  echo "$TARGET did not report bitrate $BITRATE after activation." >&2
  exit 1
fi
if ! grep -qE '<[^>]*UP[^>]*>' <<<"$DETAILS"; then
  echo "$TARGET is not UP after activation." >&2
  exit 1
fi

if ((SAVE)); then
  mkdir -p "$STATE_DIR"
  {
    printf 'SERIAL=%s\n' "$SERIAL"
    printf 'USB_BUS=%s\n' "$USB_BUS"
  } >"$STATE_FILE"
  chmod 600 "$STATE_FILE"
  echo "Saved adapter identity to $STATE_FILE"
fi

echo "CAN ready:"
ip -details link show "$TARGET"
