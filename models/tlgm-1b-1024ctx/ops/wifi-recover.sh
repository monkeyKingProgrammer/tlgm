#!/usr/bin/env bash
set -u

DEVICE="${WIFI_DEVICE:?Set WIFI_DEVICE in /etc/default/tlgm-wifi-recover}"
CONNECTION_UUID="${WIFI_CONNECTION_UUID:?Set WIFI_CONNECTION_UUID in /etc/default/tlgm-wifi-recover}"
active_uuid=$(nmcli -t -f UUID,DEVICE connection show --active 2>/dev/null \
  | awk -F: -v device="$DEVICE" '$2 == device { print $1; exit }')
device_state=$(nmcli -g GENERAL.STATE device show "$DEVICE" 2>/dev/null || true)
device_state="${device_state%% *}"
# Do not use a single gateway ping as the health signal. A router reboot, ICMP
# filtering, or a transient packet loss can make that ping fail while the Wi-Fi
# association is healthy. NetworkManager already handles AP loss and retries.
if [ "$active_uuid" = "$CONNECTION_UUID" ] \
  && [ "$device_state" = "100" ]; then
  exit 0
fi

logger -t wifi-recover \
  "Wi-Fi $DEVICE is not fully activated (uuid=$active_uuid state=$device_state); forcing reconnect"
nmcli radio wifi on || true
nmcli device set "$DEVICE" managed yes || true
nmcli connection down uuid "$CONNECTION_UUID" >/dev/null 2>&1 || true
nmcli device disconnect "$DEVICE" >/dev/null 2>&1 || true
sleep 2

if nmcli --wait 60 connection up uuid "$CONNECTION_UUID" ifname "$DEVICE"; then
  logger -t wifi-recover "Wi-Fi $DEVICE reconnected successfully"
  exit 0
fi

logger -t wifi-recover "Wi-Fi $DEVICE reconnect failed; the timer will retry"
exit 1
