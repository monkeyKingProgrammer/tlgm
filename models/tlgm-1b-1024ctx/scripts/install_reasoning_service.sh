#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "$0")/.." && pwd)"
sudo install -m 0644 "$project_dir/deploy/tlgm1b-reasoning.service" /etc/systemd/system/tlgm1b-reasoning.service
sudo install -m 0644 "$project_dir/deploy/tlgm1b-fair-ppl.service" /etc/systemd/system/tlgm1b-fair-ppl.service
sudo systemctl daemon-reload
sudo systemctl enable tlgm1b-reasoning.service
echo "Installed and enabled tlgm1b-reasoning.service."
echo "Start:   sudo systemctl start tlgm1b-reasoning.service"
echo "Stop:    sudo systemctl stop tlgm1b-reasoning.service"
echo "Status:  systemctl status tlgm1b-reasoning.service"
echo "Logs:    journalctl -u tlgm1b-reasoning.service -f"
echo "Comparison logs: journalctl -u tlgm1b-fair-ppl.service -f"
