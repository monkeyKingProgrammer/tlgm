#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python chat_cli.py "$@"
