#!/usr/bin/env bash
set -e

cd "$HOME/yolo_device_monitor"
source .venv/bin/activate

exec python scripts/indicator_inference_ui.py \
  --host 127.0.0.1 \
  --port 8300

