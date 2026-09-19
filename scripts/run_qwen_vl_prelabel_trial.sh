#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/yolo_device_monitor"
source qwen_vl_env/bin/activate
DEVICE="${1:?请传设备目录名}"
LIMIT="${2:-10}"
python scripts/build_manual_indicator_schema.py --device "$DEVICE" --device-map cuda:0
python scripts/prelabel_images_with_qwen_vl.py --device "$DEVICE" --device-map cuda:0 --limit "$LIMIT"

