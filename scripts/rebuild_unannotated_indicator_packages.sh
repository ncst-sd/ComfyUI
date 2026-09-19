#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/yolo_device_monitor"
source .venv/bin/activate

python scripts/rebuild_unannotated_indicator_packages.py \
  --interval-sec 0.8 \
  --clarity-floor 28 \
  --clarity-percentile 20 \
  --duplicate-max-hamming 7 \
  --duplicate-min-hist-corr 0.965 \
  --min-per-video 12 \
  --max-per-video 36 \
  --max-per-device 200

