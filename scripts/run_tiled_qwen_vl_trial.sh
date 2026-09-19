#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/yolo_device_monitor"
source qwen_vl_env/bin/activate

DEVICE="${1:?用法: $0 '<设备目录名>' [图片数量]}"
LIMIT="${2:-4}"
MODEL="$HOME/yolo_device_monitor/models/Qwen2.5-VL-7B-Instruct"

python scripts/prelabel_indicator_with_tiled_qwen_vl.py \
  --device "$DEVICE" \
  --model "$MODEL" \
  --device-map cuda:0 \
  --tile-size 1024 \
  --overlap 0.25 \
  --context-scale 5 \
  --limit "$LIMIT"

