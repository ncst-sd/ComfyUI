#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/yolo_device_monitor"
source qwen_vl_env/bin/activate
mkdir -p "$HOME/yolo_device_monitor/models/grounding-dino-tiny"
hf download IDEA-Research/grounding-dino-tiny --local-dir "$HOME/yolo_device_monitor/models/grounding-dino-tiny"

