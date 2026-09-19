#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/yolo_device_monitor"
source .venv/bin/activate
if [[ $# -lt 1 ]]; then
  echo "用法: $0 '<设备目录名>'"
  echo "示例: $0 '数据网柜_H3C_SR6608_华三'"
  exit 1
fi
python scripts/prelabel_from_manual.py --device "$1"

