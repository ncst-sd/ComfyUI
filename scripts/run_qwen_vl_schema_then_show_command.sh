#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/yolo_device_monitor"
source qwen_vl_env/bin/activate

DEVICE="${1:?用法: $0 '<设备目录名>' [图片数量]}"
LIMIT="${2:-5}"
MODEL="$HOME/yolo_device_monitor/models/Qwen2.5-VL-7B-Instruct"

echo "=== 第1步：重新生成严格物理灯类别表 ==="
python scripts/build_manual_indicator_schema.py \
  --device "$DEVICE" \
  --model "$MODEL" \
  --device-map cuda:0

echo
echo "请先检查："
echo "$HOME/yolo_device_monitor/annotation_packages_best_frames_fast/$DEVICE/manual_indicator_schema.json"
echo
echo "确认类别正确后，再手动运行图片预标注："
echo "python scripts/prelabel_images_with_qwen_vl.py \\"
echo "  --device \"$DEVICE\" \\"
echo "  --model \"$MODEL\" \\"
echo "  --device-map cuda:0 \\"
echo "  --limit $LIMIT"

