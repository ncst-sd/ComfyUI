#!/usr/bin/env bash
set -e

PROJECT="$HOME/yolo_device_monitor"

cd "$PROJECT/scripts"
source "$PROJECT/llm_env/bin/activate"

exec uvicorn \
  device_state_api:app \
  --host 127.0.0.1 \
  --port 8100 \
  --workers 1

