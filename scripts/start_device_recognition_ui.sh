#!/usr/bin/env bash
set -e

PROJECT="$HOME/yolo_device_monitor"
VENV="$PROJECT/.venv"
PYTHON="$VENV/bin/python"
PIP="$VENV/bin/pip"
SCRIPT="$PROJECT/scripts/device_recognition_ui.py"

cd "$PROJECT"

if [ ! -x "$PYTHON" ]; then
    echo "找不到虚拟环境：$VENV"
    exit 1
fi

if [ ! -f "$SCRIPT" ]; then
    echo "找不到前端脚本：$SCRIPT"
    exit 1
fi

if ! "$PYTHON" -c "import fastapi, uvicorn, multipart" >/dev/null 2>&1; then
    echo "正在安装前端依赖 fastapi / uvicorn / python-multipart ..."
    "$PIP" install fastapi uvicorn python-multipart
fi

echo "启动设备识别前端..."
echo "浏览器打开：http://127.0.0.1:8200"
echo "按 Ctrl+C 停止"
echo

exec "$PYTHON" "$SCRIPT" \
    --host 127.0.0.1 \
    --port 8200

