#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/yolo_device_monitor"
python3 -m venv qwen_vl_env
source qwen_vl_env/bin/activate
python -m pip install -U pip
pip install "transformers>=4.49.0" accelerate qwen-vl-utils pillow pypdf python-docx
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

