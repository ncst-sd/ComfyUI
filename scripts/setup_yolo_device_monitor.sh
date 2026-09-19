#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${HOME}/yolo_device_monitor"
VENV_DIR="${PROJECT_DIR}/.venv"

# 国内 PyPI 镜像
# 默认优先华为云；若失败自动切换清华、USTC、阿里云
MIRRORS=(
  "https://repo.huaweicloud.com/repository/pypi/simple"
  "https://pypi.tuna.tsinghua.edu.cn/simple"
  "https://pypi.mirrors.ustc.edu.cn/simple"
  "https://mirrors.aliyun.com/pypi/simple"
)

echo "========================================"
echo " YOLO Device Monitor 一键环境安装"
echo " 国内镜像自动切换版"
echo "========================================"
echo

install_with_mirrors() {
    local desc="$1"
    shift
    local packages=("$@")

    echo
    echo "安装 ${desc}..."

    for mirror in "${MIRRORS[@]}"; do
        echo
        echo "尝试镜像: ${mirror}"
        if python -m pip install \
            --index-url "${mirror}" \
            --timeout 60 \
            --retries 2 \
            "${packages[@]}"; then
            echo
            echo "${desc} 安装成功，使用镜像: ${mirror}"
            return 0
        fi
        echo "该镜像安装失败，自动尝试下一个..."
    done

    echo
    echo "错误: 所有国内镜像均安装失败：${desc}"
    return 1
}

echo "[1/8] 检查系统环境..."
command -v python3 >/dev/null 2>&1 || {
    echo "错误: 未找到 python3"
    exit 1
}

command -v nvidia-smi >/dev/null 2>&1 || {
    echo "错误: 未找到 nvidia-smi"
    exit 1
}

echo
python3 --version

echo
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

echo
echo "[2/8] 检查 Python venv..."
if ! python3 - <<'PY' >/dev/null 2>&1
import venv
PY
then
    sudo apt-get update
    sudo apt-get install -y python3-venv
fi

echo
echo "[3/8] 检查 FFmpeg..."
if ! command -v ffmpeg >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y ffmpeg
else
    ffmpeg -version | head -n 1
fi

echo
echo "[4/8] 创建项目目录..."
mkdir -p "${PROJECT_DIR}"/{data/videos,data/images,data/labels,models,outputs,logs,scripts}

echo
echo "[5/8] 创建/复用 Python 虚拟环境..."
if [ ! -d "${VENV_DIR}" ]; then
    python3 -m venv "${VENV_DIR}"
else
    echo "虚拟环境已存在: ${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

echo
echo "[6/8] 升级 pip / setuptools / wheel..."
install_with_mirrors "基础打包工具" --upgrade pip setuptools wheel

echo
echo "[7/8] 安装 PyTorch 和 YOLO 依赖..."

# 这里直接从国内 PyPI 镜像安装 torch。
# Linux x86_64 的 PyTorch wheel 会自动解析其 CUDA 相关依赖。
install_with_mirrors "PyTorch" torch torchvision

install_with_mirrors "YOLO及常用依赖" \
    ultralytics \
    opencv-python \
    pandas \
    numpy \
    pillow \
    tqdm \
    pyyaml \
    matplotlib

echo
echo "[8/8] 验证 GPU / CUDA / YOLO..."

python - <<'PY'
import sys
import torch
import cv2

print("=" * 70)
print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("PyTorch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("OpenCV:", cv2.__version__)

if not torch.cuda.is_available():
    print()
    print("错误: torch 已安装，但 CUDA 不可用。")
    print("请把以上输出发给我，我们再针对 PyTorch CUDA 版本处理。")
    raise SystemExit(2)

print("GPU:", torch.cuda.get_device_name(0))
props = torch.cuda.get_device_properties(0)
print("VRAM GB:", round(props.total_memory / 1024**3, 2))

print()
print("CUDA 矩阵计算测试...")
x = torch.randn(2048, 2048, device="cuda")
y = x @ x
torch.cuda.synchronize()
print("CUDA 计算测试: OK", tuple(y.shape))

from ultralytics import YOLO
print("Ultralytics 导入: OK")

print("=" * 70)
PY

echo
echo "========================================"
echo " 安装完成"
echo "========================================"
echo
echo "项目目录:"
echo "  ${PROJECT_DIR}"
echo
echo "以后进入环境:"
echo "  cd ${PROJECT_DIR}"
echo "  source .venv/bin/activate"
echo
echo "视频目录:"
echo "  ${PROJECT_DIR}/data/videos/"
echo
echo "输出目录:"
echo "  ${PROJECT_DIR}/outputs/"
