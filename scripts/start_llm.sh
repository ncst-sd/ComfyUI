#!/usr/bin/env bash
set -euo pipefail

PROJECT="$HOME/yolo_device_monitor"
LLM_ENV="$PROJECT/llm_env"
VLLM_BIN="$LLM_ENV/bin/vllm"
MODELS_DIR="$PROJECT/models"

echo "============================================================"
echo "            LLM / vLLM 交互式启动器"
echo "============================================================"
echo

if [ ! -x "$VLLM_BIN" ]; then
    echo "错误：找不到 vLLM：$VLLM_BIN"
    exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "错误：找不到 nvidia-smi。"
    exit 1
fi

if [ ! -d "$MODELS_DIR" ]; then
    echo "错误：模型目录不存在：$MODELS_DIR"
    exit 1
fi

mapfile -t GPU_ROWS < <(
    nvidia-smi       --query-gpu=index,name,uuid,pci.bus_id,memory.total,memory.free       --format=csv,noheader,nounits
)

if [ "${#GPU_ROWS[@]}" -eq 0 ]; then
    echo "错误：没有检测到 NVIDIA GPU。"
    exit 1
fi

echo "检测到以下物理 GPU："
echo

for i in "${!GPU_ROWS[@]}"; do
    IFS=',' read -r IDX NAME UUID BUS TOTAL FREE <<< "${GPU_ROWS[$i]}"
    IDX="$(echo "$IDX" | xargs)"
    NAME="$(echo "$NAME" | xargs)"
    UUID="$(echo "$UUID" | xargs)"
    BUS="$(echo "$BUS" | xargs)"
    TOTAL="$(echo "$TOTAL" | xargs)"
    FREE="$(echo "$FREE" | xargs)"

    echo "  $((i+1))) $NAME"
    echo "      nvidia-smi index : $IDX"
    echo "      UUID             : $UUID"
    echo "      PCI Bus          : $BUS"
    echo "      显存总量         : ${TOTAL} MiB"
    echo "      当前空闲         : ${FREE} MiB"
    echo
done

echo "可选择一张或多张 GPU，例如：1 或 1,2"
read -r -p "请选择用于 LLM 的 GPU [默认 1]: " GPU_CHOICES
GPU_CHOICES="${GPU_CHOICES:-1}"
GPU_CHOICES="$(echo "$GPU_CHOICES" | tr -d ' ')"

IFS=',' read -ra SELECTED <<< "$GPU_CHOICES"
VISIBLE_UUIDS=()
SELECTED_NAMES=()

for CHOICE in "${SELECTED[@]}"; do
    if ! [[ "$CHOICE" =~ ^[0-9]+$ ]]; then
        echo "错误：GPU 选择必须是数字。"
        exit 1
    fi

    POS=$((CHOICE - 1))

    if [ "$POS" -lt 0 ] || [ "$POS" -ge "${#GPU_ROWS[@]}" ]; then
        echo "错误：不存在 GPU 选项 $CHOICE。"
        exit 1
    fi

    IFS=',' read -r IDX NAME UUID BUS TOTAL FREE <<< "${GPU_ROWS[$POS]}"
    NAME="$(echo "$NAME" | xargs)"
    UUID="$(echo "$UUID" | xargs)"

    VISIBLE_UUIDS+=("$UUID")
    SELECTED_NAMES+=("$NAME")
done

CUDA_VISIBLE="$(IFS=,; echo "${VISIBLE_UUIDS[*]}")"
TP_SIZE="${#VISIBLE_UUIDS[@]}"

echo
echo "已选择 GPU："
for i in "${!VISIBLE_UUIDS[@]}"; do
    echo "  vLLM cuda:$i -> ${SELECTED_NAMES[$i]}"
    echo "                  ${VISIBLE_UUIDS[$i]}"
done
echo

mapfile -t MODEL_DIRS < <(
    find "$MODELS_DIR"       -mindepth 1       -maxdepth 2       -type f       -name "config.json"       -printf '%h\n'       2>/dev/null       | sort -u
)

MODELS=()
for M in "${MODEL_DIRS[@]}"; do
    B="$(basename "$M")"
    case "$B" in
        *bge*|*embedding*|*Embedding*)
            continue
            ;;
    esac
    MODELS+=("$M")
done

echo "本地模型："
echo

if [ "${#MODELS[@]}" -gt 0 ]; then
    for i in "${!MODELS[@]}"; do
        M="${MODELS[$i]}"
        SIZE="$(du -sh "$M" 2>/dev/null | awk '{print $1}')"
        echo "  $((i+1))) $(basename "$M") [$SIZE]"
        echo "      $M"
    done

    echo "  0) 手动输入模型路径"
    echo

    read -r -p "请选择模型 [默认 1]: " MODEL_CHOICE
    MODEL_CHOICE="${MODEL_CHOICE:-1}"

    if [ "$MODEL_CHOICE" = "0" ]; then
        read -r -p "请输入模型完整路径: " MODEL_PATH
    else
        if ! [[ "$MODEL_CHOICE" =~ ^[0-9]+$ ]]; then
            echo "错误：模型编号必须是数字。"
            exit 1
        fi

        POS=$((MODEL_CHOICE - 1))

        if [ "$POS" -lt 0 ] || [ "$POS" -ge "${#MODELS[@]}" ]; then
            echo "错误：模型编号不存在。"
            exit 1
        fi

        MODEL_PATH="${MODELS[$POS]}"
    fi
else
    read -r -p "未自动发现 LLM，请输入模型完整路径: " MODEL_PATH
fi

MODEL_PATH="$(realpath -m "$MODEL_PATH")"

if [ ! -d "$MODEL_PATH" ]; then
    echo "错误：模型目录不存在：$MODEL_PATH"
    exit 1
fi

echo
echo "已选择模型：$(basename "$MODEL_PATH")"
echo "路径：$MODEL_PATH"
echo

read -r -p "served-model-name [默认 device-monitor-llm]: " SERVED_NAME
SERVED_NAME="${SERVED_NAME:-device-monitor-llm}"

read -r -p "端口 [默认 8000]: " PORT
PORT="${PORT:-8000}"

read -r -p "max-model-len [默认 8192]: " MAX_MODEL_LEN
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

read -r -p "gpu-memory-utilization [默认 0.80]: " GPU_UTIL
GPU_UTIL="${GPU_UTIL:-0.80}"

echo
echo "dtype："
echo "  1) bfloat16"
echo "  2) float16"
echo "  3) auto"
read -r -p "请选择 dtype [默认 1]: " DTYPE_CHOICE
DTYPE_CHOICE="${DTYPE_CHOICE:-1}"

case "$DTYPE_CHOICE" in
    1) DTYPE="bfloat16" ;;
    2) DTYPE="float16" ;;
    3) DTYPE="auto" ;;
    *) echo "错误：dtype 选项无效。"; exit 1 ;;
esac

EXTRA_ARGS=()

echo
read -r -p "是否启用 trust-remote-code？[y/N]: " TRUST_REMOTE
TRUST_REMOTE="${TRUST_REMOTE:-N}"
if [[ "$TRUST_REMOTE" =~ ^[Yy]$ ]]; then
    EXTRA_ARGS+=("--trust-remote-code")
fi

echo
read -r -p "是否输入额外 vLLM 参数？[y/N]: " EXTRA_CONFIRM
EXTRA_CONFIRM="${EXTRA_CONFIRM:-N}"

if [[ "$EXTRA_CONFIRM" =~ ^[Yy]$ ]]; then
    echo "例如：--enforce-eager --max-num-seqs 8"
    read -r -p "额外参数: " EXTRA_INPUT
    if [ -n "$EXTRA_INPUT" ]; then
        # shellcheck disable=SC2206
        EXTRA_ARRAY=($EXTRA_INPUT)
        EXTRA_ARGS+=("${EXTRA_ARRAY[@]}")
    fi
fi

echo
echo "============================================================"
echo "最终启动配置"
echo "============================================================"

for i in "${!VISIBLE_UUIDS[@]}"; do
    echo "  cuda:$i -> ${SELECTED_NAMES[$i]}"
    echo "            ${VISIBLE_UUIDS[$i]}"
done

echo
echo "Tensor Parallel       : $TP_SIZE"
echo "模型                  : $MODEL_PATH"
echo "served-model-name     : $SERVED_NAME"
echo "host                  : 127.0.0.1"
echo "port                  : $PORT"
echo "dtype                 : $DTYPE"
echo "max-model-len         : $MAX_MODEL_LEN"
echo "gpu-memory-utilization: $GPU_UTIL"
echo
echo "OpenAI 兼容 API："
echo "  http://127.0.0.1:$PORT/v1"
echo

read -r -p "确认启动？[Y/n]: " CONFIRM
CONFIRM="${CONFIRM:-Y}"

if ! [[ "$CONFIRM" =~ ^[Yy]$ ]]; then
    echo "已取消。"
    exit 0
fi

export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE"

CMD=(
    "$VLLM_BIN"
    serve
    "$MODEL_PATH"
    --served-model-name "$SERVED_NAME"
    --host 127.0.0.1
    --port "$PORT"
    --dtype "$DTYPE"
    --tensor-parallel-size "$TP_SIZE"
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_UTIL"
)

if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
    CMD+=("${EXTRA_ARGS[@]}")
fi

echo
echo "============================================================"
echo "正在启动 vLLM"
echo "按 Ctrl+C 停止"
echo "============================================================"
echo
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE"
echo

exec "${CMD[@]}"

