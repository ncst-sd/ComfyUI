#!/usr/bin/env bash
set -euo pipefail

PROJECT="$HOME/yolo_device_monitor"
SCRIPT="$PROJECT/scripts/prelabel_indicator_grounding_dino_qwen.py"

cd "$PROJECT"
source qwen_vl_env/bin/activate

echo "============================================================"
echo " AI 指示灯预标注 - 交互式运行"
echo "============================================================"

# ------------------------------------------------------------
# 1. 选择设备
#    只列出已经存在 manual_indicator_schema.json 的设备
# ------------------------------------------------------------
PACKAGE_ROOT="$PROJECT/annotation_packages_best_frames_fast"

if [[ $# -ge 1 && -n "${1:-}" ]]; then
  DEVICE="$1"
else
  echo
  echo "------------------------------------------------------------"
  echo "设备选择"
  echo "仅显示已经存在 manual_indicator_schema.json 的设备"
  echo "------------------------------------------------------------"

  mapfile -t DEVICE_DIRS < <(
    find "$PACKAGE_ROOT" \
      -mindepth 1 -maxdepth 1 -type d \
      -exec test -f "{}/manual_indicator_schema.json" \; \
      -print \
      | sort
  )

  if [[ ${#DEVICE_DIRS[@]} -eq 0 ]]; then
    echo "错误：没有找到包含 manual_indicator_schema.json 的设备目录。"
    exit 1
  fi

  for i in "${!DEVICE_DIRS[@]}"; do
    dir="${DEVICE_DIRS[$i]}"
    name="$(basename "$dir")"

    if [[ -f "$dir/indicator_annotations.json" ]]; then
      state="已有 indicator_annotations.json"
    else
      state="可运行"
    fi

    printf "  %2d) %s | %s\n" "$((i + 1))" "$name" "$state"
  done

  echo

  while true; do
    read -r -p "请选择设备编号: " DEVICE_CHOICE

    if [[ "$DEVICE_CHOICE" =~ ^[1-9][0-9]*$ ]] \
       && (( DEVICE_CHOICE >= 1 )) \
       && (( DEVICE_CHOICE <= ${#DEVICE_DIRS[@]} )); then
      PACKAGE_DIR="${DEVICE_DIRS[$((DEVICE_CHOICE - 1))]}"
      DEVICE="$(basename "$PACKAGE_DIR")"
      break
    fi

    echo "请输入列表中的有效编号。"
  done
fi

PACKAGE_DIR="$PACKAGE_ROOT/$DEVICE"
IMAGES_DIR="$PACKAGE_DIR/images"
SCHEMA_FILE="$PACKAGE_DIR/manual_indicator_schema.json"
FINAL_ANN="$PACKAGE_DIR/indicator_annotations.json"

if [[ ! -d "$PACKAGE_DIR" ]]; then
  echo "错误：设备目录不存在："
  echo "  $PACKAGE_DIR"
  exit 1
fi

if [[ ! -f "$SCHEMA_FILE" ]]; then
  echo "错误：该设备没有 manual_indicator_schema.json："
  echo "  $SCHEMA_FILE"
  exit 1
fi

# 已有正式人工标注则立即终止，避免误覆盖
if [[ -f "$FINAL_ANN" ]]; then
  echo
  echo "安全停止：该设备已经存在正式 indicator_annotations.json："
  echo "  $FINAL_ANN"
  exit 0
fi

if [[ ! -d "$IMAGES_DIR" ]]; then
  echo "错误：images 文件夹不存在："
  echo "  $IMAGES_DIR"
  exit 1
fi

# ------------------------------------------------------------
# 2. 选择处理图片数量
# ------------------------------------------------------------
echo
echo "------------------------------------------------------------"
echo "图片处理范围"
echo "------------------------------------------------------------"
echo "排序规则：photo 开头图片优先，并按 1,2,3,...,10 自然正序处理"

TOTAL_IMAGES=$(
  find "$IMAGES_DIR" -maxdepth 1 -type f \
    \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.bmp' -o -iname '*.webp' \) \
    | wc -l
)

FAR_IMAGES=$(
  find "$IMAGES_DIR" -maxdepth 1 -type f \
    \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.bmp' -o -iname '*.webp' \) \
    | grep -c '__far__' || true
)

NORMAL_IMAGES=$((TOTAL_IMAGES - FAR_IMAGES))

echo "images 文件夹图片总数 : $TOTAL_IMAGES"
echo "其中普通图片           : $NORMAL_IMAGES"
echo "其中 __far__ 图片      : $FAR_IMAGES"
echo
echo "1) 自定义处理图片数量"
echo "2) 处理全部图片"
echo

while true; do
  read -r -p "请选择 [1/2]: " RANGE_CHOICE

  case "$RANGE_CHOICE" in
    1)
      while true; do
        read -r -p "请输入要处理的图片数量: " COUNT
        if [[ "$COUNT" =~ ^[1-9][0-9]*$ ]]; then
          LIMIT="$COUNT"
          break
        fi
        echo "请输入大于 0 的整数。"
      done
      break
      ;;
    2)
      LIMIT=0
      break
      ;;
    *)
      echo "请输入 1 或 2。"
      ;;
  esac
done

# ------------------------------------------------------------
# 3. 是否包含 __far__
# ------------------------------------------------------------
echo
echo "------------------------------------------------------------"
echo "是否处理 __far__ 视频抽帧"
echo "------------------------------------------------------------"
echo "1) 不处理 __far__（推荐先用于近距离清晰图）"
echo "2) 包含 __far__，处理 images 下所有图片"
echo

while true; do
  read -r -p "请选择 [1/2]: " FAR_CHOICE

  case "$FAR_CHOICE" in
    1)
      INCLUDE_FAR=0
      break
      ;;
    2)
      INCLUDE_FAR=1
      break
      ;;
    *)
      echo "请输入 1 或 2。"
      ;;
  esac
done

# ------------------------------------------------------------
# 4. 检测显卡并交互选择
# ------------------------------------------------------------
echo
echo "------------------------------------------------------------"
echo "显卡选择"
echo "------------------------------------------------------------"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "错误：没有找到 nvidia-smi，无法检测 NVIDIA GPU。"
  exit 1
fi

mapfile -t GPU_LINES < <(
  nvidia-smi \
    --query-gpu=index,name,memory.total,memory.free \
    --format=csv,noheader,nounits
)

if [[ ${#GPU_LINES[@]} -eq 0 ]]; then
  echo "错误：没有检测到 NVIDIA GPU。"
  exit 1
fi

echo "检测到以下显卡："
echo

for line in "${GPU_LINES[@]}"; do
  IFS=',' read -r idx name mem_total mem_free <<< "$line"

  idx="$(echo "$idx" | xargs)"
  name="$(echo "$name" | xargs)"
  mem_total="$(echo "$mem_total" | xargs)"
  mem_free="$(echo "$mem_free" | xargs)"

  printf "  GPU %s : %-30s 总显存 %s MiB / 空闲 %s MiB\n" \
    "$idx" "$name" "$mem_total" "$mem_free"
done

echo
echo "可以让 DINO 和 Qwen 使用同一张卡，也可以分开使用。"
echo

valid_gpu_index() {
  local target="$1"
  local line idx

  for line in "${GPU_LINES[@]}"; do
    IFS=',' read -r idx _ <<< "$line"
    idx="$(echo "$idx" | xargs)"
    if [[ "$idx" == "$target" ]]; then
      return 0
    fi
  done

  return 1
}

while true; do
  read -r -p "Grounding DINO 使用哪张 GPU？请输入编号: " DINO_GPU
  if valid_gpu_index "$DINO_GPU"; then
    break
  fi
  echo "GPU 编号不存在，请重新输入。"
done

while true; do
  read -r -p "Qwen-VL 使用哪张 GPU？请输入编号: " QWEN_GPU
  if valid_gpu_index "$QWEN_GPU"; then
    break
  fi
  echo "GPU 编号不存在，请重新输入。"
done

DINO_DEVICE="cuda:$DINO_GPU"
QWEN_DEVICE="cuda:$QWEN_GPU"

# ------------------------------------------------------------
# 5. 确认
# ------------------------------------------------------------
echo
echo "============================================================"
echo "运行参数确认"
echo "============================================================"
echo "设备           : $DEVICE"

if [[ "$LIMIT" -eq 0 ]]; then
  echo "处理数量       : 全部"
else
  echo "处理数量       : 前 $LIMIT 张"
fi

if [[ "$INCLUDE_FAR" -eq 1 ]]; then
  echo "__far__        : 包含"
else
  echo "__far__        : 跳过"
fi

echo "DINO GPU       : GPU $DINO_GPU ($DINO_DEVICE)"
echo "Qwen-VL GPU    : GPU $QWEN_GPU ($QWEN_DEVICE)"
echo "输出           : $PACKAGE_DIR/pre_annotations.json"
echo "============================================================"

read -r -p "确认开始？[Y/n]: " CONFIRM
CONFIRM="${CONFIRM:-Y}"

case "${CONFIRM,,}" in
  y|yes)
    ;;
  *)
    echo "已取消。"
    exit 0
    ;;
esac

# ------------------------------------------------------------
# 6. 组装参数并运行
# ------------------------------------------------------------
ARGS=(
  --device "$DEVICE"
  --dino-model "$PROJECT/models/grounding-dino-tiny"
  --qwen-model "$PROJECT/models/Qwen2.5-VL-7B-Instruct"
  --dino-device "$DINO_DEVICE"
  --qwen-device-map "$QWEN_DEVICE"
  --tile-size 1280
  --overlap 0.25
  --box-threshold 0.18
  --text-threshold 0.18
  --nms-iou 0.35
  --min-side 5
  --max-side 150
  --max-area 14000
  --max-aspect 3
  --context-scale 5
  --tight-scale 1.8
  --limit "$LIMIT"
  --save-debug
)

if [[ "$INCLUDE_FAR" -eq 1 ]]; then
  ARGS+=(--include-far)
fi

echo
echo "开始运行..."
echo

python "$SCRIPT" "${ARGS[@]}"

echo
echo "============================================================"
echo "AI 标注完成"
echo "结果文件："
echo "  $PACKAGE_DIR/pre_annotations.json"
echo "============================================================"


