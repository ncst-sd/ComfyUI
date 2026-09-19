# scripts 目录文件用途与状态说明

> 项目：`ncst-sd/ComfyUI`
>
> 目录：`scripts/`
>
> 说明：本文用于区分当前正式使用脚本、后续阶段脚本、备用工具和历史测试脚本，避免误运行旧版本。

---

## 一、当前指示灯标注主线

当前推荐流程：

```text
说明书
  ↓
build_manual_indicator_schema.py
  ↓
manual_indicator_schema.json
  ↓
run_grounding_dino_qwen_trial.sh
  ↓
prelabel_indicator_grounding_dino_qwen.py
  ↓
pre_annotations.json
  ↓
annotator.html 人工复核
  ↓
indicator_annotations.json
```

| 文件 | 状态 | 用途 |
|---|---|---|
| `build_manual_indicator_schema.py` | ✅ 最新版 / 正式使用 | 读取设备说明书，使用 Qwen2.5-VL 提取物理指示灯类别，处理状态与类别分离，并执行去重，最终生成 `manual_indicator_schema.json`。 |
| `prelabel_indicator_grounding_dino_qwen.py` | ✅ 最新版 / 正式使用 | 当前核心 AI 预标注脚本。使用 Grounding DINO 找候选灯，Qwen-VL 过滤螺丝/反光/端口等非灯目标，再结合 `manual_indicator_schema.json` 做类别判断。支持 `photo_*` 高权重模板、视频跨帧一致性、图片自然排序。 |
| `run_grounding_dino_qwen_trial.sh` | ✅ 最新版 / 推荐入口 | 交互式运行入口。自动列出已有 `manual_indicator_schema.json` 的设备，可选择设备、图片数量、是否包含 `__far__`、DINO GPU、Qwen GPU，然后调用 `prelabel_indicator_grounding_dino_qwen.py`。 |
| `rebuild_unannotated_indicator_packages.py` | ✅ 当前有效 | 重建设备标注包，只处理尚未存在正式 `indicator_annotations.json` 的设备，避免破坏已经完成人工标注的设备。 |
| `rebuild_unannotated_indicator_packages.sh` | ✅ 当前有效 | `rebuild_unannotated_indicator_packages.py` 的运行入口。 |
| `update_all_annotator_html.py` | ✅ 当前有效 | 将统一的最新版 `annotator.html` 批量复制到各设备目录。建议统一通过此脚本更新前端。 |
| `download_grounding_dino_tiny.sh` | ✅ 工具脚本 | 下载 Grounding DINO Tiny 本地模型。模型已存在时无需重复运行。 |
| `setup_qwen_vl_env.sh` | ✅ 工具脚本 | 创建或修复 Qwen-VL Python 环境。环境正常时无需重复运行。 |

---

## 二、设备识别与状态判定正式功能

这一组不是旧脚本，而是另一条正式主线。

### 设备识别

```text
原始设备图片 / 视频
  ↓
prepare_device_cls_dataset.py
  ↓
train_device_classifier.py
  ↓
test_device_classifier.py
  ↓
infer_device_video.py / device_recognition_ui.py
```

| 文件 | 状态 | 用途 |
|---|---|---|
| `prepare_equipment_dataset.py` | ✅ 保留 | 从原始设备目录整理 workspace，抽取图片/视频帧、说明书等基础数据。 |
| `prepare_device_cls_dataset.py` | ✅ 正式使用 | 构建设备分类 YOLO 数据集。 |
| `train_device_classifier.py` | ✅ 正式使用 | 训练设备分类模型。 |
| `test_device_classifier.py` | ✅ 正式使用 | 测试设备分类模型。 |
| `infer_device_video.py` | ✅ 正式使用 | 使用设备分类模型对图片或视频进行推理。 |
| `device_recognition_ui.py` | ✅ 正式使用 | 设备识别 Web UI。 |
| `start_device_recognition_ui.sh` | ✅ 正式使用 | 启动设备识别 UI。 |

### 规则、RAG、状态引擎

```text
巡检总表 / 说明书
  ↓
parse_inspection_rules.py / build_rag_index.py
  ↓
规则库 + RAG
  ↓
device_state_engine.py
  ↓
device_state_api.py
  ↓
yolo_state_bridge.py
```

| 文件 | 状态 | 用途 |
|---|---|---|
| `parse_inspection_rules.py` | ✅ 正式使用 | 从巡检总表提取设备、指示灯、颜色、状态、正常/异常规则。 |
| `build_device_alias_map.py` | ✅ 正式使用 | 建立规则库设备名与 workspace/实际设备目录名之间的映射。 |
| `add_device_rule.py` | ✅ 正式使用 | 增加现场确认的自定义确定性规则。 |
| `build_rag_index.py` | ✅ 正式使用 | 为说明书、规则等资料建立 Chroma RAG 索引。 |
| `device_state_engine.py` | ✅ 正式使用 | 规则优先 + RAG + 本地 LLM 的状态判定核心。 |
| `device_state_api.py` | ✅ 正式使用 | 提供常驻状态判定 API。 |
| `start_device_state_api.sh` | ✅ 正式使用 | 启动状态判定 API。 |
| `yolo_state_bridge.py` | ✅ 正式使用 | 将视觉检测结果转换为状态引擎可接受的输入。 |
| `start_llm.sh` | ✅ 正式使用 | 启动本地 LLM 服务。 |

---

## 三、后续训练指示灯 YOLO 时再使用

当前阶段仍以 AI 预标注 + 人工复核为主。只有当多个设备已经得到可信的 `indicator_annotations.json` 后，再进入这一组流程。

```text
indicator_annotations.json
  ↓
build_indicator_yolo_dataset.py
  ↓
YOLO 数据集
  ↓
train_indicator_detector.py
  ↓
best.pt
```

| 文件 | 状态 | 用途 |
|---|---|---|
| `generate_indicator_annotation_manifest.py` | 🟡 后续使用 | 生成指示灯人工标注清单。 |
| `prepare_indicator_detection_task.py` | 🟡 后续使用 | 为单设备指示灯检测准备首轮 YOLO 标注任务。 |
| `import_semantic_annotations.py` | 🟡 后续使用 | 将语义标注 JSON 转换为 YOLO 标签。 |
| `finalize_indicator_detection_dataset.py` | 🟡 后续使用 | 人工标注完成后切分 train / val / test。 |
| `build_indicator_yolo_dataset.py` | 🟡 后续使用 | 将 `indicator_annotations.json` 转换为 Ultralytics YOLO 数据集。 |
| `train_indicator_detector.py` | 🟡 后续使用 | 训练指示灯检测模型。 |
| `infer_indicator_video.py` | 🟡 后续使用 | 对图片/视频运行指示灯检测模型。 |
| `indicator_inference_ui.py` | 🟡 后续使用 | 指示灯检测 Web UI。 |
| `start_indicator_inference_ui.sh` | 🟡 后续使用 | 启动指示灯推理 UI。 |

> 当前人工标注尚未全部完成时，不建议提前运行 `build_indicator_yolo_dataset.py` 和 `train_indicator_detector.py`。

---

## 四、备用工具

| 文件 | 状态 | 用途 |
|---|---|---|
| `update_annotator_html_shortcuts.py` | 🟠 备用 | 给 `annotator.html` 增加自定义类别快捷键。不是主流程必要步骤。 |

建议以后将快捷键能力直接合并进最终版 `annotator.html`，避免长期依赖补丁脚本。

---

## 五、历史测试版 / 建议归档

以下脚本属于前期实验方案，目前已经被新的 Grounding DINO + Qwen 流程替代。

| 文件 | 状态 | 原用途 / 原因 |
|---|---|---|
| `dedupe_manual_indicator_schema.py` | ❌ 已被合并 | 原来单独对 `manual_indicator_schema.json` 去重。现在功能已合并进 `build_manual_indicator_schema.py`，不再需要单独运行。 |
| `prelabel_from_manual.py` | ❌ 旧版 | OpenCV 亮点检测 + OCR + 关键词匹配的早期预标注方案，误检较多。 |
| `prelabel_images_with_qwen_vl.py` | ❌ 测试版 | 逐类别让 Qwen-VL 直接找 bbox，稳定性较差。 |
| `prelabel_indicator_with_tiled_qwen_vl.py` | ❌ 测试版 | 纯 Qwen tiled 找灯 + 分类的实验方案，已被 DINO + Qwen 取代。 |
| `patch_prelabel_bbox_parser.py` | ❌ 一次性补丁 | 只用于修复旧 `prelabel_images_with_qwen_vl.py` 的 `bbox_2d` 解析问题。旧脚本不用后，该补丁也不需要。 |
| `run_manual_prelabel.sh` | ❌ 旧入口 | 对应早期 manual prelabel 流程。 |
| `run_qwen_vl_prelabel_trial.sh` | ❌ 旧入口 | 对应早期 Qwen-VL 直接预标注实验。 |
| `run_tiled_qwen_vl_trial.sh` | ❌ 旧入口 | 对应 tiled Qwen-VL 实验。 |
| `run_qwen_vl_schema_then_show_command.sh` | ❌ 旧流程 | 早期 schema + Qwen-VL 流程辅助脚本。 |
| `rebuild_annotation_packages_best_frames.py` | ❌ 旧版 | 早期标注包重建脚本。 |
| `rebuild_annotation_packages_best_frames_fast.py` | ❌ 旧版 | 上一个脚本的快速版，后来又被 `rebuild_unannotated_indicator_packages.py` 替代。 |
| `export_all_other_devices_for_annotation.py` | ❌ 旧版 | 早期离线人工标注包导出方案。 |
| `build_semantic_annotation_package.py` | ❌ 旧版 | 早期语义标注包构建方式。 |
| `update_annotator_html_only.py` | ❌ 不建议继续使用 | 内部嵌入旧版 HTML，可能把当前新版 `annotator.html` 覆盖回旧版本。 |

建议将这一组移动到：

```text
scripts/archive/
```

先归档，不急着删除。

---

## 六、不应提交到 Git 的文件

以下属于 Python 缓存文件，不应保留在仓库中：

```text
scripts/__pycache__/
*.pyc
```

建议 `.gitignore` 中加入：

```gitignore
__pycache__/
*.pyc
```

---

## 七、当前最重要的三份脚本

现在进行指示灯 AI 标注时，最主要只需要记住：

```text
1. build_manual_indicator_schema.py
   ↓
   生成 manual_indicator_schema.json

2. run_grounding_dino_qwen_trial.sh
   ↓
   交互式选择设备、图片、GPU

3. prelabel_indicator_grounding_dino_qwen.py
   ↓
   Grounding DINO + Qwen 生成 pre_annotations.json
```

其中：

- `build_manual_indicator_schema.py`：负责“说明书 → 类别表”
- `prelabel_indicator_grounding_dino_qwen.py`：负责“图片 → AI 预标注”
- `run_grounding_dino_qwen_trial.sh`：负责“方便、安全地启动 AI 预标注”

---

## 八、状态标记说明

| 标记 | 含义 |
|---|---|
| ✅ | 当前正式使用 / 应保留 |
| 🟡 | 后续阶段使用，当前暂时不用 |
| 🟠 | 备用工具 |
| ❌ | 旧版、测试版、一次性补丁或已被替代 |

---

## 九、建议的 scripts 目录整理方式

```text
scripts/
├── current/
│   ├── build_manual_indicator_schema.py
│   ├── prelabel_indicator_grounding_dino_qwen.py
│   ├── run_grounding_dino_qwen_trial.sh
│   ├── rebuild_unannotated_indicator_packages.py
│   ├── rebuild_unannotated_indicator_packages.sh
│   └── update_all_annotator_html.py
│
├── device_recognition/
│   ├── prepare_device_cls_dataset.py
│   ├── train_device_classifier.py
│   ├── test_device_classifier.py
│   ├── infer_device_video.py
│   └── device_recognition_ui.py
│
├── state_engine/
│   ├── parse_inspection_rules.py
│   ├── build_device_alias_map.py
│   ├── add_device_rule.py
│   ├── build_rag_index.py
│   ├── device_state_engine.py
│   ├── device_state_api.py
│   └── yolo_state_bridge.py
│
├── future_indicator_yolo/
│   ├── build_indicator_yolo_dataset.py
│   ├── train_indicator_detector.py
│   ├── infer_indicator_video.py
│   └── ...
│
└── archive/
    ├── prelabel_from_manual.py
    ├── prelabel_images_with_qwen_vl.py
    ├── prelabel_indicator_with_tiled_qwen_vl.py
    ├── dedupe_manual_indicator_schema.py
    └── ...
```

如果暂时不想改目录结构，也可以只保留本文档，通过状态表判断哪个脚本应该运行。

