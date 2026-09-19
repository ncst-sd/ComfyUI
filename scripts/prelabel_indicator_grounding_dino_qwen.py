#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
prelabel_indicator_tiled_grounding_dino_qwen_v2.py

相较 v1 的关键改进：
1) Grounding DINO 只使用更窄的提示词：
      "small led indicator light."
2) 对 DINO 候选增加：
      - label 过滤
      - 面积过滤
      - 长宽比过滤
      - 过大边长过滤
3) Qwen 验证时，不再只给“普通 crop”。
   会生成一张带红色矩形框的 context 图，明确告诉 Qwen：
      “只判断红框里的目标”
   同时再给一张“红框目标的紧裁剪图”。
4) 验证 prompt 明确：
      - 不得根据红框外的 RUN/ALM 丝印判断红框对象
      - 红框内必须本身是 LED/灯窗/灯珠
5) 分类阶段：
      - 同样使用“红框 context + tight crop”
      - 必须依据红框目标附近的局部丝印
      - 如果只能在较远处看到 RUN，不允许把当前目标分类为 RUN
6) 增加重复中心点抑制，减少同一个灯附近多框。
"""

import argparse
import json
import math
import re
import shutil
import tempfile
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
    Qwen2_5_VLForConditionalGeneration,
)

from qwen_vl_utils import process_vision_info


HOME = Path.home()
PROJECT = HOME / "yolo_device_monitor"

DEFAULT_PACKAGE_ROOT = PROJECT / "annotation_packages_best_frames_fast"
DEFAULT_QWEN_MODEL = PROJECT / "models" / "Qwen2.5-VL-7B-Instruct"
DEFAULT_DINO_MODEL = PROJECT / "models" / "grounding-dino-tiny"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def natural_image_name_key(name: str):
    """
    图片自然排序：
    1) photo 开头的原始照片永远排在最前面；
    2) 文件名中的数字按数值排序：1,2,3,...,10，而不是 1,10,2；
    3) 其它图片排在 photo 后面，同样使用自然排序。
    """
    name = str(name)
    stem = Path(name).stem.lower()

    photo_priority = 0 if stem.startswith("photo") else 1

    parts = re.split(r"(\d+)", stem)
    natural_parts = []

    for part in parts:
        if not part:
            continue
        if part.isdigit():
            natural_parts.append((0, int(part)))
        else:
            natural_parts.append((1, part))

    return (photo_priority, natural_parts, Path(name).suffix.lower())


def natural_image_path_key(path: Path):
    return natural_image_name_key(path.name)


def list_images_sorted(images_dir: Path):
    """
    统一读取 images/：
    photo_1, photo_2, ... photo_10 优先，
    然后其它图片继续按自然数字顺序。
    """
    return sorted(
        [
            p for p in images_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=natural_image_path_key,
    )


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--device", default=None, help="设备目录名；不填写时交互选择")
    p.add_argument("--package-root", type=Path, default=DEFAULT_PACKAGE_ROOT)

    p.add_argument("--dino-model", default=str(DEFAULT_DINO_MODEL))
    p.add_argument("--qwen-model", default=str(DEFAULT_QWEN_MODEL))

    p.add_argument("--dino-device", default="cuda:0")
    p.add_argument("--qwen-device-map", default="cuda:0")

    p.add_argument("--limit", type=int, default=4)
    p.add_argument("--include-far", action="store_true")

    p.add_argument("--tile-size", type=int, default=1280)
    p.add_argument("--overlap", type=float, default=0.25)

    p.add_argument("--box-threshold", type=float, default=0.18)
    p.add_argument("--text-threshold", type=float, default=0.18)
    p.add_argument("--nms-iou", type=float, default=0.35)

    p.add_argument("--min-side", type=float, default=5.0)
    p.add_argument("--max-side", type=float, default=150.0)
    p.add_argument("--max-area", type=float, default=14000.0)
    p.add_argument("--max-aspect", type=float, default=3.0)

    p.add_argument("--center-distance-ratio", type=float, default=0.65,
                   help="中心过近候选抑制阈值，相对于两框平均长边")

    p.add_argument("--context-scale", type=float, default=5.0)
    p.add_argument("--min-context-width", type=int, default=320)
    p.add_argument("--min-context-height", type=int, default=240)

    p.add_argument("--tight-scale", type=float, default=1.8,
                   help="候选本体紧裁剪放大倍数")

    p.add_argument("--save-debug", action="store_true")
    p.add_argument("--skip-classification", action="store_true")

    return p.parse_args()



def list_devices_with_schema(package_root: Path):
    """
    列出 package_root 下所有已经存在 manual_indicator_schema.json 的设备目录。
    """
    if not package_root.exists():
        return []

    devices = []

    for p in sorted(
        [x for x in package_root.iterdir() if x.is_dir()],
        key=lambda x: x.name.lower()
    ):
        schema = p / "manual_indicator_schema.json"
        if schema.exists():
            devices.append(p)

    return devices


def choose_device_with_schema(package_root: Path) -> str:
    devices = list_devices_with_schema(package_root)

    if not devices:
        raise SystemExit(
            f"没有找到包含 manual_indicator_schema.json 的设备目录: {package_root}"
        )

    print()
    print("=" * 80)
    print("可运行 AI 指示灯预标注的设备")
    print("仅显示已存在 manual_indicator_schema.json 的设备")
    print("=" * 80)

    for i, p in enumerate(devices, 1):
        ann = p / "indicator_annotations.json"

        if ann.exists():
            state = "已有 indicator_annotations.json（选择后会安全停止）"
        else:
            state = "可运行"

        print(f"{i:2d}) {p.name} | {state}")

    while True:
        s = input("\n请选择设备编号: ").strip()

        if not s.isdigit():
            print("请输入有效编号。")
            continue

        idx = int(s)

        if not (1 <= idx <= len(devices)):
            print("请输入列表中的有效编号。")
            continue

        selected = devices[idx - 1]

        # 保留原脚本的安全原则：
        # 已有正式标注的设备不允许继续跑 AI 预标注。
        if (selected / "indicator_annotations.json").exists():
            raise SystemExit(
                "\n安全停止：该设备已经存在正式 indicator_annotations.json：\n"
                f"{selected / 'indicator_annotations.json'}"
            )

        return selected.name


def extract_json_object(text: str):
    text = (text or "").strip()

    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S | re.I)
    if m:
        text = m.group(1).strip()

    a = text.find("{")
    b = text.rfind("}")

    if a >= 0 and b > a:
        try:
            obj = json.loads(text[a:b + 1])
            return obj if isinstance(obj, dict) else {}
        except Exception:
            pass

    return {}


def build_tiles(width, height, tile_size, overlap):
    step = max(1, int(round(tile_size * (1 - overlap))))

    def positions(total):
        if total <= tile_size:
            return [0]

        pos = list(range(0, total - tile_size + 1, step))
        last = total - tile_size

        if not pos or pos[-1] != last:
            pos.append(last)

        return sorted(set(pos))

    tiles = []

    for y in positions(height):
        for x in positions(width):
            tiles.append({
                "x1": x,
                "y1": y,
                "x2": min(width, x + tile_size),
                "y2": min(height, y + tile_size),
            })

    return tiles


def clamp_box(box, W, H):
    x1, y1, x2, y2 = map(float, box)

    x1 = max(0.0, min(float(W), x1))
    y1 = max(0.0, min(float(H), y1))
    x2 = max(0.0, min(float(W), x2))
    y2 = max(0.0, min(float(H), y2))

    if x2 <= x1 or y2 <= y1:
        return None

    return [x1, y1, x2, y2]


def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    inter = iw * ih
    if inter <= 0:
        return 0.0

    aa = (ax2 - ax1) * (ay2 - ay1)
    bb = (bx2 - bx1) * (by2 - by1)

    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


def center_distance(a, b):
    acx = (a[0] + a[2]) / 2
    acy = (a[1] + a[3]) / 2

    bcx = (b[0] + b[2]) / 2
    bcy = (b[1] + b[3]) / 2

    return math.hypot(acx - bcx, acy - bcy)


def suppress_duplicates(candidates, iou_threshold, center_ratio):
    ordered = sorted(candidates, key=lambda x: x["score"], reverse=True)

    kept = []
    rejected = []

    for c in ordered:
        reject_info = None

        for k in kept:
            ov = bbox_iou(c["box"], k["box"])

            cw = c["box"][2] - c["box"][0]
            ch = c["box"][3] - c["box"][1]
            kw = k["box"][2] - k["box"][0]
            kh = k["box"][3] - k["box"][1]

            avg_long = max(
                1.0,
                (max(cw, ch) + max(kw, kh)) / 2
            )

            cd = center_distance(c["box"], k["box"])

            if ov >= iou_threshold or cd <= avg_long * center_ratio:
                reject_info = {
                    "iou": ov,
                    "center_distance": cd,
                    "kept_box": k["box"],
                    "kept_score": k["score"],
                }
                break

        if reject_info is None:
            kept.append(c)
        else:
            r = dict(c)
            r["duplicate_conflict"] = reject_info
            rejected.append(r)

    return kept, rejected


def label_ok(label):
    s = (label or "").lower()

    # 纯 panel / status / signal 直接不要
    positive = (
        "led" in s
        or "indicator light" in s
        or "status light" in s
        or "signal light" in s
        or s.strip() == "light"
        or "light indicator" in s
    )

    if not positive:
        return False

    # 如果模型明确只给 panel，不要
    if "panel" in s and "light" not in s:
        return False

    return True


def geometric_check(box, min_side, max_side, max_area, max_aspect):
    x1, y1, x2, y2 = box

    w = x2 - x1
    h = y2 - y1
    area = w * h

    if w < min_side or h < min_side:
        return False, "too_small"

    if w > max_side or h > max_side:
        return False, "too_large_side"

    if area > max_area:
        return False, "too_large_area"

    aspect = max(w / max(h, 1e-6), h / max(w, 1e-6))

    if aspect > max_aspect:
        return False, "bad_aspect"

    return True, ""


def expand_context(box, W, H, scale, min_w, min_h):
    x1, y1, x2, y2 = box

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)

    cw = max(min_w, bw * scale)
    ch = max(min_h, bh * scale)

    nx1 = max(0, int(round(cx - cw / 2)))
    ny1 = max(0, int(round(cy - ch / 2)))
    nx2 = min(W, int(round(cx + cw / 2)))
    ny2 = min(H, int(round(cy + ch / 2)))

    return nx1, ny1, nx2, ny2


def expand_tight(box, W, H, scale):
    x1, y1, x2, y2 = box

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)

    tw = bw * scale
    th = bh * scale

    nx1 = max(0, int(round(cx - tw / 2)))
    ny1 = max(0, int(round(cy - th / 2)))
    nx2 = min(W, int(round(cx + tw / 2)))
    ny2 = min(H, int(round(cy + th / 2)))

    return nx1, ny1, nx2, ny2


def draw_candidate_context(image, global_box, context_box, save_path):
    cx1, cy1, cx2, cy2 = context_box
    context = image.crop((cx1, cy1, cx2, cy2))

    gx1, gy1, gx2, gy2 = global_box

    local_box = [
        gx1 - cx1,
        gy1 - cy1,
        gx2 - cx1,
        gy2 - cy1,
    ]

    draw = ImageDraw.Draw(context)

    # 明显粗框，明确告诉 Qwen 只看这个框
    for i in range(5):
        draw.rectangle(
            [
                local_box[0] - i,
                local_box[1] - i,
                local_box[2] + i,
                local_box[3] + i,
            ],
            outline="red",
            width=1,
        )

    context.save(save_path, quality=96)


def load_schema(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    indicators = data.get("indicators", [])

    if not isinstance(indicators, list) or not indicators:
        raise SystemExit("manual_indicator_schema.json 中没有 indicators")

    return indicators


def build_allowed_classes_text(indicators):
    lines = []

    for x in indicators:
        cid = int(x["class_id"])
        name = str(x.get("name", "")).strip()

        parts = [f"{cid}: {name}"]

        module = str(x.get("module", "")).strip()
        if module:
            parts.append("模块=" + module)

        loc = str(x.get("location_hint", "")).strip()
        if loc:
            parts.append("位置=" + loc)

        aliases = x.get("aliases", [])
        if isinstance(aliases, list) and aliases:
            parts.append("别名=" + " / ".join(map(str, aliases[:6])))

        evidence = str(x.get("evidence", "")).strip()
        if evidence:
            parts.append("说明书=" + evidence[:180])

        lines.append("；".join(parts))

    return "\n".join(lines)


def post_process_grounding_dino(
    processor,
    outputs,
    input_ids,
    target_sizes,
    box_threshold,
    text_threshold,
):
    try:
        return processor.post_process_grounded_object_detection(
            outputs,
            input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=target_sizes,
        )
    except TypeError:
        return processor.post_process_grounded_object_detection(
            outputs,
            input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=target_sizes,
        )


def detect_tile_with_dino(
    model,
    processor,
    tile_image,
    device,
    box_threshold,
    text_threshold,
):
    # 比 v1 更窄，避免 panel / status / signal 被大量独立解析
    text_prompt = "small led indicator light."

    inputs = processor(
        images=tile_image,
        text=text_prompt,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        outputs = model(**inputs)

    target_sizes = torch.tensor(
        [tile_image.size[::-1]],
        device=device,
    )

    result = post_process_grounding_dino(
        processor,
        outputs,
        inputs.input_ids,
        target_sizes,
        box_threshold,
        text_threshold,
    )[0]

    boxes = result["boxes"].detach().cpu().tolist()
    scores = result["scores"].detach().cpu().tolist()

    labels = result.get(
        "text_labels",
        result.get("labels", [])
    )

    if hasattr(labels, "detach"):
        labels = labels.detach().cpu().tolist()

    output = []

    for i, box in enumerate(boxes):
        output.append({
            "box": [float(v) for v in box],
            "score": float(scores[i]),
            "label": str(labels[i]) if i < len(labels) else "",
        })

    return output


def run_qwen(model, processor, messages, max_new_tokens):
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids
        in zip(inputs.input_ids, output)
    ]

    return processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
    )[0]


def verify_candidate(
    model,
    processor,
    marked_context_path,
    tight_crop_path,
):
    """
    内部六分类验证：
      led / screw / reflection / port / button / other

    只有 led 才进入正式指示灯分类。
    这些内部类别不会写进前端类别表。
    """
    prompt = """
你会看到两张图：

第1张：设备局部上下文，当前候选目标被红色矩形框精确标出。
第2张：红框目标本身的紧裁剪。

你的任务不是判断 RUN/ALM 等类别，而是先判断：
“红框里的这个物体本身到底是什么”。

只能从下面 6 个内部类别中选择一个：

led
- 真实物理指示灯、LED灯珠、灯帽、灯窗、半透明发光窗口、明确发光指示点

screw
- 螺丝、螺钉、螺母、铆钉、金属固定件
- 包括圆头螺丝、十字槽、一字槽、六角结构、带金属高光的固定件

reflection
- 金属反光、塑料高光、镜面亮点、玻璃反射、非灯体发光点

port
- 网口、光口、接口孔、插槽、连接器、光纤头、端口结构本身

button
- 按钮、按键、开关、拨码、旋钮

other
- 其它任何不是明确物理指示灯的东西
- 或者图像太模糊、无法确认

严格规则：
1. 只判断红框里的目标本身。
2. 红框外即使存在 RUN、ALM、PWR 等文字，也不能证明红框内目标是灯。
3. 红框外即使存在其它真正的灯，也不能把它们算到当前候选。
4. 如果红框目标是圆形金属件，并且有槽纹、螺帽轮廓、金属镜面反光，应优先判 screw。
5. 如果只是一个亮点但看不到灯窗/灯珠/半透明灯帽等明确灯体结构，应判 reflection 或 other，不要判 led。
6. 只有能看出“明确物理指示灯结构”时才允许判 led。
7. 不确定时优先 other，不要猜 led。

只输出 JSON，不要解释：

{"object_type":"led","reason":"红框内可见明确LED灯窗"}

或者：

{"object_type":"screw","reason":"红框内是圆头金属固定螺丝"}

或者：

{"object_type":"reflection","reason":"红框内只是金属高光"}

或者：

{"object_type":"port","reason":"红框内是接口结构"}

或者：

{"object_type":"button","reason":"红框内是按钮/开关"}

或者：

{"object_type":"other","reason":"无法确认是物理指示灯"}
""".strip()

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": str(marked_context_path)},
            {"type": "image", "image": str(tight_crop_path)},
            {"type": "text", "text": prompt},
        ],
    }]

    raw = run_qwen(
        model,
        processor,
        messages,
        max_new_tokens=220,
    )

    data = extract_json_object(raw)

    object_type = str(
        data.get("object_type", "other")
    ).strip().lower()

    valid_types = {
        "led",
        "screw",
        "reflection",
        "port",
        "button",
        "other",
    }

    if object_type not in valid_types:
        object_type = "other"

    reason = str(
        data.get("reason", "")
    ).strip()

    return object_type, reason, raw



def normalize_text(s):
    s = str(s or "").upper()
    s = s.replace("／", "/")
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[^A-Z0-9+\-/一-龥]", "", s)
    return s


def token_set(s):
    raw = str(s or "")
    up = raw.upper()
    tokens = set()
    for t in re.findall(r"[A-Z][A-Z0-9/+_-]{1,24}", up):
        t = normalize_text(t)
        if t:
            tokens.add(t)
    for t in re.findall(r"[\u4e00-\u9fff]{2,8}", raw):
        t = normalize_text(t)
        if t:
            tokens.add(t)
    whole = normalize_text(raw)
    if whole:
        tokens.add(whole)
    return tokens


def build_schema_retrieval_index(indicators):
    entries = []
    for x in indicators:
        cid = int(x["class_id"])
        name = str(x.get("name", "")).strip()
        aliases = list(x.get("aliases", []) or [])
        modules = list(x.get("modules", []) or [])
        module = str(x.get("module", "")).strip()
        if module and module not in modules:
            modules.insert(0, module)
        locations = list(x.get("location_hints", []) or [])
        location = str(x.get("location_hint", "")).strip()
        if location and location not in locations:
            locations.insert(0, location)
        evidence = str(x.get("evidence", "")).strip()
        states = x.get("states", [])
        states_text = " ".join(str(v) for v in states) if isinstance(states, list) else ""
        manual_text = " | ".join(p for p in [
            name,
            " ".join(map(str, aliases)),
            " ".join(map(str, modules)),
            " ".join(map(str, locations)),
            evidence,
            states_text,
        ] if p)
        entries.append({
            "class_id": cid,
            "class_name": name,
            "aliases": aliases,
            "modules": modules,
            "locations": locations,
            "evidence": evidence,
            "manual_text": manual_text,
            "manual_tokens": sorted(token_set(manual_text)),
        })
    return entries


def extract_local_evidence(model, processor, marked_context_path, tight_crop_path):
    prompt = """
你会看到两张图：

第1张：设备局部上下文，当前真实指示灯被红框精确标出。
第2张：红框灯本体的紧裁剪。

现在不要判断它属于哪个类别。
请只提取“红框灯附近能真实看到的证据”，用于之后对照设备说明书。

请尽量提取：
1. label_text：与红框灯直接对应的丝印，如 RUN、ALM、ACT、LINK/ACT、STAT、PWR、GE、FE、SFP、USB 等。
2. nearby_text：红框附近能看清的其它文字，尤其是接口名、端口名、板卡型号、模块名。
3. module_text：如果能判断红框属于哪个板卡/模块，记录可见模块文字。
4. interface_text：如果红框明显靠近某个 GE/FE/SFP/CONSOLE/AUX/USB/光口/网口，记录可见接口文字。
5. relative_position：例如“模块右侧”“接口左侧”“面板右上”“两个端口之间”等，只描述图中可见相对位置。
6. relation：label_text 相对红框的 left/right/up/down。

严格规则：
- 只记录图中真实可见的信息。
- 不得根据经验补全。
- 远处其它灯的丝印不能当作当前灯证据。
- 看不清的字段必须为 null 或 []。
- 不得返回类别 ID。

只输出 JSON。
""".strip()
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": str(marked_context_path)},
            {"type": "image", "image": str(tight_crop_path)},
            {"type": "text", "text": prompt},
        ],
    }]
    raw = run_qwen(model, processor, messages, max_new_tokens=320)
    data = extract_json_object(raw)
    nearby = data.get("nearby_text", [])
    if not isinstance(nearby, list):
        nearby = [nearby] if nearby else []
    out = {
        "label_text": data.get("label_text", None),
        "nearby_text": [str(v).strip() for v in nearby if str(v).strip()],
        "module_text": data.get("module_text", None),
        "interface_text": data.get("interface_text", None),
        "relative_position": data.get("relative_position", None),
        "relation": data.get("relation", None),
        "evidence_valid": bool(data.get("evidence_valid", False)),
        "reason": str(data.get("reason", "")).strip(),
        "raw": raw,
    }
    for k in ("label_text", "module_text", "interface_text", "relative_position", "relation"):
        if out[k] is not None:
            out[k] = str(out[k]).strip() or None
    return out


def score_manual_candidate(evidence, entry):
    score = 0.0
    reasons = []
    label = normalize_text(evidence.get("label_text"))
    module = normalize_text(evidence.get("module_text"))
    interface = normalize_text(evidence.get("interface_text"))
    relpos = normalize_text(evidence.get("relative_position"))
    nearby_tokens = set()
    for x in evidence.get("nearby_text", []):
        nearby_tokens |= token_set(x)
    manual_text_norm = normalize_text(entry["manual_text"])
    manual_tokens = set(entry["manual_tokens"])
    if label:
        if label in manual_tokens:
            score += 100
            reasons.append(f"丝印精确匹配:{evidence.get('label_text')}")
        elif len(label) >= 3 and label in manual_text_norm:
            score += 75
            reasons.append(f"丝印包含匹配:{evidence.get('label_text')}")
    if module and module in manual_text_norm:
        score += 38
        reasons.append(f"模块匹配:{evidence.get('module_text')}")
    if interface and interface in manual_text_norm:
        score += 32
        reasons.append(f"接口匹配:{evidence.get('interface_text')}")
    overlap = nearby_tokens & manual_tokens
    if overlap:
        score += min(30, 8 * len(overlap))
        reasons.append("附近文字匹配:" + ",".join(sorted(overlap)[:4]))
    if relpos:
        rel_overlap = token_set(evidence.get("relative_position")) & manual_tokens
        if rel_overlap:
            score += min(12, 4 * len(rel_overlap))
            reasons.append("位置提示匹配")
    return score, reasons


def retrieve_manual_candidates(evidence, schema_index, top_k=5):
    scored = []
    for entry in schema_index:
        score, reasons = score_manual_candidate(evidence, entry)
        scored.append({**entry, "retrieval_score": score, "retrieval_reasons": reasons})
    scored.sort(key=lambda x: x["retrieval_score"], reverse=True)
    return scored[:top_k]


def format_candidate_manual(candidates):
    chunks = []
    for c in candidates:
        parts = [
            f'class_id={c["class_id"]}',
            f'类别={c["class_name"]}',
            f'检索分={c["retrieval_score"]:.1f}',
        ]
        if c.get("aliases"):
            parts.append("别名=" + " / ".join(map(str, c["aliases"][:6])))
        if c.get("modules"):
            parts.append("模块=" + " / ".join(map(str, c["modules"][:6])))
        if c.get("locations"):
            parts.append("位置=" + " / ".join(map(str, c["locations"][:6])))
        if c.get("evidence"):
            parts.append("说明书证据=" + str(c["evidence"])[:260])
        chunks.append("；".join(parts))
    return "\n".join(chunks)


def resolve_with_manual_candidates(model, processor, marked_context_path, tight_crop_path, evidence, candidates):
    if not candidates:
        return None, "", "没有说明书候选", ""
    candidate_text = format_candidate_manual(candidates)
    evidence_json = json.dumps({
        "label_text": evidence.get("label_text"),
        "nearby_text": evidence.get("nearby_text", []),
        "module_text": evidence.get("module_text"),
        "interface_text": evidence.get("interface_text"),
        "relative_position": evidence.get("relative_position"),
    }, ensure_ascii=False)
    prompt = f"""
设备指示灯分类任务。

你会看到两张图：
第1张：当前灯被红框标出的上下文。
第2张：灯本体紧裁剪。

程序已经从设备说明书中检索出最相关的候选类别。
你不能从其它类别中选择，只能在下面候选中选择一个，或者返回 null。

图像证据：
{evidence_json}

说明书候选：
{candidate_text}

判断规则：
1. 优先使用与红框灯直接相邻的丝印。
2. 如果丝印不清晰，可以结合模块名、接口名、板卡位置、说明书位置描述判断。
3. 候选类别必须同时与图像位置和说明书描述相容。
4. 如果候选之间无法区分，返回 null。
5. 不得因为某个类别常见就优先选择。
6. 不得因为画面其它地方有 RUN/ALM 字样，就给当前灯使用。
7. 说明书证据优先于猜测。

只输出 JSON：
确定：{{"class_id": 5, "reason": "红框旁可见LINK/ACT丝印，且说明书候选5描述与该接口位置一致"}}
不确定：{{"class_id": null, "reason": "候选之间无法唯一确定"}}
""".strip()
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": str(marked_context_path)},
            {"type": "image", "image": str(tight_crop_path)},
            {"type": "text", "text": prompt},
        ],
    }]
    raw = run_qwen(model, processor, messages, max_new_tokens=260)
    data = extract_json_object(raw)
    cid = data.get("class_id", None)
    if cid is not None:
        try:
            cid = int(cid)
        except Exception:
            cid = None
    allowed_ids = {int(c["class_id"]) for c in candidates}
    if cid not in allowed_ids:
        cid = None
    reason = str(data.get("reason", "")).strip()
    if cid is None:
        return None, "", reason or "说明书候选无法唯一确定", raw
    chosen = next(c for c in candidates if int(c["class_id"]) == cid)
    return cid, chosen["class_name"], reason, raw


def classify_candidate(model, processor, marked_context_path, tight_crop_path, schema_index, top_k=5):
    evidence = extract_local_evidence(model, processor, marked_context_path, tight_crop_path)
    candidates = retrieve_manual_candidates(evidence, schema_index, top_k=top_k)
    class_id, class_name, reason, resolver_raw = resolve_with_manual_candidates(
        model, processor, marked_context_path, tight_crop_path, evidence, candidates
    )
    return class_id, class_name, reason, {
        "evidence": evidence,
        "manual_candidates": [
            {
                "class_id": c["class_id"],
                "class_name": c["class_name"],
                "retrieval_score": c["retrieval_score"],
                "retrieval_reasons": c["retrieval_reasons"],
                "evidence": c.get("evidence", ""),
                "modules": c.get("modules", []),
                "locations": c.get("locations", []),
            }
            for c in candidates
        ],
        "resolver_raw": resolver_raw,
    }


def _video_group_key(filename):
    """
    一致性分组：
    - 同一 video_xxx 的抽帧归为同组；
    - photo_xxxx 静态照片归为 photo 组，但后续使用更严格的空间匹配，
      只有位置高度重合时才认为是同一个物理灯。
    """
    m = re.search(r"__(video_\d+)__", filename, re.I)
    if m:
        return m.group(1).lower()

    if re.match(r"^photo_\d+", filename, re.I):
        return "__photos__"

    return None


def _frame_time(filename):
    m = re.search(r"__t(\d+(?:\.\d+)?)__", filename, re.I)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass

    m = re.search(r"__f(\d+)__", filename, re.I)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass

    return 0.0


def _ann_box_xyxy(a):
    x1 = float(a["x"])
    y1 = float(a["y"])
    return [
        x1,
        y1,
        x1 + float(a["w"]),
        y1 + float(a["h"]),
    ]


def _box_iou2(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    inter = iw * ih
    if inter <= 0:
        return 0.0

    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


def _normalized_center_distance(a, b, W, H):
    ax = (a[0] + a[2]) / 2.0
    ay = (a[1] + a[3]) / 2.0
    bx = (b[0] + b[2]) / 2.0
    by = (b[1] + b[3]) / 2.0

    dx = (ax - bx) / max(float(W), 1.0)
    dy = (ay - by) / max(float(H), 1.0)

    return (dx * dx + dy * dy) ** 0.5


def _size_ratio_ok(a, b):
    aw = max(1.0, a[2] - a[0])
    ah = max(1.0, a[3] - a[1])
    bw = max(1.0, b[2] - b[0])
    bh = max(1.0, b[3] - b[1])

    rw = max(aw / bw, bw / aw)
    rh = max(ah / bh, bh / ah)

    return rw <= 2.2 and rh <= 2.2


def _classification_weight(ann):
    """
    分类一致性投票权重。
    有较高说明书检索分的结果权重更高。
    """
    s = ann.get("_classification_score", 0.0)
    try:
        s = float(s)
    except Exception:
        s = 0.0

    return 1.0 + min(max(s, 0.0), 200.0) / 100.0



def _feature_image(path, max_dim=1600):
    """
    为模板配准读取缩小图，避免直接在 6240x3512 原图上做特征匹配。
    返回:
      gray, sx, sy
    其中 resized_x = original_x * sx
    """
    import cv2
    import numpy as np

    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, 1.0, 1.0

    h, w = img.shape[:2]
    scale = min(1.0, float(max_dim) / max(w, h))

    if scale < 1.0:
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        img = cv2.resize(
            img,
            (nw, nh),
            interpolation=cv2.INTER_AREA,
        )

    rh, rw = img.shape[:2]
    sx = rw / float(w)
    sy = rh / float(h)

    return img, sx, sy


def _create_feature_detector():
    """
    优先 SIFT；不可用时退回 ORB。
    """
    import cv2

    if hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create(nfeatures=5000), "sift"

    return cv2.ORB_create(nfeatures=6000), "orb"


def _compute_features(gray, detector):
    if gray is None:
        return [], None

    kp, des = detector.detectAndCompute(gray, None)
    return kp or [], des


def _homography_template_to_target(
    template_data,
    target_gray,
    target_sx,
    target_sy,
    detector_kind,
    target_kp,
    target_des,
):
    """
    从“原始静态照片”映射到“视频帧”。

    H_small: template resized -> target resized
    H_orig : template original -> target original
    """
    import cv2
    import numpy as np

    if (
        template_data.get("des") is None
        or target_des is None
        or len(template_data.get("kp", [])) < 8
        or len(target_kp) < 8
    ):
        return None

    norm = (
        cv2.NORM_L2
        if detector_kind == "sift"
        else cv2.NORM_HAMMING
    )

    matcher = cv2.BFMatcher(norm)

    try:
        pairs = matcher.knnMatch(
            template_data["des"],
            target_des,
            k=2,
        )
    except Exception:
        return None

    good = []
    ratio = 0.72 if detector_kind == "sift" else 0.78

    for pair in pairs:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)

    if len(good) < 10:
        return None

    src_pts = np.float32(
        [template_data["kp"][m.queryIdx].pt for m in good]
    ).reshape(-1, 1, 2)

    dst_pts = np.float32(
        [target_kp[m.trainIdx].pt for m in good]
    ).reshape(-1, 1, 2)

    H_small, mask = cv2.findHomography(
        src_pts,
        dst_pts,
        cv2.RANSAC,
        4.0,
    )

    if H_small is None or mask is None:
        return None

    inliers = int(mask.ravel().sum())
    inlier_ratio = inliers / max(len(good), 1)

    if inliers < 10 or inlier_ratio < 0.22:
        return None

    # small coordinate = S * original coordinate
    S_t = np.array([
        [template_data["sx"], 0.0, 0.0],
        [0.0, template_data["sy"], 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    S_v_inv = np.array([
        [1.0 / max(target_sx, 1e-12), 0.0, 0.0],
        [0.0, 1.0 / max(target_sy, 1e-12), 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    H_orig = S_v_inv @ H_small @ S_t

    quality = min(
        1.0,
        (inliers / 30.0) * 0.55
        + min(inlier_ratio / 0.55, 1.0) * 0.45,
    )

    return {
        "H": H_orig,
        "inliers": inliers,
        "good_matches": len(good),
        "inlier_ratio": inlier_ratio,
        "quality": quality,
    }


def _project_box(box, H, W, Hh):
    import cv2
    import numpy as np

    x1, y1, x2, y2 = box

    pts = np.float32([
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2],
    ]).reshape(-1, 1, 2)

    try:
        proj = cv2.perspectiveTransform(
            pts,
            H.astype("float64"),
        ).reshape(-1, 2)
    except Exception:
        return None

    if not np.isfinite(proj).all():
        return None

    px1 = float(proj[:, 0].min())
    py1 = float(proj[:, 1].min())
    px2 = float(proj[:, 0].max())
    py2 = float(proj[:, 1].max())

    px1 = max(0.0, min(float(W), px1))
    py1 = max(0.0, min(float(Hh), py1))
    px2 = max(0.0, min(float(W), px2))
    py2 = max(0.0, min(float(Hh), py2))

    if px2 <= px1 or py2 <= py1:
        return None

    # 过大的投影通常是坏 homography
    if (
        (px2 - px1) > W * 0.20
        or (py2 - py1) > Hh * 0.20
    ):
        return None

    return [px1, py1, px2, py2]


def _template_spatial_score(candidate_box, projected_box):
    """
    判断视频候选框是否对应模板照片中的同一个灯。
    """
    import math

    iou = _box_iou2(
        candidate_box,
        projected_box,
    )

    ccx = (candidate_box[0] + candidate_box[2]) / 2.0
    ccy = (candidate_box[1] + candidate_box[3]) / 2.0
    pcx = (projected_box[0] + projected_box[2]) / 2.0
    pcy = (projected_box[1] + projected_box[3]) / 2.0

    dist = math.hypot(
        ccx - pcx,
        ccy - pcy,
    )

    pw = max(
        4.0,
        projected_box[2] - projected_box[0],
    )
    ph = max(
        4.0,
        projected_box[3] - projected_box[1],
    )
    diag = math.hypot(pw, ph)

    # 同一个小灯：投影框应该重合，或中心落在约一个灯框尺度内。
    if iou < 0.05 and dist > max(18.0, diag * 0.85):
        return 0.0

    center_score = max(
        0.0,
        1.0 - dist / max(diag * 1.1, 24.0),
    )

    return min(
        1.0,
        iou * 1.8 + center_score * 0.8,
    )


def apply_photo_template_priors(
    annotations,
    image_shapes,
    images_dir,
    names,
    template_weight=8.0,
):
    """
    用原始单独保存的 photo_*.jpg/png 作为“高权重语义模板”。

    原理：
    1. photo_ 图片先按现有 AI 流程得到类别；
    2. 对每个视频抽帧，使用 SIFT/ORB + Homography
       将 photo_ 中的指示灯框投影到视频帧；
    3. 视频候选框若与投影位置一致，
       photo_ 中该灯的类别获得高权重；
    4. 多张 photo_ 模板共同投票；
    5. 高置信度模板票可覆盖视频帧单张图片的偶发错误分类。

    这样“同一个 RUN 灯”不会因为某一帧文字模糊，
    突然变成 GE / STAT / 其它类别。
    """
    try:
        import cv2
        import numpy as np
    except Exception as e:
        print(
            "  模板一致性跳过：无法导入 OpenCV:",
            e,
        )
        return {
            "photo_template_count": 0,
            "template_aligned_frames": 0,
            "template_corrected_boxes": 0,
            "template_filled_uncertain": 0,
        }

    template_files = sorted(
        (
            fn
            for fn in annotations
            if re.match(r"^photo_\d+", fn, re.I)
            and annotations.get(fn)
        ),
        key=natural_image_name_key,
    )

    if not template_files:
        return {
            "photo_template_count": 0,
            "template_aligned_frames": 0,
            "template_corrected_boxes": 0,
            "template_filled_uncertain": 0,
        }

    detector, detector_kind = _create_feature_detector()

    templates = []

    for fn in template_files:
        path = images_dir / fn
        if not path.exists():
            continue

        gray, sx, sy = _feature_image(path)
        kp, des = _compute_features(
            gray,
            detector,
        )

        usable_anns = []

        for a in annotations.get(fn, []):
            cid = a.get("class_id", None)
            if cid is None:
                continue

            try:
                cid = int(cid)
            except Exception:
                continue

            usable_anns.append({
                "class_id": cid,
                "class_name": names.get(cid, ""),
                "box": _ann_box_xyxy(a),
                "source_filename": fn,
                "base_weight": template_weight
                * _classification_weight(a),
            })

        if not usable_anns:
            continue

        templates.append({
            "filename": fn,
            "path": path,
            "gray": gray,
            "sx": sx,
            "sy": sy,
            "kp": kp,
            "des": des,
            "annotations": usable_anns,
        })

    if not templates:
        return {
            "photo_template_count": 0,
            "template_aligned_frames": 0,
            "template_corrected_boxes": 0,
            "template_filled_uncertain": 0,
        }

    corrected = 0
    filled = 0
    aligned_frames = 0

    target_files = sorted(
        (
            fn
            for fn in annotations
            if _video_group_key(fn)
            not in (None, "__photos__")
            and annotations.get(fn)
        ),
        key=natural_image_name_key,
    )

    print()
    print(
        f"  原始照片模板: {len(templates)} 张 "
        f"(权重 x{template_weight:g})"
    )

    for idx, fn in enumerate(
        target_files,
        start=1,
    ):
        path = images_dir / fn
        shape = image_shapes.get(fn)

        if not path.exists() or not shape:
            continue

        W, Hh = shape

        gray, sx, sy = _feature_image(path)
        target_kp, target_des = _compute_features(
            gray,
            detector,
        )

        alignments = []

        for t in templates:
            hdata = _homography_template_to_target(
                t,
                gray,
                sx,
                sy,
                detector_kind,
                target_kp,
                target_des,
            )

            if hdata is None:
                continue

            alignments.append(
                (t, hdata)
            )

        if not alignments:
            continue

        aligned_frames += 1

        # 把所有模板灯投影一次
        projected = []

        for t, hd in alignments:
            for ta in t["annotations"]:
                pb = _project_box(
                    ta["box"],
                    hd["H"],
                    W,
                    Hh,
                )

                if pb is None:
                    continue

                projected.append({
                    "class_id": ta["class_id"],
                    "class_name": ta["class_name"],
                    "box": pb,
                    "template_filename": t["filename"],
                    "alignment_quality": hd["quality"],
                    "inliers": hd["inliers"],
                    "inlier_ratio": hd["inlier_ratio"],
                    "base_weight": ta["base_weight"],
                })

        for a in annotations.get(fn, []):
            cb = _ann_box_xyxy(a)

            votes = {}
            supports = {}

            for p in projected:
                spatial = _template_spatial_score(
                    cb,
                    p["box"],
                )

                if spatial <= 0:
                    continue

                cid = p["class_id"]

                weight = (
                    p["base_weight"]
                    * p["alignment_quality"]
                    * spatial
                )

                votes[cid] = (
                    votes.get(cid, 0.0)
                    + weight
                )

                supports.setdefault(
                    cid,
                    [],
                ).append({
                    "template": p["template_filename"],
                    "weight": round(weight, 3),
                    "spatial_score": round(spatial, 3),
                    "alignment_quality": round(
                        p["alignment_quality"],
                        3,
                    ),
                    "inliers": p["inliers"],
                })

            if not votes:
                continue

            ranked = sorted(
                votes.items(),
                key=lambda kv: kv[1],
                reverse=True,
            )

            winner, winner_score = ranked[0]
            second_score = (
                ranked[1][1]
                if len(ranked) > 1
                else 0.0
            )

            # 高权重模板条件：
            # - 至少有一张照片模板形成强匹配；
            # - 若有冲突，第一名必须明显领先。
            if winner_score < 2.8:
                continue

            if (
                second_score > 0
                and winner_score
                < second_score * 1.35
            ):
                continue

            old = a.get("class_id", None)

            if old is None:
                filled += 1
            else:
                try:
                    old_int = int(old)
                except Exception:
                    old_int = None

                if old_int != winner:
                    corrected += 1

            a["class_id"] = winner
            a["class_name"] = names.get(
                winner,
                "",
            )
            a["template_prior"] = {
                "winner_class_id": winner,
                "winner_score": round(
                    winner_score,
                    3,
                ),
                "second_score": round(
                    second_score,
                    3,
                ),
                "supports": supports.get(
                    winner,
                    [],
                ),
            }

            reason = str(
                a.get(
                    "classification_reason",
                    "",
                )
            ).strip()

            prefix = "原始照片模板高权重校正"

            if prefix not in reason:
                a["classification_reason"] = (
                    (reason + "；" if reason else "")
                    + prefix
                )

        if (
            idx == 1
            or idx % 20 == 0
            or idx == len(target_files)
        ):
            print(
                f"    模板匹配进度 "
                f"{idx}/{len(target_files)}"
            )

    print(
        "  模板校正完成:"
        f" aligned_frames={aligned_frames},"
        f" corrected={corrected},"
        f" filled_uncertain={filled}"
    )

    return {
        "photo_template_count": len(templates),
        "photo_template_weight": template_weight,
        "template_aligned_frames": aligned_frames,
        "template_corrected_boxes": corrected,
        "template_filled_uncertain": filled,
    }


def enforce_temporal_class_consistency(
    annotations,
    image_shapes,
    names,
):
    """
    同一视频抽帧、以及高度对齐的静态照片中，
    相同空间位置的灯应尽量保持同一语义类别。

    处理方式：
    1. 在连续视频帧间按 IoU / 归一化中心距离建立 track；
    2. 对每条 track 汇总已有 class_id；
    3. 若一个类别明显占优，则统一整条 track 的类别；
    4. 原始 photo 的语义优先由照片模板机制传播到视频帧；
       普通时序一致性作为第二层稳定。
    """
    groups = {}

    for fn in annotations:
        g = _video_group_key(fn)
        if g:
            groups.setdefault(g, []).append(fn)

    changed = 0
    track_count = 0

    for g, fns in groups.items():
        fns.sort(key=lambda n: (_frame_time(n), natural_image_name_key(n)))

        tracks = []
        active = []

        for frame_idx, fn in enumerate(fns):
            shape = image_shapes.get(fn)
            if not shape:
                continue

            W, H = shape
            frame_boxes = annotations.get(fn, [])

            # 只保留最近 3 帧仍然活跃的轨迹
            active = [
                t for t in active
                if frame_idx - t["last_frame_idx"] <= 3
            ]

            used_tracks = set()

            for ann_idx, a in enumerate(frame_boxes):
                ab = _ann_box_xyxy(a)

                best = None
                best_score = -1.0

                for ti, t in enumerate(active):
                    if ti in used_tracks:
                        continue

                    prev = t["last_box"]
                    if not _size_ratio_ok(ab, prev):
                        continue

                    iou = _box_iou2(ab, prev)
                    cd = _normalized_center_distance(
                        ab, prev, W, H
                    )

                    # 视频帧允许轻微运动；静态照片更严格，
                    # 只有位置高度重合时才做类别一致化，避免串灯。
                    if g == "__photos__":
                        if iou < 0.45 and cd > 0.003:
                            continue
                    else:
                        if iou < 0.18 and cd > 0.006:
                            continue

                    score = iou * 3.0 + max(
                        0.0,
                        0.006 - cd
                    ) / 0.006

                    if score > best_score:
                        best_score = score
                        best = (ti, t)

                if best is None:
                    t = {
                        "members": [],
                        "last_box": ab,
                        "last_frame_idx": frame_idx,
                    }
                    tracks.append(t)
                    active.append(t)
                    t["members"].append((fn, ann_idx))
                else:
                    ti, t = best
                    used_tracks.add(ti)
                    t["members"].append((fn, ann_idx))
                    t["last_box"] = ab
                    t["last_frame_idx"] = frame_idx

        for t in tracks:
            members = t["members"]
            if len(members) < 2:
                continue

            votes = {}
            counts = {}

            for fn, ai in members:
                a = annotations[fn][ai]
                cid = a.get("class_id", None)

                if cid is None:
                    continue

                try:
                    cid = int(cid)
                except Exception:
                    continue

                w = _classification_weight(a)
                votes[cid] = votes.get(cid, 0.0) + w
                counts[cid] = counts.get(cid, 0) + 1

            if not votes:
                continue

            ranked = sorted(
                votes.items(),
                key=lambda kv: kv[1],
                reverse=True,
            )

            winner, winner_score = ranked[0]
            second_score = ranked[1][1] if len(ranked) > 1 else 0.0
            winner_count = counts.get(winner, 0)

            # 至少满足以下之一：
            # - 同一类别出现 >=2 次且明显占优；
            # - 只有一种非空类别证据，允许给 uncertain 帧补类别
            unique_classes = len(votes)

            stable = False

            if unique_classes == 1:
                stable = True
            elif winner_count >= 2 and winner_score >= second_score * 1.35:
                stable = True

            if not stable:
                continue

            track_count += 1
            winner_name = names.get(winner, "")

            for fn, ai in members:
                a = annotations[fn][ai]
                old = a.get("class_id", None)

                if old != winner:
                    changed += 1

                a["class_id"] = winner
                a["class_name"] = winner_name
                a["classification_reason"] = (
                    str(a.get("classification_reason", "")).strip()
                    + "；视频跨帧一致性校正"
                ).strip("；")
                a["temporal_consistency"] = True

    return {
        "video_tracks_stabilized": track_count,
        "boxes_class_corrected": changed,
    }


def schema_classes_for_output(indicators):
    """
    完整保留 manual_indicator_schema.json 中 indicators 的原始顺序。
    前端必须按该顺序展示，而不是只显示 AI 实际预测到的类别。
    """
    out = []

    for order_index, x in enumerate(indicators):
        out.append({
            "order": order_index,
            "class_id": int(x["class_id"]),
            "name": str(x.get("name", "")),
            "short_label": str(x.get("short_label", "")),
        })

    return out

def main():
    args = parse_args()

    args.package_root = args.package_root.expanduser()

    if not args.device:
        args.device = choose_device_with_schema(args.package_root)

    pkg = args.package_root / args.device

    if not pkg.exists():
        raise SystemExit(f"设备目录不存在: {pkg}")

    if (pkg / "indicator_annotations.json").exists():
        raise SystemExit(
            "安全停止：已有正式 indicator_annotations.json"
        )

    images_dir = pkg / "images"

    schema_path = pkg / "manual_indicator_schema.json"

    names = {}
    valid_ids = set()
    allowed_text = ""
    schema_index = []
    indicators = []

    if not args.skip_classification:
        indicators = load_schema(schema_path)

        names = {
            int(x["class_id"]): str(x.get("name", ""))
            for x in indicators
        }

        valid_ids = set(names)
        allowed_text = build_allowed_classes_text(indicators)
        schema_index = build_schema_retrieval_index(indicators)

    images = list_images_sorted(images_dir)

    if not args.include_far:
        images = [
            p for p in images
            if "__far__" not in p.name.lower()
        ]

    if args.limit > 0:
        images = images[:args.limit]

    print("加载 Grounding DINO:", args.dino_model)

    dproc = AutoProcessor.from_pretrained(
        args.dino_model,
        local_files_only=True,
    )

    dino = (
        AutoModelForZeroShotObjectDetection
        .from_pretrained(
            args.dino_model,
            local_files_only=True,
        )
        .to(args.dino_device)
    )

    dino.eval()

    print("加载 Qwen-VL:", args.qwen_model)

    qproc = AutoProcessor.from_pretrained(
        args.qwen_model,
        trust_remote_code=True,
        local_files_only=True,
    )

    qwen = (
        Qwen2_5_VLForConditionalGeneration
        .from_pretrained(
            args.qwen_model,
            torch_dtype="auto",
            device_map=args.qwen_device_map,
            trust_remote_code=True,
            local_files_only=True,
        )
    )

    debug_root = pkg / "_tiled_dino_qwen_v2_debug"

    if args.save_debug:
        if debug_root.exists():
            shutil.rmtree(debug_root)

        debug_root.mkdir(
            parents=True,
            exist_ok=True,
        )

    annotations = {}
    raw_outputs = {}
    image_shapes = {}

    stats = {
        "raw_dino_count": 0,
        "label_pass_count": 0,
        "geometry_pass_count": 0,
        "duplicate_kept_count": 0,
        "duplicate_rejected_count": 0,
        "verified_indicator_count": 0,
        "qwen_rejected_count": 0,
        "reject_type_counts": {
            "screw": 0,
            "reflection": 0,
            "port": 0,
            "button": 0,
            "other": 0,
        },
        "classified_count": 0,
        "uncertain_count": 0,
    }

    for image_index, image_path in enumerate(images, 1):
        print(f"\n[{image_index}/{len(images)}] {image_path.name}")

        with Image.open(image_path) as im0:
            image = im0.convert("RGB")

        W, H = image.size
        image_shapes[image_path.name] = [W, H]

        tiles = build_tiles(
            W,
            H,
            args.tile_size,
            args.overlap,
        )

        print(f"  原图={W}x{H} | tiles={len(tiles)}")

        if args.save_debug:
            work_dir = debug_root / image_path.stem
            work_dir.mkdir(parents=True, exist_ok=True)
        else:
            work_dir = Path(
                tempfile.mkdtemp(prefix="dino_qwen_v2_")
            )

        candidates = []
        rejected_logs = []

        for ti, tile in enumerate(tiles):
            tx1, ty1 = tile["x1"], tile["y1"]
            tx2, ty2 = tile["x2"], tile["y2"]

            tile_image = image.crop(
                (tx1, ty1, tx2, ty2)
            )

            local_candidates = detect_tile_with_dino(
                dino,
                dproc,
                tile_image,
                args.dino_device,
                args.box_threshold,
                args.text_threshold,
            )

            stats["raw_dino_count"] += len(local_candidates)

            pass_count = 0

            for lc in local_candidates:
                label = lc["label"]

                if not label_ok(label):
                    rejected_logs.append({
                        "stage": "label",
                        "label": label,
                        "score": lc["score"],
                    })
                    continue

                stats["label_pass_count"] += 1

                lx1, ly1, lx2, ly2 = lc["box"]

                box = clamp_box(
                    [
                        lx1 + tx1,
                        ly1 + ty1,
                        lx2 + tx1,
                        ly2 + ty1,
                    ],
                    W,
                    H,
                )

                if box is None:
                    continue

                ok, reason = geometric_check(
                    box,
                    args.min_side,
                    args.max_side,
                    args.max_area,
                    args.max_aspect,
                )

                if not ok:
                    rejected_logs.append({
                        "stage": "geometry",
                        "reason": reason,
                        "box": box,
                        "label": label,
                        "score": lc["score"],
                    })
                    continue

                stats["geometry_pass_count"] += 1
                pass_count += 1

                candidates.append({
                    "box": box,
                    "score": lc["score"],
                    "label": label,
                    "tile_index": ti,
                })

            print(
                f"    tile {ti+1}/{len(tiles)}: "
                f"raw={len(local_candidates)} pass={pass_count}"
            )

        kept, duplicate_rejected = suppress_duplicates(
            candidates,
            args.nms_iou,
            args.center_distance_ratio,
        )

        stats["duplicate_kept_count"] += len(kept)
        stats["duplicate_rejected_count"] += len(duplicate_rejected)

        print(
            f"  汇总: raw={stats['raw_dino_count']} "
            f"image_candidates={len(candidates)} "
            f"-> dedup={len(kept)}"
        )

        image_annotations = []
        candidate_logs = []

        for ci, cand in enumerate(kept):
            box = cand["box"]

            context_box = expand_context(
                box,
                W,
                H,
                args.context_scale,
                args.min_context_width,
                args.min_context_height,
            )

            tight_box = expand_tight(
                box,
                W,
                H,
                args.tight_scale,
            )

            marked_path = (
                work_dir
                / f"candidate_{ci:03d}_marked.jpg"
            )

            tight_path = (
                work_dir
                / f"candidate_{ci:03d}_tight.jpg"
            )

            draw_candidate_context(
                image,
                box,
                context_box,
                marked_path,
            )

            tight_crop = image.crop(tight_box)
            tight_crop.save(tight_path, quality=96)

            (
                object_type,
                verify_reason,
                verify_raw,
            ) = verify_candidate(
                qwen,
                qproc,
                marked_path,
                tight_path,
            )

            log = {
                "candidate_index": ci,
                "bbox": box,
                "dino_score": cand["score"],
                "dino_label": cand["label"],
                "object_type": object_type,
                "verify_reason": verify_reason,
                "verify_raw": verify_raw,
            }

            if object_type != "led":
                stats["qwen_rejected_count"] += 1
                stats["reject_type_counts"][object_type] = (
                    stats["reject_type_counts"].get(object_type, 0) + 1
                )

                log["accepted"] = False
                candidate_logs.append(log)

                print(
                    f"    cand {ci}: REJECT[{object_type}] | "
                    f"{verify_reason}"
                )
                continue

            stats["verified_indicator_count"] += 1

            class_id = None
            class_name = ""
            class_reason = ""
            class_raw = ""
            classification_evidence = {}

            if not args.skip_classification:
                (
                    class_id,
                    class_name,
                    class_reason,
                    classification_evidence,
                ) = classify_candidate(
                    qwen,
                    qproc,
                    marked_path,
                    tight_path,
                    schema_index,
                )

                class_raw = classification_evidence.get("resolver_raw", "")

                if class_id is None:
                    stats["uncertain_count"] += 1
                    print(
                        f"    cand {ci}: VERIFIED -> uncertain | "
                        f"{class_reason}"
                    )
                else:
                    stats["classified_count"] += 1
                    print(
                        f"    cand {ci}: VERIFIED -> "
                        f"{class_id} {class_name} | "
                        f"{class_reason}"
                    )

            selected_retrieval_score = 0.0
            if class_id is not None and classification_evidence:
                for mc in classification_evidence.get("manual_candidates", []):
                    try:
                        if int(mc.get("class_id")) == int(class_id):
                            selected_retrieval_score = float(
                                mc.get("retrieval_score", 0.0)
                            )
                            break
                    except Exception:
                        pass

            x1, y1, x2, y2 = box

            image_annotations.append({
                "class_id": class_id,
                "class_name": class_name,
                "x": round(x1, 2),
                "y": round(y1, 2),
                "w": round(x2 - x1, 2),
                "h": round(y2 - y1, 2),
                "source": "tiled_grounding_dino_qwen_v5",
                "reviewed": False,
                "needs_review": True,
                "classification_reason": class_reason,
                "_classification_score": selected_retrieval_score,
            })

            log["accepted"] = True
            log["class_id"] = class_id
            log["class_name"] = class_name
            log["classification_reason"] = class_reason
            log["classification_evidence"] = classification_evidence
            log["classification_raw"] = class_raw

            candidate_logs.append(log)

        annotations[image_path.name] = image_annotations

        raw_outputs[image_path.name] = {
            "candidates_before_dedup": candidates,
            "duplicate_rejected": duplicate_rejected,
            "other_rejected": rejected_logs,
            "candidates_after_dedup": kept,
            "qwen_results": candidate_logs,
        }

        if not args.save_debug:
            try:
                shutil.rmtree(work_dir)
            except Exception:
                pass

    # --------------------------------------------------------
    # 原始静态照片模板高权重：
    # photo_*.jpg/png 作为语义模板，先校正视频抽帧类别。
    # --------------------------------------------------------
    template_stats = apply_photo_template_priors(
        annotations,
        image_shapes,
        images_dir,
        names,
        template_weight=8.0,
    )
    stats.update(template_stats)

    # --------------------------------------------------------
    # 视频跨帧类别一致性：
    # 同一个物理灯在连续帧中不应一会儿 RUN、一会儿变成其它类别。
    # --------------------------------------------------------
    consistency_stats = enforce_temporal_class_consistency(
        annotations,
        image_shapes,
        names,
    )
    stats.update(consistency_stats)

    # 前端不需要看到内部一致性权重字段
    for _fn, _arr in annotations.items():
        for _a in _arr:
            _a.pop("_classification_score", None)

    all_images = list_images_sorted(images_dir)

    for p in all_images:
        annotations.setdefault(p.name, [])

    result = {
        "format": "tiled_grounding_dino_qwen_prelabel_v5",
        "device_name": args.device,

        # 完整类别表：来自 manual_indicator_schema.json
        # 注意：不是“AI 实际识别到几类”，而是该设备允许使用的全部类别。
        "schema_classes": schema_classes_for_output(indicators),
        "class_order": [
            int(x["class_id"])
            for x in indicators
        ],
        "names": {
            str(int(x["class_id"])): str(x.get("name", ""))
            for x in indicators
        },

        "annotations": annotations,
        "needs_human_review": True,
        "meta": {
            "dino_model": args.dino_model,
            "qwen_model": args.qwen_model,
            "tile_size": args.tile_size,
            "overlap": args.overlap,
            "box_threshold": args.box_threshold,
            "text_threshold": args.text_threshold,
            "nms_iou": args.nms_iou,
            "min_side": args.min_side,
            "max_side": args.max_side,
            "max_area": args.max_area,
            "max_aspect": args.max_aspect,
            **stats,
            "note":
            "Candidate highlighted with red box; "
            "Qwen first filters object type; for verified LEDs it extracts local visual/text evidence; "
            "the program retrieves top-k classes from manual_indicator_schema.json and Qwen resolves only among those manual-backed candidates; "
            "original photo_ images act as high-weight semantic templates for video frames; "
            "video-frame classes are stabilized across time; full schema classes are exported for frontend display.",
        },
        "raw_outputs": raw_outputs,
    }

    output_path = pkg / "pre_annotations.json"

    output_path.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print("完成:", output_path)
    for k, v in stats.items():
        print(f"{k}: {v}")
    print("=" * 80)


if __name__ == "__main__":
    main()


