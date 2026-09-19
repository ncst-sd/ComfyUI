#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prelabel_images_with_qwen_vl.py v2

改进：
- 不再让模型一次找所有灯；
- 对 schema 中每一个物理灯，逐类单独询问 Qwen-VL；
- 必须输出 bbox，否则不接受；
- 类别名完全来自 manual_indicator_schema.json；
- 默认跳过文件名含 __far__ 的远景帧；
- 不输出 confidence，不输出 unknown_indicator；
- pre_annotations.json 仅为人工复核草稿。
"""

import argparse
import json
import re
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

PROJECT = Path.home() / "yolo_device_monitor"
PKG = PROJECT / "annotation_packages_best_frames_fast"
DEFAULT_LOCAL_MODEL = PROJECT / "models/Qwen2.5-VL-7B-Instruct"
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def extract_json_any(text: str):
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S | re.I)
    if m:
        text = m.group(1).strip()

    # 优先数组
    a, b = text.find("["), text.rfind("]")
    if a >= 0 and b > a:
        try:
            return json.loads(text[a:b+1])
        except Exception:
            pass

    # 允许单对象
    a, b = text.find("{"), text.rfind("}")
    if a >= 0 and b > a:
        try:
            return [json.loads(text[a:b+1])]
        except Exception:
            pass

    return []

def clamp_box(box, W, H):
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = map(float, box)
    except Exception:
        return None
    x1 = max(0.0, min(float(W), x1))
    y1 = max(0.0, min(float(H), y1))
    x2 = max(0.0, min(float(W), x2))
    y2 = max(0.0, min(float(H), y2))
    if x2 <= x1 or y2 <= y1:
        return None

    # 极大框一般说明模型框的是模块/面板而不是小灯，直接拒绝
    bw, bh = x2-x1, y2-y1
    if bw * bh > W * H * 0.08:
        return None
    if bw > W * 0.45 or bh > H * 0.45:
        return None

    return x1, y1, x2, y2

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw * ih
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union if union > 0 else 0.0

def dedup_same_class(items):
    kept = []
    for x in items:
        box = (x["x"], x["y"], x["x"]+x["w"], x["y"]+x["h"])
        if all(iou(box, (k["x"],k["y"],k["x"]+k["w"],k["y"]+k["h"])) < 0.60 for k in kept):
            kept.append(x)
    return kept

def ask_one_indicator(model, proc, img_path: Path, schema, indicator, W, H):
    states = indicator.get("states", [])
    state_text = ""
    if states:
        state_text = "\n说明书中的状态信息（仅用于帮助辨认该灯，不能当成类别）：\n" + \
            "\n".join(f'- {s.get("state","")}: {s.get("meaning","")}' for s in states[:12])

    prompt = f"""
你正在对工业设备“{schema["device_name"]}”做指示灯 bbox 预标注。

本次只寻找一个目标类别：
类别ID：{indicator["class_id"]}
正式名称：{indicator["name"]}
丝印/短名：{indicator.get("short_label","")}
所属模块：{indicator.get("module","")}
位置提示：{indicator.get("location_hint","")}
说明书证据：{indicator.get("evidence","")}
{state_text}

原图尺寸：{W}x{H}

严格规则：
1. 只寻找“{indicator["name"]}”，不要寻找其他灯。
2. 只有明确看到该物理指示灯本体时才输出。
3. 如果看不清、目标太小、不能确认、被遮挡或只有说明文字而看不到灯本体，输出 []。
4. 不把接口/端口本体、按钮、螺丝、文字、屏幕、开关、反光、模块边框、插槽、连接器当成灯。
5. bbox 只框灯本体，不要把旁边丝印文字、端口或整个模块一起框入。
6. 如果图片中同一类别有多个独立物理灯（例如多个端口各自有同类灯），可返回多个 bbox。
7. 不得输出其他类别名，也不得修改正式名称。
8. bbox 必须使用原图像素坐标 [x1,y1,x2,y2]。
9. 宁可漏标，不要误标。

只输出 JSON 数组，不要解释：
[
  {{"bbox":[x1,y1,x2,y2]}}
]

没有明确目标则严格输出：
[]
""".strip()

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": str(img_path)},
            {"type": "text", "text": prompt},
        ],
    }]

    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = proc(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt"
    ).to(model.device)

    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=450, do_sample=False)

    trimmed = [o[len(i0):] for i0, o in zip(inputs.input_ids, generated)]
    raw = proc.batch_decode(trimmed, skip_special_tokens=True)[0]
    arr = extract_json_any(raw)

    out = []
    for obj in arr:
        if not isinstance(obj, dict):
            continue

        # Qwen2.5-VL 常见输出字段是 bbox_2d；
        # 旧提示词/其它模型也可能输出 bbox。
        raw_box = None
        if "bbox_2d" in obj:
            raw_box = obj["bbox_2d"]
        elif "bbox" in obj:
            raw_box = obj["bbox"]

        if raw_box is None:
            continue

        box = clamp_box(raw_box, W, H)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        out.append({
            "class_id": int(indicator["class_id"]),
            "class_name": indicator["name"],
            "x": round(x1, 2),
            "y": round(y1, 2),
            "w": round(x2-x1, 2),
            "h": round(y2-y1, 2),
            "source": "qwen_vl_manual_prelabel_v2",
            "reviewed": False,
            "needs_review": True,
        })
    return dedup_same_class(out), raw

def choose_images(images, limit, skip_far):
    selected = []
    skipped_far = []
    for p in images:
        if skip_far and "__far__" in p.name.lower():
            skipped_far.append(p)
            continue
        selected.append(p)
        if limit > 0 and len(selected) >= limit:
            break
    return selected, skipped_far

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", required=True)
    ap.add_argument("--package-root", type=Path, default=PKG)
    ap.add_argument("--model", default=str(DEFAULT_LOCAL_MODEL))
    ap.add_argument("--device-map", default="cuda:0")
    ap.add_argument("--limit", type=int, default=10, help="处理非far图片前N张；0=全部")
    ap.add_argument("--include-far", action="store_true", help="也处理文件名含 __far__ 的远景帧")
    ap.add_argument("--class-id", type=int, action="append",
                    help="只测试指定类别，可重复，例如 --class-id 0 --class-id 8")
    args = ap.parse_args()

    pkg = args.package_root.expanduser() / args.device
    if (pkg / "indicator_annotations.json").exists():
        raise SystemExit("安全停止：该设备已存在 indicator_annotations.json")

    schema_path = pkg / "manual_indicator_schema.json"
    if not schema_path.exists():
        raise SystemExit(f"缺少 {schema_path}")

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    indicators = schema.get("indicators", [])
    if args.class_id:
        ids = set(args.class_id)
        indicators = [x for x in indicators if int(x["class_id"]) in ids]
    if not indicators:
        raise SystemExit("schema 中没有可用物理灯类别")

    all_images = sorted(
        p for p in (pkg / "images").iterdir()
        if p.is_file() and p.suffix.lower() in EXT
    )
    images, far_skipped = choose_images(all_images, args.limit, not args.include_far)

    print("模型:", args.model)
    print("设备:", args.device)
    print("本次类别数:", len(indicators))
    print("本次图片数:", len(images))
    if not args.include_far:
        print("far远景默认跳过:", len(far_skipped))

    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype="auto", device_map=args.device_map,
        trust_remote_code=True, local_files_only=True
    )

    annotations = {}
    raw_outputs = {}

    for i, p in enumerate(images, 1):
        with Image.open(p) as im:
            W, H = im.size

        print(f"\n[{i}/{len(images)}] {p.name}")
        image_anns = []
        raw_outputs[p.name] = {}

        for j, indicator in enumerate(indicators, 1):
            cid = int(indicator["class_id"])
            name = indicator["name"]
            print(f"  [{j}/{len(indicators)}] 类别 {cid}: {name} ...", end="", flush=True)
            try:
                anns, raw = ask_one_indicator(model, proc, p, schema, indicator, W, H)
                image_anns.extend(anns)
                raw_outputs[p.name][str(cid)] = raw
                print(f" {len(anns)} 个")
            except Exception as e:
                raw_outputs[p.name][str(cid)] = f"ERROR: {e}"
                print(" 失败:", e)

        annotations[p.name] = image_anns
        print("  本图合计:", len(image_anns))

    # 对未处理的图保留空数组，方便前端文件名对应；far明确标记为 skipped
    for p in all_images:
        annotations.setdefault(p.name, [])

    result = {
        "format": "qwen_vl_manual_prelabel_v2",
        "device_name": args.device,
        "names": {str(x["class_id"]): x["name"] for x in schema.get("indicators", [])},
        "annotations": annotations,
        "needs_human_review": True,
        "meta": {
            "processed_images": [p.name for p in images],
            "skipped_far_images": [p.name for p in far_skipped],
            "processed_class_ids": [int(x["class_id"]) for x in indicators],
            "strategy": "one_indicator_per_query; strict_bbox; no_unknown; far_skipped_by_default",
        },
        "raw_outputs": raw_outputs,
    }

    out = pkg / "pre_annotations.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n生成:", out)

if __name__ == "__main__":
    main()

