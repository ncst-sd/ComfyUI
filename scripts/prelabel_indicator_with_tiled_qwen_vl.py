#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
prelabel_indicator_with_tiled_qwen_vl.py

目标：
从“零人工标注”起步，用 Qwen2.5-VL 做更稳定的指示灯预标注。

核心流程：
1) 原图切成重叠 tile；
2) 每个 tile 只做“找物理指示灯”，不分类；
3) tile bbox 映射回原图坐标；
4) NMS 去掉跨 tile 重复框；
5) 对每个候选灯裁剪更大上下文；
6) Qwen-VL + manual_indicator_schema.json 只做类别判断；
7) 不确定则 class_id=null；
8) 输出 pre_annotations.json，供前端人工复核。

特点：
- 不需要先有人工标注；
- 不需要先训练 YOLO；
- 不允许 Qwen 在分类阶段重新画框；
- 第一阶段类别无关，只找“灯”；
- 第二阶段类别严格来自 manual_indicator_schema.json；
- 默认跳过 __far__ 远景帧；
- 不输出 confidence 到前端类别名；
- bbox 使用 x/y/w/h，兼容你当前 annotator 前端。

建议：
- tile-size 1024
- overlap 0.25
- context-scale 5
- limit 4 先小规模测试
"""

import argparse
import json
import math
import re
import tempfile
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


HOME = Path.home()
PROJECT = HOME / "yolo_device_monitor"
DEFAULT_PKG = PROJECT / "annotation_packages_best_frames_fast"
DEFAULT_MODEL = PROJECT / "models" / "Qwen2.5-VL-7B-Instruct"

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", required=True)
    ap.add_argument("--package-root", type=Path, default=DEFAULT_PKG)
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--device-map", default="cuda:0")
    ap.add_argument("--limit", type=int, default=4, help="处理图片数；0=全部")
    ap.add_argument("--include-far", action="store_true")
    ap.add_argument("--tile-size", type=int, default=1024)
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--tile-max-new-tokens", type=int, default=700)
    ap.add_argument("--classify-max-new-tokens", type=int, default=180)
    ap.add_argument("--nms-iou", type=float, default=0.45)
    ap.add_argument("--min-box-size", type=float, default=3.0,
                    help="原图像素下最小边长")
    ap.add_argument("--max-box-area-ratio", type=float, default=0.02,
                    help="候选框占原图面积上限，防止模型框整个面板")
    ap.add_argument("--context-scale", type=float, default=5.0)
    ap.add_argument("--min-context-width", type=int, default=240)
    ap.add_argument("--min-context-height", type=int, default=180)
    ap.add_argument("--save-debug", action="store_true",
                    help="保留 tile/crop 调试图")
    ap.add_argument("--skip-classification", action="store_true",
                    help="只找灯位置，不做类别判断")
    return ap.parse_args()


def extract_json_array(text: str):
    text = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S | re.I)
    if m:
        text = m.group(1).strip()

    a, b = text.find("["), text.rfind("]")
    if a >= 0 and b > a:
        try:
            obj = json.loads(text[a:b+1])
            if isinstance(obj, list):
                return obj
        except Exception:
            pass

    # 允许单对象
    a, b = text.find("{"), text.rfind("}")
    if a >= 0 and b > a:
        try:
            obj = json.loads(text[a:b+1])
            return [obj] if isinstance(obj, dict) else []
        except Exception:
            pass

    return []


def extract_json_object(text: str):
    text = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S | re.I)
    if m:
        text = m.group(1).strip()

    a, b = text.find("{"), text.rfind("}")
    if a >= 0 and b > a:
        try:
            obj = json.loads(text[a:b+1])
            return obj if isinstance(obj, dict) else {}
        except Exception:
            pass
    return {}


def build_tiles(W, H, tile_size, overlap):
    if tile_size <= 0:
        raise ValueError("tile-size 必须 > 0")
    if not (0 <= overlap < 1):
        raise ValueError("overlap 必须在 [0,1)")

    step = max(1, int(round(tile_size * (1.0 - overlap))))

    xs = list(range(0, max(1, W - tile_size + 1), step))
    ys = list(range(0, max(1, H - tile_size + 1), step))

    if not xs or xs[-1] != max(0, W - tile_size):
        xs.append(max(0, W - tile_size))
    if not ys or ys[-1] != max(0, H - tile_size):
        ys.append(max(0, H - tile_size))

    xs = sorted(set(xs))
    ys = sorted(set(ys))

    tiles = []
    for y in ys:
        for x in xs:
            x2 = min(W, x + tile_size)
            y2 = min(H, y + tile_size)
            tiles.append((x, y, x2, y2))
    return tiles


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
    return x1, y1, x2, y2


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    aa = (ax2 - ax1) * (ay2 - ay1)
    bb = (bx2 - bx1) * (by2 - by1)
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


def nms(boxes, threshold=0.45):
    """
    无置信度 NMS：
    - 优先保留面积较小的框（对小灯更合理，减少面板大框）
    - 高 IoU 重复框去掉
    """
    if not boxes:
        return []

    order = sorted(
        range(len(boxes)),
        key=lambda i: (boxes[i][2]-boxes[i][0]) * (boxes[i][3]-boxes[i][1])
    )

    kept = []
    for idx in order:
        b = boxes[idx]
        if all(iou(b, k) < threshold for k in kept):
            kept.append(b)
    return kept


def load_schema(schema_path: Path):
    data = json.loads(schema_path.read_text(encoding="utf-8"))
    inds = data.get("indicators", [])
    if not isinstance(inds, list) or not inds:
        raise SystemExit("manual_indicator_schema.json 中没有可用 indicators")
    return data, inds


def allowed_classes_text(indicators):
    lines = []
    for x in indicators:
        cid = int(x["class_id"])
        name = str(x.get("name", "")).strip()

        modules = x.get("modules", [])
        module = str(x.get("module", "")).strip()
        if module and module not in modules:
            modules = [module] + list(modules)

        locs = x.get("location_hints", [])
        loc = str(x.get("location_hint", "")).strip()
        if loc and loc not in locs:
            locs = [loc] + list(locs)

        aliases = x.get("aliases", [])
        evidence = str(x.get("evidence", "")).strip()

        parts = [f"{cid}: {name}"]
        if modules:
            parts.append("模块=" + " / ".join(map(str, modules[:6])))
        if locs:
            parts.append("位置=" + " / ".join(map(str, locs[:6])))
        if aliases:
            parts.append("别名=" + " / ".join(map(str, aliases[:6])))
        if evidence:
            parts.append("说明书=" + evidence[:180])

        lines.append("；".join(parts))
    return "\n".join(lines)


def run_vl(model, proc, messages, max_new_tokens):
    text = proc.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = proc(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )

    trimmed = [o[len(i0):] for i0, o in zip(inputs.input_ids, output)]
    return proc.batch_decode(trimmed, skip_special_tokens=True)[0]


def detect_lights_in_tile(model, proc, tile_path: Path, tw, th, max_new_tokens):
    prompt = f"""
这是一张工业设备局部图，尺寸 {tw}x{th}。

你的任务只有一个：
找出画面中“真正的物理指示灯灯体”。

严格规则：
1. 只框灯本体，不框旁边文字。
2. 不做类别判断，不需要知道 RUN/ALM/ACT/GE 等类型。
3. 不把端口、接口、插槽、按钮、螺丝、模块、屏幕、文字、标签、连接器、反光、金属高光、塑料件当成灯。
4. 优先找具有明确灯体结构、发光点、半透明灯帽、LED窗口的目标。
5. 看不清或不确定就不要标。
6. 宁可漏标，不要误标。
7. bbox 必须是本 tile 的像素坐标 [x1,y1,x2,y2]。
8. 如果没有明确指示灯，输出 []。

只输出 JSON 数组，不要解释：
[
  {{"bbox_2d":[x1,y1,x2,y2]}}
]
""".strip()

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": str(tile_path)},
            {"type": "text", "text": prompt},
        ]
    }]

    raw = run_vl(model, proc, messages, max_new_tokens)
    arr = extract_json_array(raw)

    boxes = []
    for obj in arr:
        if not isinstance(obj, dict):
            continue
        raw_box = obj.get("bbox_2d", obj.get("bbox", None))
        box = clamp_box(raw_box, tw, th)
        if box:
            boxes.append(box)
    return boxes, raw


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


def classify_candidate(model, proc, crop_path: Path, device_name, allowed_text,
                       valid_ids, max_new_tokens):
    prompt = f"""
设备：{device_name}

这张图是某个“已经确定位置的指示灯候选框”周围的局部上下文。
候选灯位于图像中央附近。

你现在只做类别判断，不要重新找灯，也不要重新画 bbox。

允许类别严格限定为：
{allowed_text}

规则：
1. 只能从上面类别中选择。
2. 结合候选灯旁边丝印、模块型号、接口位置、面板布局和说明书位置提示判断。
3. 不得创造新类别。
4. 如果无法确认、多个类别都可能、看不清丝印、位置不匹配，必须返回 null。
5. 宁可 uncertain，也不要猜。
6. 不根据灯的颜色或亮灭状态判断类别。
7. 只判断“这个灯是什么”，不判断正常/异常。

只输出 JSON 对象：
确定：
{{"class_id": 3, "reason": "旁边可见GE丝印，位置与说明书一致"}}

不确定：
{{"class_id": null, "reason": "无法确认"}}
""".strip()

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": str(crop_path)},
            {"type": "text", "text": prompt},
        ]
    }]

    raw = run_vl(model, proc, messages, max_new_tokens)
    obj = extract_json_object(raw)

    cid = obj.get("class_id", None)
    if cid is not None:
        try:
            cid = int(cid)
        except Exception:
            cid = None

    if cid not in valid_ids:
        cid = None

    reason = str(obj.get("reason", "")).strip()
    return cid, reason, raw


def main():
    args = parse_args()

    pkg = args.package_root.expanduser() / args.device
    if not pkg.exists():
        raise SystemExit(f"设备目录不存在: {pkg}")

    if (pkg / "indicator_annotations.json").exists():
        raise SystemExit("安全停止：该设备已有正式 indicator_annotations.json")

    images_dir = pkg / "images"
    if not images_dir.exists():
        raise SystemExit(f"缺少 images/: {images_dir}")

    schema_path = pkg / "manual_indicator_schema.json"
    if not args.skip_classification and not schema_path.exists():
        raise SystemExit(f"缺少: {schema_path}")

    schema = None
    indicators = []
    names = {}
    valid_ids = set()
    allowed_text = ""

    if not args.skip_classification:
        schema, indicators = load_schema(schema_path)
        names = {int(x["class_id"]): str(x["name"]) for x in indicators}
        valid_ids = set(names)
        allowed_text = allowed_classes_text(indicators)

    imgs = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in EXT
    )

    if not args.include_far:
        imgs = [p for p in imgs if "__far__" not in p.name.lower()]

    if args.limit > 0:
        imgs = imgs[:args.limit]

    if not imgs:
        raise SystemExit("没有可处理图片")

    print("加载 Qwen-VL:", args.model)
    proc = AutoProcessor.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=True
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map=args.device_map,
        trust_remote_code=True,
        local_files_only=True
    )

    debug_root = pkg / "_tiled_qwen_debug"
    if args.save_debug:
        debug_root.mkdir(exist_ok=True)

    annotations = {}
    raw_outputs = {}
    meta_images = {}

    total_raw = 0
    total_nms = 0
    total_classified = 0
    total_uncertain = 0

    for image_index, img_path in enumerate(imgs, 1):
        print(f"\n[{image_index}/{len(imgs)}] {img_path.name}")

        with Image.open(img_path) as im0:
            im = im0.convert("RGB")
            W, H = im.size

        tiles = build_tiles(W, H, args.tile_size, args.overlap)
        print(f"  原图: {W}x{H} | tiles={len(tiles)}")

        raw_boxes_global = []
        raw_outputs[img_path.name] = {
            "tile_outputs": [],
            "classify_outputs": [],
        }

        if args.save_debug:
            img_debug = debug_root / img_path.stem
            img_debug.mkdir(exist_ok=True)
        else:
            img_debug = Path(tempfile.mkdtemp(prefix="qwen_tiles_"))

        for ti, (tx1, ty1, tx2, ty2) in enumerate(tiles):
            tile = im.crop((tx1, ty1, tx2, ty2))
            tile_path = img_debug / f"tile_{ti:03d}_{tx1}_{ty1}_{tx2}_{ty2}.jpg"
            tile.save(tile_path, quality=95)

            tw, th = tile.size
            try:
                local_boxes, raw = detect_lights_in_tile(
                    model, proc, tile_path, tw, th,
                    args.tile_max_new_tokens
                )
            except Exception as e:
                local_boxes, raw = [], f"ERROR: {e}"

            accepted = 0
            for lx1, ly1, lx2, ly2 in local_boxes:
                gx1 = lx1 + tx1
                gy1 = ly1 + ty1
                gx2 = lx2 + tx1
                gy2 = ly2 + ty1

                bw = gx2 - gx1
                bh = gy2 - gy1
                area = bw * bh

                if bw < args.min_box_size or bh < args.min_box_size:
                    continue
                if area > W * H * args.max_box_area_ratio:
                    continue

                raw_boxes_global.append((gx1, gy1, gx2, gy2))
                accepted += 1

            raw_outputs[img_path.name]["tile_outputs"].append({
                "tile_index": ti,
                "tile": [tx1, ty1, tx2, ty2],
                "raw": raw,
                "accepted_boxes": accepted,
            })

            print(f"    tile {ti+1}/{len(tiles)}: {accepted}")

            if not args.save_debug:
                try:
                    tile_path.unlink()
                except Exception:
                    pass

        total_raw += len(raw_boxes_global)

        merged_boxes = nms(raw_boxes_global, args.nms_iou)
        total_nms += len(merged_boxes)

        print(f"  候选框: raw={len(raw_boxes_global)} -> nms={len(merged_boxes)}")

        ann_list = []

        for bi, box in enumerate(merged_boxes):
            x1, y1, x2, y2 = box
            cid = None
            cname = ""
            reason = ""

            if not args.skip_classification:
                cx1, cy1, cx2, cy2 = expand_context(
                    box, W, H,
                    args.context_scale,
                    args.min_context_width,
                    args.min_context_height
                )

                crop = im.crop((cx1, cy1, cx2, cy2))
                crop_path = img_debug / f"candidate_{bi:03d}.jpg"
                crop.save(crop_path, quality=95)

                try:
                    cid, reason, raw = classify_candidate(
                        model, proc, crop_path,
                        args.device,
                        allowed_text,
                        valid_ids,
                        args.classify_max_new_tokens
                    )
                except Exception as e:
                    cid, reason, raw = None, f"classify error: {e}", f"ERROR: {e}"

                raw_outputs[img_path.name]["classify_outputs"].append({
                    "candidate_index": bi,
                    "bbox": [x1, y1, x2, y2],
                    "context": [cx1, cy1, cx2, cy2],
                    "raw": raw,
                })

                if cid is None:
                    total_uncertain += 1
                    print(f"    cand {bi}: uncertain")
                else:
                    total_classified += 1
                    cname = names[cid]
                    print(f"    cand {bi}: {cid} {cname}")

                if not args.save_debug:
                    try:
                        crop_path.unlink()
                    except Exception:
                        pass

            ann_list.append({
                "class_id": cid,
                "class_name": cname,
                "x": round(x1, 2),
                "y": round(y1, 2),
                "w": round(x2 - x1, 2),
                "h": round(y2 - y1, 2),
                "source": "tiled_qwen_vl",
                "reviewed": False,
                "needs_review": True,
                "classification_reason": reason,
            })

        annotations[img_path.name] = ann_list
        meta_images[img_path.name] = {
            "width": W,
            "height": H,
            "tile_count": len(tiles),
            "raw_candidate_count": len(raw_boxes_global),
            "nms_candidate_count": len(merged_boxes),
        }

        if not args.save_debug:
            try:
                img_debug.rmdir()
            except Exception:
                pass

    # 其它未处理图片保持空数组，方便前端按文件名加载
    all_imgs = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in EXT
    )
    for p in all_imgs:
        annotations.setdefault(p.name, [])

    result = {
        "format": "tiled_qwen_vl_prelabel_v1",
        "device_name": args.device,
        "annotations": annotations,
        "needs_human_review": True,
        "meta": {
            "model": str(args.model),
            "tile_size": args.tile_size,
            "overlap": args.overlap,
            "nms_iou": args.nms_iou,
            "context_scale": args.context_scale,
            "processed_images": [p.name for p in imgs],
            "skip_classification": args.skip_classification,
            "raw_candidate_count": total_raw,
            "nms_candidate_count": total_nms,
            "classified_count": total_classified,
            "uncertain_count": total_uncertain,
            "per_image": meta_images,
            "note": "stage1=tiled light detection; stage2=manual-schema classification; bbox not redrawn during classification",
        },
        "raw_outputs": raw_outputs,
    }

    out_path = pkg / "pre_annotations.json"
    out_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("\n" + "=" * 80)
    print("完成:", out_path)
    print("raw candidate:", total_raw)
    print("after NMS:", total_nms)
    if not args.skip_classification:
        print("classified:", total_classified)
        print("uncertain:", total_uncertain)
    print("=" * 80)


if __name__ == "__main__":
    main()

