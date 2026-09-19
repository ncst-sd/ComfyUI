#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_indicator_yolo_dataset.py

把 annotation_packages_best_frames_fast/<设备>/indicator_annotations.json
转换成 Ultralytics YOLO 检测数据集。

特点：
- 按设备通用，不再写死某一台设备
- 保留 JSON 中原始 class_id，不做合并
- 空标注图片生成空 txt，作为负样本保留
- 视频帧按 source video 分组，避免同一视频泄漏到 train/val/test
- 静态照片按单张独立分组
- 默认硬链接图片，失败时自动复制
- 输出 data.yaml / split_manifest.csv / class_stats.csv / dataset_summary.csv

默认输入：
  ~/yolo_device_monitor/annotation_packages_best_frames_fast/<device>/

默认输出：
  ~/yolo_device_monitor/data/indicator_yolo/<device>/
"""

import argparse
import csv
import json
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

PROJECT = Path.home() / "yolo_device_monitor"
DEFAULT_SRC_ROOT = PROJECT / "annotation_packages_best_frames_fast"
DEFAULT_DST_ROOT = PROJECT / "data" / "indicator_yolo"


def video_group_key(filename: str) -> str:
    """
    video_0017__设备名__video_002__t...png
    -> video::video_002

    photo_0003__设备名__DSC08484.jpg
    -> photo::<filename>
    """
    name = Path(filename).name

    if name.startswith("video_"):
        parts = name.split("__")
        vids = [p for p in parts if re.fullmatch(r"video_\d+", p)]
        if len(vids) >= 2:
            return f"video::{vids[1]}"
        if len(vids) == 1:
            return f"video::{vids[0]}"

    return f"photo::{name}"


def safe_link_or_copy(src: Path, dst: Path, force_copy: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if force_copy:
        shutil.copy2(src, dst)
        return "copy"

    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def split_groups(groups, train_ratio, val_ratio, seed):
    """
    以 group 为单位随机划分。
    目标是避免视频帧泄漏，而不是保证图片数严格按比例。
    """
    rng = random.Random(seed)

    keys = list(groups.keys())
    rng.shuffle(keys)

    total_images = sum(len(groups[k]) for k in keys)
    target_train = total_images * train_ratio
    target_val = total_images * val_ratio

    splits = {"train": [], "val": [], "test": []}
    counts = {"train": 0, "val": 0, "test": 0}

    for k in keys:
        n = len(groups[k])

        # 优先填还没达到目标的 train / val，剩余进 test
        if counts["train"] < target_train:
            target = "train"
        elif counts["val"] < target_val:
            target = "val"
        else:
            target = "test"

        splits[target].append(k)
        counts[target] += n

    # 尽量保证三个 split 都有 group
    nonempty = [s for s in splits if splits[s]]
    if len(nonempty) < 3 and len(keys) >= 3:
        for missing in [s for s in splits if not splits[s]]:
            donor = max(splits, key=lambda x: len(splits[x]))
            if len(splits[donor]) > 1:
                splits[missing].append(splits[donor].pop())

    return splits


def clamp_box(x, y, w, h, W, H):
    x1 = max(0.0, min(float(x), float(W)))
    y1 = max(0.0, min(float(y), float(H)))
    x2 = max(0.0, min(float(x) + float(w), float(W)))
    y2 = max(0.0, min(float(y) + float(h), float(H)))

    if x2 <= x1 or y2 <= y1:
        return None

    bw = x2 - x1
    bh = y2 - y1
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0

    return (
        cx / W,
        cy / H,
        bw / W,
        bh / H,
    )


def image_size(path: Path):
    # Pillow 更轻便；失败再尝试 cv2
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size
    except Exception:
        import cv2
        img = cv2.imread(str(path))
        if img is None:
            return None
        h, w = img.shape[:2]
        return w, h


def write_yaml(path: Path, out_dir: Path, names: dict):
    ordered = {int(k): v for k, v in names.items()}
    max_id = max(ordered) if ordered else -1

    # 保证 Ultralytics names 列表索引 == class_id
    name_list = []
    for i in range(max_id + 1):
        name_list.append(ordered.get(i, f"class_{i}"))

    lines = [
        f"path: {out_dir}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]

    for i, name in enumerate(name_list):
        safe = str(name).replace('"', '\\"')
        lines.append(f'  {i}: "{safe}"')

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_one(device, src_root, dst_root, train_ratio, val_ratio, seed, force_copy):
    src_dir = src_root / device
    ann_path = src_dir / "indicator_annotations.json"
    images_dir = src_dir / "images"

    if not ann_path.exists():
        raise FileNotFoundError(f"缺少标注文件: {ann_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"缺少图片目录: {images_dir}")

    data = json.loads(ann_path.read_text(encoding="utf-8"))

    json_device = data.get("device_name", "")
    if json_device and json_device != device:
        print(f"WARNING: JSON device_name={json_device!r}，目录名={device!r}")

    names = data.get("names", {})
    annotations = data.get("annotations", {})

    if not isinstance(names, dict) or not names:
        raise RuntimeError("JSON 中 names 为空")
    if not isinstance(annotations, dict) or not annotations:
        raise RuntimeError("JSON 中 annotations 为空")

    groups = defaultdict(list)
    missing_images = []

    for filename in annotations:
        img = images_dir / filename
        if not img.exists():
            missing_images.append(filename)
            continue
        groups[video_group_key(filename)].append(filename)

    if missing_images:
        print(f"WARNING: 有 {len(missing_images)} 张标注图片在 images/ 中找不到，将跳过")

    split_groups_map = split_groups(groups, train_ratio, val_ratio, seed)

    out_dir = dst_root / device

    # 只重建当前设备输出目录
    if out_dir.exists():
        shutil.rmtree(out_dir)

    for split in ("train", "val", "test"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    class_counts = Counter()
    split_image_counts = Counter()
    split_box_counts = Counter()
    split_empty_counts = Counter()
    link_modes = Counter()

    for split, group_keys in split_groups_map.items():
        for g in group_keys:
            for filename in groups[g]:
                src_img = images_dir / filename
                dst_img = out_dir / "images" / split / filename

                mode = safe_link_or_copy(src_img, dst_img, force_copy)
                link_modes[mode] += 1

                size = image_size(src_img)
                if not size:
                    print(f"WARNING: 无法读取图片尺寸，跳过: {src_img}")
                    continue
                W, H = size

                boxes = annotations.get(filename, []) or []
                yolo_lines = []
                valid_boxes = 0

                for b in boxes:
                    try:
                        cid = int(b["class_id"])
                        norm = clamp_box(
                            b["x"], b["y"], b["w"], b["h"], W, H
                        )
                    except Exception:
                        norm = None

                    if norm is None:
                        continue

                    cx, cy, bw, bh = norm
                    yolo_lines.append(
                        f"{cid} {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}"
                    )
                    valid_boxes += 1
                    class_counts[cid] += 1

                label_path = out_dir / "labels" / split / (Path(filename).stem + ".txt")
                label_path.write_text(
                    "\n".join(yolo_lines) + ("\n" if yolo_lines else ""),
                    encoding="utf-8",
                )

                split_image_counts[split] += 1
                split_box_counts[split] += valid_boxes
                if valid_boxes == 0:
                    split_empty_counts[split] += 1

                manifest_rows.append({
                    "split": split,
                    "group": g,
                    "filename": filename,
                    "boxes": valid_boxes,
                })

    write_yaml(out_dir / "data.yaml", out_dir, names)

    with (out_dir / "split_manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["split", "group", "filename", "boxes"])
        w.writeheader()
        w.writerows(manifest_rows)

    with (out_dir / "class_stats.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "class_name", "instances"])
        for k in sorted((int(x) for x in names.keys())):
            w.writerow([k, names.get(str(k), names.get(k, "")), class_counts[k]])

    with (out_dir / "dataset_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["item", "train", "val", "test", "total"])
        for label, counter in [
            ("images", split_image_counts),
            ("boxes", split_box_counts),
            ("empty_images", split_empty_counts),
        ]:
            vals = [counter[s] for s in ("train", "val", "test")]
            w.writerow([label, *vals, sum(vals)])

    print()
    print("=" * 72)
    print("设备:", device)
    print("完成:", out_dir)
    print("类别数:", len(names))
    print(
        "split:",
        {s: split_image_counts[s] for s in ("train", "val", "test")}
    )
    print(
        "空标注:",
        {s: split_empty_counts[s] for s in ("train", "val", "test")}
    )
    print("总框数:", sum(split_box_counts.values()))
    print("链接方式:", dict(link_modes))
    print("data.yaml:", out_dir / "data.yaml")

    return out_dir


def discover_ready_devices(src_root: Path):
    out = []
    if not src_root.exists():
        return out
    for d in sorted(src_root.iterdir()):
        if d.is_dir() and (d / "indicator_annotations.json").exists() and (d / "images").exists():
            out.append(d.name)
    return out


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--device",
        action="append",
        help="设备目录名。可重复传多个；传 all 表示处理所有已有 indicator_annotations.json 的设备",
    )
    ap.add_argument("--src-root", type=Path, default=DEFAULT_SRC_ROOT)
    ap.add_argument("--dst-root", type=Path, default=DEFAULT_DST_ROOT)
    ap.add_argument("--train-ratio", type=float, default=0.70)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copy", action="store_true", help="强制复制图片，不使用硬链接")

    args = ap.parse_args()

    src_root = args.src_root.expanduser()
    dst_root = args.dst_root.expanduser()

    devices = args.device or []

    if not devices:
        raise SystemExit(
            "请指定 --device，例如：\n"
            '  --device "地震预警_CRES-LMA_101_交大铁发"\n'
            '  --device "FAS_MDS3400DLX_V3.1_佳讯飞鸿"'
        )

    if "all" in devices:
        devices = discover_ready_devices(src_root)

    if not devices:
        raise SystemExit("没有找到可生成的数据集")

    for device in devices:
        build_one(
            device=device,
            src_root=src_root,
            dst_root=dst_root,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
            force_copy=args.copy,
        )


if __name__ == "__main__":
    main()

