#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
finalize_indicator_detection_dataset.py

人工完成 YOLO bbox 标注后，把 to_label 数据切成 train/val/test。

要求：
- images/to_label/<name>.jpg
- labels/to_label/<name>.txt
- YOLO txt 格式：
    class_id x_center y_center width height
  坐标为 0~1 归一化值

特点：
- 按原 annotation_manifest.csv 中的 video_group 优先分组切分
- 尽量避免同一视频组泄漏到不同集合
- 输出 images/train|val|test 和 labels/train|val|test
"""

import argparse
import csv
import os
import shutil
from collections import defaultdict
from pathlib import Path


def link_or_copy(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        os.symlink(src, dst)
    else:
        try:
            os.link(src, dst)
        except Exception:
            shutil.copy2(src, dst)


def validate_label(path: Path, class_count: int):
    """
    空标签允许：代表该图没有标注目标。
    非空行必须是标准 YOLO bbox。
    """
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return True, ""

    for lineno, line in enumerate(text.splitlines(), 1):
        parts = line.split()
        if len(parts) != 5:
            return False, f"{path.name}:{lineno} 列数不是5"
        try:
            cls = int(parts[0])
            vals = [float(x) for x in parts[1:]]
        except Exception:
            return False, f"{path.name}:{lineno} 解析失败"

        if cls < 0 or cls >= class_count:
            return False, f"{path.name}:{lineno} class_id越界: {cls}"
        if not all(0.0 <= x <= 1.0 for x in vals):
            return False, f"{path.name}:{lineno} 坐标不在0~1"

    return True, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--project",
        type=Path,
        default=Path.home() / "yolo_device_monitor"
    )
    ap.add_argument(
        "--workspace-name",
        default="数据网柜_H3C_SR6608_华三"
    )
    ap.add_argument("--train-ratio", type=float, default=0.70)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument(
        "--mode",
        choices=["hardlink", "copy", "symlink"],
        default="hardlink"
    )
    ap.add_argument(
        "--reset-splits",
        action="store_true"
    )
    args = ap.parse_args()

    if args.train_ratio <= 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio >= 1:
        raise SystemExit("切分比例非法")

    root = (
        args.project.expanduser().resolve()
        / "data"
        / "indicator_detection"
        / args.workspace_name
    )

    manifest = root / "annotation_manifest.csv"
    data_yaml = root / "data.yaml"
    if not manifest.exists():
        raise SystemExit(f"缺少 manifest: {manifest}")
    if not data_yaml.exists():
        raise SystemExit(f"缺少 data.yaml: {data_yaml}")

    import yaml
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    names = cfg.get("names") or {}
    class_count = len(names)

    rows = []
    with manifest.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    # 校验标签
    errors = []
    valid_rows = []
    for row in rows:
        img = Path(row["label_image"])
        lbl = Path(row["label_file"])
        if not img.exists():
            errors.append(f"缺少图片: {img}")
            continue
        if not lbl.exists():
            errors.append(f"缺少标签: {lbl}")
            continue
        ok, msg = validate_label(lbl, class_count)
        if not ok:
            errors.append(msg)
            continue
        valid_rows.append(row)

    if errors:
        print("发现标签问题：")
        for e in errors[:50]:
            print(" -", e)
        raise SystemExit(f"共有 {len(errors)} 个错误，请修正后再切分")

    if args.reset_splits:
        for split in ["train", "val", "test"]:
            for kind in ["images", "labels"]:
                p = root / kind / split
                if p.exists():
                    shutil.rmtree(p)

    groups = defaultdict(list)
    for row in valid_rows:
        groups[row["video_group"]].append(row)

    group_items = sorted(
        groups.items(),
        key=lambda kv: len(kv[1]),
        reverse=True
    )

    total = len(valid_rows)
    targets = {
        "train": round(total * args.train_ratio),
        "val": round(total * args.val_ratio),
    }
    targets["test"] = total - targets["train"] - targets["val"]

    assigned = {"train": [], "val": [], "test": []}
    counts = {"train": 0, "val": 0, "test": 0}

    # 先确保三个集合有组
    if len(group_items) >= 3:
        for split, item in zip(["train", "val", "test"], group_items[:3]):
            g, items = item
            assigned[split].append((g, items))
            counts[split] += len(items)
        remaining = group_items[3:]
    else:
        remaining = group_items

    for g, items in remaining:
        split = max(
            ["train", "val", "test"],
            key=lambda s: targets[s] - counts[s]
        )
        assigned[split].append((g, items))
        counts[split] += len(items)

    for split, grouped in assigned.items():
        for _, items in grouped:
            for row in items:
                src_img = Path(row["label_image"])
                src_lbl = Path(row["label_file"])
                dst_img = root / "images" / split / src_img.name
                dst_lbl = root / "labels" / split / src_lbl.name
                link_or_copy(src_img, dst_img, args.mode)
                link_or_copy(src_lbl, dst_lbl, args.mode)

    print("=" * 88)
    print("YOLO detection 数据集切分完成")
    print("=" * 88)
    print(f"train={counts['train']} val={counts['val']} test={counts['test']}")
    print("data.yaml:", data_yaml)


if __name__ == "__main__":
    main()

