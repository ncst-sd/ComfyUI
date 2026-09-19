#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path
import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def sanitize_class_name(name):
    name = str(name).strip().replace("/", "_").replace("\\", "_")
    return re.sub(r"\s+", "_", name)

def infer_video_group(path):
    stem = path.stem
    m = re.search(r"(.+?__video_\d+)", stem)
    if m:
        return m.group(1)
    return path.parent.name or stem

def link_or_copy(src, dst, mode):
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

def target_counts(n, train_ratio, val_ratio):
    train = int(round(n * train_ratio))
    val = int(round(n * val_ratio))
    test = n - train - val
    if n >= 3:
        train = max(train, 1)
        val = max(val, 1)
        test = max(test, 1)
        while train + val + test > n:
            if train >= val and train >= test and train > 1:
                train -= 1
            elif val >= test and val > 1:
                val -= 1
            elif test > 1:
                test -= 1
            else:
                break
        while train + val + test < n:
            train += 1
    return {"train": train, "val": val, "test": test}

def split_by_images(images, train_ratio, val_ratio):
    images = sorted(images, key=lambda p: p.name)
    targets = target_counts(len(images), train_ratio, val_ratio)
    t_end = targets["train"]
    v_end = t_end + targets["val"]
    out = {}
    for i, img in enumerate(images):
        out[img] = "train" if i < t_end else "val" if i < v_end else "test"
    return out, "image_level"

def split_by_groups(groups, train_ratio, val_ratio):
    group_items = sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True)
    total = sum(len(v) for _, v in group_items)
    targets = target_counts(total, train_ratio, val_ratio)
    assigned = {"train": [], "val": [], "test": []}
    counts = {"train": 0, "val": 0, "test": 0}

    if len(group_items) >= 3:
        for split, item in zip(["train", "val", "test"], group_items[:3]):
            g, imgs = item
            assigned[split].append((g, imgs))
            counts[split] += len(imgs)
        remaining = group_items[3:]
    else:
        remaining = group_items

    for g, imgs in remaining:
        candidate = max(
            ["train", "val", "test"],
            key=lambda s: targets[s] - counts[s]
        )
        assigned[candidate].append((g, imgs))
        counts[candidate] += len(imgs)

    out = {}
    for split, group_list in assigned.items():
        for _, imgs in group_list:
            for img in imgs:
                out[img] = split
    return out, "video_group"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=Path, default=Path.home() / "yolo_device_monitor")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--train-ratio", type=float, default=0.70)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--mode", choices=["hardlink", "copy", "symlink"], default="hardlink")
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()

    if args.train_ratio <= 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio >= 1:
        raise SystemExit("split比例非法")

    project = args.project.expanduser().resolve()
    workspace = project / "workspace"
    output = args.output.expanduser().resolve() if args.output else project / "data" / "device_cls"

    if args.reset and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    classes, records = [], []
    counts = defaultdict(lambda: defaultdict(int))
    strategies = {}

    for device_dir in sorted(workspace.iterdir()):
        if not device_dir.is_dir() or device_dir.name.startswith("_"):
            continue
        frames_dir = device_dir / "extracted_frames"
        if not frames_dir.exists():
            continue

        images = sorted([
            p for p in frames_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        ])
        if not images:
            continue

        class_name = sanitize_class_name(device_dir.name)
        classes.append({"class_name": class_name, "workspace_name": device_dir.name})

        groups = defaultdict(list)
        for img in images:
            groups[infer_video_group(img)].append(img)

        if len(groups) >= 3:
            image_splits, strategy = split_by_groups(groups, args.train_ratio, args.val_ratio)
        else:
            image_splits, strategy = split_by_images(images, args.train_ratio, args.val_ratio)

        strategies[class_name] = {
            "strategy": strategy,
            "video_groups": len(groups),
            "images": len(images),
        }

        for img in images:
            split = image_splits[img]
            dst = output / split / class_name / img.name
            link_or_copy(img, dst, args.mode)
            counts[class_name][split] += 1
            records.append({
                "workspace_name": device_dir.name,
                "class_name": class_name,
                "video_group": infer_video_group(img),
                "split": split,
                "split_strategy": strategy,
                "source": str(img),
                "target": str(dst),
            })

    if not classes:
        raise SystemExit("没有发现 extracted_frames 图像")

    (output / "classes.yaml").write_text(
        yaml.safe_dump({
            "task": "classification",
            "class_count": len(classes),
            "classes": classes,
            "split_strategy": strategies,
        }, allow_unicode=True, sort_keys=False, width=140),
        encoding="utf-8"
    )

    manifest = output / "split_manifest.csv"
    with manifest.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "workspace_name", "class_name", "video_group",
            "split", "split_strategy", "source", "target"
        ])
        writer.writeheader()
        writer.writerows(records)

    lines = [f"设备类别数: {len(classes)}", f"总图片数: {len(records)}", ""]
    for c in sorted(counts):
        s = strategies[c]
        lines.append(
            f"{c}: train={counts[c]['train']} val={counts[c]['val']} test={counts[c]['test']} "
            f"| strategy={s['strategy']} | video_groups={s['video_groups']}"
        )

    (output / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("\n输出:", output)

if __name__ == "__main__":
    main()

