#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
prepare_indicator_detection_task.py

为“单台设备的指示灯检测”准备首轮 YOLO 标注集。

默认目标：
- workspace: 数据网柜_H3C_SR6608_华三
- 目标抽样: 200 张
- 从多个视频组中尽量均匀抽样
- 自动读取 verified alias -> rule_name
- 自动读取规则库中的 indicator 名称作为检测类别
- 不自动生成 bbox，labels/to_label/*.txt 为空文件，等待人工标注

输出：
~/yolo_device_monitor/data/indicator_detection/<workspace_name>/
├── images/to_label/*.jpg
├── labels/to_label/*.txt
├── data.yaml
├── annotation_manifest.csv
└── summary.txt
"""

import argparse
import csv
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path

import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def infer_video_group(path: Path) -> str:
    stem = path.stem
    m = re.search(r"(.+?__video_\d+)", stem)
    if m:
        return m.group(1)
    return path.parent.name or stem


def load_yaml(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


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


def resolve_rule_name(workspace_name: str, alias_yaml: dict):
    for device_id, info in (alias_yaml.get("devices") or {}).items():
        if not info.get("verified", False):
            continue
        if workspace_name in (info.get("workspace_names") or []):
            return {
                "device_id": device_id,
                "canonical_name": info.get("canonical_name", ""),
                "rule_name": info.get("rule_name", ""),
            }
    return None


def balanced_sample(groups: dict, target: int):
    """
    尽量跨视频组均匀抽样。
    第一轮先每组轮流取1张，再逐轮增加。
    """
    ordered = {}
    for g, imgs in groups.items():
        imgs = sorted(imgs)
        ordered[g] = imgs

    selected = []
    positions = {g: 0 for g in ordered}
    group_names = sorted(ordered)

    while len(selected) < target:
        added = 0
        for g in group_names:
            if len(selected) >= target:
                break
            pos = positions[g]
            imgs = ordered[g]
            if pos >= len(imgs):
                continue

            # 每轮在剩余序列上尽量取分散位置
            remaining = len(imgs) - pos
            step = max(1, remaining // max(1, (target - len(selected)) // max(1, len(group_names))))
            idx = min(pos, len(imgs) - 1)

            selected.append((g, imgs[idx]))
            positions[g] = min(len(imgs), pos + step)
            added += 1

        if added == 0:
            break

    # 如果因跳步导致不足，再补未选图
    chosen = {p for _, p in selected}
    if len(selected) < target:
        for g in group_names:
            for p in ordered[g]:
                if p not in chosen:
                    selected.append((g, p))
                    chosen.add(p)
                    if len(selected) >= target:
                        break
            if len(selected) >= target:
                break

    return selected[:target]


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
    ap.add_argument(
        "--count",
        type=int,
        default=200
    )
    ap.add_argument(
        "--mode",
        choices=["hardlink", "copy", "symlink"],
        default="hardlink"
    )
    ap.add_argument(
        "--reset",
        action="store_true"
    )
    args = ap.parse_args()

    project = args.project.expanduser().resolve()
    workspace_dir = project / "workspace" / args.workspace_name
    frames_dir = workspace_dir / "extracted_frames"
    alias_path = project / "rules" / "device_alias_map.yaml"
    rules_path = project / "rules" / "device_indicator_rules.yaml"

    if not frames_dir.exists():
        raise SystemExit(f"找不到 extracted_frames: {frames_dir}")

    alias_yaml = load_yaml(alias_path)
    rules_yaml = load_yaml(rules_path)

    resolved = resolve_rule_name(args.workspace_name, alias_yaml)
    if not resolved:
        raise SystemExit(
            f"workspace 未找到 verified alias 映射: {args.workspace_name}"
        )

    rule_name = resolved["rule_name"]
    device_rule = (rules_yaml.get("devices") or {}).get(rule_name)
    if not device_rule:
        raise SystemExit(f"规则库中找不到设备: {rule_name}")

    indicators = list((device_rule.get("indicators") or {}).keys())
    if not indicators:
        raise SystemExit(f"设备没有 indicator 定义: {rule_name}")

    images = sorted(
        p for p in frames_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )
    if not images:
        raise SystemExit("没有可用图片")

    groups = defaultdict(list)
    for img in images:
        groups[infer_video_group(img)].append(img)

    target = min(args.count, len(images))
    selected = balanced_sample(groups, target)

    out = project / "data" / "indicator_detection" / args.workspace_name
    if args.reset and out.exists():
        shutil.rmtree(out)

    images_out = out / "images" / "to_label"
    labels_out = out / "labels" / "to_label"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    rows = []

    for i, (group, src) in enumerate(selected, start=1):
        # 避免不同视频出现同名帧
        new_name = f"{i:04d}__{src.name}"
        dst_img = images_out / new_name
        dst_lbl = labels_out / (Path(new_name).stem + ".txt")

        link_or_copy(src, dst_img, args.mode)
        dst_lbl.touch(exist_ok=True)

        rows.append({
            "index": i,
            "video_group": group,
            "source_image": str(src),
            "label_image": str(dst_img),
            "label_file": str(dst_lbl),
            "status": "pending",
        })

    # YOLO detect YAML
    data_yaml = {
        "path": str(out),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(indicators)},
    }
    (out / "data.yaml").write_text(
        yaml.safe_dump(data_yaml, allow_unicode=True, sort_keys=False),
        encoding="utf-8"
    )

    manifest = out / "annotation_manifest.csv"
    with manifest.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "video_group",
                "source_image",
                "label_image",
                "label_file",
                "status",
            ]
        )
        writer.writeheader()
        writer.writerows(rows)

    summary_lines = [
        f"workspace_name: {args.workspace_name}",
        f"canonical_name: {resolved['canonical_name']}",
        f"rule_name: {rule_name}",
        f"总原始图片: {len(images)}",
        f"视频组数: {len(groups)}",
        f"首轮抽样图片: {len(selected)}",
        f"指示灯类别数: {len(indicators)}",
        "",
        "类别：",
    ]
    summary_lines.extend(
        f"{i}: {name}" for i, name in enumerate(indicators)
    )

    (out / "summary.txt").write_text(
        "\n".join(summary_lines),
        encoding="utf-8"
    )

    print("=" * 88)
    print("指示灯检测标注任务已生成")
    print("=" * 88)
    print("\n".join(summary_lines))
    print()
    print("图片目录:", images_out)
    print("标签目录:", labels_out)
    print("data.yaml:", out / "data.yaml")
    print("manifest:", manifest)


if __name__ == "__main__":
    main()

