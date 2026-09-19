#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_indicator_annotation_manifest.py

为后续“指示灯检测/识别”生成待标注清单。

它不会自动猜 bounding box。
它负责把：
- 设备目录
- 图片路径
- 对应规则库里的指示灯名称
整理成一个 CSV，方便人工标注工具或后续脚本使用。

输出：
~/yolo_device_monitor/data/indicator_annotation/
├── annotation_manifest.csv
├── indicator_classes.yaml
└── summary.txt
"""

import argparse
import csv
from pathlib import Path

import yaml


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--project",
        type=Path,
        default=Path.home() / "yolo_device_monitor"
    )
    ap.add_argument(
        "--max-per-device",
        type=int,
        default=300,
        help="每个设备最多抽多少张图用于首轮指示灯标注"
    )
    args = ap.parse_args()

    project = args.project.expanduser().resolve()
    workspace = project / "workspace"
    rules_path = project / "rules" / "device_indicator_rules.yaml"
    alias_path = project / "rules" / "device_alias_map.yaml"

    outdir = project / "data" / "indicator_annotation"
    outdir.mkdir(parents=True, exist_ok=True)

    rules = yaml.safe_load(
        rules_path.read_text(encoding="utf-8")
    ) or {}
    aliases = yaml.safe_load(
        alias_path.read_text(encoding="utf-8")
    ) or {}

    rule_devices = rules.get("devices", {})
    alias_devices = aliases.get("devices", {})

    # workspace -> canonical/rule_name
    ws_map = {}
    for _, info in alias_devices.items():
        if not info.get("verified", False):
            continue
        for ws in info.get("workspace_names", []) or []:
            ws_map[ws] = {
                "canonical_name": info.get("canonical_name"),
                "rule_name": info.get("rule_name"),
            }

    rows = []
    all_indicator_classes = set()
    summary_lines = []

    for ws_dir in sorted(workspace.iterdir()):
        if not ws_dir.is_dir() or ws_dir.name.startswith("_"):
            continue

        frames_dir = ws_dir / "extracted_frames"
        if not frames_dir.exists():
            continue

        images = sorted(
            p for p in frames_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        )
        if not images:
            continue

        mapping = ws_map.get(ws_dir.name)
        if mapping:
            rule_name = mapping["rule_name"]
            canonical_name = mapping["canonical_name"]
            indicators = sorted(
                rule_devices
                .get(rule_name, {})
                .get("indicators", {})
                .keys()
            )
        else:
            rule_name = ""
            canonical_name = ""
            indicators = []

        # 均匀抽样 max_per_device
        if len(images) > args.max_per_device:
            step = len(images) / args.max_per_device
            picked = [
                images[min(int(i * step), len(images) - 1)]
                for i in range(args.max_per_device)
            ]
        else:
            picked = images

        expected = " | ".join(indicators)
        all_indicator_classes.update(
            f"{rule_name}::{x}"
            for x in indicators
            if rule_name
        )

        for img in picked:
            rows.append({
                "workspace_name": ws_dir.name,
                "canonical_device": canonical_name,
                "rule_name": rule_name,
                "image_path": str(img),
                "expected_indicators": expected,
                "mapped": bool(mapping),
                "annotation_status": "pending",
            })

        summary_lines.append(
            f"{ws_dir.name}: "
            f"images={len(images)}, "
            f"selected={len(picked)}, "
            f"mapped={bool(mapping)}, "
            f"indicators={len(indicators)}"
        )

    manifest = outdir / "annotation_manifest.csv"
    with manifest.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "workspace_name",
                "canonical_device",
                "rule_name",
                "image_path",
                "expected_indicators",
                "mapped",
                "annotation_status",
            ]
        )
        writer.writeheader()
        writer.writerows(rows)

    classes_yaml = outdir / "indicator_classes.yaml"
    classes_yaml.write_text(
        yaml.safe_dump(
            {
                "strategy": "device_scoped_indicator_classes",
                "note": "建议首轮按设备范围标注指示灯，不要把不同设备同名灯强行合并。",
                "classes": sorted(all_indicator_classes),
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8"
    )

    (outdir / "summary.txt").write_text(
        "\n".join(summary_lines),
        encoding="utf-8"
    )

    print("=" * 88)
    print("指示灯标注清单生成完成")
    print("=" * 88)
    print("待标注图片:", len(rows))
    print("输出:", manifest)
    print("类别文件:", classes_yaml)


if __name__ == "__main__":
    main()

