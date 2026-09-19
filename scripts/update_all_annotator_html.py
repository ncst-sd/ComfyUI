#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
update_all_annotator_html.py

作用：
把一个统一的 annotator.html 模板，覆盖复制到所有设备文件夹中，
文件名保持不变，仍然叫 annotator.html。

默认适配你的项目结构：
~/yolo_device_monitor/annotation_packages_best_frames_fast/
└── 设备A/
    ├── annotator.html
    ├── images/
    └── ...
└── 设备B/
    ├── annotator.html
    ├── images/
    └── ...

特点：
- 不做备份
- 直接覆盖
- 只更新“设备子目录”中的 annotator.html
- 如果某个设备目录下原本没有 annotator.html，也会自动创建
"""

import argparse
import shutil
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="统一更新所有设备目录下的 annotator.html")
    p.add_argument(
        "--root",
        default=str(Path.home() / "yolo_device_monitor" / "annotation_packages_best_frames_fast"),
        help="所有设备文件夹的根目录"
    )
    p.add_argument(
        "--template",
        required=True,
        help="新的 annotator.html 模板文件路径"
    )
    p.add_argument(
        "--only-missing",
        action="store_true",
        help="只给缺少 annotator.html 的设备目录创建，不覆盖已有文件"
    )
    return p.parse_args()


def is_device_dir(p: Path) -> bool:
    if not p.is_dir():
        return False
    # 设备目录通常至少有 images 目录，或者本身已有 annotator.html / json 文件
    if (p / "images").is_dir():
        return True
    if (p / "annotator.html").exists():
        return True
    if (p / "indicator_annotations.json").exists():
        return True
    if (p / "pre_annotations.json").exists():
        return True
    if (p / "manual_indicator_schema.json").exists():
        return True
    return False


def main():
    args = parse_args()

    root = Path(args.root).expanduser().resolve()
    template = Path(args.template).expanduser().resolve()

    if not root.exists():
        raise SystemExit(f"根目录不存在: {root}")
    if not template.exists():
        raise SystemExit(f"模板文件不存在: {template}")
    if template.name.lower() != "annotator.html":
        print(f"提示：模板文件名不是 annotator.html，而是 {template.name}，但这不影响覆盖生成。")

    device_dirs = [p for p in sorted(root.iterdir()) if is_device_dir(p)]
    if not device_dirs:
        raise SystemExit(f"在根目录下没有找到设备目录: {root}")

    updated = []
    created = []
    skipped = []

    for dev in device_dirs:
        target = dev / "annotator.html"

        if args.only_missing and target.exists():
            skipped.append(str(target))
            continue

        existed = target.exists()
        shutil.copy2(template, target)

        if existed:
            updated.append(str(target))
            print(f"[覆盖] {target}")
        else:
            created.append(str(target))
            print(f"[创建] {target}")

    print("\n" + "=" * 80)
    print("处理完成")
    print("=" * 80)
    print(f"设备目录数: {len(device_dirs)}")
    print(f"覆盖更新: {len(updated)}")
    print(f"新建文件: {len(created)}")
    print(f"跳过数量: {len(skipped)}")

    if skipped:
        print("\n已跳过（only-missing 模式）:")
        for x in skipped:
            print("  ", x)


if __name__ == "__main__":
    main()

