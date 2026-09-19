#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_device_classifier.py

训练 13 类机房设备图像分类模型（Ultralytics Classification）。

数据目录默认：
~/yolo_device_monitor/data/device_cls/
├── train/<class>/*.jpg
├── val/<class>/*.jpg
└── test/<class>/*.jpg

示例：
python scripts/train_device_classifier.py

常用覆盖：
python scripts/train_device_classifier.py \
  --model yolo11s-cls.pt \
  --epochs 80 \
  --imgsz 320 \
  --batch 64
"""

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="训练设备分类模型")

    p.add_argument(
        "--project-root",
        type=Path,
        default=Path.home() / "yolo_device_monitor"
    )

    p.add_argument(
        "--data",
        type=Path,
        default=None,
        help="分类数据集根目录，默认 data/device_cls"
    )

    p.add_argument(
        "--model",
        default="yolo11s-cls.pt",
        help="Ultralytics 分类预训练模型"
    )

    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="0")
    p.add_argument("--patience", type=int, default=15)

    p.add_argument(
        "--name",
        default="device_cls_yolo11s"
    )

    p.add_argument(
        "--cache",
        action="store_true",
        help="缓存图像；内存充足时可开启"
    )

    return p.parse_args()


def check_dataset(data_dir: Path):
    required = [
        data_dir / "train",
        data_dir / "val",
        data_dir / "test",
    ]

    for p in required:
        if not p.exists():
            raise SystemExit(f"缺少数据目录: {p}")

    classes = sorted(
        p.name
        for p in (data_dir / "train").iterdir()
        if p.is_dir()
    )

    if not classes:
        raise SystemExit("train 下没有类别目录")

    # 确认 val/test 类别目录基本一致
    for split in ["val", "test"]:
        split_classes = {
            p.name
            for p in (data_dir / split).iterdir()
            if p.is_dir()
        }
        missing = set(classes) - split_classes
        if missing:
            raise SystemExit(
                f"{split} 缺少类别目录: {sorted(missing)}"
            )

    return classes


def main():
    args = parse_args()

    project_root = args.project_root.expanduser().resolve()
    data_dir = (
        args.data.expanduser().resolve()
        if args.data
        else project_root / "data" / "device_cls"
    )

    classes = check_dataset(data_dir)

    runs_dir = project_root / "runs" / "classify"
    runs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("设备分类训练")
    print("=" * 88)
    print("数据集:", data_dir)
    print("类别数:", len(classes))
    print("模型:", args.model)
    print("epochs:", args.epochs)
    print("imgsz:", args.imgsz)
    print("batch:", args.batch)
    print("device:", args.device)
    print("输出:", runs_dir / args.name)
    print()

    model = YOLO(args.model)

    results = model.train(
        data=str(data_dir),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        project=str(runs_dir),
        name=args.name,
        exist_ok=True,
        cache=args.cache,
        pretrained=True,
        verbose=True,

        # 分类增强：保持保守，避免把设备面板扭曲得过头
        hsv_h=0.01,
        hsv_s=0.25,
        hsv_v=0.20,
        degrees=2.0,
        translate=0.03,
        scale=0.10,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.0,
        erasing=0.10,
    )

    run_dir = Path(results.save_dir)
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"

    meta = {
        "data": str(data_dir),
        "classes": classes,
        "model": args.model,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "run_dir": str(run_dir),
        "best": str(best),
        "last": str(last),
    }

    (run_dir / "training_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 88)
    print("训练结束")
    print("=" * 88)
    print("run_dir:", run_dir)
    print("best:", best)
    print("last:", last)

    # 使用 test split 做一次最终评估
    if best.exists():
        print()
        print("开始使用 best.pt 在 test split 上评估...")

        best_model = YOLO(str(best))

        metrics = best_model.val(
            data=str(data_dir),
            split="test",
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            project=str(runs_dir),
            name=args.name + "_test",
            exist_ok=True,
        )

        print()
        print("test评估完成。")
        print("top1:", getattr(metrics, "top1", "N/A"))
        print("top5:", getattr(metrics, "top5", "N/A"))


if __name__ == "__main__":
    main()

