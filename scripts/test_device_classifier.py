#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_device_classifier.py

对训练后的设备分类模型做快速测试。

示例：
python scripts/test_device_classifier.py \
  --model runs/classify/device_cls_yolo11s/weights/best.pt \
  --source "/path/to/image.jpg"
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--model",
        type=Path,
        default=Path.home()
        / "yolo_device_monitor"
        / "runs"
        / "classify"
        / "device_cls_yolo11s"
        / "weights"
        / "best.pt"
    )

    p.add_argument("--source", required=True)
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--device", default="0")
    p.add_argument("--topk", type=int, default=5)

    args = p.parse_args()

    model = YOLO(
        str(args.model.expanduser().resolve())
    )

    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        device=args.device,
        verbose=False,
    )

    for result in results:
        probs = result.probs

        if probs is None:
            print("没有分类结果")
            continue

        names = result.names
        top5 = probs.top5[:args.topk]
        confs = probs.top5conf[:args.topk]

        print("source:", result.path)

        for rank, (idx, conf) in enumerate(
            zip(top5, confs),
            start=1
        ):
            print(
                f"{rank}. {names[int(idx)]}: "
                f"{float(conf):.4f}"
            )


if __name__ == "__main__":
    main()

