#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_indicator_detector.py

通用多设备指示灯检测训练脚本。

示例：
python scripts/train_indicator_detector.py \
  --device "地震预警_CRES-LMA_101_交大铁发"

python scripts/train_indicator_detector.py \
  --device "FAS_MDS3400DLX_V3.1_佳讯飞鸿"

也可一次训练两个：
python scripts/train_indicator_detector.py \
  --device "地震预警_CRES-LMA_101_交大铁发" \
  --device "FAS_MDS3400DLX_V3.1_佳讯飞鸿"

说明：
- 默认使用本地 ~/yolo_device_monitor/models/yolo/yolo11s.pt
- 默认 amp=False，避免无网络机器触发 Ultralytics AMP 检查下载 yolo26n.pt
- 每个设备训练结束后自动用 best.pt 对 test split 做验证
"""

import argparse
import re
from pathlib import Path

from ultralytics import YOLO

PROJECT = Path.home() / "yolo_device_monitor"
DATA_ROOT = PROJECT / "data" / "indicator_yolo"
RUN_ROOT = PROJECT / "runs" / "detect"
DEFAULT_MODEL = PROJECT / "models" / "yolo" / "yolo11s.pt"

KNOWN_RUN_NAMES = {
    "地震预警_CRES-LMA_101_交大铁发": "indicator_cres_lma101_v2_yolo11s",
    "FAS_MDS3400DLX_V3.1_佳讯飞鸿": "indicator_fas_mds3400dlx_yolo11s",
}


def slugify_device(name: str):
    if name in KNOWN_RUN_NAMES:
        return KNOWN_RUN_NAMES[name]

    slug = re.sub(r"[^0-9A-Za-z_-]+", "_", name).strip("_")
    if not slug:
        slug = "device"
    return f"indicator_{slug}_yolo11s"


def train_one(args, device):
    data_yaml = DATA_ROOT / device / "data.yaml"

    if not data_yaml.exists():
        raise FileNotFoundError(
            f"数据集不存在: {data_yaml}\n"
            "请先运行 build_indicator_yolo_dataset.py"
        )

    model_path = Path(args.model).expanduser()

    if not model_path.exists():
        raise FileNotFoundError(
            f"找不到预训练检测模型: {model_path}\n"
            "注意：yolo11s-cls.pt 是分类模型，不能用于检测训练。"
        )

    run_name = args.name or slugify_device(device)

    print()
    print("=" * 72)
    print("设备:", device)
    print("数据集:", data_yaml)
    print("预训练模型:", model_path)
    print("输出:", RUN_ROOT / run_name)
    print("GPU device:", args.gpu)
    print("imgsz:", args.imgsz)
    print("batch:", args.batch)
    print("epochs:", args.epochs)
    print("AMP:", args.amp)
    print("=" * 72)
    print()

    model = YOLO(str(model_path))

    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.gpu,
        project=str(RUN_ROOT),
        name=run_name,
        exist_ok=args.exist_ok,
        patience=args.patience,

        # 避免无网络环境的 AMP 检查下载
        amp=args.amp,

        # 适合指示灯/面板小目标的温和增强
        mosaic=0.5,
        close_mosaic=10,
        degrees=0.0,
        translate=0.08,
        scale=0.30,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.0,

        hsv_h=0.01,
        hsv_s=0.25,
        hsv_v=0.20,

        plots=True,
        verbose=True,
    )

    best = RUN_ROOT / run_name / "weights" / "best.pt"

    if not best.exists():
        print(f"WARNING: 没找到 best.pt: {best}")
        return

    print()
    print("训练完成，开始 test split 验证：")
    print(best)

    best_model = YOLO(str(best))
    metrics = best_model.val(
        data=str(data_yaml),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.gpu,
        plots=True,
        project=str(RUN_ROOT),
        name=f"{run_name}_test",
    )

    print()
    print("test results_dict:")
    print(metrics.results_dict)
    print()
    print("best.pt:")
    print(best)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--device",
        action="append",
        required=True,
        help="设备名，可重复传多个",
    )
    ap.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help="本地 YOLO11 detection 预训练权重",
    )
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--name", default=None, help="只训练单设备时可自定义 run name")
    ap.add_argument("--exist-ok", action="store_true")

    amp_group = ap.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", dest="amp", action="store_true")
    amp_group.add_argument("--no-amp", dest="amp", action="store_false")
    ap.set_defaults(amp=False)

    args = ap.parse_args()

    if args.name and len(args.device) > 1:
        raise SystemExit("--name 只能在单设备训练时使用")

    for device in args.device:
        train_one(args, device)


if __name__ == "__main__":
    main()

