#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from collections import Counter, deque
from pathlib import Path
import cv2
from ultralytics import YOLO

def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        type=Path,
        default=Path.home() / "yolo_device_monitor" / "runs" / "classify" /
        "device_cls_yolo11s" / "weights" / "best.pt"
    )
    p.add_argument("--source", required=True)
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--device", default="0")
    p.add_argument("--sample-seconds", type=float, default=0.5)
    p.add_argument("--window", type=int, default=8)
    p.add_argument("--min-frame-conf", type=float, default=0.60)
    p.add_argument("--min-vote-ratio", type=float, default=0.75)
    p.add_argument("--min-mean-conf", type=float, default=0.80)
    p.add_argument("--pretty", action="store_true")
    args = p.parse_args()

    model_path = args.model.expanduser().resolve()
    if not model_path.exists():
        raise SystemExit(f"模型不存在: {model_path}")

    model = YOLO(str(model_path))
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"无法打开视频: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_every = max(1, int(round(fps * args.sample_seconds)))

    history = deque(maxlen=args.window)
    sampled = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % sample_every != 0:
            frame_idx += 1
            continue

        result = model.predict(
            frame,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )[0]

        probs = result.probs
        if probs is None:
            frame_idx += 1
            continue

        top_idx = int(probs.top1)
        conf = float(probs.top1conf)
        name = result.names[top_idx]

        item = {
            "frame": frame_idx,
            "time_sec": round(frame_idx / fps, 3),
            "class_name": name,
            "confidence": round(conf, 6),
        }
        sampled.append(item)

        if conf >= args.min_frame_conf:
            history.append((name, conf))

        frame_idx += 1

    cap.release()

    stable = False
    final_name = None
    vote_ratio = 0.0
    mean_conf = 0.0

    if history:
        counts = Counter(name for name, _ in history)
        final_name, count = counts.most_common(1)[0]
        vote_ratio = count / len(history)

        winner_confs = [
            conf for name, conf in history if name == final_name
        ]
        mean_conf = sum(winner_confs) / len(winner_confs)

        stable = (
            len(history) >= min(3, args.window)
            and vote_ratio >= args.min_vote_ratio
            and mean_conf >= args.min_mean_conf
        )

    output = {
        "source": args.source,
        "model": str(model_path),
        "fps": fps,
        "total_frames": total_frames,
        "sample_seconds": args.sample_seconds,
        "sampled_count": len(sampled),
        "decision": {
            "workspace_name": final_name if stable else None,
            "candidate_name": final_name,
            "stable": stable,
            "vote_ratio": round(vote_ratio, 6),
            "mean_confidence": round(mean_conf, 6),
            "window_size": len(history),
            "thresholds": {
                "min_frame_conf": args.min_frame_conf,
                "min_vote_ratio": args.min_vote_ratio,
                "min_mean_conf": args.min_mean_conf,
            },
        },
        "recent_samples": sampled[-args.window:],
    }

    print(json.dumps(
        output,
        ensure_ascii=False,
        indent=2 if args.pretty else None
    ))

if __name__ == "__main__":
    main()

