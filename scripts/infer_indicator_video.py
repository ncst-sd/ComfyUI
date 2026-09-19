#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, time
from pathlib import Path
import cv2
from ultralytics import YOLO

PROJECT = Path.home() / "yolo_device_monitor"
DEFAULT_MODEL = PROJECT / "runs" / "detect" / "indicator_cres_lma101_yolo11s" / "weights" / "best.pt"
DEFAULT_OUTPUT = PROJECT / "runs" / "indicator_infer"

def draw(frame, result, names, fps=None):
    if result.boxes is not None:
        for box in result.boxes:
            x1,y1,x2,y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
            cid = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            label = f"{names.get(cid, str(cid))} {conf:.2f}"
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,label,(x1,max(20,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2,cv2.LINE_AA)
    if fps is not None:
        cv2.putText(frame,f"Infer FPS: {fps:.1f}",(20,40),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,255),2,cv2.LINE_AA)
    return frame

def infer_image(model, src, out_dir, args):
    img = cv2.imread(str(src))
    if img is None:
        raise SystemExit(f"无法读取图片: {src}")
    t0 = time.perf_counter()
    res = model.predict(img, imgsz=args.imgsz, conf=args.conf, iou=args.iou, device=args.device, verbose=False)[0]
    fps = 1.0 / max(time.perf_counter()-t0, 1e-9)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{src.stem}_pred{src.suffix}"
    cv2.imwrite(str(out), draw(img.copy(),res,model.names,fps))
    print("完成:", out)
    print("检测数:", len(res.boxes) if res.boxes is not None else 0)

def infer_video(model, src, out_dir, args):
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise SystemExit(f"无法打开视频: {src}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{src.stem}_pred.mp4"
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps_src, (w,h))
    if not writer.isOpened():
        raise SystemExit(f"无法创建输出视频: {out}")

    times = []
    i = 0
    print(f"输入: {src}")
    print(f"分辨率: {w}x{h}  FPS: {fps_src:.2f}  总帧: {total}")
    print(f"输出: {out}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        res = model.predict(frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou, device=args.device, verbose=False)[0]
        dt = time.perf_counter() - t0
        times.append(dt)
        if len(times) > 30:
            times.pop(0)
        fps = 1.0 / max(sum(times)/len(times), 1e-9)
        writer.write(draw(frame.copy(),res,model.names,fps))
        i += 1
        if i % 100 == 0:
            pct = i/total*100 if total else 0
            print(f"{i}/{total} ({pct:.1f}%) infer_fps={fps:.1f}")

    cap.release()
    writer.release()
    print("推理完成:", out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.50)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    if not args.model.exists():
        raise SystemExit(f"找不到模型: {args.model}")
    if not args.source.exists():
        raise SystemExit(f"找不到输入文件: {args.source}")

    model = YOLO(str(args.model))
    ext = args.source.suffix.lower()
    if ext in {".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}:
        infer_image(model,args.source,args.output_dir,args)
    elif ext in {".mp4",".avi",".mov",".mkv",".m4v",".ts",".mts",".m2ts",".wmv",".flv"}:
        infer_video(model,args.source,args.output_dir,args)
    else:
        raise SystemExit(f"暂不支持格式: {ext}")

if __name__ == "__main__":
    main()

