#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv", ".flv", ".ts", ".mts", ".m2ts"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MANUAL_EXTS = {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".txt", ".md"}

MANUAL_DIR_NAMES = {"说明书", "手册", "manual", "manuals", "文档"}
IMAGE_DIR_NAMES = {"图片", "图像", "照片", "images", "image", "photos"}
VIDEO_DIR_NAMES = {"视频", "录像", "video", "videos"}
GLOBAL_HINTS = ("巡检总表", "指示灯巡检", "设备指示灯")


def parse_args():
    p = argparse.ArgumentParser(description="按设备文件夹名称整理设备监测数据集")
    p.add_argument("--root", type=Path, required=True, help="原始数据根目录")
    p.add_argument("--output", type=Path,
                   default=Path.home() / "yolo_device_monitor" / "workspace")
    p.add_argument("--interval", type=float, default=1.0, help="抽帧间隔（秒）")
    p.add_argument("--topk", type=int, default=120, help="每个视频最多保留代表帧数")
    p.add_argument("--blur-threshold", type=float, default=40.0)
    p.add_argument("--brightness-min", type=float, default=20.0)
    p.add_argument("--brightness-max", type=float, default=240.0)
    p.add_argument("--copy-videos", action="store_true",
                   help="把原视频复制到 workspace 并按设备名重命名")
    p.add_argument("--keep-all-frames", action="store_true")
    return p.parse_args()


def normalize_name(name):
    name = name.strip()
    name = re.sub(r'[\\/:*?"<>|\r\n\t]+', "_", name)
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("._") or "unknown_device"


def find_subdir(device_dir, candidates):
    lowers = {x.lower() for x in candidates}
    for child in device_dir.iterdir():
        if child.is_dir() and child.name.lower() in lowers:
            return child
    return None


def collect_files(folder, exts):
    if folder is None or not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*")
                  if p.is_file() and p.suffix.lower() in exts)


def calc_metrics(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast = float(gray.std())
    return blur, brightness, contrast


def quality_ok(blur, brightness, args):
    reasons = []
    if blur < args.blur_threshold:
        reasons.append("blurry")
    if brightness < args.brightness_min:
        reasons.append("too_dark")
    if brightness > args.brightness_max:
        reasons.append("too_bright")
    return not reasons, ("ok" if not reasons else ",".join(reasons))


def save_jpg(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise RuntimeError(f"无法保存图片: {path}")


def select_frames(df, topk):
    if df.empty:
        return df
    good = df[df["passed_quality"] == True].copy()
    if good.empty:
        good = df.copy()
    good = good.sort_values("timestamp_sec").reset_index(drop=True)
    if len(good) <= topk:
        return good

    bins = np.linspace(0, len(good), topk + 1, dtype=int)
    chosen = []
    for i in range(topk):
        part = good.iloc[bins[i]:bins[i + 1]]
        if not part.empty:
            chosen.append(
                part.sort_values(["blur_score", "contrast"],
                                 ascending=False).head(1)
            )
    return pd.concat(chosen, ignore_index=True).sort_values("timestamp_sec")


def make_contact_sheet(paths, output, cols=4, cell_width=320):
    cards = []
    for p in list(paths):
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = cell_width / max(w, 1)
        nh = max(1, int(h * scale))
        img = cv2.resize(img, (cell_width, nh))
        header = 32
        card = np.full((nh + header, cell_width, 3), 255, np.uint8)
        card[header:, :] = img
        cv2.putText(card, p.name[:48], (5, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (0, 0, 0), 1, cv2.LINE_AA)
        cards.append(card)

    if not cards:
        return

    max_h = max(x.shape[0] for x in cards)
    rows = math.ceil(len(cards) / cols)
    sheet = np.full((rows * max_h, cols * cell_width, 3), 255, np.uint8)

    for i, card in enumerate(cards):
        r, c = divmod(i, cols)
        h, w = card.shape[:2]
        y = r * max_h
        x = c * cell_width
        sheet[y:y+h, x:x+w] = card

    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), sheet)


def process_video(src, device_name, device_safe, video_idx,
                  device_out, args, mapping_rows):

    video_id = f"{device_safe}__video_{video_idx:03d}"
    normalized_video_name = f"{video_id}{src.suffix.lower()}"
    normalized_video_path = device_out / "videos" / normalized_video_name
    normalized_video_path.parent.mkdir(parents=True, exist_ok=True)

    if args.copy_videos and not normalized_video_path.exists():
        shutil.copy2(src, normalized_video_path)

    mapping_rows.append({
        "device_name": device_name,
        "file_type": "video",
        "source_path": str(src),
        "source_name": src.name,
        "normalized_name": normalized_video_name,
        "normalized_path": str(normalized_video_path),
        "copied": bool(args.copy_videos),
    })

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {src}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if fps <= 0:
        cap.release()
        raise RuntimeError(f"无法读取 FPS: {src}")

    duration = frame_count / fps if frame_count else 0
    step = max(1, int(round(args.interval * fps)))

    tmp_dir = device_out / "_tmp_frames" / video_id
    final_dir = device_out / "extracted_frames" / video_id
    tmp_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % step == 0:
            ts = frame_idx / fps
            blur, brightness, contrast = calc_metrics(frame)
            passed, reason = quality_ok(blur, brightness, args)

            frame_name = (
                f"{device_safe}__video_{video_idx:03d}"
                f"__t{ts:09.3f}__f{frame_idx:08d}.jpg"
            )
            temp_path = tmp_dir / frame_name
            save_jpg(temp_path, frame)

            rows.append({
                "device_name": device_name,
                "video_index": video_idx,
                "source_video": str(src),
                "source_video_name": src.name,
                "normalized_video_name": normalized_video_name,
                "frame_index": frame_idx,
                "timestamp_sec": round(ts, 3),
                "width": width,
                "height": height,
                "blur_score": round(blur, 3),
                "brightness": round(brightness, 3),
                "contrast": round(contrast, 3),
                "passed_quality": bool(passed),
                "quality_reason": reason,
                "frame_name": frame_name,
            })

        frame_idx += 1

    cap.release()

    df = pd.DataFrame(rows)
    if df.empty:
        return {
            "source_video": src.name,
            "normalized_video": normalized_video_name,
            "duration_sec": round(duration, 3),
            "sampled_frames": 0,
            "quality_passed": 0,
            "selected_frames": 0,
        }, df, []

    selected = select_frames(df, args.topk)
    selected_names = set(selected["frame_name"].tolist())

    final_paths = []
    for name in selected_names:
        src_img = tmp_dir / name
        dst_img = final_dir / name
        if not dst_img.exists():
            shutil.copy2(src_img, dst_img)
        final_paths.append(dst_img)

    df["selected"] = df["frame_name"].isin(selected_names)

    if args.keep_all_frames:
        all_dir = device_out / "all_sampled_frames" / video_id
        all_dir.mkdir(parents=True, exist_ok=True)
        for p in tmp_dir.glob("*.jpg"):
            dst = all_dir / p.name
            if not dst.exists():
                shutil.copy2(p, dst)

    summary = {
        "source_video": src.name,
        "normalized_video": normalized_video_name,
        "width": width,
        "height": height,
        "fps": round(fps, 3),
        "duration_sec": round(duration, 3),
        "sampled_frames": len(df),
        "quality_passed": int(df["passed_quality"].sum()),
        "selected_frames": len(selected_names),
    }
    return summary, df, sorted(final_paths)


def copy_and_rename(files, device_name, device_safe, device_out,
                    file_type, mapping_rows):
    if file_type == "image":
        out_dir = device_out / "source_images"
    else:
        out_dir = device_out / "manuals"

    out_dir.mkdir(parents=True, exist_ok=True)
    copied = []

    for idx, src in enumerate(files, 1):
        dst_name = f"{device_safe}__{file_type}_{idx:03d}{src.suffix.lower()}"
        dst = out_dir / dst_name
        if not dst.exists():
            shutil.copy2(src, dst)

        mapping_rows.append({
            "device_name": device_name,
            "file_type": file_type,
            "source_path": str(src),
            "source_name": src.name,
            "normalized_name": dst.name,
            "normalized_path": str(dst),
            "copied": True,
        })
        copied.append(dst)

    return copied


def main():
    args = parse_args()
    root = args.root.expanduser().resolve()
    output = args.output.expanduser().resolve()

    if not root.is_dir():
        raise SystemExit(f"根目录不存在: {root}")

    output.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("数据根目录:", root)
    print("输出目录:", output)
    print("默认不会修改任何原始文件。")
    print("新视频/图片/抽帧名称均依据设备文件夹名称生成。")
    print("=" * 88)

    mapping_rows = []
    dataset_rows = []

    # 根目录巡检总表
    global_dir = output / "_global"
    global_dir.mkdir(parents=True, exist_ok=True)
    global_files = [
        p for p in root.iterdir()
        if p.is_file()
        and p.suffix.lower() in MANUAL_EXTS
        and any(h in p.name for h in GLOBAL_HINTS)
    ]
    for idx, src in enumerate(sorted(global_files), 1):
        dst = global_dir / f"global_inspection_table_{idx:03d}{src.suffix.lower()}"
        if not dst.exists():
            shutil.copy2(src, dst)
        mapping_rows.append({
            "device_name": "_GLOBAL_",
            "file_type": "global_inspection_table",
            "source_path": str(src),
            "source_name": src.name,
            "normalized_name": dst.name,
            "normalized_path": str(dst),
            "copied": True,
        })

    devices = sorted([p for p in root.iterdir() if p.is_dir()],
                     key=lambda p: p.name)

    print(f"\n发现设备目录: {len(devices)} 个")

    for n, device_dir in enumerate(devices, 1):
        device_name = device_dir.name
        device_safe = normalize_name(device_name)
        device_out = output / device_safe
        device_out.mkdir(parents=True, exist_ok=True)

        print(f"\n[{n}/{len(devices)}] {device_name}")

        manual_dir = find_subdir(device_dir, MANUAL_DIR_NAMES)
        image_dir = find_subdir(device_dir, IMAGE_DIR_NAMES)
        video_dir = find_subdir(device_dir, VIDEO_DIR_NAMES)

        manuals = collect_files(manual_dir, MANUAL_EXTS)
        images = collect_files(image_dir, IMAGE_EXTS)
        videos = collect_files(video_dir, VIDEO_EXTS)

        print(f"  说明书: {len(manuals)}")
        print(f"  图片:   {len(images)}")
        print(f"  视频:   {len(videos)}")

        copied_manuals = copy_and_rename(
            manuals, device_name, device_safe,
            device_out, "manual", mapping_rows
        )
        copied_images = copy_and_rename(
            images, device_name, device_safe,
            device_out, "image", mapping_rows
        )

        all_meta = []
        video_summaries = []
        contact_candidates = list(copied_images[:20])

        for i, video in enumerate(videos, 1):
            print(f"    视频 {i}: {video.name}")
            try:
                summary, meta, frames = process_video(
                    video, device_name, device_safe, i,
                    device_out, args, mapping_rows
                )
                video_summaries.append(summary)
                if not meta.empty:
                    all_meta.append(meta)
                contact_candidates.extend(frames[:12])
                print(f"      -> {summary['normalized_video']}")
                print(f"      -> 代表帧: {summary['selected_frames']}")
            except Exception as e:
                print(f"      [ERROR] {e}")

        device_meta = pd.concat(all_meta, ignore_index=True) if all_meta else pd.DataFrame()
        device_meta.to_csv(device_out / "metadata.csv",
                           index=False, encoding="utf-8-sig")
        pd.DataFrame(video_summaries).to_csv(
            device_out / "videos_summary.csv",
            index=False, encoding="utf-8-sig"
        )

        make_contact_sheet(contact_candidates[:60],
                           device_out / "contact_sheet.jpg")

        total_duration = sum(x.get("duration_sec", 0) for x in video_summaries)
        total_sampled = sum(x.get("sampled_frames", 0) for x in video_summaries)
        total_quality = sum(x.get("quality_passed", 0) for x in video_summaries)
        total_selected = sum(x.get("selected_frames", 0) for x in video_summaries)

        (device_out / "summary.txt").write_text(
            f"""设备名称: {device_name}
原始设备目录: {device_dir}

说明书数量: {len(manuals)}
已有图片数量: {len(images)}
视频数量: {len(videos)}
视频总时长: {total_duration:.3f} 秒

采样帧数量: {total_sampled}
质量通过帧数量: {total_quality}
最终代表帧数量: {total_selected}
总视觉样本数量: {len(copied_images) + total_selected}

说明:
- 原始文件未被修改。
- workspace 中图片、说明书、视频映射、抽帧均依据设备文件夹名称统一命名。
- 原始文件名和规范文件名对应关系见 workspace/file_mapping.csv。
""",
            encoding="utf-8"
        )

        dataset_rows.append({
            "device_name": device_name,
            "source_device_dir": str(device_dir),
            "manual_count": len(manuals),
            "source_image_count": len(images),
            "video_count": len(videos),
            "video_duration_sec": round(total_duration, 3),
            "sampled_frame_count": total_sampled,
            "quality_passed_frame_count": total_quality,
            "selected_frame_count": total_selected,
            "total_visual_samples": len(copied_images) + total_selected,
            "workspace_dir": str(device_out),
        })

    pd.DataFrame(dataset_rows).to_csv(
        output / "dataset_index.csv",
        index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(mapping_rows).to_csv(
        output / "file_mapping.csv",
        index=False, encoding="utf-8-sig"
    )

    print("\n" + "=" * 88)
    print("处理完成")
    print("总索引:", output / "dataset_index.csv")
    print("名称映射:", output / "file_mapping.csv")
    print("=" * 88)


if __name__ == "__main__":
    main()

