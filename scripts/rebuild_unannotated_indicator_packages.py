#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
rebuild_unannotated_indicator_packages.py

用途：
- 只重建“尚未开始标注”的设备待标注图片。
- 只要目标设备目录里已经存在 indicator_annotations.json，就整台设备完全跳过。
- 不修改已开始标注设备的 images/、annotator.html、JSON 或其它文件。

默认策略：
- 候选帧间隔：0.8 秒
- 清晰度：Laplacian variance，自适应过滤最差 20%，最低阈值 28
- 去重：dHash <= 7 且灰度直方图相关系数 >= 0.965 视为重复
- 近/中/远景：ORB 特征尺度在“单段视频内部”按 33%/67% 分位数做相对分桶
- 每段视频：至少尽量保留 12 张，最多 36 张
- 每台设备：静态照片 + 视频代表帧总数最多 200 张
- 静态照片优先；视频帧按不同视频轮流分配剩余额度，避免一个长视频占满 200 张
- 视频帧保存为 PNG
- 原始设备目录与待标注目录通过标准化关键字匹配，不要求空格/下划线完全一致
- 匹配不唯一或相似度不足时自动跳过，避免误覆盖
- 直接覆盖未标注设备原来的 images/
"""

import argparse
import csv
import json
import math
import os
import re
import shutil
import unicodedata
from difflib import SequenceMatcher
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


PROJECT = Path.home() / "yolo_device_monitor"
DEFAULT_SOURCE_ROOT = Path("/home/crscs/桌面/数字孪生机房巡检识别模型测试视频")
DEFAULT_OUTPUT_ROOT = PROJECT / "annotation_packages_best_frames_fast"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv", ".flv", ".ts", ".mts", ".m2ts"}


@dataclass
class Candidate:
    video_name: str
    video_idx: int
    frame_idx: int
    time_sec: float
    clarity: float
    dhash: int
    hist: np.ndarray
    scale_score: float
    bucket: str = "mid"
    reject_reason: str = ""
    selected: bool = False
    output_name: str = ""


def sanitize(s: str) -> str:
    return re.sub(r'[\\/:*?"<>|\r\n\t]+', "_", s)


def list_files(root: Path, exts: set) -> List[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def find_device_dirs(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir())


def normalize_device_name(name: str) -> str:
    """
    用于设备目录匹配的标准化名称：
    - Unicode NFKC
    - 转小写
    - 去掉空格、下划线、连字符及其它标点
    - 仅保留中文、英文字母、数字

    例如：
      FAS MDS3400DLX V3.1 佳讯飞鸿
      FAS_MDS3400DLX_V3.1_佳讯飞鸿
    会得到同一个 key。
    """
    s = unicodedata.normalize("NFKC", str(name)).lower()
    return "".join(ch for ch in s if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")


def device_keywords(name: str) -> List[str]:
    """
    辅助模糊匹配。
    把空格、下划线、横线、斜杠和常见标点都当作分隔符。
    """
    s = unicodedata.normalize("NFKC", str(name)).lower()
    parts = re.split(r"[\s_\-./\\()\[\]（）【】,，:：;；]+", s)
    return [p for p in parts if p]


def keyword_overlap(a: str, b: str) -> float:
    ka = set(device_keywords(a))
    kb = set(device_keywords(b))
    if not ka or not kb:
        return 0.0
    return len(ka & kb) / len(ka | kb)


def match_source_to_package_dir(source_name: str, package_dirs: List[Path]):
    """
    将原始数据目录名匹配到 annotation_packages_best_frames_fast 已有目录。

    优先级：
    1. normalize_device_name 完全一致 -> 直接匹配
    2. 否则用 canonical SequenceMatcher + 关键词重合度打分
    3. 只有“唯一且足够强”的候选才自动匹配
    4. 模糊或歧义时返回 None，绝不猜测覆盖
    """
    src_key = normalize_device_name(source_name)

    exact = [p for p in package_dirs if normalize_device_name(p.name) == src_key]
    if len(exact) == 1:
        return exact[0], 1.0, "normalized_exact"
    if len(exact) > 1:
        return None, 0.0, "ambiguous_normalized_exact"

    scored = []
    for p in package_dirs:
        dst_key = normalize_device_name(p.name)
        seq = SequenceMatcher(None, src_key, dst_key).ratio() if src_key and dst_key else 0.0
        overlap = keyword_overlap(source_name, p.name)
        score = 0.75 * seq + 0.25 * overlap
        scored.append((score, seq, overlap, p))

    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored:
        return None, 0.0, "no_package_dirs"

    best = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else 0.0

    # 自动匹配必须满足：
    # - 综合分 >= 0.82
    # - canonical 相似度 >= 0.85
    # - 与第二名至少拉开 0.08，避免误覆盖
    if best[0] >= 0.82 and best[1] >= 0.85 and (best[0] - second_score) >= 0.08:
        return best[3], best[0], "fuzzy_unique"

    return None, best[0], "no_safe_unique_match"


def build_device_mapping(source_root: Path, output_root: Path):
    """
    以 output_root 中现有设备目录名为权威目标名。
    原始数据目录通过关键字/标准化名称匹配到对应目标目录。
    """
    source_dirs = find_device_dirs(source_root)
    package_dirs = sorted(p for p in output_root.iterdir() if p.is_dir()) if output_root.exists() else []

    mappings = []
    unmatched = []

    for src in source_dirs:
        dst, score, method = match_source_to_package_dir(src.name, package_dirs)
        if dst is None:
            unmatched.append({
                "source": src.name,
                "best_score": round(score, 4),
                "reason": method,
            })
        else:
            mappings.append({
                "source_dir": src,
                "package_dir": dst,
                "score": score,
                "method": method,
            })

    return mappings, unmatched


def clarity_score(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    scale = min(1.0, 1280.0 / max(h, w))
    if scale < 1.0:
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def dhash64(frame: np.ndarray) -> int:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    value = 0
    for b in diff.flatten():
        value = (value << 1) | int(bool(b))
    return value


def hamming64(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def gray_hist(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
    return hist.flatten().astype(np.float32)


def hist_corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(cv2.compareHist(a, b, cv2.HISTCMP_CORREL))


def orb_scale_score(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    if max(h, w) > 1280:
        s = 1280.0 / max(h, w)
        gray = cv2.resize(gray, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)

    orb = cv2.ORB_create(nfeatures=800, fastThreshold=12)
    kps = orb.detect(gray, None)
    if not kps:
        return 0.0
    sizes = np.asarray([kp.size for kp in kps], dtype=np.float32)
    return float(np.percentile(sizes, 60))


def read_frame(cap: cv2.VideoCapture, frame_idx: int) -> Optional[np.ndarray]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    return frame if ok and frame is not None else None


def extract_candidates(video: Path, video_idx: int, interval_sec: float, min_candidates: int):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return [], {"error": "open_failed"}

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if fps <= 0 or total <= 0:
        cap.release()
        return [], {"error": "invalid_meta"}

    step = max(1, int(round(interval_sec * fps)))
    frame_ids = list(range(0, total, step))

    if len(frame_ids) < min_candidates and total > 1:
        n = min(min_candidates, total)
        frame_ids = sorted(set(int(round(x)) for x in np.linspace(0, total - 1, n)))

    out = []
    for fi in frame_ids:
        frame = read_frame(cap, fi)
        if frame is None:
            continue
        out.append(Candidate(
            video_name=video.name,
            video_idx=video_idx,
            frame_idx=fi,
            time_sec=fi / fps,
            clarity=clarity_score(frame),
            dhash=dhash64(frame),
            hist=gray_hist(frame),
            scale_score=orb_scale_score(frame),
        ))

    cap.release()
    return out, {
        "fps": fps,
        "total_frames": total,
        "duration_sec": total / fps,
        "candidates": len(out),
    }


def classify_buckets(cands: List[Candidate]):
    vals = np.asarray([c.scale_score for c in cands if c.scale_score > 0], dtype=np.float32)
    if len(vals) < 3:
        for c in cands:
            c.bucket = "mid"
        return

    q1, q2 = np.quantile(vals, [0.33, 0.67])
    for c in cands:
        if c.scale_score <= 0:
            c.bucket = "mid"
        elif c.scale_score <= q1:
            c.bucket = "far"
        elif c.scale_score >= q2:
            c.bucket = "near"
        else:
            c.bucket = "mid"


def adaptive_clarity_threshold(cands: List[Candidate], floor: float, percentile: float) -> float:
    if not cands:
        return floor
    vals = np.asarray([c.clarity for c in cands], dtype=np.float32)
    return max(float(floor), float(np.percentile(vals, percentile)))


def quality_dedup(
    cands: List[Candidate],
    clarity_threshold: float,
    max_hamming: int,
    min_hist_corr: float,
) -> List[Candidate]:
    reps: List[Candidate] = []

    for c in sorted(cands, key=lambda x: x.time_sec):
        if c.clarity < clarity_threshold:
            c.reject_reason = f"blur<{clarity_threshold:.1f}"
            continue

        duplicate = None
        for r in reps[-8:]:
            if hamming64(c.dhash, r.dhash) <= max_hamming and hist_corr(c.hist, r.hist) >= min_hist_corr:
                duplicate = r
                break

        if duplicate is None:
            reps.append(c)
            continue

        if c.clarity > duplicate.clarity * 1.18:
            duplicate.reject_reason = f"replaced_by_clearer@{c.time_sec:.2f}s"
            reps.remove(duplicate)
            reps.append(c)
        else:
            c.reject_reason = f"duplicate@{duplicate.time_sec:.2f}s"

    return sorted(reps, key=lambda x: x.time_sec)


def temporal_pick(items: List[Candidate], k: int) -> List[Candidate]:
    if k <= 0 or not items:
        return []
    if len(items) <= k:
        return list(items)

    items = sorted(items, key=lambda x: x.time_sec)
    t0, t1 = items[0].time_sec, items[-1].time_sec

    if t1 <= t0:
        return sorted(items, key=lambda x: x.clarity, reverse=True)[:k]

    bins = [[] for _ in range(k)]
    for c in items:
        idx = min(k - 1, int((c.time_sec - t0) / (t1 - t0 + 1e-9) * k))
        bins[idx].append(c)

    chosen = [max(b, key=lambda x: x.clarity) for b in bins if b]

    if len(chosen) < k:
        rest = [x for x in items if x not in chosen]
        rest.sort(key=lambda x: x.clarity, reverse=True)
        chosen.extend(rest[:k-len(chosen)])

    return chosen[:k]


def select_video_frames(
    cands: List[Candidate],
    reps: List[Candidate],
    min_per_video: int,
    max_per_video: int,
    min_per_bucket: int,
) -> List[Candidate]:
    by = {"far": [], "mid": [], "near": []}
    for c in reps:
        by.setdefault(c.bucket, []).append(c)

    if len(reps) <= max_per_video:
        selected = list(reps)
    else:
        base = max_per_video // 3
        targets = {
            "far": base,
            "mid": base,
            "near": max_per_video - base * 2,
        }
        selected = []
        for bucket in ("far", "mid", "near"):
            if by[bucket]:
                k = min(len(by[bucket]), max(min_per_bucket, targets[bucket]))
                selected.extend(temporal_pick(by[bucket], k))

        if len(selected) < max_per_video:
            rest = [c for c in reps if c not in selected]
            rest.sort(key=lambda x: x.clarity, reverse=True)
            selected.extend(rest[:max_per_video-len(selected)])

        selected = selected[:max_per_video]

    # 数量不足时自动回填
    if len(selected) < min_per_video:
        existing_times = [c.time_sec for c in selected]
        bucket_counts = Counter(c.bucket for c in selected)
        pool = [c for c in cands if c not in selected]

        def time_distance(c):
            if not existing_times:
                return 9999.0
            return min(abs(c.time_sec - t) for t in existing_times)

        def score(c):
            reason = c.reject_reason or ""
            # 被去重的清晰帧优先于模糊帧
            duplicate_bonus = 3.0 if "duplicate" in reason or "replaced" in reason else 0.0
            blur_penalty = -3.0 if reason.startswith("blur<") else 0.0
            bucket_bonus = 2.0 / (1.0 + bucket_counts.get(c.bucket, 0))
            time_bonus = min(3.0, time_distance(c) / 2.0)
            clarity_bonus = math.log1p(max(0.0, c.clarity)) / 4.0
            return duplicate_bonus + blur_penalty + bucket_bonus + time_bonus + clarity_bonus

        pool.sort(key=score, reverse=True)

        for c in pool:
            if len(selected) >= min_per_video:
                break
            selected.append(c)
            existing_times.append(c.time_sec)
            bucket_counts[c.bucket] += 1

    selected = sorted(selected[:max_per_video], key=lambda x: x.time_sec)

    for c in cands:
        c.selected = c in selected

    return selected


def choose_static_photos(photos: List[Path], max_count: int) -> List[Path]:
    if len(photos) <= max_count:
        return photos

    # 如果静态照片本身超过设备总上限，优先清晰图片，同时兼顾原顺序。
    scored = []
    for i, p in enumerate(photos):
        img = cv2.imread(str(p))
        score = clarity_score(img) if img is not None else 0.0
        scored.append((score, i, p))

    scored.sort(key=lambda x: (-x[0], x[1]))
    chosen = [p for _, _, p in scored[:max_count]]
    chosen_set = set(chosen)
    return [p for p in photos if p in chosen_set]


def cap_video_frames_per_device(
    selected_per_video: List[List[Candidate]],
    budget: int,
) -> List[List[Candidate]]:
    """
    视频总预算按 round-robin 分配：
    每轮每个视频拿 1 张，避免长视频占完设备总额度。
    每个视频内部按照时间顺序覆盖全时段。
    """
    if budget <= 0:
        return [[] for _ in selected_per_video]

    queues = []
    for items in selected_per_video:
        # 先把该视频的已选帧重新按时间均匀排序
        queues.append(list(sorted(items, key=lambda c: c.time_sec)))

    kept = [[] for _ in queues]
    remaining = budget

    # 为了时间覆盖，比单纯从头拿更好：先生成每视频“从时间轴均匀展开”的顺序
    ordered_queues = []
    for q in queues:
        if len(q) <= 1:
            ordered_queues.append(q)
            continue

        order = []
        left, right = 0, len(q) - 1
        take_left = True
        while left <= right:
            if take_left:
                order.append(q[left]); left += 1
            else:
                order.append(q[right]); right -= 1
            take_left = not take_left
        ordered_queues.append(order)

    while remaining > 0:
        progressed = False
        for i, q in enumerate(ordered_queues):
            if remaining <= 0:
                break
            if q:
                kept[i].append(q.pop(0))
                remaining -= 1
                progressed = True
        if not progressed:
            break

    for x in kept:
        x.sort(key=lambda c: c.time_sec)

    return kept


def render_video_frames(
    video: Path,
    selected: List[Candidate],
    out_images: Path,
    device_name: str,
    global_counter: int,
) -> Tuple[List[str], int]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return [], global_counter

    names = []
    for c in selected:
        frame = read_frame(cap, c.frame_idx)
        if frame is None:
            continue

        global_counter += 1
        name = (
            f"video_{global_counter:04d}__{sanitize(device_name)}__"
            f"video_{c.video_idx:03d}__{c.bucket}__"
            f"t{c.time_sec:09.3f}__f{c.frame_idx:08d}.png"
        )
        cv2.imwrite(
            str(out_images / name),
            frame,
            [cv2.IMWRITE_PNG_COMPRESSION, 3],
        )
        c.output_name = name
        names.append(name)

    cap.release()
    return names, global_counter


def rebuild_device(args, device_dir: Path, package_dir: Path):
    device = package_dir.name
    ann_path = package_dir / "indicator_annotations.json"

    # 最重要的保护规则
    if ann_path.exists():
        print(f"[保护/跳过] {device}  (已存在 indicator_annotations.json)")
        return None

    print(f"[开始重建] {device}")

    package_dir.mkdir(parents=True, exist_ok=True)
    out_images = package_dir / "images"

    # 未标注设备才允许覆盖 images/
    if out_images.exists():
        shutil.rmtree(out_images)
    out_images.mkdir(parents=True, exist_ok=True)

    # 1. 静态照片
    photo_root = device_dir / "图片"
    photos = list_files(photo_root, IMAGE_EXTS)
    if not photos:
        photos = [
            p for p in list_files(device_dir, IMAGE_EXTS)
            if "说明书" not in p.parts and "视频" not in p.parts
        ]

    photos = choose_static_photos(photos, args.max_per_device)

    photo_names = []
    for i, src in enumerate(photos, 1):
        name = f"photo_{i:04d}__{sanitize(device)}__{sanitize(src.stem)}{src.suffix.lower()}"
        shutil.copy2(src, out_images / name)
        photo_names.append(name)

    remaining_budget = max(0, args.max_per_device - len(photo_names))

    # 2. 视频候选
    video_root = device_dir / "视频"
    videos = list_files(video_root, VIDEO_EXTS)
    if not videos:
        videos = [p for p in list_files(device_dir, VIDEO_EXTS) if "说明书" not in p.parts]

    all_candidates = []
    selected_per_video = []
    thresholds = {}
    video_meta = []

    for vid_idx, video in enumerate(videos, 1):
        cands, meta = extract_candidates(
            video,
            vid_idx,
            args.interval_sec,
            args.min_candidates,
        )

        if not cands:
            all_candidates.append([])
            selected_per_video.append([])
            video_meta.append({"video": video.name, **meta, "selected_before_device_cap": 0})
            continue

        classify_buckets(cands)
        threshold = adaptive_clarity_threshold(
            cands,
            args.clarity_floor,
            args.clarity_percentile,
        )
        thresholds[video.name] = threshold

        reps = quality_dedup(
            cands,
            threshold,
            args.duplicate_max_hamming,
            args.duplicate_min_hist_corr,
        )

        selected = select_video_frames(
            cands,
            reps,
            args.min_per_video,
            args.max_per_video,
            args.min_per_bucket,
        )

        all_candidates.append(cands)
        selected_per_video.append(selected)
        video_meta.append({
            "video": video.name,
            **meta,
            "clarity_threshold": threshold,
            "after_quality_dedup": len(reps),
            "selected_before_device_cap": len(selected),
            "far": sum(c.bucket == "far" for c in selected),
            "mid": sum(c.bucket == "mid" for c in selected),
            "near": sum(c.bucket == "near" for c in selected),
        })

    # 3. 应用“每设备最多 200 张”的总预算
    kept_per_video = cap_video_frames_per_device(selected_per_video, remaining_budget)

    for cands in all_candidates:
        for c in cands:
            c.selected = False

    for kept in kept_per_video:
        for c in kept:
            c.selected = True

    # 4. 真正写出视频 PNG
    video_names = []
    global_counter = 0
    for video, kept in zip(videos, kept_per_video):
        names, global_counter = render_video_frames(
            video, kept, out_images, device, global_counter
        )
        video_names.extend(names)

    # 5. 报告
    report_rows = []
    for cands in all_candidates:
        for c in cands:
            reason = c.reject_reason
            if not c.selected and not reason:
                reason = "dropped_by_device_max_or_balance"
            report_rows.append({
                "device": device,
                "video": c.video_name,
                "frame_idx": c.frame_idx,
                "time_sec": f"{c.time_sec:.3f}",
                "clarity": f"{c.clarity:.3f}",
                "clarity_threshold": f"{thresholds.get(c.video_name, 0):.3f}",
                "scale_score": f"{c.scale_score:.3f}",
                "scale_bucket": c.bucket,
                "selected": int(c.selected),
                "reject_reason": reason,
                "output_name": c.output_name,
            })

    with (package_dir / "frame_selection_report.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        fields = [
            "device", "video", "frame_idx", "time_sec", "clarity",
            "clarity_threshold", "scale_score", "scale_bucket",
            "selected", "reject_reason", "output_name",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(report_rows)

    all_names = photo_names + video_names

    summary = {
        "device": device,
        "strategy": {
            "interval_sec": args.interval_sec,
            "clarity_floor": args.clarity_floor,
            "clarity_percentile": args.clarity_percentile,
            "duplicate_max_hamming": args.duplicate_max_hamming,
            "duplicate_min_hist_corr": args.duplicate_min_hist_corr,
            "min_per_video": args.min_per_video,
            "max_per_video": args.max_per_video,
            "max_per_device": args.max_per_device,
            "scale_method": "ORB relative quantiles inside each video",
        },
        "static_photos": len(photo_names),
        "video_frames": len(video_names),
        "total_images": len(all_names),
        "video_meta": video_meta,
    }

    (package_dir / "frame_selection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 不创建 indicator_annotations.json。
    # 这样“存在 indicator_annotations.json = 已经开始标注”的保护规则保持清晰。
    print(
        f"[完成] {device}: 静态照片 {len(photo_names)} + "
        f"视频代表帧 {len(video_names)} = {len(all_names)} 张 "
        f"(设备上限 {args.max_per_device})"
    )
    return summary


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)

    ap.add_argument("--interval-sec", type=float, default=0.8)
    ap.add_argument("--min-candidates", type=int, default=16)

    ap.add_argument("--clarity-floor", type=float, default=28.0)
    ap.add_argument("--clarity-percentile", type=float, default=20.0)

    ap.add_argument("--duplicate-max-hamming", type=int, default=7)
    ap.add_argument("--duplicate-min-hist-corr", type=float, default=0.965)

    ap.add_argument("--min-per-video", type=int, default=12)
    ap.add_argument("--max-per-video", type=int, default=36)
    ap.add_argument("--min-per-bucket", type=int, default=4)

    ap.add_argument("--max-per-device", type=int, default=200)

    args = ap.parse_args()
    args.source_root = args.source_root.expanduser()
    args.output_root = args.output_root.expanduser()

    if not args.source_root.exists():
        raise SystemExit(f"原始数据目录不存在: {args.source_root}")

    mappings, unmatched = build_device_mapping(args.source_root, args.output_root)
    if not mappings and not unmatched:
        raise SystemExit("没有找到设备目录")

    print("原始数据:", args.source_root)
    print("输出目录:", args.output_root)
    print("目录匹配: 标准化关键字匹配，不要求空格/下划线完全一致")
    print("保护规则: 已存在 indicator_annotations.json 的设备完全跳过")
    print("设备图片总上限:", args.max_per_device)
    print()

    print("设备目录匹配结果:")
    for m in mappings:
        print(
            f"  [匹配] {m['source_dir'].name}  ->  {m['package_dir'].name} "
            f"({m['method']}, score={m['score']:.3f})"
        )
    for u in unmatched:
        print(
            f"  [未安全匹配/跳过] {u['source']} "
            f"(best_score={u['best_score']:.3f}, {u['reason']})"
        )
    print()

    processed = []
    skipped = []

    for m in mappings:
        d = m["source_dir"]
        package_dir = m["package_dir"]
        ann = package_dir / "indicator_annotations.json"

        if ann.exists():
            print(
                f"[保护/跳过] {package_dir.name}  "
                f"(源目录: {d.name}; 已存在 indicator_annotations.json)"
            )
            skipped.append(package_dir.name)
            continue

        try:
            result = rebuild_device(args, d, package_dir)
            if result:
                processed.append(result)
        except Exception as e:
            print(f"[失败] {d.name} -> {package_dir.name}: {e}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    global_summary = {
        "processed_devices": [x["device"] for x in processed],
        "skipped_with_indicator_annotations": skipped,
        "unmatched_source_devices": unmatched,
        "max_per_device": args.max_per_device,
        "summaries": processed,
    }
    (args.output_root / "rebuild_unannotated_summary.json").write_text(
        json.dumps(global_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("处理完成")
    print("重建设备:", len(processed))
    print("保护跳过:", len(skipped))
    print("每台设备最终图片不会超过:", args.max_per_device)


if __name__ == "__main__":
    main()

