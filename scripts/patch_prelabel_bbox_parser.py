#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

target = Path.home() / "yolo_device_monitor" / "scripts" / "prelabel_images_with_qwen_vl.py"
text = target.read_text(encoding="utf-8")

old = '''    for obj in arr:
        if not isinstance(obj, dict) or "bbox" not in obj:
            continue
        box = clamp_box(obj["bbox"], W, H)
        if box is None:
            continue
'''

new = '''    for obj in arr:
        if not isinstance(obj, dict):
            continue

        # Qwen2.5-VL 常见输出字段是 bbox_2d；
        # 旧提示词/其它模型也可能输出 bbox。
        raw_box = None
        if "bbox_2d" in obj:
            raw_box = obj["bbox_2d"]
        elif "bbox" in obj:
            raw_box = obj["bbox"]

        if raw_box is None:
            continue

        box = clamp_box(raw_box, W, H)
        if box is None:
            continue
'''

if old not in text:
    raise SystemExit(
        "没有找到旧的 bbox 解析代码，可能脚本版本不同。\\n"
        "请把当前 ~/yolo_device_monitor/scripts/prelabel_images_with_qwen_vl.py 发给我。"
    )

text = text.replace(old, new)
target.write_text(text, encoding="utf-8")

print("已修复:", target)
print("现在同时支持 bbox_2d 和 bbox。")

