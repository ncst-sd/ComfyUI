#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
yolo_state_bridge.py

视觉层 -> 状态判定 API 桥接程序

用途：
1. 接收 YOLO/视觉模块输出的结构化 JSON
2. 校验字段
3. 过滤低置信度观测
4. 转换为 device_state_api.py 的 /judge 请求
5. 输出状态判定结果

当前阶段可先用 mock JSON 测试；
后续接真实 YOLO 时，只要让 YOLO 输出同样结构即可。

示例：
python scripts/yolo_state_bridge.py \
  --input examples/mock_yolo_output.json

也支持 stdin:
cat examples/mock_yolo_output.json | \
python scripts/yolo_state_bridge.py --stdin
"""

import argparse
import json
import sys
from pathlib import Path

import requests
import yaml


DEFAULT_PROJECT = Path.home() / "yolo_device_monitor"
DEFAULT_CONFIG = DEFAULT_PROJECT / "config" / "vision_output_schema.yaml"
DEFAULT_API = "http://127.0.0.1:8100/judge"


def load_yaml(path):
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def normalize_payload(payload, cfg):
    if not isinstance(payload, dict):
        raise ValueError("输入必须是 JSON object")

    device = payload.get("device")
    if not device:
        raise ValueError("缺少 device")

    observations = payload.get("observations")
    if not isinstance(observations, list) or not observations:
        raise ValueError("observations 必须是非空数组")

    min_conf = float(
        cfg.get("filters", {}).get("min_observation_confidence", 0.60)
    )

    allowed_colors = set(
        cfg.get("allowed_values", {}).get("colors", [])
    )
    allowed_states = set(
        cfg.get("allowed_values", {}).get("states", [])
    )

    normalized = []
    dropped = []

    for idx, obs in enumerate(observations):
        if not isinstance(obs, dict):
            dropped.append({
                "index": idx,
                "reason": "observation_not_object"
            })
            continue

        indicator = str(obs.get("indicator", "")).strip()
        color = str(obs.get("color", "unknown")).strip()
        state = str(obs.get("state", "unknown")).strip()
        confidence = float(obs.get("confidence", 1.0))

        if not indicator:
            dropped.append({
                "index": idx,
                "reason": "missing_indicator"
            })
            continue

        if confidence < min_conf:
            dropped.append({
                "index": idx,
                "indicator": indicator,
                "reason": "low_confidence",
                "confidence": confidence,
                "threshold": min_conf,
            })
            continue

        if allowed_colors and color not in allowed_colors:
            dropped.append({
                "index": idx,
                "indicator": indicator,
                "reason": "unsupported_color",
                "value": color,
            })
            continue

        if allowed_states and state not in allowed_states:
            dropped.append({
                "index": idx,
                "indicator": indicator,
                "reason": "unsupported_state",
                "value": state,
            })
            continue

        normalized.append({
            "indicator": indicator,
            "color": color,
            "state": state,
        })

    if not normalized:
        raise ValueError(
            "过滤后没有可提交的有效 observations"
        )

    return {
        "device": str(device).strip(),
        "observations": normalized,
    }, dropped


def main():
    p = argparse.ArgumentParser(
        description="YOLO视觉输出 -> 状态判定API 桥接"
    )

    p.add_argument(
        "--input",
        type=Path,
        help="YOLO输出JSON文件"
    )

    p.add_argument(
        "--stdin",
        action="store_true",
        help="从stdin读取JSON"
    )

    p.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG
    )

    p.add_argument(
        "--api",
        default=DEFAULT_API
    )

    p.add_argument(
        "--timeout",
        type=float,
        default=30
    )

    p.add_argument(
        "--pretty",
        action="store_true",
        help="格式化输出JSON"
    )

    args = p.parse_args()

    if bool(args.input) == bool(args.stdin):
        raise SystemExit(
            "必须二选一：--input <file> 或 --stdin"
        )

    cfg = load_yaml(
        args.config.expanduser().resolve()
    )

    if args.stdin:
        raw = sys.stdin.read()
    else:
        input_path = args.input.expanduser().resolve()
        if not input_path.exists():
            raise SystemExit(
                f"输入文件不存在: {input_path}"
            )
        raw = input_path.read_text(
            encoding="utf-8"
        )

    try:
        payload = json.loads(raw)
    except Exception as e:
        raise SystemExit(
            f"JSON解析失败: {e}"
        )

    try:
        judge_payload, dropped = normalize_payload(
            payload,
            cfg
        )
    except Exception as e:
        raise SystemExit(
            f"视觉输出校验失败: {e}"
        )

    try:
        resp = requests.post(
            args.api,
            json=judge_payload,
            timeout=args.timeout,
        )
    except Exception as e:
        raise SystemExit(
            f"状态API调用失败: {e}"
        )

    if not resp.ok:
        raise SystemExit(
            f"状态API返回错误 HTTP {resp.status_code}: {resp.text}"
        )

    result = {
        "vision_input": payload,
        "judge_request": judge_payload,
        "dropped_observations": dropped,
        "judge_result": resp.json(),
    }

    if args.pretty:
        print(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2
            )
        )
    else:
        print(
            json.dumps(
                result,
                ensure_ascii=False
            )
        )


if __name__ == "__main__":
    main()

