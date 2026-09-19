#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
add_device_rule.py

向 rules/custom_device_rules.yaml 添加/修改自定义确定性规则。

示例：
python scripts/add_device_rule.py \
  --device "H3C SR6608 路由器 AR02" \
  --indicator "业务板 ACT 激活灯" \
  --color yellow \
  --state steady \
  --meaning "经现场确认：黄色常亮表示XXX" \
  --severity warning

规则优先级：
custom_device_rules.yaml > device_indicator_rules.yaml
"""

import argparse
import shutil
from datetime import datetime
from pathlib import Path
import yaml


VALID_COLORS = {
    "green", "red", "yellow", "blue", "orange",
    "orange_red", "yellow_green", "unknown"
}
VALID_STATES = {
    "steady", "blinking", "off", "on",
    "steady_or_blinking", "unknown"
}
VALID_SEVERITIES = {
    "normal", "info", "warning", "critical", "unknown"
}


def load_yaml(path):
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def main():
    p = argparse.ArgumentParser(description="添加/更新设备确定性规则")
    p.add_argument("--project", type=Path, default=Path.home() / "yolo_device_monitor")
    p.add_argument("--device", required=True)
    p.add_argument("--indicator", required=True)
    p.add_argument("--color", required=True)
    p.add_argument("--state", required=True)
    p.add_argument("--meaning", required=True)
    p.add_argument("--severity", required=True, choices=sorted(VALID_SEVERITIES))
    p.add_argument("--alarm", choices=["true", "false"], default=None)
    p.add_argument("--note", default="")
    args = p.parse_args()

    project = args.project.expanduser().resolve()
    rules_dir = project / "rules"
    base_path = rules_dir / "device_indicator_rules.yaml"
    custom_path = rules_dir / "custom_device_rules.yaml"

    if not base_path.exists():
        raise SystemExit(f"基础规则库不存在: {base_path}")

    color = args.color.strip().lower()
    state = args.state.strip().lower()

    if color not in VALID_COLORS:
        raise SystemExit(f"不支持的 color={color}，允许: {sorted(VALID_COLORS)}")
    if state not in VALID_STATES:
        raise SystemExit(f"不支持的 state={state}，允许: {sorted(VALID_STATES)}")

    base = load_yaml(base_path)
    base_devices = base.get("devices", {})

    if args.device not in base_devices:
        raise SystemExit(
            f"设备名不在基础规则库中: {args.device}\n"
            "请使用规则库中的 canonical rule_name。"
        )

    # 指示灯允许新增，但提醒
    indicator_exists = args.indicator in base_devices[args.device].get("indicators", {})

    custom = load_yaml(custom_path)
    if not custom:
        custom = {
            "schema_version": 1,
            "description": "人工确认后追加的确定性规则；优先级高于基础规则库。",
            "devices": {},
        }

    # 备份
    if custom_path.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = custom_path.with_name(f"custom_device_rules.{ts}.bak.yaml")
        shutil.copy2(custom_path, backup)
        print("已备份:", backup)

    dev = custom.setdefault("devices", {}).setdefault(
        args.device,
        {"indicators": {}}
    )
    ind = dev.setdefault("indicators", {}).setdefault(
        args.indicator,
        {"rules": []}
    )

    if args.alarm is None:
        alarm = args.severity in {"warning", "critical"}
    else:
        alarm = args.alarm == "true"

    new_rule = {
        "color": color,
        "color_raw": color,
        "state": state,
        "state_raw": state,
        "meaning": args.meaning,
        "severity": args.severity,
        "alarm": alarm,
        "note": args.note,
        "verified_by_human": True,
    }

    # 同 color/state 则覆盖，否则追加
    replaced = False
    for i, old in enumerate(ind["rules"]):
        if (
            str(old.get("color", "")).strip() == color
            and str(old.get("state", "")).strip() == state
        ):
            ind["rules"][i] = new_rule
            replaced = True
            break

    if not replaced:
        ind["rules"].append(new_rule)

    custom_path.write_text(
        yaml.safe_dump(
            custom,
            allow_unicode=True,
            sort_keys=False,
            width=140,
        ),
        encoding="utf-8",
    )

    print()
    print("规则已保存:", custom_path)
    print("设备:", args.device)
    print("指示灯:", args.indicator)
    print("color/state:", color, state)
    print("severity:", args.severity)
    print("alarm:", alarm)
    print("动作:", "覆盖已有自定义规则" if replaced else "新增自定义规则")
    if not indicator_exists:
        print("注意：该指示灯名不在基础规则库中，本次作为新指示灯加入。")
    print()
    print("无需重建RAG；下次启动 device_state_engine.py 时会自动加载。")


if __name__ == "__main__":
    main()

