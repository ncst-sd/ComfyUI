#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dedupe_manual_indicator_schema.py

对 manual_indicator_schema.json 做“保守去重”：
1) 自动合并完全重复 / 仅空格标点差异的类别；
2) 自动合并一批明确同义的常见指示灯名称；
3) 不自动合并模块语义明显不同的类别；
4) 合并后重新连续编号 class_id；
5) 保留 aliases / modules / location_hints / evidence / states；
6) 输出 ambiguous_candidates.json，供人工确认近似但不安全自动合并的类别；
7) 修改前自动备份 manual_indicator_schema.json.bak。

建议：去重后重新跑 prelabel，不要继续使用旧 pre_annotations.json。
"""

import argparse
import json
import re
import shutil
import unicodedata
from pathlib import Path
from difflib import SequenceMatcher


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s)).upper()
    s = s.replace("（", "(").replace("）", ")")
    s = re.sub(r"[\s_\-./\\:：,，;；()（）\[\]【】]+", "", s)
    return s


def semantic_key(name: str, short_label: str = "") -> str:
    """
    对常见灯做语义归一。优先匹配更具体的类别，避免 LINK/ACT 被 ACT 吞掉。
    """
    s = f"{short_label} {name}".upper()
    n = norm(s)

    patterns = [
        ("LINK_ACT", [r"LINK.?ACT", r"链路状态.*数据传输", r"载波信号"]),
        ("ALM",      [r"\bALM\b", r"告警指示灯", r"系统告警"]),
        ("PALM",     [r"\bPALM\b", r"功率管理告警"]),
        ("PWR",      [r"\bPWR\b", r"\bPOWER\b", r"电源指示灯"]),
        ("RUN",      [r"\bRUN\b", r"运行状态指示灯", r"运行指示灯"]),
        ("ACT",      [r"\bACT\b", r"主.?备用状态指示灯", r"激活灯"]),
        ("STAT",     [r"\bSTAT\b", r"状态指示灯"]),
        ("GE",       [r"\bGE\b.*指示灯", r"GE接口指示灯"]),
        ("FE",       [r"\bFE\b.*指示灯", r"FE接口指示灯"]),
        ("SFP",      [r"\bSFP\+?\b.*指示灯", r"SFP接口指示灯"]),
        ("XFP",      [r"\bXFP\b.*指示灯", r"XFP接口.*指示灯"]),
        ("USB",      [r"\bUSB\d*\b.*指示灯", r"USB接口.*指示灯", r"USB\d*状态指示灯"]),
        ("CF",       [r"\bCF\b.*指示灯", r"CF卡.*指示灯"]),
        ("MANAGEMENT", [r"MANAGEMENT.*指示灯", r"管理以太网口指示灯"]),
        ("CONSOLE_AUX", [r"CONSOLE.?AUX.*指示灯"]),
        ("CONSOLE",  [r"CONSOLE口指示灯"]),
        ("AUX",      [r"AUX口指示灯"]),
        ("POE",      [r"\bPOE\b.*指示灯"]),
    ]

    for key, regs in patterns:
        for rg in regs:
            if re.search(rg, s, re.I):
                return key

    # 完全归一后的名称兜底
    return "RAW:" + norm(name)


def canonical_name(key: str, items):
    preferred = {
        "LINK_ACT": "LINK/ACT 链路/数据传输指示灯",
        "ALM": "ALM 告警指示灯",
        "PALM": "PALM 功率管理告警指示灯",
        "PWR": "PWR 电源指示灯",
        "RUN": "RUN 运行指示灯",
        "ACT": "ACT 主/备用状态指示灯",
        "STAT": "STAT 状态指示灯",
        "GE": "GE 接口指示灯",
        "FE": "FE 接口指示灯",
        "SFP": "SFP 接口指示灯",
        "XFP": "XFP 接口指示灯",
        "USB": "USB 状态指示灯",
        "CF": "CF 卡状态指示灯",
        "MANAGEMENT": "MANAGEMENT 管理以太网口指示灯",
        "CONSOLE_AUX": "CONSOLE/AUX 口指示灯",
        "CONSOLE": "CONSOLE 口指示灯",
        "AUX": "AUX 口指示灯",
        "POE": "POE 接口指示灯",
    }
    if key in preferred:
        return preferred[key]

    # raw 类别：取最短的非空名称，避免带模块前缀的冗长名字作为主名称
    names = [str(x.get("name", "")).strip() for x in items if str(x.get("name", "")).strip()]
    return min(names, key=len) if names else key.replace("RAW:", "")


def uniq_text(values):
    out, seen = [], set()
    for v in values:
        v = str(v).strip()
        if not v:
            continue
        k = norm(v)
        if k not in seen:
            seen.add(k)
            out.append(v)
    return out


def uniq_states(states):
    out, seen = [], set()
    for s in states:
        if not isinstance(s, dict):
            continue
        key = json.dumps(s, ensure_ascii=False, sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


def should_keep(item):
    name = str(item.get("name", "")).strip()
    if not name:
        return False

    # 明显只是接口本体、没有“灯/指示/状态”语义的条目不自动删除，
    # 只放入 ambiguous 供人工看，避免误删真实丝印。
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", required=True, help="manual_indicator_schema.json 路径")
    ap.add_argument("--apply-semantic-merge", action="store_true",
                    help="启用常见语义类别合并；不加则只做严格重复去重")
    args = ap.parse_args()

    schema_path = Path(args.schema).expanduser().resolve()
    if not schema_path.exists():
        raise SystemExit(f"不存在: {schema_path}")

    data = json.loads(schema_path.read_text(encoding="utf-8"))
    indicators = data.get("indicators", [])
    if not isinstance(indicators, list):
        raise SystemExit("schema 中 indicators 不是数组")

    backup = schema_path.with_suffix(schema_path.suffix + ".bak")
    shutil.copy2(schema_path, backup)

    groups = {}
    for item in indicators:
        if not isinstance(item, dict) or not should_keep(item):
            continue

        if args.apply_semantic_merge:
            key = semantic_key(item.get("name", ""), item.get("short_label", ""))
        else:
            key = "STRICT:" + norm(item.get("name", ""))

        groups.setdefault(key, []).append(item)

    merged = []
    merge_report = []

    for key, items in groups.items():
        base = dict(items[0])

        names = uniq_text([x.get("name", "") for x in items])
        aliases = uniq_text(
            [a for x in items for a in x.get("aliases", [])] + names
        )
        modules = uniq_text(
            [x.get("module", "") for x in items] +
            [m for x in items for m in x.get("modules", [])]
        )
        locs = uniq_text(
            [x.get("location_hint", "") for x in items] +
            [m for x in items for m in x.get("location_hints", [])]
        )
        evidences = uniq_text([x.get("evidence", "") for x in items])
        states = uniq_states([s for x in items for s in x.get("states", [])])

        base["name"] = canonical_name(key, items)
        base["aliases"] = [x for x in aliases if norm(x) != norm(base["name"])]
        base["modules"] = modules
        base["location_hints"] = locs
        base["evidence_list"] = evidences
        base["states"] = states

        # 兼容旧字段
        base["module"] = modules[0] if modules else ""
        base["location_hint"] = locs[0] if locs else ""
        base["evidence"] = evidences[0] if evidences else ""

        merged.append(base)

        if len(items) > 1:
            merge_report.append({
                "semantic_key": key,
                "merged_to": base["name"],
                "old_class_ids": [x.get("class_id") for x in items],
                "old_names": names,
                "modules": modules,
            })

    # 稳定排序并重新编号
    merged.sort(key=lambda x: (semantic_key(x.get("name", ""), x.get("short_label", "")),
                               norm(x.get("name", ""))))
    for i, item in enumerate(merged):
        item["class_id"] = i

    # 找“高相似但未自动合并”的项，供人工确认
    ambiguous = []
    for i in range(len(merged)):
        for j in range(i + 1, len(merged)):
            a, b = merged[i], merged[j]
            na, nb = norm(a.get("name", "")), norm(b.get("name", ""))
            if not na or not nb:
                continue
            ratio = SequenceMatcher(None, na, nb).ratio()
            if ratio >= 0.72 and semantic_key(a.get("name", ""), a.get("short_label", "")) != \
               semantic_key(b.get("name", ""), b.get("short_label", "")):
                ambiguous.append({
                    "class_a": a["class_id"],
                    "name_a": a["name"],
                    "class_b": b["class_id"],
                    "name_b": b["name"],
                    "similarity": round(ratio, 3),
                })

    data["indicators"] = merged
    data["dedupe"] = {
        "original_count": len(indicators),
        "new_count": len(merged),
        "semantic_merge": bool(args.apply_semantic_merge),
        "merge_report": merge_report,
    }

    schema_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    amb_path = schema_path.parent / "ambiguous_indicator_candidates.json"
    amb_path.write_text(json.dumps(ambiguous, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"备份: {backup}")
    print(f"原类别数: {len(indicators)}")
    print(f"去重后: {len(merged)}")
    print(f"自动合并组数: {len(merge_report)}")
    print(f"待人工确认的近似类别对: {len(ambiguous)}")
    print(f"更新: {schema_path}")
    print(f"人工确认列表: {amb_path}")

    if merge_report:
        print("\n自动合并:")
        for r in merge_report:
            print(f'- {r["old_class_ids"]} {r["old_names"]} -> {r["merged_to"]}')


if __name__ == "__main__":
    main()

