#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_device_alias_map.py

为“规则库设备名”和“workspace设备目录名”建立统一映射。

输入：
  ~/yolo_device_monitor/rules/device_indicator_rules.yaml
  ~/yolo_device_monitor/workspace/
  ~/yolo_device_monitor/workspace/dataset_index.csv   (如果存在则辅助使用)

输出：
  ~/yolo_device_monitor/rules/device_alias_map.yaml
  ~/yolo_device_monitor/rules/device_alias_map_review.csv

设计目标：
- canonical_name：规则库中的标准设备名
- rule_name：规则库设备名
- workspace_name：workspace 里的实际设备目录名
- aliases：常见别名（自动生成）
- confidence：自动匹配置信度
- needs_review：低置信度时要求人工复核

注意：
- 本脚本只生成“候选映射”，不会修改原规则库和RAG数据库。
- 建议人工查看 review.csv 后再用于生产。
"""

import argparse
import csv
import re
from pathlib import Path

import yaml

try:
    import pandas as pd
except Exception:
    pd = None

try:
    from rapidfuzz import fuzz, process
except Exception:
    fuzz = None
    process = None


def clean_text(s):
    if s is None:
        return ""
    s = str(s).replace("\xa0", " ").replace("\u3000", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_name(s):
    s = clean_text(s).lower()

    replacements = {
        "（": "(",
        "）": ")",
        "－": "-",
        "—": "-",
        "–": "-",
        "_": " ",
        "/": " ",
        "\\": " ",
    }

    for k, v in replacements.items():
        s = s.replace(k, v)

    # 常见无意义词弱化
    noise = [
        "设备",
        "机柜",
        "模块",
        "单元",
        "装置",
        "主机",
        "系统",
        "巡检",
    ]

    for n in noise:
        s = s.replace(n, " ")

    s = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    return s


def compact_name(s):
    return re.sub(
        r"[^0-9a-z\u4e00-\u9fff]",
        "",
        normalize_name(s),
    )


def tokenize(s):
    n = normalize_name(s)
    tokens = set()

    for part in n.split():
        if part:
            tokens.add(part)

    # 提取型号类token，如 SR6608、UPS2000、UT5202
    for m in re.findall(r"[a-z]+\s*[-_]?\s*\d+[a-z0-9-]*", n, flags=re.I):
        tokens.add(re.sub(r"\s+", "", m).lower())

    # 提取纯数字+字母组合
    for m in re.findall(r"[a-z0-9-]{3,}", n, flags=re.I):
        tokens.add(m.lower())

    return tokens


def score_pair(rule_name, workspace_name):
    """
    组合评分：
    - 规范化全名相似度
    - compact字符串相似度
    - 型号/词元重叠
    """

    rn = normalize_name(rule_name)
    wn = normalize_name(workspace_name)

    rc = compact_name(rule_name)
    wc = compact_name(workspace_name)

    rt = tokenize(rule_name)
    wt = tokenize(workspace_name)

    scores = []

    if fuzz is not None:
        scores.append(fuzz.WRatio(rn, wn))
        scores.append(fuzz.ratio(rc, wc))
        scores.append(fuzz.token_set_ratio(rn, wn))

    if rc and wc:
        if rc == wc:
            scores.append(100)
        elif rc in wc or wc in rc:
            scores.append(96)

    overlap = rt & wt

    # 型号token非常重要
    model_tokens = {
        t for t in overlap
        if re.search(r"[a-z]", t) and re.search(r"\d", t)
    }

    bonus = 0

    if model_tokens:
        bonus += 10

    if overlap:
        bonus += min(10, len(overlap) * 2)

    base = max(scores) if scores else 0
    final = min(100, base + bonus)

    return int(round(final)), sorted(overlap), sorted(model_tokens)


def generate_aliases(canonical_name):
    aliases = set()

    raw = clean_text(canonical_name)
    aliases.add(raw)

    # 去括号内容
    no_brackets = re.sub(r"（.*?）|\(.*?\)", "", raw).strip()
    if no_brackets:
        aliases.add(no_brackets)

    # 提取英文/型号组合
    english_model = re.findall(
        r"(?:[A-Za-z]+(?:[- ]?[A-Za-z0-9]+)*\s*)+",
        raw
    )

    for x in english_model:
        x = clean_text(x)
        if len(x) >= 3:
            aliases.add(x)

    # 型号
    for x in re.findall(
        r"[A-Za-z]{1,10}[- ]?\d+[A-Za-z0-9-]*",
        raw
    ):
        aliases.add(x.replace(" ", ""))

    # 常见简称片段
    for token in re.split(r"[\s（）()、,/]+", raw):
        token = token.strip()
        if len(token) >= 3:
            aliases.add(token)

    aliases = sorted(
        x for x in aliases
        if x and x != raw
    )

    return aliases


def read_workspace_names(workspace_dir):
    names = []

    if not workspace_dir.exists():
        return names

    for p in sorted(workspace_dir.iterdir()):
        if not p.is_dir():
            continue
        if p.name.startswith("_"):
            continue
        names.append(p.name)

    return names


def read_dataset_index(index_path):
    rows = []

    if not index_path.exists() or pd is None:
        return rows

    try:
        df = pd.read_csv(
            index_path,
            dtype=str,
            keep_default_na=False
        )
    except Exception:
        return rows

    return df.to_dict("records")


def maybe_get_index_names(index_rows):
    """
    尽可能从 dataset_index.csv 中发现设备名列。
    """
    if not index_rows:
        return []

    possible_cols = [
        "device_name",
        "device",
        "equipment_name",
        "equipment",
        "设备名称",
        "设备",
        "folder_name",
        "source_device",
    ]

    cols = list(index_rows[0].keys())

    selected = None

    for c in possible_cols:
        if c in cols:
            selected = c
            break

    if selected is None:
        return []

    vals = []

    for r in index_rows:
        v = clean_text(r.get(selected))
        if v:
            vals.append(v)

    return sorted(set(vals))


def main():
    ap = argparse.ArgumentParser(
        description="生成设备名称统一映射"
    )

    ap.add_argument(
        "--project",
        type=Path,
        default=Path.home() / "yolo_device_monitor",
    )

    ap.add_argument(
        "--rules",
        type=Path,
        default=None,
    )

    ap.add_argument(
        "--workspace",
        type=Path,
        default=None,
    )

    ap.add_argument(
        "--review-threshold",
        type=int,
        default=82,
        help="低于该分数标记 needs_review=true",
    )

    args = ap.parse_args()

    project = args.project.expanduser().resolve()

    rules_path = (
        args.rules.expanduser().resolve()
        if args.rules
        else project / "rules" / "device_indicator_rules.yaml"
    )

    workspace_dir = (
        args.workspace.expanduser().resolve()
        if args.workspace
        else project / "workspace"
    )

    output_yaml = (
        project / "rules" / "device_alias_map.yaml"
    )

    output_csv = (
        project / "rules" / "device_alias_map_review.csv"
    )

    if not rules_path.exists():
        raise SystemExit(
            f"规则文件不存在: {rules_path}"
        )

    rules = yaml.safe_load(
        rules_path.read_text(
            encoding="utf-8"
        )
    ) or {}

    rule_devices = list(
        rules.get(
            "devices",
            {}
        ).keys()
    )

    if not rule_devices:
        raise SystemExit(
            "规则文件中没有找到 devices"
        )

    workspace_names = read_workspace_names(
        workspace_dir
    )

    index_rows = read_dataset_index(
        workspace_dir / "dataset_index.csv"
    )

    index_names = maybe_get_index_names(
        index_rows
    )

    # 目录名优先，dataset_index 做补充
    all_workspace_names = sorted(
        set(
            workspace_names
            + index_names
        )
    )

    if not all_workspace_names:
        raise SystemExit(
            f"没有发现 workspace 设备目录: {workspace_dir}"
        )

    result = {
        "schema_version": 1,
        "source_rules": str(rules_path),
        "source_workspace": str(workspace_dir),
        "devices": {},
    }

    review_rows = []

    used_workspace = set()

    for rule_name in rule_devices:

        candidates = []

        for ws_name in all_workspace_names:
            score, overlap, models = score_pair(
                rule_name,
                ws_name
            )

            candidates.append({
                "workspace_name": ws_name,
                "score": score,
                "overlap_tokens": overlap,
                "model_tokens": models,
            })

        candidates.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        best = candidates[0]

        # 如果第一名已被占用且第二名接近，仍保留第一名，但要求复核
        duplicate = (
            best["workspace_name"]
            in used_workspace
        )

        needs_review = (
            best["score"]
            < args.review_threshold
            or duplicate
        )

        canonical_id = re.sub(
            r"[^0-9a-z]+",
            "_",
            normalize_name(
                rule_name
            )
        ).strip("_")

        if not canonical_id:
            canonical_id = (
                "device_"
                + str(
                    len(result["devices"])
                    + 1
                )
            )

        aliases = generate_aliases(
            rule_name
        )

        result["devices"][canonical_id] = {
            "canonical_name":
                rule_name,

            "rule_name":
                rule_name,

            "workspace_name":
                best[
                    "workspace_name"
                ],

            "aliases":
                aliases,

            "confidence":
                best[
                    "score"
                ],

            "matched_tokens":
                best[
                    "overlap_tokens"
                ],

            "matched_model_tokens":
                best[
                    "model_tokens"
                ],

            "needs_review":
                bool(
                    needs_review
                ),

            "top_candidates": [
                {
                    "workspace_name":
                        x[
                            "workspace_name"
                        ],
                    "score":
                        x[
                            "score"
                        ]
                }
                for x in candidates[:3]
            ]
        }

        if not duplicate:
            used_workspace.add(
                best[
                    "workspace_name"
                ]
            )

        review_rows.append({
            "canonical_id":
                canonical_id,
            "rule_name":
                rule_name,
            "workspace_name":
                best[
                    "workspace_name"
                ],
            "confidence":
                best[
                    "score"
                ],
            "needs_review":
                needs_review,
            "candidate_2":
                candidates[1]["workspace_name"]
                if len(candidates) > 1
                else "",
            "candidate_2_score":
                candidates[1]["score"]
                if len(candidates) > 1
                else "",
            "candidate_3":
                candidates[2]["workspace_name"]
                if len(candidates) > 2
                else "",
            "candidate_3_score":
                candidates[2]["score"]
                if len(candidates) > 2
                else "",
        })

    output_yaml.write_text(
        yaml.safe_dump(
            result,
            allow_unicode=True,
            sort_keys=False,
            width=140,
        ),
        encoding="utf-8"
    )

    with output_csv.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                review_rows[0].keys()
            )
        )

        writer.writeheader()
        writer.writerows(
            review_rows
        )

    review_count = sum(
        1
        for x in review_rows
        if x["needs_review"]
    )

    print("=" * 88)
    print("设备名称映射生成完成")
    print("=" * 88)
    print("规则设备数量:", len(rule_devices))
    print("workspace候选数量:", len(all_workspace_names))
    print("需人工复核:", review_count)
    print()
    print("YAML:")
    print(output_yaml)
    print()
    print("Review CSV:")
    print(output_csv)
    print()
    print("建议先执行:")
    print(
        f'column -s, -t < "{output_csv}" | less -S'
    )


if __name__ == "__main__":
    main()

