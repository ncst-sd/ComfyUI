#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_manual_indicator_schema.py v2

目标：
1) 从说明书中只提取“物理上可画框的指示灯”，不把颜色/常亮/闪烁等状态当类别；
2) 同一物理灯的不同状态写入 states，不拆成多个类别；
3) 合并明显重复的同一物理灯；
4) 自动执行类别去重，可选择“语义去重 / 严格去重 / 不去重”；
5) 只生成 manual_indicator_schema.json；
   近似类别直接写入该 JSON 的 dedupe.ambiguous_candidates。

安全：
- 若设备已有 indicator_annotations.json，直接停止；
- 不修改人工标注文件。
"""

import argparse
import json
import re
import unicodedata
import subprocess
from pathlib import Path
from difflib import SequenceMatcher

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

PROJECT = Path.home() / "yolo_device_monitor"
SRC = Path("/home/crscs/桌面/数字孪生机房巡检识别模型测试视频")
PKG = PROJECT / "annotation_packages_best_frames_fast"
DEFAULT_LOCAL_MODEL = PROJECT / "models/Qwen2.5-VL-7B-Instruct"

KEYS = [
    "指示灯", "PWR", "POWER", "RUN", "ALM", "ALARM", "ACT", "LINK", "LNK",
    "LOS", "FAIL", "FAULT", "STATUS", "STAT", "COM", "SYS", "NET", "LAN",
    "WAN", "RX", "TX", "电源", "运行", "告警", "故障", "状态", "业务",
    "通信", "网口", "链路", "同步", "工作", "正常", "常亮", "闪烁", "熄灭"
]

STATE_ONLY_RE = re.compile(
    r"^(红色|绿色|黄色|橙色|蓝色|白色)?\s*"
    r"(常亮|闪烁|慢闪|快闪|熄灭|灭|亮|正常|异常|故障)$",
    re.I
)

def norm(s):
    s = unicodedata.normalize("NFKC", str(s)).lower()
    return "".join(ch for ch in s if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")

def norm_label(s):
    s = unicodedata.normalize("NFKC", str(s)).upper()
    s = re.sub(r"[\s_\-./\\()（）【】\[\]:：]+", "", s)
    return s

def find_src(root: Path, name: str) -> Path:
    hits = [p for p in root.iterdir() if p.is_dir() and norm(p.name) == norm(name)]
    if len(hits) != 1:
        raise SystemExit(f"无法唯一匹配原始设备目录: {name}")
    return hits[0]

def read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
        txt = "\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)
        if txt.strip():
            return txt
    except Exception:
        pass
    try:
        cp = subprocess.run(
            ["pdftotext", "-layout", str(path), "-"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120
        )
        if cp.returncode == 0 and cp.stdout.strip():
            return cp.stdout
    except Exception:
        pass
    return ""

def read_docx(path: Path) -> str:
    try:
        from docx import Document
        d = Document(str(path))
        parts = []
        for p in d.paragraphs:
            t = p.text.strip()
            if t:
                parts.append(t)
        for table in d.tables:
            for row in table.rows:
                vals = []
                for cell in row.cells:
                    t = re.sub(r"\\s+", " ", cell.text).strip()
                    if t:
                        vals.append(t)
                if vals:
                    parts.append(" | ".join(vals))
        return "\n".join(parts)
    except Exception:
        return ""

def read_manuals(root: Path):
    out = {}
    if not root.exists():
        return out
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        text = ""
        if ext == ".pdf":
            text = read_pdf(p)
        elif ext == ".docx":
            text = read_docx(p)
        elif ext in {".txt", ".md"}:
            text = p.read_text(encoding="utf-8", errors="ignore")
        if text.strip():
            out[str(p)] = text
    return out

def evidence_windows(text: str, radius=1000):
    low = text.lower()
    points = []
    for k in KEYS:
        start = 0
        needle = k.lower()
        while True:
            i = low.find(needle, start)
            if i < 0:
                break
            points.append(i)
            start = i + max(1, len(needle))
    chunks = []
    for i in sorted(set(points)):
        chunk = text[max(0, i-radius):min(len(text), i+radius)]
        chunk = re.sub(r"\n{3,}", "\n\n", chunk).strip()
        if chunk and chunk not in chunks:
            chunks.append(chunk)
    return chunks[:100]

def extract_json_array(text: str):
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S | re.I)
    if m:
        text = m.group(1).strip()
    a, b = text.find("["), text.rfind("]")
    if a >= 0 and b > a:
        return json.loads(text[a:b+1])
    raise ValueError("模型输出中没有 JSON 数组")

def canonical_short_label(item):
    short = str(item.get("short_label", "")).strip()
    if short:
        return norm_label(short)
    name = str(item.get("name", ""))
    english = re.findall(r"\b[A-Z][A-Z0-9/+-]{1,20}\b", name.upper())
    if english:
        return norm_label(english[0])
    return ""

def looks_like_state_only(item):
    name = str(item.get("name", "")).strip()
    if not name:
        return True
    if STATE_ONLY_RE.match(name):
        return True
    # 没有任何“灯/indicator/LED”等目标语义，且主要由状态词构成时排除
    state_words = ("常亮", "闪烁", "慢闪", "快闪", "熄灭", "黄色", "绿色", "红色", "橙色")
    target_words = ("灯", "指示", "LED", "PWR", "RUN", "ALM", "ACT", "LINK", "STAT",
                    "STATUS", "COM", "SYS", "NET", "LAN", "WAN", "RX", "TX")
    if any(w in name for w in state_words) and not any(w.upper() in name.upper() for w in target_words):
        return True
    return False

def clean_states(states):
    if not isinstance(states, list):
        return []
    out = []
    seen = set()
    for s in states:
        if isinstance(s, dict):
            status = str(s.get("state", "")).strip()
            meaning = str(s.get("meaning", "")).strip()
            color = str(s.get("color", "")).strip()
            obj = {"state": status, "meaning": meaning}
            if color:
                obj["color"] = color
            key = json.dumps(obj, ensure_ascii=False, sort_keys=True)
        else:
            obj = {"state": str(s).strip(), "meaning": ""}
            key = json.dumps(obj, ensure_ascii=False, sort_keys=True)
        if obj["state"] and key not in seen:
            seen.add(key)
            out.append(obj)
    return out

def merge_candidates(items):
    """
    合并规则尽量保守：
    - 状态项剔除；
    - 同 module 且 short_label 相同 -> 同一物理灯；
    - 若无 short_label，则 name 完全标准化相同 -> 合并；
    """
    merged = {}
    for raw in items:
        if not isinstance(raw, dict) or looks_like_state_only(raw):
            continue

        item = {
            "name": str(raw.get("name", "")).strip(),
            "short_label": str(raw.get("short_label", "")).strip(),
            "module": str(raw.get("module", "")).strip(),
            "location_hint": str(raw.get("location_hint", "")).strip(),
            "evidence": str(raw.get("evidence", "")).strip(),
            "states": clean_states(raw.get("states", [])),
        }
        if not item["name"]:
            continue

        sl = canonical_short_label(item)
        module_key = norm(item["module"])
        if sl:
            key = ("short_module", sl, module_key)
        else:
            key = ("name", norm(item["name"]))

        if key not in merged:
            merged[key] = item
        else:
            old = merged[key]
            # 选更完整的名称/位置/evidence
            if len(item["name"]) > len(old["name"]):
                old["name"] = item["name"]
            for f in ("short_label", "module", "location_hint", "evidence"):
                if not old.get(f) and item.get(f):
                    old[f] = item[f]
            old["states"] = clean_states(old.get("states", []) + item.get("states", []))

    # 二次合并：完全同 module，且一个名称明确包含另一个 short label
    vals = list(merged.values())
    result = []
    used = [False] * len(vals)
    for i, a in enumerate(vals):
        if used[i]:
            continue
        for j in range(i+1, len(vals)):
            if used[j]:
                continue
            b = vals[j]
            if norm(a.get("module", "")) != norm(b.get("module", "")):
                continue
            sa, sb = canonical_short_label(a), canonical_short_label(b)
            if sa and sb and sa == sb:
                if len(b["name"]) > len(a["name"]):
                    a["name"] = b["name"]
                a["states"] = clean_states(a.get("states", []) + b.get("states", []))
                if not a.get("location_hint") and b.get("location_hint"):
                    a["location_hint"] = b["location_hint"]
                if not a.get("evidence") and b.get("evidence"):
                    a["evidence"] = b["evidence"]
                used[j] = True
        result.append(a)
        used[i] = True

    # 稳定排序：先按 module，再按 name
    result.sort(key=lambda x: (norm(x.get("module", "")), norm(x["name"])))
    for i, x in enumerate(result):
        x["class_id"] = i
    return result



def list_device_folders(package_root: Path):
    """列出 annotation_packages_best_frames_fast 下所有设备文件夹。"""
    if not package_root.exists():
        return []

    return sorted(
        [p for p in package_root.iterdir() if p.is_dir()],
        key=lambda p: p.name
    )


def choose_device_interactively(package_root: Path) -> str:
    folders = list_device_folders(package_root)

    if not folders:
        raise SystemExit(f"没有找到设备文件夹: {package_root}")

    print()
    print("=" * 80)
    print("请选择要生成 manual_indicator_schema.json 的设备文件夹")
    print("=" * 80)

    for i, p in enumerate(folders, 1):
        ann = p / "indicator_annotations.json"
        state = "已有 indicator_annotations.json" if ann.exists() else "可处理"
        print(f"{i:2d}) {p.name} | {state}")

    while True:
        s = input("\n请选择设备编号: ").strip()

        if not s.isdigit():
            print("请输入有效编号。")
            continue

        idx = int(s)
        if not (1 <= idx <= len(folders)):
            print("请输入列表中的有效编号。")
            continue

        selected = folders[idx - 1]
        ann = selected / "indicator_annotations.json"

        if ann.exists():
            raise SystemExit(
                "\n安全停止：选中的设备文件夹已经存在 indicator_annotations.json：\n"
                f"{ann}\n"
                "本次不生成或修改任何文件。"
            )

        return selected.name

def choose_gpu_interactively(default_index=0) -> str:
    try:
        cp = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=15
        )
        lines = [x.strip() for x in cp.stdout.splitlines() if x.strip()] if cp.returncode == 0 else []
    except Exception:
        lines = []

    if not lines:
        print("未检测到 nvidia-smi，默认使用 cuda:0")
        return "cuda:0"

    valid = set()
    print("\\n检测到以下 GPU：")
    for line in lines:
        vals = [x.strip() for x in line.split(",")]
        if len(vals) < 4:
            continue
        try:
            idx = int(vals[0])
        except Exception:
            continue
        valid.add(idx)
        print(f"GPU {idx}: {vals[1]} | 总显存 {vals[2]} MiB | 空闲 {vals[3]} MiB")

    while True:
        s = input(f"请选择 Qwen-VL 使用的 GPU 编号 [{default_index}]: ").strip()
        idx = default_index if not s else (int(s) if s.isdigit() else -1)
        if idx in valid:
            return f"cuda:{idx}"
        print("请输入有效 GPU 编号。")


def backup_existing_schema(path: Path):
    if not path.exists():
        return None
    bak = path.with_suffix(path.suffix + ".bak")
    n = 1
    while bak.exists():
        bak = path.with_suffix(path.suffix + f".bak{n}")
        n += 1
    bak.write_bytes(path.read_bytes())
    return bak



def dedupe_norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s)).upper()
    s = s.replace("（", "(").replace("）", ")")
    s = re.sub(r"[\s_\-./\\:：,，;；()（）\[\]【】]+", "", s)
    return s


def dedupe_semantic_key(name: str, short_label: str = "") -> str:
    s = f"{short_label} {name}".upper()

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

    return "RAW:" + dedupe_norm(name)


def dedupe_canonical_name(key: str, items):
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

    names = [
        str(x.get("name", "")).strip()
        for x in items
        if str(x.get("name", "")).strip()
    ]
    return min(names, key=len) if names else key.replace("RAW:", "")


def dedupe_uniq_text(values):
    out, seen = [], set()
    for v in values:
        v = str(v).strip()
        if not v:
            continue
        k = dedupe_norm(v)
        if k not in seen:
            seen.add(k)
            out.append(v)
    return out


def dedupe_uniq_states(states):
    out, seen = [], set()
    for s in states:
        if not isinstance(s, dict):
            continue
        key = json.dumps(s, ensure_ascii=False, sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


def dedupe_indicators(indicators, semantic_merge=True):
    """
    将原 dedupe_manual_indicator_schema.py 的逻辑合并进来。

    semantic_merge=True:
        常见同义灯做语义合并；
    semantic_merge=False:
        只合并严格重复类别。
    """
    groups = {}

    for item in indicators:
        if not isinstance(item, dict):
            continue

        name = str(item.get("name", "")).strip()
        if not name:
            continue

        if semantic_merge:
            key = dedupe_semantic_key(
                item.get("name", ""),
                item.get("short_label", "")
            )
        else:
            key = "STRICT:" + dedupe_norm(item.get("name", ""))

        groups.setdefault(key, []).append(item)

    merged = []
    merge_report = []

    for key, items in groups.items():
        base = dict(items[0])

        names = dedupe_uniq_text(
            [x.get("name", "") for x in items]
        )
        aliases = dedupe_uniq_text(
            [a for x in items for a in x.get("aliases", [])]
            + names
        )
        modules = dedupe_uniq_text(
            [x.get("module", "") for x in items]
            + [m for x in items for m in x.get("modules", [])]
        )
        locs = dedupe_uniq_text(
            [x.get("location_hint", "") for x in items]
            + [m for x in items for m in x.get("location_hints", [])]
        )
        evidences = dedupe_uniq_text(
            [x.get("evidence", "") for x in items]
            + [e for x in items for e in x.get("evidence_list", [])]
        )
        states = dedupe_uniq_states(
            [s for x in items for s in x.get("states", [])]
        )

        base["name"] = dedupe_canonical_name(key, items)
        base["aliases"] = [
            x for x in aliases
            if dedupe_norm(x) != dedupe_norm(base["name"])
        ]
        base["modules"] = modules
        base["location_hints"] = locs
        base["evidence_list"] = evidences
        base["states"] = states

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

    merged.sort(
        key=lambda x: (
            dedupe_semantic_key(
                x.get("name", ""),
                x.get("short_label", "")
            ),
            dedupe_norm(x.get("name", "")),
        )
    )

    for i, item in enumerate(merged):
        item["class_id"] = i

    ambiguous = []

    for i in range(len(merged)):
        for j in range(i + 1, len(merged)):
            a, b = merged[i], merged[j]

            na = dedupe_norm(a.get("name", ""))
            nb = dedupe_norm(b.get("name", ""))

            if not na or not nb:
                continue

            ratio = SequenceMatcher(None, na, nb).ratio()

            ka = dedupe_semantic_key(
                a.get("name", ""),
                a.get("short_label", "")
            )
            kb = dedupe_semantic_key(
                b.get("name", ""),
                b.get("short_label", "")
            )

            if ratio >= 0.72 and ka != kb:
                ambiguous.append({
                    "class_a": a["class_id"],
                    "name_a": a["name"],
                    "class_b": b["class_id"],
                    "name_b": b["name"],
                    "similarity": round(ratio, 3),
                })

    return merged, merge_report, ambiguous


def choose_dedupe_mode_interactively():
    print()
    print("=" * 80)
    print("类别去重方式")
    print("=" * 80)
    print("1) 语义去重（推荐）：合并 RUN / ALM / GE / FE / LINK/ACT 等明显同义类别")
    print("2) 严格去重：只合并名称完全重复或仅标点空格不同的类别")
    print("3) 不去重：保留 build 阶段生成的类别")
    print()

    while True:
        s = input("请选择 [1/2/3，默认1]: ").strip()
        if not s:
            s = "1"

        if s == "1":
            return "semantic"
        if s == "2":
            return "strict"
        if s == "3":
            return "none"

        print("请输入 1、2 或 3。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None, help="不填写时交互选择设备")
    ap.add_argument("--source-root", type=Path, default=SRC)
    ap.add_argument("--package-root", type=Path, default=PKG)
    ap.add_argument("--model", default=str(DEFAULT_LOCAL_MODEL))
    ap.add_argument("--device-map", default=None, help="如 cuda:0；不填写时交互选择")
    ap.add_argument("--overwrite", action="store_true", help="允许覆盖已有 schema；不生成额外备份文件")
    ap.add_argument(
        "--dedupe-mode",
        choices=["semantic", "strict", "none"],
        default=None,
        help="semantic=语义去重；strict=严格去重；none=不去重；不填则交互选择"
    )
    args = ap.parse_args()

    args.source_root = args.source_root.expanduser()
    args.package_root = args.package_root.expanduser()

    if not args.device:
        args.device = choose_device_interactively(args.package_root)
    if not args.device_map:
        args.device_map = choose_gpu_interactively()

    if not args.dedupe_mode:
        args.dedupe_mode = choose_dedupe_mode_interactively()

    pkg = args.package_root / args.device
    if (pkg / "indicator_annotations.json").exists():
        raise SystemExit(f"安全停止：该设备文件夹已存在 indicator_annotations.json：{pkg / 'indicator_annotations.json'}")

    schema_out = pkg / "manual_indicator_schema.json"
    if schema_out.exists() and not args.overwrite:
        ans = input(f"已存在 {schema_out.name}，是否重新生成并备份旧文件？[y/N]: ").strip().lower()
        if ans not in {"y", "yes"}:
            raise SystemExit("已取消。")
        args.overwrite = True
    if schema_out.exists() and args.overwrite:
        print("将覆盖已有 manual_indicator_schema.json（不生成额外备份文件）")

    src = find_src(args.source_root, args.device)
    manuals = read_manuals(src / "说明书")
    if not manuals:
        raise SystemExit("没有读取到说明书文本")

    chunks = []
    for filename, text in manuals.items():
        for w in evidence_windows(text):
            chunks.append(f"[来源:{Path(filename).name}]\n{w}")
    if not chunks:
        raise SystemExit("说明书中没有找到明显指示灯相关段落")

    batches, cur = [], ""
    for c in chunks:
        if len(cur) + len(c) > 15000 and cur:
            batches.append(cur)
            cur = c
        else:
            cur += "\n" + c
    if cur:
        batches.append(cur)

    print("\n设备:", args.device)
    print("Qwen-VL GPU:", args.device_map)
    print("加载模型:", args.model)
    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype="auto", device_map=args.device_map,
        trust_remote_code=True, local_files_only=True
    )

    all_items, raw_outputs = [], []
    for i, batch in enumerate(batches, 1):
        print(f"[{i}/{len(batches)}] 解析说明书...")
        prompt = f"""
你正在从工业设备说明书中建立“物理指示灯目标检测类别表”。

必须严格区分：
A. 物理灯目标：可以在设备照片中用一个 bbox 框住的灯，例如 RUN运行指示灯、ALM告警指示灯、GE接口指示灯。
B. 灯的状态：黄色常亮、绿色闪烁、熄灭、红色等。状态绝对不能成为目标检测类别，只能写入该物理灯的 states。

严格规则：
1. 只提取说明书明确存在的物理指示灯。
2. 不得根据常识补充说明书没有的灯。
3. “黄色常亮 / 黄色闪烁 / 绿色 / 红色 / 熄灭”等只能放在 states，禁止作为 name。
4. 同一物理灯如果在说明书不同位置被重复描述，只输出一次。
5. 英文丝印与中文功能名合并为正式名称，如“RUN 运行指示灯”。
6. 不把端口本体、接口、按钮、屏幕、开关、模块、文字标签当成灯。
7. 对接口灯，要写清所属模块/接口类型，避免把不同模块的 LINK/ACT 混在一起。
8. evidence 必须来自说明书原意；证据不足就不要输出。
9. name 是以后 YOLO 的正式类别名称，所以必须描述“物理灯”，不能描述状态。

只输出 JSON 数组，不要解释：
[
  {{
    "name": "RUN 运行指示灯",
    "short_label": "RUN",
    "module": "主控板",
    "location_hint": "ALM旁边",
    "states": [
      {{"state":"绿色常亮","meaning":"正常"}},
      {{"state":"绿色闪烁","meaning":"运行中"}}
    ],
    "evidence": "说明书原意的简短证据"
  }}
]

说明书证据：
{batch}
""".strip()

        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], padding=True, return_tensors="pt").to(model.device)

        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=1800, do_sample=False)

        trimmed = [o[len(i0):] for i0, o in zip(inputs.input_ids, generated)]
        raw = proc.batch_decode(trimmed, skip_special_tokens=True)[0]
        raw_outputs.append(raw)
        try:
            all_items.extend(extract_json_array(raw))
        except Exception as e:
            print("  JSON解析失败:", e)

    raw_indicators = merge_candidates(all_items)

    # --------------------------------------------------------
    # 第二阶段：去重
    # --------------------------------------------------------
    if args.dedupe_mode == "none":
        indicators = raw_indicators
        merge_report = []
        ambiguous = []
        semantic_merge = False

    else:
        semantic_merge = args.dedupe_mode == "semantic"

        indicators, merge_report, ambiguous = dedupe_indicators(
            raw_indicators,
            semantic_merge=semantic_merge,
        )

    result = {
        "format": "manual_indicator_schema_v2",
        "device_name": args.device,
        "source_device_dir": str(src),
        "model": str(args.model),
        "indicators": indicators,
        "manual_sources": list(manuals.keys()),
        "needs_human_review": True,
        "notes": "类别只表示物理指示灯；颜色/常亮/闪烁等已放入 states，不作为类别。",
        "raw_model_outputs": raw_outputs,
        "dedupe": {
            "original_count": len(raw_indicators),
            "new_count": len(indicators),
            "mode": args.dedupe_mode,
            "semantic_merge": bool(
                args.dedupe_mode == "semantic"
            ),
            "merge_report": merge_report,
            "ambiguous_candidates": ambiguous,
        },
    }

    out = schema_out
    out.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print("\n" + "=" * 80)
    print("生成完成")
    print("=" * 80)
    print("schema:", out)
    print("去重模式:", args.dedupe_mode)
    print("build 初始类别数:", len(raw_indicators))
    print("最终类别数:", len(indicators))
    print("自动合并组数:", len(merge_report))
    print("待人工确认近似类别对:", len(ambiguous))

    print("\n最终类别：")
    for x in indicators:
        print(
            f'  {x["class_id"]}: '
            f'{x["name"]} | '
            f'module={x.get("module","")}'
        )
        if x.get("states"):
            print(
                "     states:",
                "；".join(
                    s.get("state", "")
                    for s in x["states"]
                    if isinstance(s, dict)
                )
            )

    if merge_report:
        print("\n自动合并：")
        for r in merge_report:
            print(
                f'- {r["old_class_ids"]} '
                f'{r["old_names"]} '
                f'-> {r["merged_to"]}'
            )

if __name__ == "__main__":
    main()


