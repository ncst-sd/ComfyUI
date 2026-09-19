#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
device_state_engine.py  (v3)

规则优先 + 自定义规则扩展 + 设备映射 + RAG + 本地LLM + LLM安全告警闸门

新增：
1. 自动加载 rules/custom_device_rules.yaml
2. custom_device_rules.yaml 中同设备/同指示灯/同颜色/同状态规则优先于基础规则
3. LLM 单独推断出的 warning/fault 不允许直接正式告警：
   - alarm 强制 false
   - needs_review 强制 true
   - status/severity 对外降为 uncertain/unknown
   - 原始 LLM 判断保存在 llm_candidate 中
4. 只有确定性规则命中的 warning/critical 才能正式 alarm=true
"""

import argparse
import copy
import json
import re
from pathlib import Path

import chromadb
import requests
import yaml
from sentence_transformers import SentenceTransformer

try:
    from rapidfuzz import fuzz, process
except Exception:
    fuzz = None
    process = None


COLOR_ALIASES = {
    "绿": "green", "绿色": "green", "green": "green",
    "红": "red", "红色": "red", "red": "red",
    "黄": "yellow", "黄色": "yellow", "yellow": "yellow",
    "蓝": "blue", "蓝色": "blue", "blue": "blue",
    "橙": "orange", "橙色": "orange", "orange": "orange",
    "橙红": "orange_red",
    "黄/绿": "yellow_green", "绿/黄": "yellow_green",
    "yellow_green": "yellow_green",
    "unknown": "unknown",
}

STATE_ALIASES = {
    "常亮": "steady", "长亮": "steady", "steady": "steady",
    "闪烁": "blinking", "闪": "blinking", "blinking": "blinking",
    "熄灭": "off", "灭": "off", "不亮": "off", "off": "off",
    "亮": "on", "点亮": "on", "on": "on",
    "常亮/闪烁": "steady_or_blinking",
    "常亮 / 闪烁": "steady_or_blinking",
    "steady_or_blinking": "steady_or_blinking",
    "unknown": "unknown",
}

SEVERITY_RANK = {
    "normal": 0,
    "info": 1,
    "warning": 2,
    "critical": 3,
    "unknown": -1,
}


def clean_text(s):
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s).replace("\xa0", " ").replace("\u3000", " ")).strip()


def norm_compact(s):
    return re.sub(r"[\s_\-—–/（）()\[\]【】]+", "", clean_text(s)).lower()


def normalize_color(s):
    raw = clean_text(s)
    if raw in COLOR_ALIASES:
        return COLOR_ALIASES[raw]
    low = raw.lower()
    if low in COLOR_ALIASES:
        return COLOR_ALIASES[low]
    for k, v in COLOR_ALIASES.items():
        if k and k in raw:
            return v
    return low or "unknown"


def normalize_state(s):
    raw = clean_text(s)
    if raw in STATE_ALIASES:
        return STATE_ALIASES[raw]
    low = raw.lower()
    if low in STATE_ALIASES:
        return STATE_ALIASES[low]
    if "常亮" in raw and "闪烁" in raw:
        return "steady_or_blinking"
    for k, v in STATE_ALIASES.items():
        if k and k in raw:
            return v
    return low or "unknown"


def fuzzy_best(query, choices, threshold=72):
    if not query or not choices:
        return None, 0

    nq = norm_compact(query)

    for c in choices:
        if norm_compact(c) == nq:
            return c, 100

    for c in choices:
        nc = norm_compact(c)
        if nq and (nq in nc or nc in nq):
            return c, 95

    if process is None:
        return None, 0

    result = process.extractOne(query, choices, scorer=fuzz.WRatio)
    if result and result[1] >= threshold:
        return result[0], int(result[1])

    return None, int(result[1]) if result else 0


def parse_observe(text):
    if "=" not in text:
        raise ValueError(f"--observe 格式错误: {text}")

    indicator, rhs = text.split("=", 1)
    parts = [clean_text(x) for x in re.split(r"[,，]", rhs) if clean_text(x)]

    if len(parts) >= 2:
        color, state = parts[0], parts[1]
    elif len(parts) == 1:
        color, state = "unknown", parts[0]
    else:
        color, state = "unknown", "unknown"

    return {
        "indicator": clean_text(indicator),
        "color": normalize_color(color),
        "state": normalize_state(state),
    }


def extract_json_from_text(content):
    if content is None:
        return None

    content = str(content).strip()
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.S | re.I).strip()
    content = re.sub(r"^```json\s*", "", content, flags=re.I)
    content = re.sub(r"^```\s*", "", content)
    content = re.sub(r"\s*```$", "", content).strip()

    try:
        return json.loads(content)
    except Exception:
        pass

    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(content[start:end + 1])
        except Exception:
            pass

    return None


def load_yaml(path):
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def merge_custom_rules(base_devices, custom_obj):
    """
    自定义规则覆盖策略：
    - device/indicator 不存在时允许新增
    - 同一 indicator 下，若 custom 的 color+state 与基础规则一致，则替换基础规则
    - 否则追加
    - 每条规则附加 rule_source=base/custom
    """
    merged = copy.deepcopy(base_devices)

    for dev_name, dev_info in merged.items():
        for ind_name, ind_info in dev_info.get("indicators", {}).items():
            for r in ind_info.get("rules", []):
                r.setdefault("rule_source", "base")

    custom_devices = custom_obj.get("devices", {}) if isinstance(custom_obj, dict) else {}

    for dev_name, dev_info in custom_devices.items():
        merged.setdefault(dev_name, {"normal_state_description": "", "indicators": {}})
        if dev_info.get("normal_state_description"):
            merged[dev_name]["normal_state_description"] = dev_info["normal_state_description"]

        for ind_name, ind_info in dev_info.get("indicators", {}).items():
            merged[dev_name].setdefault("indicators", {})
            merged[dev_name]["indicators"].setdefault(ind_name, {"rules": []})

            base_rules = merged[dev_name]["indicators"][ind_name].setdefault("rules", [])

            for crule in ind_info.get("rules", []):
                crule = copy.deepcopy(crule)
                crule["rule_source"] = "custom"
                ckey = (
                    clean_text(crule.get("color", "unknown")),
                    clean_text(crule.get("state", "unknown")),
                )

                replaced = False
                for i, brule in enumerate(base_rules):
                    bkey = (
                        clean_text(brule.get("color", "unknown")),
                        clean_text(brule.get("state", "unknown")),
                    )
                    if bkey == ckey:
                        base_rules[i] = crule
                        replaced = True
                        break

                if not replaced:
                    base_rules.append(crule)

    return merged


class DeviceStateEngine:
    def __init__(
        self,
        project,
        embedding_model,
        llm_base_url,
        llm_model,
        collection_name,
    ):
        self.project = project

        self.rules_path = project / "rules" / "device_indicator_rules.yaml"
        self.custom_rules_path = project / "rules" / "custom_device_rules.yaml"
        self.alias_path = project / "rules" / "device_alias_map.yaml"
        self.chroma_dir = project / "rag" / "chroma_db"

        for p, label in [
            (self.rules_path, "基础规则库"),
            (self.alias_path, "设备映射文件"),
            (self.chroma_dir, "RAG数据库"),
            (embedding_model, "Embedding模型"),
        ]:
            if not p.exists():
                raise FileNotFoundError(f"{label}不存在: {p}")

        base_rules = load_yaml(self.rules_path)
        custom_rules = load_yaml(self.custom_rules_path)

        self.rules = base_rules
        self.devices = merge_custom_rules(
            base_rules.get("devices", {}),
            custom_rules
        )
        self.device_names = list(self.devices.keys())

        self.alias_map = load_yaml(self.alias_path)
        self.alias_devices = self.alias_map.get("devices", {})

        self.alias_search_index = {}
        for device_id, info in self.alias_devices.items():
            names = set()

            for field in ("canonical_name", "rule_name"):
                name = clean_text(info.get(field, ""))
                if name:
                    names.add(name)

            for a in info.get("aliases", []) or []:
                a = clean_text(a)
                if a:
                    names.add(a)

            for ws in info.get("workspace_names", []) or []:
                ws = clean_text(ws)
                if ws:
                    names.add(ws)

            for name in names:
                self.alias_search_index[name] = device_id

        print("Loading embedding model...")
        self.embedding = SentenceTransformer(str(embedding_model), device="cuda")

        self.chroma = chromadb.PersistentClient(path=str(self.chroma_dir))
        self.collection = self.chroma.get_collection(collection_name)

        self.llm_base_url = llm_base_url.rstrip("/")
        self.llm_model = llm_model

    def resolve_device(self, requested):
        requested = clean_text(requested)

        alias_names = list(self.alias_search_index.keys())
        matched_alias, score = fuzzy_best(requested, alias_names, threshold=65)

        if matched_alias:
            device_id = self.alias_search_index[matched_alias]
            info = self.alias_devices[device_id]
            return {
                "device_id": device_id,
                "canonical_name": info.get("canonical_name"),
                "rule_name": info.get("rule_name"),
                "workspace_names": info.get("workspace_names", []) or [],
                "verified": bool(info.get("verified", False)),
                "matched_alias": matched_alias,
                "match_score": score,
            }

        rule_name, score = fuzzy_best(requested, self.device_names, threshold=65)
        if rule_name:
            return {
                "device_id": None,
                "canonical_name": rule_name,
                "rule_name": rule_name,
                "workspace_names": [],
                "verified": False,
                "matched_alias": rule_name,
                "match_score": score,
            }

        return {
            "device_id": None,
            "canonical_name": requested,
            "rule_name": None,
            "workspace_names": [],
            "verified": False,
            "matched_alias": None,
            "match_score": 0,
        }

    def match_indicator(self, rule_name, requested):
        if not rule_name:
            return None, 0

        indicators = list(
            self.devices
            .get(rule_name, {})
            .get("indicators", {})
            .keys()
        )
        return fuzzy_best(requested, indicators, threshold=62)

    @staticmethod
    def rule_matches(rule, color, state):
        rc = clean_text(rule.get("color", "unknown"))
        rs = clean_text(rule.get("state", "unknown"))

        color_ok = (
            color == "unknown"
            or rc == "unknown"
            or color == rc
        )

        state_ok = (
            state == "unknown"
            or rs == "unknown"
            or state == rs
            or (rs == "steady_or_blinking" and state in {"steady", "blinking"})
        )

        return color_ok and state_ok

    def deterministic(self, requested_device, observations):
        resolved = self.resolve_device(requested_device)
        rule_name = resolved.get("rule_name")

        out = {
            "requested_device": requested_device,
            "device_resolution": resolved,
            "matched_device": rule_name,
            "matched_rules": [],
            "unresolved": [],
        }

        if not rule_name:
            out["unresolved"].append({"reason": "device_not_found_in_rules"})
            return out

        if rule_name not in self.devices:
            out["unresolved"].append({"reason": "resolved_rule_name_not_found"})
            return out

        for obs in observations:
            indicator_req = clean_text(obs.get("indicator"))
            color = normalize_color(obs.get("color", "unknown"))
            state = normalize_state(obs.get("state", "unknown"))

            indicator, ind_score = self.match_indicator(rule_name, indicator_req)

            if not indicator:
                out["unresolved"].append({
                    "reason": "indicator_not_found_in_rules",
                    "requested_indicator": indicator_req,
                    "color": color,
                    "state": state,
                })
                continue

            candidates = (
                self.devices[rule_name]["indicators"][indicator]
                .get("rules", [])
            )

            matched = [
                r for r in candidates
                if self.rule_matches(r, color, state)
            ]

            common = {
                "requested_indicator": indicator_req,
                "matched_indicator": indicator,
                "indicator_match_score": ind_score,
                "color": color,
                "state": state,
            }

            if len(matched) == 1:
                out["matched_rules"].append({
                    **common,
                    "rule": matched[0]
                })

            elif len(matched) > 1:
                outcomes = {
                    (
                        r.get("severity", "unknown"),
                        r.get("meaning", "")
                    )
                    for r in matched
                }

                if len(outcomes) == 1:
                    out["matched_rules"].append({
                        **common,
                        "rule": matched[0]
                    })
                else:
                    out["unresolved"].append({
                        **common,
                        "reason": "multiple_conflicting_rules",
                        "candidate_rules": matched,
                    })
            else:
                out["unresolved"].append({
                    **common,
                    "reason": "no_exact_rule",
                    "available_rules": candidates,
                })

        return out

    @staticmethod
    def aggregate(det):
        matched = det.get("matched_rules", [])
        unresolved = det.get("unresolved", [])

        if not matched:
            return {
                "complete": False,
                "status": "unknown",
                "severity": "unknown",
                "alarm": False,
            }

        severities = [
            x["rule"].get("severity", "unknown")
            for x in matched
        ]

        known = [
            s for s in severities
            if s != "unknown" and s in SEVERITY_RANK
        ]

        max_sev = (
            max(known, key=lambda s: SEVERITY_RANK[s])
            if known
            else "unknown"
        )

        complete = (
            not unresolved
            and all(s != "unknown" for s in severities)
        )

        status = {
            "critical": "fault",
            "warning": "warning",
            "info": "info",
            "normal": "normal",
        }.get(max_sev, "unknown")

        return {
            "complete": complete,
            "status": status,
            "severity": max_sev,
            "alarm": max_sev in {"warning", "critical"},
        }

    def _query_chroma(self, qemb, where, n_results):
        if n_results <= 0:
            return []

        try:
            res = self.collection.query(
                query_embeddings=[qemb],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            return []

        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]

        return [
            {
                "document": doc,
                "metadata": meta or {},
                "distance": dist,
            }
            for doc, meta, dist in zip(docs, metas, dists)
        ]

    def retrieve(
        self,
        resolved_device,
        query,
        n_per_workspace=4,
        n_global=4
    ):
        qemb = self.embedding.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0].tolist()

        items = []

        if bool(resolved_device.get("verified", False)):
            for ws_name in resolved_device.get("workspace_names", []) or []:
                items.extend(
                    self._query_chroma(
                        qemb,
                        {"device_name": ws_name},
                        n_per_workspace
                    )
                )

        items.extend(
            self._query_chroma(
                qemb,
                {"device_name": "_GLOBAL_"},
                n_global
            )
        )

        seen = set()
        dedup = []

        for x in items:
            key = (
                x["metadata"].get("source_file"),
                x["metadata"].get("page", 0),
                x["document"],
            )
            if key not in seen:
                seen.add(key)
                dedup.append(x)

        dedup.sort(key=lambda x: x.get("distance", 999))
        return dedup

    def call_llm(self, payload, contexts):
        blocks = []
        sources = []

        for i, item in enumerate(contexts, 1):
            m = item["metadata"]

            blocks.append(
                f"[资料{i}] "
                f"设备={m.get('device_name','')} | "
                f"文件={m.get('source_file','')} | "
                f"页={m.get('page',0)} | "
                f"章节={m.get('section','')}\n"
                f"{item['document']}"
            )

            sources.append({
                "index": i,
                "device_name": m.get("device_name", ""),
                "source_file": m.get("source_file", ""),
                "page": m.get("page", 0),
                "section": m.get("section", ""),
                "distance": item.get("distance"),
            })

        system_prompt = """
你是机房设备巡检状态分析器。

必须遵守：
1. 只能依据输入的确定性规则结果和检索到的资料判断。
2. 不得凭常识补充设备规则，不得编造。
3. 已命中的确定性规则优先级最高，不允许推翻。
4. 如果资料没有明确说明当前观测状态的含义，必须输出 uncertain。
5. “没有出现在正常状态定义中”不能自动推导为异常。
6. 不允许因为颜色本身直接推断故障。
7. 业务上下文相关状态不能擅自判为故障。
8. 输出必须是一个JSON对象，不要Markdown，不要额外文字。

JSON结构：
{
  "status": "normal|info|warning|fault|uncertain",
  "severity": "normal|info|warning|critical|unknown",
  "alarm": true,
  "needs_review": false,
  "reason": ["原因1", "原因2"],
  "possible_fault": null,
  "evidence": [
    {"source_index": 1, "statement": "依据"}
  ]
}
""".strip()

        user_prompt = (
            "【设备与视觉观测】\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
            + "\n\n【检索资料】\n"
            + "\n\n".join(blocks)
        )

        request_body = {
            "model": self.llm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 1000,
            "chat_template_kwargs": {
                "enable_thinking": False
            },
            "response_format": {
                "type": "json_object"
            },
        }

        resp = requests.post(
            f"{self.llm_base_url}/chat/completions",
            headers={"Content-Type": "application/json"},
            json=request_body,
            timeout=180,
        )
        resp.raise_for_status()

        message = resp.json()["choices"][0]["message"]
        content = (message.get("content") or "").strip()
        parsed = extract_json_from_text(content)

        if parsed is None:
            parsed = {
                "status": "uncertain",
                "severity": "unknown",
                "alarm": False,
                "needs_review": True,
                "reason": ["LLM未返回可解析JSON"],
                "possible_fault": None,
                "evidence": [],
                "raw_llm_output": content,
            }

        return parsed, sources

    def judge(self, device, observations):
        det = self.deterministic(device, observations)
        agg = self.aggregate(det)
        resolved = det.get("device_resolution", {})

        # 完整确定性规则命中
        if agg["complete"]:
            reasons = []

            for x in det["matched_rules"]:
                r = x["rule"]
                src = r.get("rule_source", "base")
                reasons.append(
                    f"{x['matched_indicator']}："
                    f"{r.get('color_raw','')} "
                    f"{r.get('state_raw','')} -> "
                    f"{r.get('meaning','')} "
                    f"({r.get('severity','unknown')}, source={src})"
                )

            return {
                "device": resolved.get("canonical_name"),
                "requested_device": device,
                "device_resolution": resolved,
                "status": agg["status"],
                "severity": agg["severity"],
                "alarm": agg["alarm"],
                "decision_source": "deterministic_rule",
                "llm_used": False,
                "needs_review": False,
                "reason": reasons,
                "deterministic": det,
            }

        # RAG + LLM
        canonical_name = resolved.get("canonical_name") or device

        query_lines = [
            f"设备：{canonical_name}",
            "请根据设备说明书和巡检规则判断以下观测状态：",
        ]

        for obs in observations:
            query_lines.append(
                f"{obs.get('indicator','')} "
                f"{obs.get('color','unknown')} "
                f"{obs.get('state','unknown')}"
            )

        for u in det.get("unresolved", []):
            query_lines.append(
                "规则未确定原因：" + u.get("reason", "")
            )

        query = "\n".join(query_lines)
        contexts = self.retrieve(resolved, query)

        payload = {
            "requested_device": device,
            "resolved_device": resolved,
            "observations": observations,
            "deterministic_rule_result": det,
            "rule_aggregate": agg,
        }

        llm_result, sources = self.call_llm(payload, contexts)

        # 已命中的确定性 warning/critical 绝不能被 LLM 降级
        matched_sevs = [
            x["rule"].get("severity", "unknown")
            for x in det.get("matched_rules", [])
        ]
        known = [
            s for s in matched_sevs
            if s != "unknown" and s in SEVERITY_RANK
        ]
        strongest = (
            max(known, key=lambda s: SEVERITY_RANK[s])
            if known
            else "unknown"
        )

        deterministic_alarm_support = strongest in {"warning", "critical"}

        # --------------------------------------------------
        # LLM安全告警闸门
        # --------------------------------------------------
        llm_candidate = {
            "status": llm_result.get("status", "uncertain"),
            "severity": llm_result.get("severity", "unknown"),
            "alarm": bool(llm_result.get("alarm", False)),
            "needs_review": bool(llm_result.get("needs_review", True)),
            "reason": llm_result.get("reason", []),
            "possible_fault": llm_result.get("possible_fault"),
            "evidence": llm_result.get("evidence", []),
        }

        llm_abnormal = (
            llm_candidate["status"] in {"warning", "fault"}
            or llm_candidate["severity"] in {"warning", "critical"}
            or llm_candidate["alarm"] is True
        )

        if deterministic_alarm_support:
            # 有确定性告警支撑：最终等级至少保持确定性等级
            final_severity = strongest
            final_status = "fault" if strongest == "critical" else "warning"

            return {
                "device": canonical_name,
                "requested_device": device,
                "device_resolution": resolved,
                "status": final_status,
                "severity": final_severity,
                "alarm": True,
                "decision_source": "deterministic_rule+llm_rag",
                "llm_used": True,
                "needs_review": bool(llm_candidate["needs_review"]),
                "reason": (
                    llm_candidate["reason"]
                    + ["正式告警由已命中的确定性规则支撑，LLM仅用于补充解释。"]
                ),
                "possible_fault": llm_candidate["possible_fault"],
                "evidence": llm_candidate["evidence"],
                "llm_candidate": llm_candidate,
                "rag_sources": sources,
                "deterministic": det,
            }

        if llm_abnormal:
            # 只有 LLM 认为异常：禁止直接告警
            return {
                "device": canonical_name,
                "requested_device": device,
                "device_resolution": resolved,
                "status": "uncertain",
                "severity": "unknown",
                "alarm": False,
                "decision_source": "llm_rag_candidate",
                "llm_used": True,
                "needs_review": True,
                "reason": (
                    llm_candidate["reason"]
                    + ["LLM推断的异常没有确定性规则支撑，已被安全闸门降为待复核候选，不触发正式告警。"]
                ),
                "possible_fault": llm_candidate["possible_fault"],
                "evidence": llm_candidate["evidence"],
                "llm_candidate": llm_candidate,
                "rag_sources": sources,
                "deterministic": det,
            }

        # LLM normal/info/uncertain：允许作为解释结果，但仍保留其 review 标志
        result = {
            "device": canonical_name,
            "requested_device": device,
            "device_resolution": resolved,
            "status": llm_candidate["status"],
            "severity": llm_candidate["severity"],
            "alarm": False,
            "decision_source": "llm_rag",
            "llm_used": True,
            "needs_review": llm_candidate["needs_review"],
            "reason": llm_candidate["reason"],
            "possible_fault": llm_candidate["possible_fault"],
            "evidence": llm_candidate["evidence"],
            "llm_candidate": llm_candidate,
            "rag_sources": sources,
            "deterministic": det,
        }

        if "raw_llm_output" in llm_result:
            result["raw_llm_output"] = llm_result["raw_llm_output"]

        return result


def main():
    p = argparse.ArgumentParser(
        description="设备状态：规则优先 + 自定义规则 + 设备映射 + RAG + LLM安全闸门"
    )

    p.add_argument(
        "--project",
        type=Path,
        default=Path.home() / "yolo_device_monitor"
    )
    p.add_argument(
        "--embedding-model",
        type=Path,
        default=Path.home() / "yolo_device_monitor" / "models" / "bge-small-zh-v1.5"
    )
    p.add_argument(
        "--llm-base-url",
        default="http://127.0.0.1:8000/v1"
    )
    p.add_argument(
        "--llm-model",
        default="device-monitor-llm"
    )
    p.add_argument(
        "--collection",
        default="device_manuals"
    )
    p.add_argument("--device")
    p.add_argument("--observe", action="append", default=[])
    p.add_argument("--json", dest="json_input")

    args = p.parse_args()

    if args.json_input:
        payload = json.loads(args.json_input)
        device = payload["device"]
        observations = payload.get("observations", [])

        for obs in observations:
            obs["color"] = normalize_color(obs.get("color", "unknown"))
            obs["state"] = normalize_state(obs.get("state", "unknown"))
    else:
        if not args.device:
            raise SystemExit("必须提供 --device 或 --json")

        device = args.device
        observations = [parse_observe(x) for x in args.observe]

    if not observations:
        raise SystemExit("至少提供一个状态观测")

    engine = DeviceStateEngine(
        project=args.project.expanduser().resolve(),
        embedding_model=args.embedding_model.expanduser().resolve(),
        llm_base_url=args.llm_base_url,
        llm_model=args.llm_model,
        collection_name=args.collection,
    )

    result = engine.judge(device, observations)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

