#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
device_state_api.py

常驻状态引擎 API：
- POST /judge
- GET  /health
- GET  /devices
- GET  /rules/custom
- POST /rules/add

特点：
1. 启动时只加载一次 BGE embedding 模型和 ChromaDB
2. /judge 复用同一个 DeviceStateEngine 实例
3. /rules/add 可在线增加人工确认规则
4. 新增规则后仅热重载规则，不重新加载 embedding 模型
5. custom 规则优先级高于基础规则
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import List, Optional

import requests
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# 从同目录导入现有状态引擎
from device_state_engine import (
    DeviceStateEngine,
    merge_custom_rules,
    load_yaml,
    normalize_color,
    normalize_state,
)

PROJECT = Path.home() / "yolo_device_monitor"
BASE_RULES = PROJECT / "rules" / "device_indicator_rules.yaml"
CUSTOM_RULES = PROJECT / "rules" / "custom_device_rules.yaml"
ALIAS_MAP = PROJECT / "rules" / "device_alias_map.yaml"
EMBEDDING_MODEL = PROJECT / "models" / "bge-small-zh-v1.5"
CHROMA_DB = PROJECT / "rag" / "chroma_db"

LLM_BASE_URL = "http://127.0.0.1:8000/v1"
LLM_MODEL = "device-monitor-llm"
COLLECTION = "device_manuals"

app = FastAPI(
    title="机房设备状态判定服务",
    version="1.0.0",
    description="确定性规则优先 + RAG + 本地LLM + 自定义规则扩展",
)

engine: Optional[DeviceStateEngine] = None
rules_lock = Lock()


class Observation(BaseModel):
    indicator: str
    color: str = "unknown"
    state: str = "unknown"


class JudgeRequest(BaseModel):
    device: str
    observations: List[Observation]


class AddRuleRequest(BaseModel):
    device: str = Field(..., description="规则库中的标准设备名")
    indicator: str
    color: str
    state: str
    meaning: str
    severity: str
    alarm: Optional[bool] = None
    note: str = ""


VALID_SEVERITIES = {
    "normal",
    "info",
    "warning",
    "critical",
    "unknown",
}


def ensure_custom_rule_file():
    if CUSTOM_RULES.exists():
        return

    CUSTOM_RULES.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "schema_version": 1,
        "description": "人工确认后追加的确定性规则；优先级高于 device_indicator_rules.yaml。",
        "devices": {},
    }
    CUSTOM_RULES.write_text(
        yaml.safe_dump(
            obj,
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def reload_rules_only():
    """
    热重载基础规则 + custom规则。
    不重新加载 BGE，不重新打开 ChromaDB。
    """
    global engine

    if engine is None:
        raise RuntimeError("engine 尚未初始化")

    base = load_yaml(BASE_RULES)
    custom = load_yaml(CUSTOM_RULES)

    engine.rules = base
    engine.devices = merge_custom_rules(
        base.get("devices", {}),
        custom,
    )
    engine.device_names = list(engine.devices.keys())


def initialize_engine():
    global engine

    ensure_custom_rule_file()

    engine = DeviceStateEngine(
        project=PROJECT,
        embedding_model=EMBEDDING_MODEL,
        llm_base_url=LLM_BASE_URL,
        llm_model=LLM_MODEL,
        collection_name=COLLECTION,
    )


@app.on_event("startup")
def startup_event():
    initialize_engine()


@app.get("/")
def root():
    return {
        "service": "device-state-api",
        "status": "running",
        "endpoints": [
            "GET /health",
            "GET /devices",
            "POST /judge",
            "GET /rules/custom",
            "POST /rules/add",
        ],
    }


@app.get("/health")
def health():
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="状态引擎未初始化",
        )

    llm_ok = False
    llm_detail = ""

    try:
        r = requests.get(
            f"{LLM_BASE_URL}/models",
            timeout=3,
        )
        llm_ok = r.ok
        if not r.ok:
            llm_detail = f"HTTP {r.status_code}"
    except Exception as e:
        llm_detail = str(e)

    return {
        "status": "ok" if llm_ok else "degraded",
        "engine_loaded": True,
        "embedding_model_loaded": True,
        "chroma_loaded": True,
        "llm_available": llm_ok,
        "llm_detail": llm_detail,
        "base_rules": str(BASE_RULES),
        "custom_rules": str(CUSTOM_RULES),
        "alias_map": str(ALIAS_MAP),
    }


@app.get("/devices")
def devices():
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="状态引擎未初始化",
        )

    result = []

    for name, info in engine.devices.items():
        indicators = sorted(
            info.get("indicators", {}).keys()
        )

        result.append({
            "device": name,
            "indicator_count": len(indicators),
            "indicators": indicators,
        })

    return {
        "count": len(result),
        "devices": result,
    }


@app.post("/judge")
def judge(req: JudgeRequest):
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="状态引擎未初始化",
        )

    if not req.observations:
        raise HTTPException(
            status_code=400,
            detail="observations 不能为空",
        )

    observations = []

    for obs in req.observations:
        observations.append({
            "indicator": obs.indicator,
            "color": normalize_color(obs.color),
            "state": normalize_state(obs.state),
        })

    try:
        return engine.judge(
            req.device,
            observations,
        )
    except requests.RequestException as e:
        raise HTTPException(
            status_code=502,
            detail=f"调用本地LLM失败: {e}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"状态判定失败: {e}",
        )


@app.get("/rules/custom")
def get_custom_rules():
    ensure_custom_rule_file()

    obj = load_yaml(CUSTOM_RULES)

    return {
        "path": str(CUSTOM_RULES),
        "rules": obj,
    }


@app.post("/rules/add")
def add_rule(req: AddRuleRequest):
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="状态引擎未初始化",
        )

    severity = req.severity.strip().lower()

    if severity not in VALID_SEVERITIES:
        raise HTTPException(
            status_code=400,
            detail=f"非法 severity: {severity}",
        )

    # 必须是规则库已有设备，避免拼写错误创建幽灵设备
    if req.device not in engine.devices:
        raise HTTPException(
            status_code=400,
            detail=(
                f"设备不在当前规则库中: {req.device}。"
                "请先通过 GET /devices 查看标准设备名。"
            ),
        )

    color = normalize_color(req.color)
    state = normalize_state(req.state)

    if not color:
        color = "unknown"
    if not state:
        state = "unknown"

    alarm = (
        req.alarm
        if req.alarm is not None
        else severity in {"warning", "critical"}
    )

    new_rule = {
        "color": color,
        "color_raw": req.color,
        "state": state,
        "state_raw": req.state,
        "meaning": req.meaning,
        "severity": severity,
        "alarm": bool(alarm),
        "note": req.note,
        "verified_by_human": True,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }

    with rules_lock:
        ensure_custom_rule_file()

        custom = load_yaml(CUSTOM_RULES)

        if not custom:
            custom = {
                "schema_version": 1,
                "description": "人工确认后追加的确定性规则；优先级高于 device_indicator_rules.yaml。",
                "devices": {},
            }

        # 先备份
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = CUSTOM_RULES.with_name(
            f"custom_device_rules.{ts}.bak.yaml"
        )
        shutil.copy2(
            CUSTOM_RULES,
            backup,
        )

        dev = custom.setdefault(
            "devices",
            {}
        ).setdefault(
            req.device,
            {"indicators": {}},
        )

        ind = dev.setdefault(
            "indicators",
            {}
        ).setdefault(
            req.indicator,
            {"rules": []},
        )

        replaced = False

        for i, old in enumerate(
            ind["rules"]
        ):
            if (
                str(old.get("color", "")).strip() == color
                and str(old.get("state", "")).strip() == state
            ):
                ind["rules"][i] = new_rule
                replaced = True
                break

        if not replaced:
            ind["rules"].append(
                new_rule
            )

        CUSTOM_RULES.write_text(
            yaml.safe_dump(
                custom,
                allow_unicode=True,
                sort_keys=False,
                width=140,
            ),
            encoding="utf-8",
        )

        # 热重载，不重新加载BGE
        reload_rules_only()

    return {
        "ok": True,
        "action": (
            "updated"
            if replaced
            else "added"
        ),
        "backup": str(backup),
        "custom_rules": str(CUSTOM_RULES),
        "rule": {
            "device": req.device,
            "indicator": req.indicator,
            **new_rule,
        },
        "hot_reload": True,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "device_state_api:app",
        host="127.0.0.1",
        port=8100,
        reload=False,
    )

