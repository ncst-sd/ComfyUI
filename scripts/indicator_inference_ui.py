#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import cv2
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from ultralytics import YOLO
import uvicorn

PROJECT = Path.home() / "yolo_device_monitor"

DEFAULT_DEVICE_MODEL = (
    PROJECT / "runs" / "classify" / "device_cls_yolo11s" / "weights" / "best.pt"
)

DEFAULT_INDICATOR_MODELS = {
    "地震预警_CRES-LMA_101_交大铁发": str(
        PROJECT / "runs" / "detect" / "indicator_cres_lma101_v2_yolo11s" / "weights" / "best.pt"
    ),
    "FAS_MDS3400DLX_V3.1_佳讯飞鸿": str(
        PROJECT / "runs" / "detect" / "indicator_fas_mds3400dlx_yolo11s" / "weights" / "best.pt"
    ),
}

app = FastAPI(title="Device -> Indicator Multi-model UI")
_model_cache = {}
_model_lock = threading.Lock()


def get_model(path: str):
    p = str(Path(path).expanduser().resolve())
    with _model_lock:
        if p not in _model_cache:
            _model_cache[p] = YOLO(p)
        return _model_cache[p]


def parse_source(source: str):
    s = source.strip()
    if s.isdigit():
        return int(s)
    if s.startswith(("rtsp://", "http://", "https://")):
        return s
    return str(Path(s).expanduser())


def seconds_text(sec):
    sec = max(0.0, float(sec or 0.0))
    return f"{int(sec // 60):02d}:{int(sec % 60):02d}"


def classify_device(model, frame, imgsz, device, topk=5):
    result = model.predict(frame, imgsz=imgsz, device=device, verbose=False)[0]
    if result.probs is None:
        return {"name": "", "class_id": -1, "confidence": 0.0, "topk": []}

    ids = result.probs.top5[:topk]
    confs = result.probs.top5conf[:topk].detach().cpu().tolist()
    items = []
    for cid, conf in zip(ids, confs):
        cid = int(cid)
        items.append({
            "class_id": cid,
            "class_name": result.names.get(cid, str(cid)),
            "confidence": float(conf),
        })

    best = items[0] if items else {"class_id": -1, "class_name": "", "confidence": 0.0}
    return {
        "name": best["class_name"],
        "class_id": best["class_id"],
        "confidence": best["confidence"],
        "topk": items,
    }


def draw_boxes(frame, result):
    details = []
    if result.boxes is None:
        return frame, details

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy().astype(int).tolist()
        cid = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        name = result.names.get(cid, str(cid))

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame, f"{cid} {conf:.2f}", (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 2, cv2.LINE_AA
        )

        details.append({
            "class_id": cid,
            "class_name": name,
            "confidence": conf,
            "bbox": [x1, y1, x2, y2],
        })
    return frame, details


def draw_runtime(frame, device_conf, routed, det_count, fps):
    cv2.putText(
        frame, f"Device conf: {device_conf:.2f}", (20, 34),
        cv2.FONT_HERSHEY_SIMPLEX, 0.72,
        (0, 255, 0) if routed else (0, 165, 255), 2, cv2.LINE_AA
    )
    cv2.putText(
        frame, f"Indicators: {det_count}  FPS: {fps:.1f}", (20, 68),
        cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 255), 2, cv2.LINE_AA
    )
    if routed:
        cv2.putText(
            frame, "Indicator model routed", (20, 102),
            cv2.FONT_HERSHEY_SIMPLEX, 0.66, (0, 255, 0), 2, cv2.LINE_AA
        )


class PipelineWorker:
    def __init__(self):
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.running = False
        self.error = ""
        self.latest_jpeg = None

        self.source = ""
        self.device = "0"
        self.device_model_path = str(DEFAULT_DEVICE_MODEL)
        self.indicator_model_map = dict(DEFAULT_INDICATOR_MODELS)

        self.device_conf_threshold = 0.70
        self.indicator_conf = 0.25
        self.indicator_iou = 0.50
        self.device_imgsz = 320
        self.indicator_imgsz = 1280
        self.device_interval = 15
        self.frame_skip = 0

        self.device_result = {"name": "", "class_id": -1, "confidence": 0.0, "topk": []}
        self.device_votes = []
        self.device_vote_name = ""
        self.routed_device = ""
        self.routed_indicator_model = ""
        self.route_status = "未路由"

        self.indicator_details = []
        self.indicator_counts = {}
        self.indicator_total_current = 0

        self.frames_processed = 0
        self.source_frame_idx = 0
        self.total_frames = 0
        self.source_fps = 0.0
        self.width = 0
        self.height = 0
        self.duration_sec = 0.0
        self.current_time_sec = 0.0
        self.progress_pct = None
        self.pipeline_stage = "idle"

        self.infer_fps = 0.0
        self.device_infer_ms = 0.0
        self.indicator_infer_ms = 0.0

    def stop(self):
        self.stop_event.set()
        t = self.thread
        if t and t.is_alive():
            t.join(timeout=3.0)
        with self.lock:
            self.running = False
        self.thread = None

    def start(self, cfg):
        self.stop()
        self.stop_event.clear()

        with self.lock:
            self.running = True
            self.error = ""
            self.latest_jpeg = None
            self.source = str(cfg.get("source", "0"))
            self.device = str(cfg.get("device", "0"))
            self.device_model_path = str(cfg.get("device_model_path", DEFAULT_DEVICE_MODEL))

            supplied = cfg.get("indicator_model_map", {})
            self.indicator_model_map = {
                str(k): str(v) for k, v in supplied.items()
                if str(k).strip() and str(v).strip()
            }

            self.device_conf_threshold = float(cfg.get("device_conf_threshold", 0.70))
            self.indicator_conf = float(cfg.get("indicator_conf", 0.25))
            self.indicator_iou = float(cfg.get("indicator_iou", 0.50))
            self.device_imgsz = int(cfg.get("device_imgsz", 320))
            self.indicator_imgsz = int(cfg.get("indicator_imgsz", 1280))
            self.device_interval = max(1, int(cfg.get("device_interval", 15)))
            self.frame_skip = max(0, int(cfg.get("frame_skip", 0)))

            self.device_result = {"name": "", "class_id": -1, "confidence": 0.0, "topk": []}
            self.device_votes = []
            self.device_vote_name = ""
            self.routed_device = ""
            self.routed_indicator_model = ""
            self.route_status = "未路由"
            self.indicator_details = []
            self.indicator_counts = {}
            self.indicator_total_current = 0
            self.frames_processed = 0
            self.source_frame_idx = 0
            self.total_frames = 0
            self.progress_pct = None
            self.pipeline_stage = "starting"

        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _vote(self, name):
        self.device_votes.append(name)
        if len(self.device_votes) > 5:
            self.device_votes.pop(0)
        self.device_vote_name = Counter(self.device_votes).most_common(1)[0][0] if self.device_votes else ""

    def _run(self):
        cap = None
        try:
            device_model_file = Path(self.device_model_path).expanduser()
            if not device_model_file.exists():
                raise RuntimeError(f"设备识别模型不存在: {device_model_file}")

            device_model = get_model(str(device_model_file))
            source = parse_source(self.source)
            cap = cv2.VideoCapture(source)

            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            if not cap.isOpened():
                raise RuntimeError(f"无法打开视频源: {self.source}")

            self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            self.source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

            is_local_file = isinstance(source, str) and Path(source).exists() and Path(source).is_file()
            self.total_frames = total_frames if is_local_file else 0
            self.duration_sec = (
                total_frames / self.source_fps
                if is_local_file and total_frames > 0 and self.source_fps > 0
                else 0.0
            )

            recent_times = []

            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    if is_local_file:
                        break
                    time.sleep(0.05)
                    continue

                source_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)

                if self.frame_skip > 0:
                    for _ in range(self.frame_skip):
                        if not cap.grab():
                            break

                t_pipeline = time.perf_counter()

                if self.frames_processed == 0 or self.frames_processed % self.device_interval == 0:
                    self.pipeline_stage = "device_recognition"
                    t0 = time.perf_counter()
                    self.device_result = classify_device(
                        device_model, frame, self.device_imgsz, self.device, topk=5
                    )
                    self.device_infer_ms = (time.perf_counter() - t0) * 1000.0
                    self._vote(self.device_result["name"])

                dres = dict(self.device_result)
                vote_name = self.device_vote_name

                route_name = ""
                route_model_path = ""
                route_status = "设备尚未稳定"

                if dres["confidence"] >= self.device_conf_threshold and dres["name"] == vote_name:
                    if vote_name in self.indicator_model_map:
                        candidate = Path(self.indicator_model_map[vote_name]).expanduser()
                        if candidate.exists():
                            route_name = vote_name
                            route_model_path = str(candidate)
                            route_status = "已路由"
                        else:
                            route_status = f"对应指示灯模型不存在: {candidate}"
                    else:
                        route_status = "该设备尚未配置指示灯模型"

                details = []
                indicator_ms = 0.0

                if route_name:
                    self.pipeline_stage = "indicator_recognition"
                    indicator_model = get_model(route_model_path)
                    t0 = time.perf_counter()
                    ires = indicator_model.predict(
                        frame,
                        imgsz=self.indicator_imgsz,
                        conf=self.indicator_conf,
                        iou=self.indicator_iou,
                        device=self.device,
                        verbose=False,
                    )[0]
                    indicator_ms = (time.perf_counter() - t0) * 1000.0
                    drawn, details = draw_boxes(frame.copy(), ires)
                else:
                    self.pipeline_stage = "indicator_skipped"
                    drawn = frame.copy()

                counts = Counter(x["class_name"] for x in details)

                dt = time.perf_counter() - t_pipeline
                recent_times.append(dt)
                if len(recent_times) > 30:
                    recent_times.pop(0)
                fps = 1.0 / max(sum(recent_times) / len(recent_times), 1e-9)

                draw_runtime(drawn, dres["confidence"], bool(route_name), len(details), fps)

                ok_jpg, enc = cv2.imencode(
                    ".jpg", drawn, [int(cv2.IMWRITE_JPEG_QUALITY), 85]
                )
                if not ok_jpg:
                    continue

                current_time = source_frame_idx / self.source_fps if self.source_fps > 0 else 0.0
                progress = (
                    min(100.0, source_frame_idx / total_frames * 100.0)
                    if is_local_file and total_frames > 0 else None
                )

                with self.lock:
                    self.latest_jpeg = enc.tobytes()
                    self.frames_processed += 1
                    self.source_frame_idx = source_frame_idx
                    self.current_time_sec = current_time
                    self.progress_pct = progress
                    self.routed_device = route_name
                    self.routed_indicator_model = route_model_path
                    self.route_status = route_status
                    self.indicator_details = details
                    self.indicator_counts = dict(counts)
                    self.indicator_total_current = len(details)
                    self.indicator_infer_ms = indicator_ms
                    self.infer_fps = fps

            with self.lock:
                self.running = False
                self.pipeline_stage = "finished" if is_local_file else "stopped"
                if is_local_file:
                    self.progress_pct = 100.0
                    self.current_time_sec = self.duration_sec

        except Exception as e:
            with self.lock:
                self.error = str(e)
                self.running = False
                self.pipeline_stage = "error"
        finally:
            if cap is not None:
                cap.release()

    def status(self):
        with self.lock:
            return {
                "running": self.running,
                "error": self.error,
                "pipeline_stage": self.pipeline_stage,
                "device_result": self.device_result,
                "device_vote_name": self.device_vote_name,
                "routed_device": self.routed_device,
                "routed_indicator_model": self.routed_indicator_model,
                "route_status": self.route_status,
                "indicator_details": self.indicator_details,
                "indicator_counts": self.indicator_counts,
                "indicator_total_current": self.indicator_total_current,
                "frames_processed": self.frames_processed,
                "source_frame_idx": self.source_frame_idx,
                "total_frames": self.total_frames,
                "current_time_text": seconds_text(self.current_time_sec),
                "duration_text": seconds_text(self.duration_sec),
                "progress_pct": self.progress_pct,
                "source_fps": self.source_fps,
                "infer_fps": self.infer_fps,
                "width": self.width,
                "height": self.height,
                "device_infer_ms": self.device_infer_ms,
                "indicator_infer_ms": self.indicator_infer_ms,
            }


worker = PipelineWorker()

INDEX = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>设备→指示灯 多模型联合识别</title>
<style>
body{font-family:Arial,"Microsoft YaHei";margin:0;background:#0f1318;color:#edf2f7}
.wrap{max-width:1600px;margin:auto;padding:20px}
.card{background:#181e26;border:1px solid #2b3540;border-radius:12px;padding:16px;margin-bottom:14px}
h1,h3{margin-top:0}.grid{display:grid;grid-template-columns:repeat(4,minmax(180px,1fr));gap:12px}
.main{display:grid;grid-template-columns:minmax(0,2.1fr) minmax(390px,1fr);gap:14px}
label{display:block;font-size:12px;color:#a9b5c3;margin-bottom:5px}
input,select{width:100%;box-sizing:border-box;padding:9px;background:#0f1318;color:#fff;border:1px solid #3b4756;border-radius:7px}
button{padding:10px 18px;border:0;border-radius:8px;cursor:pointer;color:#fff;background:#2784ff}
button.stop{background:#d44f4f}.row{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.small{color:#9ba8b7;font-size:12px}.ok{color:#7ee59a}.err{color:#ff8e8e}.big{font-size:20px;font-weight:700}
.k{color:#9ba8b7;font-size:12px}.stats{display:grid;grid-template-columns:repeat(5,1fr);gap:8px}
.stat{background:#0f1318;padding:10px;border-radius:8px}
#videoBox{min-height:480px;background:#07090c;border-radius:10px;display:flex;align-items:center;justify-content:center;overflow:hidden}
#liveView{width:100%;max-height:76vh;object-fit:contain}
.progressWrap{height:18px;background:#0c1014;border-radius:9px;overflow:hidden;border:1px solid #323c48}
#progressBar{height:100%;width:0%;background:#2784ff}
table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:7px 5px;border-bottom:1px solid #303946;text-align:left}
.tag{display:inline-block;margin:2px 4px 2px 0;padding:4px 7px;border-radius:6px;background:#25303b}
.modelrow{display:grid;grid-template-columns:1fr 1.7fr;gap:8px;margin-bottom:8px}
@media(max-width:1100px){.main{grid-template-columns:1fr}.grid,.stats{grid-template-columns:1fr 1fr}}
</style>
</head>
<body>
<div class="wrap">
<h1>设备识别 → 指示灯识别 多设备联合测试</h1>

<div class="card">
<div class="grid">
<div><label>视频源</label><input id="source" value="0"></div>
<div><label>设备分类模型</label><input id="deviceModel" value="__DEVICE_MODEL__"></div>
<div><label>设备识别最低置信度</label><input id="deviceConf" type="number" value="0.70" step="0.05"></div>
<div><label>设备识别间隔（帧）</label><input id="deviceInterval" type="number" value="15"></div>
<div><label>设备 imgsz</label><select id="deviceImgsz"><option selected>320</option><option>416</option></select></div>
<div><label>指示灯 imgsz</label><select id="indicatorImgsz"><option>960</option><option selected>1280</option><option>1536</option></select></div>
<div><label>指示灯 conf</label><input id="indicatorConf" type="number" value="0.25" step="0.05"></div>
<div><label>指示灯 IoU</label><input id="indicatorIou" type="number" value="0.50" step="0.05"></div>
<div><label>GPU device</label><input id="device" value="0"></div>
<div><label>跳帧</label><select id="frameSkip"><option value="0" selected>0</option><option value="1">1</option><option value="2">2</option></select></div>
</div>

<h3 style="margin-top:18px">设备 → 指示灯模型映射</h3>
<div class="modelrow"><input id="name1" value="地震预警_CRES-LMA_101_交大铁发"><input id="model1" value="__EARTH_MODEL__"></div>
<div class="modelrow"><input id="name2" value="FAS_MDS3400DLX_V3.1_佳讯飞鸿"><input id="model2" value="__FAS_MODEL__"></div>

<div class="row" style="margin-top:14px">
<button onclick="startPipeline()">启动联合识别</button>
<button class="stop" onclick="stopPipeline()">停止</button>
<span id="statusText" class="small">未启动</span>
</div>
</div>

<div class="card"><div class="stats">
<div class="stat"><div class="k">设备 Top1</div><div class="big" id="deviceName">-</div></div>
<div class="stat"><div class="k">设备置信度</div><div class="big" id="deviceConfidence">0.000</div></div>
<div class="stat"><div class="k">多数投票</div><div class="big" id="voteName">-</div></div>
<div class="stat"><div class="k">当前路由</div><div class="big" id="routedDevice">-</div></div>
<div class="stat"><div class="k">当前灯数量</div><div class="big" id="indicatorTotal">0</div></div>
</div></div>

<div class="card">
<div class="row" style="justify-content:space-between;margin-bottom:7px">
<div><span id="timeText">00:00 / 00:00</span><span class="small" id="frameText"></span></div>
<div id="progressText">LIVE</div></div>
<div class="progressWrap"><div id="progressBar"></div></div>
</div>

<div class="main">
<div class="card"><h3>实时画面</h3><div id="videoBox"><div id="placeholder" class="small">启动后显示画面</div><img id="liveView" style="display:none"></div></div>

<div>
<div class="card"><h3>设备识别 Top-5</h3><table><thead><tr><th>设备</th><th>置信度</th></tr></thead><tbody id="topkBody"></tbody></table></div>
<div class="card"><h3>模型路由</h3><div class="k">状态</div><div id="routeStatus">-</div><div class="k" style="margin-top:10px">当前指示灯模型</div><div id="routeModel" class="small" style="word-break:break-all">-</div></div>
<div class="card"><h3>当前指示灯识别</h3><div id="indicatorCounts"></div><table><thead><tr><th>ID</th><th>类别</th><th>置信度</th></tr></thead><tbody id="indicatorBody"></tbody></table></div>
<div class="card"><h3>性能</h3><table><tbody>
<tr><td>整体 FPS</td><td id="inferFps">0.0</td></tr>
<tr><td>设备识别</td><td id="deviceMs">0 ms</td></tr>
<tr><td>灯识别</td><td id="indicatorMs">0 ms</td></tr>
<tr><td>分辨率</td><td id="resolution">-</td></tr>
<tr><td>源 FPS</td><td id="sourceFps">-</td></tr>
</tbody></table></div>
</div>
</div>
</div>

<script>
let timer=null;
function esc(s){return String(s??"").replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;");}

async function startPipeline(){
  const modelMap={};
  for(const i of [1,2]){
    const n=document.getElementById("name"+i).value.trim();
    const p=document.getElementById("model"+i).value.trim();
    if(n&&p) modelMap[n]=p;
  }
  const body={
    source:document.getElementById("source").value,
    device_model_path:document.getElementById("deviceModel").value,
    indicator_model_map:modelMap,
    device_conf_threshold:parseFloat(document.getElementById("deviceConf").value),
    device_interval:parseInt(document.getElementById("deviceInterval").value),
    device_imgsz:parseInt(document.getElementById("deviceImgsz").value),
    indicator_imgsz:parseInt(document.getElementById("indicatorImgsz").value),
    indicator_conf:parseFloat(document.getElementById("indicatorConf").value),
    indicator_iou:parseFloat(document.getElementById("indicatorIou").value),
    device:document.getElementById("device").value,
    frame_skip:parseInt(document.getElementById("frameSkip").value)
  };
  const r=await fetch("/pipeline/start",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
  const data=await r.json();
  if(!r.ok){document.getElementById("statusText").textContent=data.error||"启动失败";return;}
  const img=document.getElementById("liveView");
  img.src="/pipeline/stream?t="+Date.now();img.style.display="block";
  document.getElementById("placeholder").style.display="none";
  if(timer) clearInterval(timer);
  timer=setInterval(updateStatus,500);updateStatus();
}

async function stopPipeline(){
  await fetch("/pipeline/stop",{method:"POST"});
  if(timer){clearInterval(timer);timer=null;}
  const img=document.getElementById("liveView");img.src="";img.style.display="none";
  document.getElementById("placeholder").style.display="block";updateStatus();
}

async function updateStatus(){
  const r=await fetch("/pipeline/status");const s=await r.json();
  const st=document.getElementById("statusText");
  if(s.error){st.textContent="错误："+s.error;st.className="small err";}
  else if(s.running){st.textContent="运行中";st.className="small ok";}
  else{st.textContent="已停止";st.className="small";}

  const d=s.device_result||{};
  document.getElementById("deviceName").textContent=d.name||"-";
  document.getElementById("deviceConfidence").textContent=Number(d.confidence||0).toFixed(3);
  document.getElementById("voteName").textContent=s.device_vote_name||"-";
  document.getElementById("routedDevice").textContent=s.routed_device||"-";
  document.getElementById("indicatorTotal").textContent=s.indicator_total_current||0;
  document.getElementById("routeStatus").textContent=s.route_status||"-";
  document.getElementById("routeModel").textContent=s.routed_indicator_model||"-";

  document.getElementById("topkBody").innerHTML=(d.topk||[]).map(x=>`<tr><td>${esc(x.class_name)}</td><td>${Number(x.confidence).toFixed(3)}</td></tr>`).join("");
  const counts=s.indicator_counts||{};
  document.getElementById("indicatorCounts").innerHTML=Object.entries(counts).length?Object.entries(counts).map(([k,v])=>`<span class="tag">${esc(k)} × ${v}</span>`).join(""):'<span class="small">当前没有指示灯结果</span>';
  document.getElementById("indicatorBody").innerHTML=(s.indicator_details||[]).map(x=>`<tr><td>${x.class_id}</td><td>${esc(x.class_name)}</td><td>${Number(x.confidence).toFixed(3)}</td></tr>`).join("");

  document.getElementById("inferFps").textContent=Number(s.infer_fps||0).toFixed(1);
  document.getElementById("deviceMs").textContent=Number(s.device_infer_ms||0).toFixed(1)+" ms";
  document.getElementById("indicatorMs").textContent=Number(s.indicator_infer_ms||0).toFixed(1)+" ms";
  document.getElementById("resolution").textContent=(s.width&&s.height)?`${s.width}×${s.height}`:"-";
  document.getElementById("sourceFps").textContent=s.source_fps?Number(s.source_fps).toFixed(1):"-";

  if(s.progress_pct===null||s.progress_pct===undefined){
    document.getElementById("progressText").textContent="LIVE";
    document.getElementById("progressBar").style.width="100%";
    document.getElementById("timeText").textContent="实时视频";
    document.getElementById("frameText").textContent=` 已处理 ${s.frames_processed||0} 帧`;
  }else{
    const pct=Math.max(0,Math.min(100,Number(s.progress_pct||0)));
    document.getElementById("progressText").textContent=pct.toFixed(1)+"%";
    document.getElementById("progressBar").style.width=pct+"%";
    document.getElementById("timeText").textContent=`${s.current_time_text||"00:00"} / ${s.duration_text||"00:00"}`;
    document.getElementById("frameText").textContent=` 帧 ${s.source_frame_idx||0} / ${s.total_frames||0}`;
  }
}
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    return (
        INDEX
        .replace("__DEVICE_MODEL__", str(DEFAULT_DEVICE_MODEL))
        .replace("__EARTH_MODEL__", DEFAULT_INDICATOR_MODELS["地震预警_CRES-LMA_101_交大铁发"])
        .replace("__FAS_MODEL__", DEFAULT_INDICATOR_MODELS["FAS_MDS3400DLX_V3.1_佳讯飞鸿"])
    )

@app.get("/health")
def health():
    return {
        "ok": True,
        "device_model": str(DEFAULT_DEVICE_MODEL),
        "device_model_exists": DEFAULT_DEVICE_MODEL.exists(),
        "indicator_models": {
            k: {"path": v, "exists": Path(v).expanduser().exists()}
            for k, v in DEFAULT_INDICATOR_MODELS.items()
        },
    }

@app.post("/pipeline/start")
async def pipeline_start(cfg: dict):
    device_model = Path(str(cfg.get("device_model_path", DEFAULT_DEVICE_MODEL))).expanduser()
    if not device_model.exists():
        return JSONResponse(status_code=400, content={"error": f"设备识别模型不存在: {device_model}"})
    worker.start(cfg)
    return {"ok": True, "status": worker.status()}

@app.post("/pipeline/stop")
def pipeline_stop():
    worker.stop()
    return {"ok": True}

@app.get("/pipeline/status")
def pipeline_status():
    return worker.status()

@app.get("/pipeline/stream")
def pipeline_stream():
    def gen():
        last = None
        while True:
            with worker.lock:
                frame = worker.latest_jpeg
                running = worker.running
            if frame is not None and frame is not last:
                last = frame
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            if not running and frame is None:
                break
            time.sleep(0.01)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8300)
    args = ap.parse_args()
    print(f"联合识别前端: http://{args.host}:{args.port}")
    print(f"设备模型: {DEFAULT_DEVICE_MODEL}")
    for name, model in DEFAULT_INDICATOR_MODELS.items():
        print(f"{name}: {model}")
    uvicorn.run(app, host=args.host, port=args.port, workers=1)

if __name__ == "__main__":
    main()

