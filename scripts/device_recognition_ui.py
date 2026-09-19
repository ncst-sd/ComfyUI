#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import shutil
import tempfile
import threading
import time
from collections import Counter, deque
from pathlib import Path
from typing import Optional

import cv2
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from ultralytics import YOLO

HTML = r"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>设备识别实时监控</title>
<style>
body{margin:0;font-family:Arial,"Microsoft YaHei",sans-serif;background:#111;color:#eee}
.header{padding:14px 20px;background:#1b1b1b;border-bottom:1px solid #333;font-size:22px;font-weight:700}
.layout{display:grid;grid-template-columns:380px 1fr;gap:16px;padding:16px}
.card{background:#1b1b1b;border:1px solid #333;border-radius:10px;overflow:hidden}
.panel{padding:16px}.section-title{font-weight:700;margin-bottom:10px;font-size:16px}
label{display:block;margin:10px 0 5px;color:#aaa;font-size:13px}
input,select{width:100%;box-sizing:border-box;padding:8px;border-radius:6px;border:1px solid #444;background:#222;color:#eee}
button{width:100%;padding:10px;margin-top:10px;border:none;border-radius:6px;cursor:pointer;font-weight:700}
.start{background:#2f8f46;color:white}.stop{background:#8f2f2f;color:white}
.video-wrap{background:#000;min-height:520px;display:flex;align-items:center;justify-content:center}
.video-wrap img{width:100%;height:auto;display:block}.grid2{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.result{padding:16px;border-top:1px solid #333}.device{font-size:26px;font-weight:700;line-height:1.35}
.ok{color:#74d680}.warn{color:#ffd166}.bad{color:#ff6b6b}.small{font-size:13px;color:#aaa;line-height:1.5}
.metric{display:grid;grid-template-columns:1fr auto;gap:10px;margin-top:10px}
.status{margin-top:10px;padding:10px;border-radius:6px;background:#252525}.hidden{display:none}
@media(max-width:1100px){.layout{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="header">设备识别实时监控</div>

<div class="layout">
  <div class="card panel">
    <div class="section-title">视频来源</div>

    <label>来源类型</label>
    <select id="sourceType" onchange="switchSourceType()">
      <option value="video">本地视频文件</option>
      <option value="camera">USB 摄像头</option>
      <option value="rtsp">RTSP 摄像头</option>
    </select>

    <div id="videoBox">
      <label>选择视频文件</label>
      <input type="file" id="videoFile" accept="video/*">
    </div>

    <div id="cameraBox" class="hidden">
      <label>摄像头编号</label>
      <input type="number" id="cameraId" value="0" min="0">
    </div>

    <div id="rtspBox" class="hidden">
      <label>RTSP 地址</label>
      <input type="text" id="rtspUrl" placeholder="rtsp://user:pass@ip/stream">
    </div>

    <div class="section-title" style="margin-top:20px">识别参数</div>

    <div class="grid2">
      <div>
        <label>imgsz</label>
        <input type="number" id="imgsz" value="320">
      </div>
      <div>
        <label>采样间隔(秒)</label>
        <input type="number" step="0.1" id="sampleSeconds" value="0.5">
      </div>
    </div>

    <div class="grid2">
      <div>
        <label>投票窗口</label>
        <input type="number" id="window" value="8">
      </div>
      <div>
        <label>单帧最低置信度</label>
        <input type="number" step="0.01" id="minFrameConf" value="0.60">
      </div>
    </div>

    <div class="grid2">
      <div>
        <label>最低投票比例</label>
        <input type="number" step="0.01" id="minVoteRatio" value="0.75">
      </div>
      <div>
        <label>最低平均置信度</label>
        <input type="number" step="0.01" id="minMeanConf" value="0.80">
      </div>
    </div>

    <button class="start" onclick="startRecognition()">启动识别</button>
    <button class="stop" onclick="stopRecognition()">停止识别</button>
    <div id="message" class="status small">未启动</div>
  </div>

  <div class="card">
    <div class="video-wrap">
      <img src="/video_feed" alt="video stream">
    </div>

    <div class="result">
      <div class="small">设备识别结果</div>
      <div id="device" class="device warn">设备身份未确认</div>
      <div class="metric"><span>当前候选设备</span><span id="candidate">-</span></div>
      <div class="metric"><span>识别状态</span><span id="stable">-</span></div>
      <div class="metric"><span>平均置信度</span><span id="confidence">0%</span></div>
      <div class="metric"><span>时序投票比例</span><span id="vote">0%</span></div>
      <div class="metric"><span>当前来源</span><span id="source">-</span></div>
      <div class="metric"><span>实时 FPS</span><span id="fps">-</span></div>
    </div>
  </div>
</div>

<script>
function switchSourceType(){
  const t=document.getElementById('sourceType').value;
  document.getElementById('videoBox').classList.toggle('hidden',t!=='video');
  document.getElementById('cameraBox').classList.toggle('hidden',t!=='camera');
  document.getElementById('rtspBox').classList.toggle('hidden',t!=='rtsp');
}

async function startRecognition(){
  const t=document.getElementById('sourceType').value;
  const fd=new FormData();

  fd.append('source_type',t);
  fd.append('imgsz',document.getElementById('imgsz').value);
  fd.append('sample_seconds',document.getElementById('sampleSeconds').value);
  fd.append('window',document.getElementById('window').value);
  fd.append('min_frame_conf',document.getElementById('minFrameConf').value);
  fd.append('min_vote_ratio',document.getElementById('minVoteRatio').value);
  fd.append('min_mean_conf',document.getElementById('minMeanConf').value);

  if(t==='video'){
    const f=document.getElementById('videoFile').files[0];
    if(!f){alert('请选择视频文件');return;}
    fd.append('video_file',f);
  }else if(t==='camera'){
    fd.append('camera_id',document.getElementById('cameraId').value);
  }else{
    const url=document.getElementById('rtspUrl').value.trim();
    if(!url){alert('请输入 RTSP 地址');return;}
    fd.append('rtsp_url',url);
  }

  document.getElementById('message').textContent='正在启动...';

  try{
    const r=await fetch('/api/start',{method:'POST',body:fd});
    const d=await r.json();
    document.getElementById('message').textContent=d.message||JSON.stringify(d);
  }catch(e){
    document.getElementById('message').textContent='启动失败: '+e;
  }
}

async function stopRecognition(){
  try{
    const r=await fetch('/api/stop',{method:'POST'});
    const d=await r.json();
    document.getElementById('message').textContent=d.message||'已停止';
  }catch(e){
    document.getElementById('message').textContent='停止失败: '+e;
  }
}

async function refreshStatus(){
  try{
    const r=await fetch('/api/status');
    const s=await r.json();

    document.getElementById('candidate').textContent=s.candidate_name||'-';
    document.getElementById('source').textContent=s.source||'-';
    document.getElementById('fps').textContent=(s.stream_fps||0).toFixed(1);

    const conf=Math.round((s.mean_confidence||0)*1000)/10;
    const vote=Math.round((s.vote_ratio||0)*1000)/10;

    document.getElementById('confidence').textContent=conf+'%';
    document.getElementById('vote').textContent=vote+'%';

    const deviceEl=document.getElementById('device');
    const stableEl=document.getElementById('stable');

    if(s.stable){
      deviceEl.textContent=s.workspace_name||'-';
      deviceEl.className='device ok';
      stableEl.textContent='已确认';
      stableEl.className='ok';
    }else{
      deviceEl.textContent='设备身份未确认';
      deviceEl.className='device warn';
      stableEl.textContent='不确定';
      stableEl.className='warn';
    }

    if(s.error){
      document.getElementById('message').textContent=s.error;
    }
  }catch(e){}
}

setInterval(refreshStatus,500);
switchSourceType();
refreshStatus();
</script>
</body>
</html>
"""


class RecognitionManager:
    def __init__(self, model_path: Path, device: str = "0"):
        self.model_path = str(model_path.expanduser().resolve())
        self.device = device
        self.model = YOLO(self.model_path)

        self.thread = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()

        self.latest_jpeg: Optional[bytes] = None
        self.temp_video: Optional[Path] = None
        self.status = self._empty_status()

    def _empty_status(self):
        return {
            "running": False,
            "source": None,
            "workspace_name": None,
            "candidate_name": None,
            "stable": False,
            "vote_ratio": 0.0,
            "mean_confidence": 0.0,
            "stream_fps": 0.0,
            "error": None,
        }

    def stop(self):
        self.stop_event.set()

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3.0)

        self.thread = None
        self.stop_event.clear()

        old_temp = self.temp_video
        self.temp_video = None

        if old_temp and old_temp.exists():
            try:
                old_temp.unlink()
            except Exception:
                pass

        with self.lock:
            self.status["running"] = False

    def start(
        self,
        *,
        source,
        source_label,
        temp_video=None,
        imgsz=320,
        sample_seconds=0.5,
        window=8,
        min_frame_conf=0.60,
        min_vote_ratio=0.75,
        min_mean_conf=0.80,
        loop_video=True,
    ):
        # 关键修复：先停止旧任务，再登记新上传视频。
        self.stop()

        self.temp_video = Path(temp_video) if temp_video else None

        with self.lock:
            self.status = self._empty_status()
            self.status["running"] = True
            self.status["source"] = source_label
            self.latest_jpeg = None

        self.thread = threading.Thread(
            target=self._run,
            kwargs={
                "source": source,
                "imgsz": imgsz,
                "sample_seconds": sample_seconds,
                "window": window,
                "min_frame_conf": min_frame_conf,
                "min_vote_ratio": min_vote_ratio,
                "min_mean_conf": min_mean_conf,
                "loop_video": loop_video,
            },
            daemon=True,
        )
        self.thread.start()

    def _run(
        self,
        source,
        imgsz,
        sample_seconds,
        window,
        min_frame_conf,
        min_vote_ratio,
        min_mean_conf,
        loop_video,
    ):
        history = deque(maxlen=window)

        try:
            print(f"[INFO] 打开视频源: {source}")

            if isinstance(source, str) and Path(source).exists():
                print(f"[INFO] 文件存在: {source}")
                print(f"[INFO] 文件大小: {Path(source).stat().st_size} bytes")

            cap = cv2.VideoCapture(source)

            if not cap.isOpened():
                raise RuntimeError(f"无法打开视频源: {source}")

            is_file = isinstance(source, str) and Path(source).is_file()
            source_fps = cap.get(cv2.CAP_PROP_FPS)

            last_infer = 0.0
            fps_count = 0
            fps_t0 = time.time()

            while not self.stop_event.is_set():
                ok, frame = cap.read()

                if not ok:
                    if is_file and loop_video:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        history.clear()
                        continue
                    break

                now = time.time()

                if now - last_infer >= sample_seconds:
                    result = self.model.predict(
                        frame,
                        imgsz=imgsz,
                        device=self.device,
                        verbose=False,
                    )[0]

                    probs = result.probs

                    if probs is not None:
                        idx = int(probs.top1)
                        conf = float(probs.top1conf)
                        name = result.names[idx]

                        if conf >= min_frame_conf:
                            history.append((name, conf))

                        if history:
                            counts = Counter(n for n, _ in history)
                            candidate, count = counts.most_common(1)[0]
                            vote_ratio = count / len(history)

                            winner_confs = [
                                c for n, c in history
                                if n == candidate
                            ]
                            mean_conf = sum(winner_confs) / len(winner_confs)

                            stable = (
                                len(history) >= min(3, window)
                                and vote_ratio >= min_vote_ratio
                                and mean_conf >= min_mean_conf
                            )

                            with self.lock:
                                self.status["candidate_name"] = candidate
                                self.status["vote_ratio"] = vote_ratio
                                self.status["mean_confidence"] = mean_conf
                                self.status["stable"] = stable
                                self.status["workspace_name"] = (
                                    candidate if stable else None
                                )

                    last_infer = now

                fps_count += 1

                if now - fps_t0 >= 1.0:
                    fps = fps_count / (now - fps_t0)

                    with self.lock:
                        self.status["stream_fps"] = fps

                    fps_count = 0
                    fps_t0 = now

                with self.lock:
                    stable = self.status["stable"]
                    vote = self.status["vote_ratio"]
                    conf = self.status["mean_confidence"]

                overlay = frame.copy()

                cv2.rectangle(
                    overlay,
                    (10, 10),
                    (430, 90),
                    (0, 0, 0),
                    -1,
                )

                cv2.putText(
                    overlay,
                    "DEVICE: CONFIRMED" if stable else "DEVICE: UNCERTAIN",
                    (22, 42),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    (0, 255, 0) if stable else (0, 215, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    overlay,
                    f"confidence={conf:.3f} vote={vote:.3f}",
                    (22, 72),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.56,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                ok_jpg, buf = cv2.imencode(
                    ".jpg",
                    overlay,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 82],
                )

                if ok_jpg:
                    with self.lock:
                        self.latest_jpeg = buf.tobytes()

                if is_file and source_fps and source_fps > 0:
                    time.sleep(max(0.0, 1.0 / source_fps))

            cap.release()

        except Exception as e:
            print("[ERROR]", repr(e))

            with self.lock:
                self.status["error"] = str(e)
                self.status["running"] = False

        finally:
            with self.lock:
                self.status["running"] = False

    def get_status(self):
        with self.lock:
            return dict(self.status)

    def get_jpeg(self):
        with self.lock:
            return self.latest_jpeg


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default=str(
            Path.home()
            / "yolo_device_monitor"
            / "runs"
            / "classify"
            / "device_cls_yolo11s"
            / "weights"
            / "best.pt"
        ),
    )

    parser.add_argument("--device", default="0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8200)

    return parser.parse_args()


args = parse_args()

manager = RecognitionManager(
    Path(args.model),
    device=args.device,
)

app = FastAPI(title="Device Recognition UI")


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


@app.get("/health")
def health():
    return {
        "ok": True,
        "model": manager.model_path,
        "device": manager.device,
    }


@app.get("/api/status")
def status():
    return manager.get_status()


@app.post("/api/stop")
def stop():
    manager.stop()

    return {
        "ok": True,
        "message": "识别已停止",
    }


@app.post("/api/start")
async def start(
    source_type: str = Form(...),
    imgsz: int = Form(320),
    sample_seconds: float = Form(0.5),
    window: int = Form(8),
    min_frame_conf: float = Form(0.60),
    min_vote_ratio: float = Form(0.75),
    min_mean_conf: float = Form(0.80),
    camera_id: Optional[int] = Form(None),
    rtsp_url: Optional[str] = Form(None),
    video_file: Optional[UploadFile] = File(None),
):
    if source_type == "video":
        if video_file is None:
            return JSONResponse(
                {"ok": False, "message": "没有选择视频文件"},
                status_code=400,
            )

        suffix = Path(video_file.filename or "video.mp4").suffix or ".mp4"

        tmp = (
            Path(tempfile.gettempdir())
            / f"device_ui_{time.time_ns()}{suffix}"
        )

        with tmp.open("wb") as f:
            shutil.copyfileobj(video_file.file, f)

        if not tmp.exists() or tmp.stat().st_size == 0:
            return JSONResponse(
                {"ok": False, "message": "视频上传失败或文件为空"},
                status_code=400,
            )

        print(f"[INFO] 上传视频: {video_file.filename}")
        print(f"[INFO] 临时文件: {tmp}")
        print(f"[INFO] 临时文件大小: {tmp.stat().st_size} bytes")

        manager.start(
            source=str(tmp),
            source_label=video_file.filename or tmp.name,
            temp_video=tmp,
            imgsz=imgsz,
            sample_seconds=sample_seconds,
            window=window,
            min_frame_conf=min_frame_conf,
            min_vote_ratio=min_vote_ratio,
            min_mean_conf=min_mean_conf,
            loop_video=True,
        )

        return {
            "ok": True,
            "message": f"识别已启动：{video_file.filename or tmp.name}",
        }

    if source_type == "camera":
        source = int(camera_id or 0)

        manager.start(
            source=source,
            source_label=f"camera:{source}",
            temp_video=None,
            imgsz=imgsz,
            sample_seconds=sample_seconds,
            window=window,
            min_frame_conf=min_frame_conf,
            min_vote_ratio=min_vote_ratio,
            min_mean_conf=min_mean_conf,
            loop_video=False,
        )

        return {
            "ok": True,
            "message": f"识别已启动：camera:{source}",
        }

    if source_type == "rtsp":
        if not rtsp_url:
            return JSONResponse(
                {"ok": False, "message": "RTSP 地址不能为空"},
                status_code=400,
            )

        manager.start(
            source=rtsp_url,
            source_label=rtsp_url,
            temp_video=None,
            imgsz=imgsz,
            sample_seconds=sample_seconds,
            window=window,
            min_frame_conf=min_frame_conf,
            min_vote_ratio=min_vote_ratio,
            min_mean_conf=min_mean_conf,
            loop_video=False,
        )

        return {
            "ok": True,
            "message": "RTSP 识别已启动",
        }

    return JSONResponse(
        {"ok": False, "message": "不支持的视频来源类型"},
        status_code=400,
    )


@app.get("/video_feed")
def video_feed():
    def generate():
        while True:
            jpg = manager.get_jpeg()

            if jpg is None:
                time.sleep(0.05)
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + jpg
                + b"\r\n"
            )

            time.sleep(0.03)

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    print("=" * 80)
    print("设备识别前端")
    print("=" * 80)
    print("模型:", manager.model_path)
    print("设备:", manager.device)
    print(f"访问地址: http://{args.host}:{args.port}")
    print()

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )

