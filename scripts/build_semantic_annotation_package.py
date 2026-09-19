#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, shutil, zipfile
from pathlib import Path
import yaml

HTML = r'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>离线语义指示灯标注工具</title>
<style>
body{font-family:Arial,"Microsoft YaHei",sans-serif;margin:0;background:#f2f2f2}
#top{padding:10px;background:#222;color:#fff;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
button,input{font-size:14px} button{padding:6px 10px;cursor:pointer}
#wrap{display:flex;height:calc(100vh - 58px)}
#left{width:330px;background:#fff;border-right:1px solid #bbb;overflow:auto;padding:10px}
#main{flex:1;display:flex;align-items:center;justify-content:center;overflow:auto;background:#777}
canvas{background:#111;cursor:crosshair}
.cls{padding:7px;margin:3px 0;border:1px solid #ccc;cursor:pointer;border-radius:4px}
.cls.active{background:#d9edf7;border-color:#31708f;font-weight:bold}
.item{font-size:12px;padding:3px 4px;border-bottom:1px solid #eee;cursor:pointer;word-break:break-all}
.item.active{background:#e8f4ff}.item.done::before{content:"✓ ";color:green;font-weight:bold}
.note{background:#fff7d6;border:1px solid #e0cc7b;padding:8px;font-size:13px;line-height:1.5;margin:8px 0}
.small{font-size:12px;color:#555;line-height:1.5}
</style>
</head>
<body>
<div id="top">
<input id="files" type="file" accept="image/*" multiple>
<button onclick="prevImage()">上一张 A</button>
<button onclick="nextImage()">下一张 D</button>
<button onclick="undo()">撤销 Z</button>
<button onclick="clearBoxes()">清空 C</button>
<button onclick="exportJSON()">导出标注 JSON</button>
<span id="status"></span>
</div>
<div id="wrap">
<div id="left">
<b>设备：</b><div id="deviceName"></div>
<div class="note">标注员需要根据专业知识选择正确的指示灯类别，再框选灯本体。</div>
<b>指示灯类别</b>
<div id="classes"></div>
<hr>
<div class="small">数字键选择类别；A上一张；D下一张；Z撤销；C清空。</div>
<hr>
<div id="list"></div>
</div>
<div id="main"><canvas id="canvas"></canvas></div>
</div>
<script>
const CONFIG = __CONFIG__;
const filesInput=document.getElementById('files'),canvas=document.getElementById('canvas'),ctx=canvas.getContext('2d');
const list=document.getElementById('list'),classesDiv=document.getElementById('classes'),statusEl=document.getElementById('status');
document.getElementById('deviceName').textContent=CONFIG.device_name;
let files=[],index=0,img=new Image(),annotations={},scale=1,classId=Number(Object.keys(CONFIG.names)[0]||0);
let drag=false,sx=0,sy=0,ex=0,ey=0,imageW=0,imageH=0;
function current(){if(!files.length)return[];const n=files[index].name;if(!annotations[n])annotations[n]=[];return annotations[n]}
function buildClasses(){classesDiv.innerHTML="";Object.keys(CONFIG.names).forEach(k=>{const d=document.createElement('div');d.className="cls"+(Number(k)===classId?" active":"");d.textContent=`${k}: ${CONFIG.names[k]}`;d.onclick=()=>{classId=Number(k);buildClasses();draw()};classesDiv.appendChild(d)})}
function buildList(){list.innerHTML="";files.forEach((f,i)=>{const d=document.createElement('div');d.className="item";if(i===index)d.classList.add("active");if((annotations[f.name]||[]).length)d.classList.add("done");d.textContent=`${i+1}. ${f.name}`;d.onclick=()=>{index=i;buildList();loadImage()};list.appendChild(d)})}
filesInput.addEventListener('change',e=>{files=[...e.target.files].sort((a,b)=>a.name.localeCompare(b.name));index=0;buildList();loadImage()});
function loadImage(){if(!files.length)return;const f=files[index],url=URL.createObjectURL(f);img=new Image();img.onload=()=>{imageW=img.naturalWidth;imageH=img.naturalHeight;const mw=Math.max(700,innerWidth-400),mh=Math.max(500,innerHeight-100);scale=Math.min(mw/imageW,mh/imageH,1);canvas.width=Math.round(imageW*scale);canvas.height=Math.round(imageH*scale);draw();URL.revokeObjectURL(url)};img.src=url}
function draw(){if(!files.length||!img.complete)return;ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(img,0,0,canvas.width,canvas.height);ctx.lineWidth=2;ctx.strokeStyle="#00ff00";ctx.fillStyle="#00ff00";ctx.font="14px sans-serif";current().forEach((b)=>{ctx.strokeRect(b.x*scale,b.y*scale,b.w*scale,b.h*scale);ctx.fillText(`${b.class_id}:${CONFIG.names[b.class_id]}`,b.x*scale+2,Math.max(14,b.y*scale-3))});if(drag){ctx.strokeStyle="#ffff00";ctx.strokeRect(sx,sy,ex-sx,ey-sy)}updateStatus()}
canvas.addEventListener('mousedown',e=>{if(!files.length)return;const r=canvas.getBoundingClientRect();sx=e.clientX-r.left;sy=e.clientY-r.top;ex=sx;ey=sy;drag=true;draw()});
canvas.addEventListener('mousemove',e=>{if(!drag)return;const r=canvas.getBoundingClientRect();ex=e.clientX-r.left;ey=e.clientY-r.top;draw()});
canvas.addEventListener('mouseup',e=>{if(!drag)return;drag=false;const r=canvas.getBoundingClientRect();ex=e.clientX-r.left;ey=e.clientY-r.top;let x1=Math.min(sx,ex)/scale,y1=Math.min(sy,ey)/scale,x2=Math.max(sx,ex)/scale,y2=Math.max(sy,ey)/scale;if(x2-x1>=3&&y2-y1>=3)current().push({class_id:classId,class_name:CONFIG.names[classId],x:x1,y:y1,w:x2-x1,h:y2-y1});buildList();draw()});
function undo(){const b=current();if(b.length){b.pop();buildList();draw()}}
function clearBoxes(){if(!files.length)return;if(confirm("清空当前图片全部标注？")){annotations[files[index].name]=[];buildList();draw()}}
function prevImage(){if(index>0){index--;buildList();loadImage()}}
function nextImage(){if(index<files.length-1){index++;buildList();loadImage()}}
function updateStatus(){if(!files.length){statusEl.textContent="请选择图片";return}const done=files.filter(f=>(annotations[f.name]||[]).length>0).length;statusEl.textContent=`${index+1}/${files.length} | 类别 ${classId}:${CONFIG.names[classId]} | 已标图片 ${done}`}
function exportJSON(){const out={format:"semantic_indicator_bbox_v1",device_name:CONFIG.device_name,names:CONFIG.names,annotations};const blob=new Blob([JSON.stringify(out,null,2)],{type:"application/json"});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download="indicator_annotations.json";a.click()}
document.addEventListener('keydown',e=>{if(/^[0-9]$/.test(e.key)&&CONFIG.names[e.key]!==undefined){classId=Number(e.key);buildClasses();draw()}else if(e.key==='a'||e.key==='A')prevImage();else if(e.key==='d'||e.key==='D')nextImage();else if(e.key==='z'||e.key==='Z')undo();else if(e.key==='c'||e.key==='C')clearBoxes()});
buildClasses();
</script>
</body>
</html>'''

README = '''离线语义指示灯标注包

1. 解压 ZIP。
2. 双击 annotator.html。
3. 点击“选择文件”，进入 images 文件夹，全选所有图片。
4. 左侧选择正确的指示灯类别。
5. 鼠标框选对应指示灯本体。
6. 完成后点击“导出标注 JSON”。
7. 将 indicator_annotations.json 发回。

标注要求：
- 标注员需要知道各指示灯的实际含义。
- 只框指示灯本体，不框文字、按钮、端口、整个面板。
- 熄灭但位置明确的灯也标注。
- 无法确认的灯不要猜。
'''

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project",type=Path,default=Path.home()/"yolo_device_monitor")
    ap.add_argument("--workspace-name",required=True)
    a=ap.parse_args()
    project=a.project.expanduser().resolve()
    root=project/"data"/"indicator_detection"/a.workspace_name
    images=root/"images"/"to_label"
    data_yaml=root/"data.yaml"
    if not images.exists(): raise SystemExit(f"图片目录不存在: {images}")
    cfg=yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    names=cfg.get("names") or {}
    if isinstance(names,list): names={i:v for i,v in enumerate(names)}
    names={str(k):str(v) for k,v in names.items()}
    if not names: raise SystemExit("data.yaml 没有 names")
    out=project/"annotation_packages"/a.workspace_name
    if out.exists(): shutil.rmtree(out)
    (out/"images").mkdir(parents=True)
    config={"device_name":a.workspace_name,"names":names}
    html=HTML.replace("__CONFIG__",json.dumps(config,ensure_ascii=False))
    (out/"annotator.html").write_text(html,encoding="utf-8")
    (out/"README.txt").write_text(README,encoding="utf-8")
    (out/"classes.txt").write_text("\n".join(f"{k}: {v}" for k,v in names.items()),encoding="utf-8")
    n=0
    for f in sorted(images.iterdir()):
        if f.is_file():
            shutil.copy2(f,out/"images"/f.name); n+=1
    zpath=out.with_suffix(".zip")
    if zpath.exists(): zpath.unlink()
    with zipfile.ZipFile(zpath,"w",zipfile.ZIP_DEFLATED) as z:
        for f in out.rglob("*"):
            if f.is_file(): z.write(f,f.relative_to(out.parent))
    print("设备:",a.workspace_name)
    print("类别:",names)
    print("图片数:",n)
    print("ZIP:",zpath)

if __name__=="__main__":
    main()

