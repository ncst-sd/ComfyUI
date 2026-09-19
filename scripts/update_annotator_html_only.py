#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
update_annotator_html_only.py

只更新每个设备文件夹中的 annotator.html：
- 不重新抽帧
- 不删除/修改 images
- 不修改 frame_quality.csv
- 不重新生成视频帧
- 不修改已有标注结果文件
- 修复“输入自定义类别时触发快捷键”的问题
- 保留滚轮缩放、Space+左键平移、左键画框
- 支持在自定义类别输入框中按 Enter 新增类别

默认目录：
~/yolo_device_monitor/annotation_packages_best_frames_fast/
"""

import argparse
import json
import re
from pathlib import Path


DEFAULT_ROOT = (
    Path.home()
    / "yolo_device_monitor"
    / "annotation_packages_best_frames_fast"
)


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>指示灯标注</title>
<style>
body{margin:0;background:#111;color:#eee;font-family:Arial,"Microsoft YaHei";overflow:hidden}
#top{height:54px;display:flex;align-items:center;gap:7px;padding:0 10px;background:#1b1b1b}
#wrap{display:flex;height:calc(100vh - 54px)}
#left{width:360px;padding:10px;overflow:auto;background:#1b1b1b}
#main{flex:1;position:relative;overflow:hidden;background:#222}
canvas{position:absolute;left:0;top:0;background:#000;cursor:crosshair}
button,input{font-size:14px}
button{padding:6px 9px}
.cls,.item{padding:6px;border-bottom:1px solid #333;cursor:pointer}
.cls.active,.item.active{background:#245d78}
.item.done:before{content:"✓ ";color:#7fe18a}
.note{padding:8px;margin:8px 0;background:#493f16;line-height:1.5}
.small{font-size:12px;color:#aaa}
#status{margin-left:auto;font-size:13px}
</style>
</head>
<body>
<div id="top">
<input id="files" type="file" accept="image/*" multiple>
<button onclick="prev()">上一张 A</button>
<button onclick="next()">下一张 D</button>
<button onclick="undo()">撤销 Z</button>
<button onclick="clearNow()">清空 C</button>
<button onclick="zoomAt(0.8,W/2,H/2)">－</button>
<span id="zl">100%</span>
<button onclick="zoomAt(1.25,W/2,H/2)">＋</button>
<button onclick="fit()">适应 F</button>
<button onclick="reset()">100% R</button>
<button onclick="exportJSON()">导出 JSON</button>
<span id="status"></span>
</div>

<div id="wrap">
<div id="left">
<b>设备：</b><div id="dev"></div>

<div class="note">
<b>滚轮：</b>放大/缩小<br>
<b>Space+左键：</b>平移<br>
<b>左键拖动：</b>画框<br>
建议先放大灯区域再标。
</div>

<b>类别</b>
<div id="classes"></div>

<div>
<input id="newc" placeholder="新类别">
<button onclick="addc()">新增</button>
</div>

<hr>
<div class="small">
0~9 选类别；A/D 切图；Z 撤销；C 清空；F 适应；R 100%<br>
在输入框里输入文字时不会触发这些快捷键，按 Enter 可直接新增类别。
</div>
<hr>

<div id="list"></div>
</div>

<div id="main"><canvas id="c"></canvas></div>
</div>

<script>
const CFG=__CFG__;

const c=document.getElementById("c");
const x=c.getContext("2d");
const main=document.getElementById("main");
const filesEl=document.getElementById("files");
const classesEl=document.getElementById("classes");
const listEl=document.getElementById("list");
const newClassEl=document.getElementById("newc");

document.getElementById("dev").textContent=CFG.device_name;

let names={...CFG.names};
let files=[];
let idx=0;
let img=new Image();
let ann={};
let cid=Object.keys(names).length ? +Object.keys(names)[0] : null;

let z=1;
let px=0;
let py=0;

let draw=false;
let pan=false;
let space=false;

let sx=0,sy=0,cx=0,cy=0;
let opx=0,opy=0,psx=0,psy=0;
let W=0,H=0;


function boxes(){
    if(!files.length)return[];
    let n=files[idx].name;
    return ann[n]||(ann[n]=[]);
}


function resize(){
    c.width=W=main.clientWidth;
    c.height=H=main.clientHeight;
}


function classes(){
    classesEl.innerHTML="";

    Object.keys(names)
        .sort((a,b)=>+a-+b)
        .forEach(k=>{
            let d=document.createElement("div");
            d.className="cls"+(+k===cid?" active":"");
            d.textContent=k+": "+names[k];

            d.onclick=()=>{
                cid=+k;
                classes();
                paint();
            };

            classesEl.appendChild(d);
        });
}


function addc(){
    let v=newClassEl.value.trim();

    if(!v)return;

    let k=0;

    while(names[k]!==undefined)k++;

    names[k]=v;
    cid=k;

    newClassEl.value="";

    classes();
    paint();

    newClassEl.focus();
}


/*
 * 输入框专用处理：
 * 1. 阻止 A/D/Z/C/F/R/数字/Space 冒泡到全局快捷键
 * 2. Enter 直接新增类别
 */
newClassEl.addEventListener("keydown", e=>{
    e.stopPropagation();

    if(e.key==="Enter"){
        e.preventDefault();
        addc();
    }
});

newClassEl.addEventListener("keyup", e=>{
    e.stopPropagation();
});


function buildList(){
    listEl.innerHTML="";

    files.forEach((f,i)=>{
        let d=document.createElement("div");

        d.className=
            "item"
            +(i===idx?" active":"")
            +((ann[f.name]||[]).length?" done":"");

        d.textContent=(i+1)+". "+f.name;

        d.onclick=()=>{
            idx=i;
            buildList();
            load();
        };

        listEl.appendChild(d);
    });
}


filesEl.onchange=e=>{
    files=[...e.target.files]
        .sort((a,b)=>a.name.localeCompare(b.name));

    idx=0;

    buildList();
    load();
};


function load(){
    if(!files.length)return;

    let u=URL.createObjectURL(files[idx]);

    img=new Image();

    img.onload=()=>{
        fit();
        URL.revokeObjectURL(u);
    };

    img.src=u;
}


function fit(){
    if(!img.naturalWidth)return;

    resize();

    z=Math.min(
        W/img.naturalWidth,
        H/img.naturalHeight
    );

    px=(W-img.naturalWidth*z)/2;
    py=(H-img.naturalHeight*z)/2;

    paint();
}


function reset(){
    if(!img.naturalWidth)return;

    resize();

    z=1;

    px=(W-img.naturalWidth)/2;
    py=(H-img.naturalHeight)/2;

    paint();
}


function zoomAt(f,qx,qy){
    let old=z;

    z=Math.max(
        .1,
        Math.min(20,z*f)
    );

    let ix=(qx-px)/old;
    let iy=(qy-py)/old;

    px=qx-ix*z;
    py=qy-iy*z;

    paint();
}


c.onwheel=e=>{
    e.preventDefault();

    let r=c.getBoundingClientRect();

    zoomAt(
        e.deltaY<0 ? 1.2 : .833,
        e.clientX-r.left,
        e.clientY-r.top
    );
};


function toImg(a,b){
    return{
        x:(a-px)/z,
        y:(b-py)/z
    };
}


function paint(){
    if(!img.complete||!img.naturalWidth)return;

    resize();

    x.clearRect(0,0,W,H);

    x.imageSmoothingEnabled=true;
    x.imageSmoothingQuality="high";

    x.drawImage(
        img,
        px,
        py,
        img.naturalWidth*z,
        img.naturalHeight*z
    );

    x.lineWidth=2;
    x.strokeStyle="#00ff00";
    x.fillStyle="#00ff00";
    x.font="14px sans-serif";

    boxes().forEach(b=>{
        let a=px+b.x*z;
        let d=py+b.y*z;
        let w=b.w*z;
        let h=b.h*z;

        x.strokeRect(a,d,w,h);

        x.fillText(
            b.class_id+":"+b.class_name,
            a+2,
            Math.max(14,d-3)
        );
    });

    if(draw){
        x.strokeStyle="#ffff00";

        x.strokeRect(
            Math.min(sx,cx),
            Math.min(sy,cy),
            Math.abs(cx-sx),
            Math.abs(cy-sy)
        );
    }

    document.getElementById("zl").textContent=
        Math.round(z*100)+"%";

    let done=files.filter(
        f=>(ann[f.name]||[]).length
    ).length;

    document.getElementById("status").textContent=
        files.length
        ? (idx+1)+"/"+files.length+" | 已标 "+done
        : "";
}


c.onmousedown=e=>{
    let r=c.getBoundingClientRect();

    let a=e.clientX-r.left;
    let b=e.clientY-r.top;

    if(space||e.button===1){
        pan=true;

        psx=a;
        psy=b;

        opx=px;
        opy=py;

        return;
    }

    if(cid===null||names[cid]===undefined){
        alert("请先选择或新增类别");
        return;
    }

    draw=true;

    sx=cx=a;
    sy=cy=b;
};


c.onmousemove=e=>{
    let r=c.getBoundingClientRect();

    let a=e.clientX-r.left;
    let b=e.clientY-r.top;

    if(pan){
        px=opx+a-psx;
        py=opy+b-psy;

        paint();
    }
    else if(draw){
        cx=a;
        cy=b;

        paint();
    }
};


c.onmouseup=e=>{
    if(pan){
        pan=false;
        return;
    }

    if(!draw)return;

    draw=false;

    let a=toImg(sx,sy);
    let b=toImg(cx,cy);

    let x1=Math.max(
        0,
        Math.min(
            img.naturalWidth,
            Math.min(a.x,b.x)
        )
    );

    let y1=Math.max(
        0,
        Math.min(
            img.naturalHeight,
            Math.min(a.y,b.y)
        )
    );

    let x2=Math.max(
        0,
        Math.min(
            img.naturalWidth,
            Math.max(a.x,b.x)
        )
    );

    let y2=Math.max(
        0,
        Math.min(
            img.naturalHeight,
            Math.max(a.y,b.y)
        )
    );

    if(x2-x1>=2&&y2-y1>=2){
        boxes().push({
            class_id:cid,
            class_name:names[cid],
            x:x1,
            y:y1,
            w:x2-x1,
            h:y2-y1
        });
    }

    buildList();
    paint();
};


function undo(){
    let b=boxes();

    if(b.length){
        b.pop();

        buildList();
        paint();
    }
}


function clearNow(){
    if(
        files.length
        &&confirm("清空当前图片？")
    ){
        ann[files[idx].name]=[];

        buildList();
        paint();
    }
}


function prev(){
    if(idx>0){
        idx--;

        buildList();
        load();
    }
}


function next(){
    if(idx<files.length-1){
        idx++;

        buildList();
        load();
    }
}


function exportJSON(){
    let o={
        format:"semantic_indicator_bbox_v3",
        device_name:CFG.device_name,
        names,
        annotations:ann
    };

    let b=new Blob(
        [JSON.stringify(o,null,2)],
        {type:"application/json"}
    );

    let a=document.createElement("a");

    a.href=URL.createObjectURL(b);
    a.download="indicator_annotations.json";

    a.click();
}


/*
 * 全局快捷键：
 * 焦点在 input / textarea / select / contenteditable 时全部禁用。
 */
document.addEventListener("keydown", e=>{

    const t=e.target;

    const typing=
        t instanceof HTMLInputElement
        || t instanceof HTMLTextAreaElement
        || t instanceof HTMLSelectElement
        || t.isContentEditable;

    if(typing){
        return;
    }

    if(e.code==="Space"){
        space=true;
        e.preventDefault();
        return;
    }

    if(/^[0-9]$/.test(e.key)&&names[e.key]!==undefined){
        cid=+e.key;

        classes();
        paint();
    }
    else if(/[aA]/.test(e.key)){
        prev();
    }
    else if(/[dD]/.test(e.key)){
        next();
    }
    else if(/[zZ]/.test(e.key)){
        undo();
    }
    else if(/[cC]/.test(e.key)){
        clearNow();
    }
    else if(/[fF]/.test(e.key)){
        fit();
    }
    else if(/[rR]/.test(e.key)){
        reset();
    }
});


document.addEventListener("keyup", e=>{

    const t=e.target;

    const typing=
        t instanceof HTMLInputElement
        || t instanceof HTMLTextAreaElement
        || t instanceof HTMLSelectElement
        || t.isContentEditable;

    if(typing){
        return;
    }

    if(e.code==="Space"){
        space=false;
    }
});


window.onblur=()=>{
    space=false;
    pan=false;
};


window.onresize=()=>{
    if(img.naturalWidth){
        fit();
    }
};


classes();
</script>
</body>
</html>
"""


def load_names(folder: Path):
    names = {}

    classes_txt = folder / "classes.txt"

    if classes_txt.exists():
        for line in classes_txt.read_text(
            encoding="utf-8",
            errors="ignore"
        ).splitlines():

            line = line.strip()

            if not line or ":" not in line:
                continue

            key, value = line.split(":", 1)

            if key.strip().isdigit():
                names[key.strip()] = value.strip()

    # 如果 classes.txt 不存在，尝试从旧 annotator.html 读取
    if not names:
        old_html = folder / "annotator.html"

        if old_html.exists():
            text = old_html.read_text(
                encoding="utf-8",
                errors="ignore"
            )

            patterns = [
                r"const\s+CFG\s*=\s*(\{.*?\});",
                r"const\s+CONFIG\s*=\s*(\{.*?\});",
            ]

            for pattern in patterns:
                m = re.search(
                    pattern,
                    text,
                    flags=re.S
                )

                if not m:
                    continue

                try:
                    cfg = json.loads(
                        m.group(1)
                    )

                    raw = cfg.get(
                        "names",
                        {}
                    )

                    names = {
                        str(k): str(v)
                        for k, v in raw.items()
                    }

                    break

                except Exception:
                    pass

    return names


def update_one(folder: Path):
    names = load_names(folder)

    config = {
        "device_name": folder.name,
        "names": names
    }

    html = HTML.replace(
        "__CFG__",
        json.dumps(
            config,
            ensure_ascii=False
        )
    )

    target = folder / "annotator.html"

    target.write_text(
        html,
        encoding="utf-8"
    )

    return len(names)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="标注包根目录"
    )

    ap.add_argument(
        "--device",
        default=None,
        help="只更新指定设备目录"
    )

    args = ap.parse_args()

    if not args.root.exists():
        raise SystemExit(
            f"目录不存在：{args.root}"
        )

    folders = sorted(
        p for p in args.root.iterdir()
        if p.is_dir()
    )

    if args.device:
        folders = [
            p for p in folders
            if p.name == args.device
        ]

        if not folders:
            raise SystemExit(
                f"找不到设备目录：{args.device}"
            )

    updated = 0

    for folder in folders:
        # 只处理真正的设备包目录
        if not (folder / "images").is_dir():
            continue

        class_count = update_one(folder)

        print(
            f"[OK] {folder.name} "
            f"| classes={class_count} "
            f"| {folder/'annotator.html'}"
        )

        updated += 1

    print()
    print(f"完成，共更新 {updated} 个 annotator.html")
    print("没有重新抽帧，没有修改 images。")


if __name__ == "__main__":
    main()

