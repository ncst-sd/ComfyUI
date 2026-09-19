#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, re, shutil, zipfile
from collections import defaultdict
from pathlib import Path
import yaml

IMG_EXTS={'.jpg','.jpeg','.png','.bmp','.webp'}

HTML=r'''<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8"><title>设备指示灯离线标注</title>
<style>body{font-family:Arial,"Microsoft YaHei",sans-serif;margin:0;background:#f2f2f2}#top{padding:10px;background:#222;color:#fff;display:flex;gap:8px;align-items:center;flex-wrap:wrap}button,input{font-size:14px}button{padding:6px 10px;cursor:pointer}#wrap{display:flex;height:calc(100vh - 58px)}#left{width:350px;background:#fff;border-right:1px solid #bbb;overflow:auto;padding:10px}#main{flex:1;display:flex;align-items:center;justify-content:center;overflow:auto;background:#777}canvas{background:#111;cursor:crosshair}.cls{padding:7px;margin:3px 0;border:1px solid #ccc;cursor:pointer;border-radius:4px}.cls.active{background:#d9edf7;border-color:#31708f;font-weight:bold}.item{font-size:12px;padding:3px 4px;border-bottom:1px solid #eee;cursor:pointer;word-break:break-all}.item.active{background:#e8f4ff}.item.done::before{content:"✓ ";color:green;font-weight:bold}.note{background:#fff7d6;border:1px solid #e0cc7b;padding:8px;font-size:13px;line-height:1.5;margin:8px 0}.small{font-size:12px;color:#555;line-height:1.5}#newClass{width:220px}</style></head><body>
<div id="top"><input id="files" type="file" accept="image/*" multiple><button onclick="prevImage()">上一张 A</button><button onclick="nextImage()">下一张 D</button><button onclick="undo()">撤销 Z</button><button onclick="clearBoxes()">清空 C</button><button onclick="exportJSON()">导出标注 JSON</button><span id="status"></span></div>
<div id="wrap"><div id="left"><b>设备：</b><div id="deviceName"></div><div class="note">先选择正确的指示灯类别，再框选灯本体。若没有预置类别，请由熟悉设备的标注员自行新增。</div><b>指示灯类别</b><div id="classes"></div><div style="margin:8px 0"><input id="newClass" placeholder="输入新类别名称，例如 PWR 电源灯"><button onclick="addClass()">新增类别</button></div><hr><div class="small">数字键0~9选类别；A上一张；D下一张；Z撤销；C清空。全部标完后统一导出 JSON。</div><hr><div id="list"></div></div><div id="main"><canvas id="canvas"></canvas></div></div>
<script>
const CONFIG=__CONFIG__,filesInput=document.getElementById('files'),canvas=document.getElementById('canvas'),ctx=canvas.getContext('2d'),list=document.getElementById('list'),classesDiv=document.getElementById('classes'),statusEl=document.getElementById('status');document.getElementById('deviceName').textContent=CONFIG.device_name;let names=Object.assign({},CONFIG.names),files=[],index=0,img=new Image(),annotations={},scale=1,classId=Object.keys(names).length?Number(Object.keys(names)[0]):null,drag=false,sx=0,sy=0,ex=0,ey=0,imageW=0,imageH=0;
function current(){if(!files.length)return[];const n=files[index].name;if(!annotations[n])annotations[n]=[];return annotations[n]}
function buildClasses(){classesDiv.innerHTML='';Object.keys(names).sort((a,b)=>Number(a)-Number(b)).forEach(k=>{const d=document.createElement('div');d.className='cls'+(Number(k)===classId?' active':'');d.textContent=`${k}: ${names[k]}`;d.onclick=()=>{classId=Number(k);buildClasses();draw()};classesDiv.appendChild(d)});if(!Object.keys(names).length)classesDiv.innerHTML='<div class="small">当前没有预置类别，请先新增类别。</div>'}
function addClass(){const i=document.getElementById('newClass'),n=i.value.trim();if(!n)return;let id=0;while(names[id]!==undefined)id++;names[id]=n;classId=id;i.value='';buildClasses();draw()}
function buildList(){list.innerHTML='';files.forEach((f,i)=>{const d=document.createElement('div');d.className='item';if(i===index)d.classList.add('active');if((annotations[f.name]||[]).length)d.classList.add('done');d.textContent=`${i+1}. ${f.name}`;d.onclick=()=>{index=i;buildList();loadImage()};list.appendChild(d)})}
filesInput.addEventListener('change',e=>{files=[...e.target.files].sort((a,b)=>a.name.localeCompare(b.name));index=0;buildList();loadImage()});
function loadImage(){if(!files.length)return;const f=files[index],url=URL.createObjectURL(f);img=new Image();img.onload=()=>{imageW=img.naturalWidth;imageH=img.naturalHeight;const mw=Math.max(700,innerWidth-420),mh=Math.max(500,innerHeight-100);scale=Math.min(mw/imageW,mh/imageH,1);canvas.width=Math.round(imageW*scale);canvas.height=Math.round(imageH*scale);draw();URL.revokeObjectURL(url)};img.src=url}
function draw(){if(!files.length||!img.complete)return;ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(img,0,0,canvas.width,canvas.height);ctx.lineWidth=2;ctx.strokeStyle='#00ff00';ctx.fillStyle='#00ff00';ctx.font='14px sans-serif';current().forEach(b=>{ctx.strokeRect(b.x*scale,b.y*scale,b.w*scale,b.h*scale);ctx.fillText(`${b.class_id}:${b.class_name}`,b.x*scale+2,Math.max(14,b.y*scale-3))});if(drag){ctx.strokeStyle='#ffff00';ctx.strokeRect(sx,sy,ex-sx,ey-sy)}updateStatus()}
canvas.addEventListener('mousedown',e=>{if(!files.length)return;if(classId===null||names[classId]===undefined){alert('请先选择或新增一个指示灯类别');return}const r=canvas.getBoundingClientRect();sx=e.clientX-r.left;sy=e.clientY-r.top;ex=sx;ey=sy;drag=true;draw()});canvas.addEventListener('mousemove',e=>{if(!drag)return;const r=canvas.getBoundingClientRect();ex=e.clientX-r.left;ey=e.clientY-r.top;draw()});canvas.addEventListener('mouseup',e=>{if(!drag)return;drag=false;const r=canvas.getBoundingClientRect();ex=e.clientX-r.left;ey=e.clientY-r.top;let x1=Math.min(sx,ex)/scale,y1=Math.min(sy,ey)/scale,x2=Math.max(sx,ex)/scale,y2=Math.max(sy,ey)/scale;if(x2-x1>=3&&y2-y1>=3)current().push({class_id:classId,class_name:names[classId],x:x1,y:y1,w:x2-x1,h:y2-y1});buildList();draw()});
function undo(){const b=current();if(b.length){b.pop();buildList();draw()}}function clearBoxes(){if(!files.length)return;if(confirm('清空当前图片全部标注？')){annotations[files[index].name]=[];buildList();draw()}}function prevImage(){if(index>0){index--;buildList();loadImage()}}function nextImage(){if(index<files.length-1){index++;buildList();loadImage()}}function updateStatus(){if(!files.length){statusEl.textContent='请选择 images 目录中的全部图片';return}const done=files.filter(f=>(annotations[f.name]||[]).length>0).length,cls=(classId!==null&&names[classId]!==undefined)?`${classId}:${names[classId]}`:'未选择类别';statusEl.textContent=`${index+1}/${files.length} | ${cls} | 已标图片 ${done}`}
function exportJSON(){if(!files.length){alert('请先导入图片');return}const out={format:'semantic_indicator_bbox_v2',device_name:CONFIG.device_name,names:names,annotations:annotations};const blob=new Blob([JSON.stringify(out,null,2)],{type:'application/json'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='indicator_annotations.json';a.click()}
document.addEventListener('keydown',e=>{if(/^[0-9]$/.test(e.key)&&names[e.key]!==undefined){classId=Number(e.key);buildClasses();draw()}else if(e.key==='a'||e.key==='A')prevImage();else if(e.key==='d'||e.key==='D')nextImage();else if(e.key==='z'||e.key==='Z')undo();else if(e.key==='c'||e.key==='C')clearBoxes()});buildClasses();
</script></body></html>'''

README='''设备指示灯离线标注\n\n1. 解压 ZIP。\n2. 双击 annotator.html。\n3. 点击“选择文件”，进入 images 文件夹，全选全部图片。\n4. 选择正确的指示灯类别。\n5. 鼠标框住对应指示灯本体。\n6. 一张标完后按 D 到下一张，不需要每张单独保存。\n7. 全部完成后点击“导出标注 JSON”。\n8. 把 indicator_annotations.json 发回。\n\n如果左侧没有正确类别，请由熟悉设备的标注员自行新增。\n'''

def load_yaml(p):
    return yaml.safe_load(p.read_text(encoding='utf-8')) or {} if p.exists() else {}

def group_of(p):
    m=re.search(r'(.+?__video_\d+)',p.stem)
    return m.group(1) if m else (p.parent.name or p.stem)

def sample_balanced(images,target):
    groups=defaultdict(list)
    for p in images: groups[group_of(p)].append(p)
    for g in groups: groups[g]=sorted(groups[g])
    selected=[]; chosen=set(); names=sorted(groups); r=0
    while len(selected)<target:
        added=0
        for g in names:
            arr=groups[g]
            if r>=len(arr): continue
            idx=min(len(arr)-1, round(r*(len(arr)-1)/max(1,min(len(arr),target)-1))) if len(arr)>1 else 0
            p=arr[idx]
            if p not in chosen:
                selected.append(p); chosen.add(p); added+=1
                if len(selected)>=target: break
        if not added: break
        r+=1
    if len(selected)<target:
        for p in images:
            if p not in chosen:
                selected.append(p); chosen.add(p)
                if len(selected)>=target: break
    return selected[:target]

def verified_indicators(workspace_name,aliases,rules):
    for info in (aliases.get('devices') or {}).values():
        if info.get('verified') is not True: continue
        if workspace_name not in (info.get('workspace_names') or []): continue
        rule_name=info.get('rule_name') or info.get('canonical_name')
        dev=(rules.get('devices') or {}).get(rule_name,{})
        inds=list((dev.get('indicators') or {}).keys())
        if inds: return inds,rule_name
    return [],''

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--project',type=Path,default=Path.home()/'yolo_device_monitor')
    ap.add_argument('--count',type=int,default=200)
    ap.add_argument('--exclude',default='数据网柜_H3C_SR6608_华三')
    a=ap.parse_args()
    project=a.project.expanduser().resolve(); ws=project/'workspace'
    aliases=load_yaml(project/'rules'/'device_alias_map.yaml'); rules=load_yaml(project/'rules'/'device_indicator_rules.yaml')
    out=project/'annotation_packages_all'; out.mkdir(parents=True,exist_ok=True)
    excludes={x.strip() for x in a.exclude.split(',') if x.strip()}
    rows=[]
    for d in sorted(p for p in ws.iterdir() if p.is_dir() and not p.name.startswith('_') and p.name not in excludes):
        frames=d/'extracted_frames'
        images=sorted(p for p in frames.rglob('*') if p.is_file() and p.suffix.lower() in IMG_EXTS) if frames.exists() else []
        if not images:
            print('[SKIP]',d.name,'无 extracted_frames 图片'); rows.append((d.name,0,0,'无图片','')); continue
        selected=sample_balanced(images,min(a.count,len(images)))
        inds,rule_name=verified_indicators(d.name,aliases,rules)
        package=out/d.name
        if package.exists(): shutil.rmtree(package)
        (package/'images').mkdir(parents=True)
        config={'device_name':d.name,'names':{str(i):v for i,v in enumerate(inds)}}
        (package/'annotator.html').write_text(HTML.replace('__CONFIG__',json.dumps(config,ensure_ascii=False)),encoding='utf-8')
        (package/'README.txt').write_text(README,encoding='utf-8')
        if inds:
            (package/'classes.txt').write_text('\n'.join(f'{i}: {v}' for i,v in enumerate(inds)),encoding='utf-8')
            status='预置语义类别'
        else:
            (package/'classes.txt').write_text('未预置类别，请由熟悉设备的标注员在网页中自行新增。\n',encoding='utf-8')
            status='类别由标注员新增'
        for i,src in enumerate(selected,1): shutil.copy2(src,package/'images'/f'{i:04d}__{src.name}')
        zpath=out/f'{d.name}.zip'
        if zpath.exists(): zpath.unlink()
        with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as z:
            for f in package.rglob('*'):
                if f.is_file(): z.write(f,f.relative_to(out))
        print(f'[OK] {d.name}: {len(selected)} 张 | {status}')
        rows.append((d.name,len(images),len(selected),status,rule_name))
    with (out/'export_summary.csv').open('w',encoding='utf-8-sig') as f:
        f.write('device_name,total_frames,exported,status,rule_name\n')
        for r in rows:
            vals=[str(x).replace('"','""') for x in r]
            f.write(','.join('"'+x+'"' for x in vals)+'\n')
    print('\n完成，输出目录:',out)

if __name__=='__main__': main()

