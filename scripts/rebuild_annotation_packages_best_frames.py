#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, csv, json, math, re, shutil, zipfile
from pathlib import Path
import cv2

VIDEO_EXTS={'.mp4','.avi','.mov','.mkv','.m4v','.ts','.mts','.m2ts','.wmv','.flv'}
DEFAULT_SOURCE_ROOT=Path.home()/ '桌面' / '数字孪生机房巡检识别模型测试视频'
DEFAULT_OLD_PACKAGES=Path.home()/ 'yolo_device_monitor' / 'annotation_packages_all'
DEFAULT_OUTPUT_ROOT=Path.home()/ 'yolo_device_monitor' / 'annotation_packages_best_frames'

ANNOTATOR_HTML="""<!DOCTYPE html>
<html lang=\"zh-CN\"><head><meta charset=\"UTF-8\"><title>设备指示灯离线标注</title>
<style>
body{margin:0;font-family:Arial,\"Microsoft YaHei\",sans-serif;background:#111;color:#eee;overflow:hidden}
#top{height:56px;box-sizing:border-box;padding:8px 12px;background:#1b1b1b;display:flex;gap:8px;align-items:center;border-bottom:1px solid #333}
button,input{font-size:14px}button{padding:7px 10px;cursor:pointer}#wrap{display:flex;height:calc(100vh - 56px)}
#left{width:370px;background:#1b1b1b;border-right:1px solid #333;overflow:auto;padding:10px;box-sizing:border-box}
#main{flex:1;position:relative;background:#222;overflow:hidden}canvas{position:absolute;left:0;top:0;background:#000;cursor:crosshair}
.cls{padding:7px;margin:3px 0;border:1px solid #555;cursor:pointer;border-radius:4px}.cls.active{background:#245d78;border-color:#62b7df;font-weight:bold}
.item{font-size:12px;padding:3px 4px;border-bottom:1px solid #333;cursor:pointer;word-break:break-all}.item.active{background:#245d78}.item.done::before{content:\"✓ \";color:#74d680;font-weight:bold}
.note{background:#493f16;border:1px solid #7d6a25;padding:8px;font-size:13px;line-height:1.55;margin:8px 0}.small{font-size:12px;color:#aaa;line-height:1.5}
#status{margin-left:auto;font-size:13px;color:#ccc}#zoomLabel{min-width:62px;text-align:center}
</style></head><body>
<div id=\"top\"><input id=\"files\" type=\"file\" accept=\"image/*\" multiple><button onclick=\"prevImage()\">上一张 A</button><button onclick=\"nextImage()\">下一张 D</button><button onclick=\"undo()\">撤销 Z</button><button onclick=\"clearBoxes()\">清空 C</button><button onclick=\"zoomOut()\">－</button><span id=\"zoomLabel\">100%</span><button onclick=\"zoomIn()\">＋</button><button onclick=\"fitImage()\">适应窗口 F</button><button onclick=\"resetZoom()\">100% R</button><button onclick=\"exportJSON()\">导出标注 JSON</button><span id=\"status\">请选择图片</span></div>
<div id=\"wrap\"><div id=\"left\"><b>设备：</b><div id=\"deviceName\"></div><div class=\"note\"><b>放大 / 缩小：</b>鼠标滚轮<br><b>平移：</b>按住 Space + 左键拖动<br><b>画框：</b>左键拖动<br><b>建议：</b>先把指示灯放大到能看清，再画框。</div><b>指示灯类别</b><div id=\"classes\"></div><div style=\"margin:8px 0\"><input id=\"newClass\" placeholder=\"输入新类别名称\" style=\"width:250px\"><button onclick=\"addClass()\">新增</button></div><hr><div class=\"small\">数字键 0~9：选择类别<br>A / D：上一张 / 下一张<br>Z：撤销　C：清空<br>F：适应窗口　R：100%</div><hr><div id=\"list\"></div></div><div id=\"main\"><canvas id=\"canvas\"></canvas></div></div>
<script>
const CONFIG=__CONFIG__;const filesInput=document.getElementById('files'),canvas=document.getElementById('canvas'),ctx=canvas.getContext('2d'),main=document.getElementById('main'),list=document.getElementById('list'),classesDiv=document.getElementById('classes'),statusEl=document.getElementById('status'),zoomLabel=document.getElementById('zoomLabel');document.getElementById('deviceName').textContent=CONFIG.device_name;
let names=Object.assign({},CONFIG.names),files=[],index=0,img=new Image(),annotations={},classId=Object.keys(names).length?Number(Object.keys(names)[0]):null,zoom=1,panX=0,panY=0,drawing=false,panning=false,spaceDown=false,start={x:0,y:0},cur={x:0,y:0},panStart={x:0,y:0},panOrigin={x:0,y:0};
function current(){if(!files.length)return[];const n=files[index].name;if(!annotations[n])annotations[n]=[];return annotations[n]}
function buildClasses(){classesDiv.innerHTML='';Object.keys(names).sort((a,b)=>Number(a)-Number(b)).forEach(k=>{const d=document.createElement('div');d.className='cls'+(Number(k)===classId?' active':'');d.textContent=`${k}: ${names[k]}`;d.onclick=()=>{classId=Number(k);buildClasses();draw()};classesDiv.appendChild(d)});if(!Object.keys(names).length)classesDiv.innerHTML='<div class=\"small\">请先新增类别。</div>'}
function addClass(){const i=document.getElementById('newClass'),name=i.value.trim();if(!name)return;let id=0;while(names[id]!==undefined)id++;names[id]=name;classId=id;i.value='';buildClasses();draw()}
filesInput.addEventListener('change',e=>{files=[...e.target.files].sort((a,b)=>a.name.localeCompare(b.name));index=0;buildList();loadImage()});
function buildList(){list.innerHTML='';files.forEach((f,i)=>{const d=document.createElement('div');d.className='item';if(i===index)d.classList.add('active');if((annotations[f.name]||[]).length)d.classList.add('done');d.textContent=`${i+1}. ${f.name}`;d.onclick=()=>{index=i;buildList();loadImage()};list.appendChild(d)})}
function resizeCanvas(){canvas.width=main.clientWidth;canvas.height=main.clientHeight}
function loadImage(){if(!files.length)return;const u=URL.createObjectURL(files[index]);img=new Image();img.onload=()=>{fitImage();URL.revokeObjectURL(u)};img.src=u}
function fitImage(){if(!img.naturalWidth)return;resizeCanvas();zoom=Math.min(canvas.width/img.naturalWidth,canvas.height/img.naturalHeight);panX=(canvas.width-img.naturalWidth*zoom)/2;panY=(canvas.height-img.naturalHeight*zoom)/2;draw()}
function resetZoom(){if(!img.naturalWidth)return;resizeCanvas();zoom=1;panX=(canvas.width-img.naturalWidth)/2;panY=(canvas.height-img.naturalHeight)/2;draw()}
function zoomAt(factor,cx,cy){const old=zoom;zoom=Math.max(.1,Math.min(20,zoom*factor));const ix=(cx-panX)/old,iy=(cy-panY)/old;panX=cx-ix*zoom;panY=cy-iy*zoom;draw()}
function zoomIn(){zoomAt(1.25,canvas.width/2,canvas.height/2)}function zoomOut(){zoomAt(.8,canvas.width/2,canvas.height/2)}
canvas.addEventListener('wheel',e=>{e.preventDefault();const r=canvas.getBoundingClientRect();zoomAt(e.deltaY<0?1.2:.833,e.clientX-r.left,e.clientY-r.top)},{passive:false});
function toImage(x,y){return{x:(x-panX)/zoom,y:(y-panY)/zoom}}
function draw(){if(!img.complete||!img.naturalWidth)return;resizeCanvas();ctx.clearRect(0,0,canvas.width,canvas.height);ctx.imageSmoothingEnabled=true;ctx.imageSmoothingQuality='high';ctx.drawImage(img,panX,panY,img.naturalWidth*zoom,img.naturalHeight*zoom);ctx.lineWidth=2;ctx.strokeStyle='#00ff00';ctx.fillStyle='#00ff00';ctx.font='14px sans-serif';current().forEach(b=>{const x=panX+b.x*zoom,y=panY+b.y*zoom,w=b.w*zoom,h=b.h*zoom;ctx.strokeRect(x,y,w,h);ctx.fillText(`${b.class_id}:${b.class_name}`,x+2,Math.max(14,y-3))});if(drawing){ctx.strokeStyle='#ffff00';ctx.strokeRect(Math.min(start.x,cur.x),Math.min(start.y,cur.y),Math.abs(cur.x-start.x),Math.abs(cur.y-start.y))}zoomLabel.textContent=Math.round(zoom*100)+'%';updateStatus()}
canvas.addEventListener('mousedown',e=>{const r=canvas.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top;if(spaceDown||e.button===1){panning=true;panStart={x,y};panOrigin={x:panX,y:panY};canvas.style.cursor='grabbing';return}if(classId===null||names[classId]===undefined){alert('请先选择或新增类别');return}drawing=true;start={x,y};cur={x,y}});
canvas.addEventListener('mousemove',e=>{const r=canvas.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top;if(panning){panX=panOrigin.x+(x-panStart.x);panY=panOrigin.y+(y-panStart.y);draw();return}if(drawing){cur={x,y};draw()}});
canvas.addEventListener('mouseup',()=>{if(panning){panning=false;canvas.style.cursor=spaceDown?'grab':'crosshair';return}if(!drawing)return;drawing=false;const a=toImage(start.x,start.y),b=toImage(cur.x,cur.y);let x1=Math.max(0,Math.min(img.naturalWidth,Math.min(a.x,b.x))),y1=Math.max(0,Math.min(img.naturalHeight,Math.min(a.y,b.y))),x2=Math.max(0,Math.min(img.naturalWidth,Math.max(a.x,b.x))),y2=Math.max(0,Math.min(img.naturalHeight,Math.max(a.y,b.y)));if(x2-x1>=2&&y2-y1>=2)current().push({class_id:classId,class_name:names[classId],x:x1,y:y1,w:x2-x1,h:y2-y1});buildList();draw()});
function undo(){const b=current();if(b.length){b.pop();buildList();draw()}}function clearBoxes(){if(files.length&&confirm('清空当前图片全部标注？')){annotations[files[index].name]=[];buildList();draw()}}function prevImage(){if(index>0){index--;buildList();loadImage()}}function nextImage(){if(index<files.length-1){index++;buildList();loadImage()}}
function updateStatus(){if(!files.length){statusEl.textContent='请选择图片';return}const done=files.filter(f=>(annotations[f.name]||[]).length>0).length,cls=(classId!==null&&names[classId]!==undefined)?`${classId}:${names[classId]}`:'未选择类别';statusEl.textContent=`${index+1}/${files.length} | ${cls} | 已标 ${done}`}
function exportJSON(){if(!files.length){alert('请先选择图片');return}const out={format:'semantic_indicator_bbox_v3',device_name:CONFIG.device_name,names,annotations},blob=new Blob([JSON.stringify(out,null,2)],{type:'application/json'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='indicator_annotations.json';a.click()}
document.addEventListener('keydown',e=>{if(e.code==='Space'){spaceDown=true;canvas.style.cursor='grab';e.preventDefault();return}if(/^[0-9]$/.test(e.key)&&names[e.key]!==undefined){classId=Number(e.key);buildClasses();draw()}else if(e.key==='a'||e.key==='A')prevImage();else if(e.key==='d'||e.key==='D')nextImage();else if(e.key==='z'||e.key==='Z')undo();else if(e.key==='c'||e.key==='C')clearBoxes();else if(e.key==='f'||e.key==='F')fitImage();else if(e.key==='r'||e.key==='R')resetZoom()});document.addEventListener('keyup',e=>{if(e.code==='Space'){spaceDown=false;canvas.style.cursor='crosshair'}});window.addEventListener('resize',()=>{if(img.naturalWidth)fitImage()});buildClasses();
</script></body></html>"""

def norm_name(s): return re.sub(r'[^0-9a-zA-Z\u4e00-\u9fff]+','',s).lower()

def load_class_names(old_package):
    names={}
    p=old_package/'classes.txt'
    if p.exists():
        for line in p.read_text(encoding='utf-8',errors='ignore').splitlines():
            if ':' in line:
                k,v=line.split(':',1)
                if k.strip().isdigit(): names[k.strip()]=v.strip()
    return names

def find_device_folder(source_root, device_name):
    direct=source_root/device_name
    if direct.is_dir(): return direct
    target=norm_name(device_name); c=[]
    for p in source_root.iterdir():
        if not p.is_dir(): continue
        n=norm_name(p.name)
        if n==target: return p
        if target in n or n in target: c.append(p)
    return c[0] if len(c)==1 else None

def video_info(path):
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened(): return None
    fps=float(cap.get(cv2.CAP_PROP_FPS) or 0); frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0); height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration=frames/fps if fps>0 and frames>0 else 0; cap.release()
    return None if duration<=0 else {'fps':fps,'frames':frames,'duration':duration,'width':width,'height':height}

def sharpness_scores(frame):
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY); full=float(cv2.Laplacian(gray,cv2.CV_64F).var())
    h,w=gray.shape; crop=gray[int(h*.15):int(h*.85),int(w*.15):int(w*.85)]
    center=float(cv2.Laplacian(crop,cv2.CV_64F).var()); return full,center,center*.8+full*.2

def best_near_time(video_path, anchor, duration, search_window, candidates):
    cap=cv2.VideoCapture(str(video_path)); best=None
    if not cap.isOpened(): return None
    start=max(0,anchor-search_window); end=min(max(0,duration-.001),anchor+search_window)
    times=[anchor] if candidates<=1 or end<=start else [start+i*(end-start)/(candidates-1) for i in range(candidates)]
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC,t*1000); ok,frame=cap.read()
        if not ok or frame is None: continue
        frame_no=int(cap.get(cv2.CAP_PROP_POS_FRAMES))-1; full,center,combined=sharpness_scores(frame)
        item={'frame':frame,'time':t,'frame_no':frame_no,'full_score':full,'center_score':center,'combined_score':combined}
        if best is None or combined>best['combined_score']: best=item
    cap.release(); return best

def allocate_counts(rows,max_images):
    n=len(rows)
    if not n:return []
    if max_images<=n:
        counts=[0]*n
        for i in sorted(range(n),key=lambda i:rows[i]['duration'],reverse=True)[:max_images]: counts[i]=1
        return counts
    counts=[1]*n; rem=max_images-n; total=sum(v['duration'] for v in rows); fr=[]; added=0
    for i,v in enumerate(rows):
        share=rem*(v['duration']/total) if total else rem/n; base=int(math.floor(share)); counts[i]+=base; added+=base; fr.append((share-base,i))
    for _,i in sorted(fr,reverse=True)[:rem-added]: counts[i]+=1
    return counts

def safe_filename(s): return re.sub(r'[\\/:*?\"<>|]+','_',s)

def process_device(device_name,source_dir,old_package,out_root,args):
    videos=[]
    for p in sorted(source_dir.rglob('*')):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            info=video_info(p)
            if info: info['path']=p; videos.append(info)
    if not videos: print('  [SKIP] 无可读视频'); return None
    counts=allocate_counts(videos,args.max_images); package=out_root/device_name
    if package.exists(): shutil.rmtree(package)
    images=package/'images'; images.mkdir(parents=True)
    names=load_class_names(old_package)
    if names: (package/'classes.txt').write_text('\n'.join(f'{k}: {names[k]}' for k in sorted(names,key=lambda x:int(x)))+'\n',encoding='utf-8')
    cfg={'device_name':device_name,'names':names}
    (package/'annotator.html').write_text(ANNOTATOR_HTML.replace('__CONFIG__',json.dumps(cfg,ensure_ascii=False)),encoding='utf-8')
    cand=[]
    for vi,(v,count) in enumerate(zip(videos,counts),1):
        if count<=0: continue
        print(f"  video {vi:03d}/{len(videos):03d} {v['path'].name} targets={count}")
        for ai in range(count):
            anchor=v['duration']*(ai+.5)/count
            b=best_near_time(v['path'],anchor,v['duration'],args.search_window,args.candidates)
            if b:
                b.update(video_idx=vi,video_path=v['path'],anchor=anchor,width=v['width'],height=v['height']); cand.append(b)
    good=[x for x in cand if x['center_score']>=args.min_center_score]; bad=[x for x in cand if x['center_score']<args.min_center_score]
    selected=good[:args.max_images]
    if args.fill_low_quality and len(selected)<args.max_images:
        bad.sort(key=lambda x:x['combined_score'],reverse=True); selected+=bad[:args.max_images-len(selected)]
    selected.sort(key=lambda x:(x['video_idx'],x['time']))
    rows=[]
    for gi,item in enumerate(selected,1):
        fname=f"{gi:04d}__{safe_filename(device_name)}__video_{item['video_idx']:03d}__t{item['time']:09.3f}__f{max(0,item['frame_no']):08d}.jpg"
        out=images/fname
        if not cv2.imwrite(str(out),item['frame'],[int(cv2.IMWRITE_JPEG_QUALITY),args.jpeg_quality]): continue
        rows.append({'image':fname,'source_video':str(item['video_path']),'video_index':item['video_idx'],'anchor_time_sec':round(item['anchor'],3),'selected_time_sec':round(item['time'],3),'frame_no':item['frame_no'],'width':item['width'],'height':item['height'],'full_sharpness':round(item['full_score'],2),'center_sharpness':round(item['center_score'],2),'combined_sharpness':round(item['combined_score'],2),'quality_pass':int(item['center_score']>=args.min_center_score)})
    if rows:
        with (package/'frame_quality.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    (package/'README_标注说明.txt').write_text(f'''设备：{device_name}\n\n清晰帧优选版：\n- 原始分辨率保存\n- 每个目标时间点在 ±{args.search_window:.2f}s 内搜索 {args.candidates} 张候选帧\n- 中央区域清晰度阈值：{args.min_center_score}\n- JPEG质量：{args.jpeg_quality}\n- 导出图片：{len(rows)}\n\n网页操作：\n- 鼠标滚轮：缩放\n- Space + 左键：平移\n- 左键拖动：画框\n- F：适应窗口\n- R：100%\n''',encoding='utf-8')
    zpath=out_root/f'{device_name}.zip'
    if zpath.exists(): zpath.unlink()
    with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as z:
        for f in package.rglob('*'):
            if f.is_file(): z.write(f,f.relative_to(out_root))
    avg=sum(r['center_sharpness'] for r in rows)/len(rows) if rows else 0
    return {'device_name':device_name,'status':'ok','videos':len(videos),'candidate_frames':len(cand),'good_candidates':len(good),'exported':len(rows),'avg_center_sharpness':round(avg,2),'zip':str(zpath)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source-root',type=Path,default=DEFAULT_SOURCE_ROOT); ap.add_argument('--old-packages',type=Path,default=DEFAULT_OLD_PACKAGES); ap.add_argument('--output-root',type=Path,default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument('--max-images',type=int,default=200); ap.add_argument('--search-window',type=float,default=.5); ap.add_argument('--candidates',type=int,default=11); ap.add_argument('--min-center-score',type=float,default=100.0); ap.add_argument('--jpeg-quality',type=int,default=98); ap.add_argument('--fill-low-quality',action='store_true'); ap.add_argument('--device',default=None)
    args=ap.parse_args(); args.output_root.mkdir(parents=True,exist_ok=True)
    old_dirs=sorted(p for p in args.old_packages.iterdir() if p.is_dir())
    if args.device: old_dirs=[p for p in old_dirs if p.name==args.device]
    results=[]
    for i,old in enumerate(old_dirs,1):
        print(f'\n[{i}/{len(old_dirs)}] {old.name}')
        src=find_device_folder(args.source_root,old.name)
        if src is None:
            print('  [SKIP] 无法唯一匹配原始设备目录'); results.append({'device_name':old.name,'status':'source_folder_not_found'}); continue
        print('  source:',src); r=process_device(old.name,src,old,args.output_root,args)
        if r: results.append(r); print(f"  [OK] exported={r['exported']} good={r['good_candidates']} avg_center={r['avg_center_sharpness']}")
    fields=['device_name','status','videos','candidate_frames','good_candidates','exported','avg_center_sharpness','zip']
    with (args.output_root/'rebuild_summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader(); w.writerows(results)
    print('\n完成：',args.output_root)

if __name__=='__main__': main()

