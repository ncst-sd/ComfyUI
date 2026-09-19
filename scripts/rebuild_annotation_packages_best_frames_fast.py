#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, csv, json, math, re, shutil, zipfile
from pathlib import Path
import cv2

VIDEO_EXTS={'.mp4','.avi','.mov','.mkv','.m4v','.ts','.mts','.m2ts','.wmv','.flv'}
IMAGE_EXTS={'.jpg','.jpeg','.png','.bmp','.webp','.tif','.tiff'}
SRC=Path.home()/'桌面'/'数字孪生机房巡检识别模型测试视频'
OLD=Path.home()/'yolo_device_monitor'/'annotation_packages_all'
OUT=Path.home()/'yolo_device_monitor'/'annotation_packages_best_frames_fast'

HTML=r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>指示灯标注</title>
<style>
body{margin:0;background:#111;color:#eee;font-family:Arial,"Microsoft YaHei";overflow:hidden}#top{height:54px;display:flex;align-items:center;gap:7px;padding:0 10px;background:#1b1b1b}#wrap{display:flex;height:calc(100vh - 54px)}#left{width:360px;padding:10px;overflow:auto;background:#1b1b1b}#main{flex:1;position:relative;overflow:hidden;background:#222}canvas{position:absolute;left:0;top:0;background:#000;cursor:crosshair}button,input{font-size:14px}button{padding:6px 9px}.cls,.item{padding:6px;border-bottom:1px solid #333;cursor:pointer}.cls.active,.item.active{background:#245d78}.item.done:before{content:"✓ ";color:#7fe18a}.note{padding:8px;margin:8px 0;background:#493f16;line-height:1.5}.small{font-size:12px;color:#aaa}#status{margin-left:auto;font-size:13px}
</style></head><body>
<div id="top"><input id="files" type="file" accept="image/*" multiple><button onclick="prev()">上一张 A</button><button onclick="next()">下一张 D</button><button onclick="undo()">撤销 Z</button><button onclick="clearNow()">清空 C</button><button onclick="zoomAt(0.8,W/2,H/2)">－</button><span id="zl">100%</span><button onclick="zoomAt(1.25,W/2,H/2)">＋</button><button onclick="fit()">适应 F</button><button onclick="reset()">100% R</button><button onclick="exportJSON()">导出 JSON</button><span id="status"></span></div>
<div id="wrap"><div id="left"><b>设备：</b><div id="dev"></div><div class="note"><b>滚轮：</b>放大/缩小<br><b>Space+左键：</b>平移<br><b>左键拖动：</b>画框<br>建议先放大灯区域再标。</div><b>类别</b><div id="classes"></div><div><input id="newc" placeholder="新类别"><button onclick="addc()">新增</button></div><hr><div class="small">0~9 选类别；A/D 切图；Z 撤销；C 清空；F 适应；R 100%</div><hr><div id="list"></div></div><div id="main"><canvas id="c"></canvas></div></div>
<script>
const CFG=__CFG__,c=document.getElementById('c'),x=c.getContext('2d'),main=document.getElementById('main');const filesEl=document.getElementById('files'),classesEl=document.getElementById('classes'),listEl=document.getElementById('list');document.getElementById('dev').textContent=CFG.device_name;let names={...CFG.names},files=[],idx=0,img=new Image(),ann={},cid=Object.keys(names).length?+Object.keys(names)[0]:null;let z=1,px=0,py=0,draw=false,pan=false,space=false,sx=0,sy=0,cx=0,cy=0,opx=0,opy=0,psx=0,psy=0;let W=0,H=0;
function boxes(){if(!files.length)return[];let n=files[idx].name;return ann[n]||(ann[n]=[])}function resize(){c.width=W=main.clientWidth;c.height=H=main.clientHeight}function classes(){classesEl.innerHTML='';Object.keys(names).sort((a,b)=>+a-+b).forEach(k=>{let d=document.createElement('div');d.className='cls'+(+k===cid?' active':'');d.textContent=k+': '+names[k];d.onclick=()=>{cid=+k;classes();paint()};classesEl.appendChild(d)})}function addc(){let e=document.getElementById('newc'),v=e.value.trim();if(!v)return;let k=0;while(names[k]!==undefined)k++;names[k]=v;cid=k;e.value='';classes();paint()}function buildList(){listEl.innerHTML='';files.forEach((f,i)=>{let d=document.createElement('div');d.className='item'+(i===idx?' active':'')+((ann[f.name]||[]).length?' done':'');d.textContent=(i+1)+'. '+f.name;d.onclick=()=>{idx=i;buildList();load()};listEl.appendChild(d)})}
filesEl.onchange=e=>{files=[...e.target.files].sort((a,b)=>a.name.localeCompare(b.name));idx=0;buildList();load()};function load(){if(!files.length)return;let u=URL.createObjectURL(files[idx]);img=new Image();img.onload=()=>{fit();URL.revokeObjectURL(u)};img.src=u}function fit(){if(!img.naturalWidth)return;resize();z=Math.min(W/img.naturalWidth,H/img.naturalHeight);px=(W-img.naturalWidth*z)/2;py=(H-img.naturalHeight*z)/2;paint()}function reset(){if(!img.naturalWidth)return;resize();z=1;px=(W-img.naturalWidth)/2;py=(H-img.naturalHeight)/2;paint()}function zoomAt(f,qx,qy){let old=z;z=Math.max(.1,Math.min(20,z*f));let ix=(qx-px)/old,iy=(qy-py)/old;px=qx-ix*z;py=qy-iy*z;paint()}c.onwheel=e=>{e.preventDefault();let r=c.getBoundingClientRect();zoomAt(e.deltaY<0?1.2:.833,e.clientX-r.left,e.clientY-r.top)};function toImg(a,b){return{x:(a-px)/z,y:(b-py)/z}}
function paint(){if(!img.complete||!img.naturalWidth)return;resize();x.clearRect(0,0,W,H);x.imageSmoothingEnabled=true;x.imageSmoothingQuality='high';x.drawImage(img,px,py,img.naturalWidth*z,img.naturalHeight*z);x.lineWidth=2;x.strokeStyle='#00ff00';x.fillStyle='#00ff00';x.font='14px sans-serif';boxes().forEach(b=>{let a=px+b.x*z,d=py+b.y*z,w=b.w*z,h=b.h*z;x.strokeRect(a,d,w,h);x.fillText(b.class_id+':'+b.class_name,a+2,Math.max(14,d-3))});if(draw){x.strokeStyle='#ffff00';x.strokeRect(Math.min(sx,cx),Math.min(sy,cy),Math.abs(cx-sx),Math.abs(cy-sy))}document.getElementById('zl').textContent=Math.round(z*100)+'%';let done=files.filter(f=>(ann[f.name]||[]).length).length;document.getElementById('status').textContent=files.length?(idx+1)+'/'+files.length+' | 已标 '+done:''}
c.onmousedown=e=>{let r=c.getBoundingClientRect(),a=e.clientX-r.left,b=e.clientY-r.top;if(space||e.button===1){pan=true;psx=a;psy=b;opx=px;opy=py;return}if(cid===null||names[cid]===undefined){alert('请先选择或新增类别');return}draw=true;sx=cx=a;sy=cy=b};c.onmousemove=e=>{let r=c.getBoundingClientRect(),a=e.clientX-r.left,b=e.clientY-r.top;if(pan){px=opx+a-psx;py=opy+b-psy;paint()}else if(draw){cx=a;cy=b;paint()}};c.onmouseup=e=>{if(pan){pan=false;return}if(!draw)return;draw=false;let a=toImg(sx,sy),b=toImg(cx,cy);let x1=Math.max(0,Math.min(img.naturalWidth,Math.min(a.x,b.x))),y1=Math.max(0,Math.min(img.naturalHeight,Math.min(a.y,b.y))),x2=Math.max(0,Math.min(img.naturalWidth,Math.max(a.x,b.x))),y2=Math.max(0,Math.min(img.naturalHeight,Math.max(a.y,b.y)));if(x2-x1>=2&&y2-y1>=2)boxes().push({class_id:cid,class_name:names[cid],x:x1,y:y1,w:x2-x1,h:y2-y1});buildList();paint()};function undo(){let b=boxes();if(b.length){b.pop();buildList();paint()}}function clearNow(){if(files.length&&confirm('清空当前图片？')){ann[files[idx].name]=[];buildList();paint()}}function prev(){if(idx>0){idx--;buildList();load()}}function next(){if(idx<files.length-1){idx++;buildList();load()}}function exportJSON(){let o={format:'semantic_indicator_bbox_v3',device_name:CFG.device_name,names,annotations:ann},b=new Blob([JSON.stringify(o,null,2)],{type:'application/json'}),a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='indicator_annotations.json';a.click()}document.onkeydown=e=>{if(e.code==='Space'){space=true;e.preventDefault();return}if(/^[0-9]$/.test(e.key)&&names[e.key]!==undefined){cid=+e.key;classes();paint()}else if(/[aA]/.test(e.key))prev();else if(/[dD]/.test(e.key))next();else if(/[zZ]/.test(e.key))undo();else if(/[cC]/.test(e.key))clearNow();else if(/[fF]/.test(e.key))fit();else if(/[rR]/.test(e.key))reset()};document.onkeyup=e=>{if(e.code==='Space')space=false};window.onresize=()=>{if(img.naturalWidth)fit()};classes();
</script></body></html>'''

def norm(s): return re.sub(r'[^0-9a-zA-Z\u4e00-\u9fff]+','',s).lower()
def safe(s): return re.sub(r'[\\/:*?"<>|]+','_',s)

def find_device(root,name):
    p=root/name
    if p.is_dir(): return p
    t=norm(name); cand=[]
    for q in root.iterdir():
        if not q.is_dir(): continue
        n=norm(q.name)
        if n==t: return q
        if t in n or n in t: cand.append(q)
    return cand[0] if len(cand)==1 else None

def load_names(pkg):
    out={}; p=pkg/'classes.txt'
    if p.exists():
        for line in p.read_text(encoding='utf-8',errors='ignore').splitlines():
            if ':' in line:
                a,b=line.split(':',1)
                if a.strip().isdigit(): out[a.strip()]=b.strip()
    return out

def find_static_images(device_dir):
    preferred=[]
    for p in device_dir.rglob('*'):
        if p.is_dir() and any(k in p.name.lower() for k in ['图片','照片','image','photo']): preferred.append(p)
    scan_dirs=preferred if preferred else [device_dir]
    imgs=[]; seen=set()
    for base in scan_dirs:
        for p in base.rglob('*'):
            if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS: continue
            rp=str(p.resolve())
            if rp not in seen:
                seen.add(rp); imgs.append(p)
    return sorted(imgs)

def info(p):
    c=cv2.VideoCapture(str(p))
    if not c.isOpened(): return None
    fps=float(c.get(cv2.CAP_PROP_FPS) or 0); frames=int(c.get(cv2.CAP_PROP_FRAME_COUNT) or 0); w=int(c.get(cv2.CAP_PROP_FRAME_WIDTH) or 0); h=int(c.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0); c.release()
    if fps<=0 or frames<=0:return None
    return dict(path=p,fps=fps,frames=frames,duration=frames/fps,width=w,height=h)

def alloc(videos,total):
    n=len(videos)
    if n==0 or total<=0:return [0]*n
    if total<=n:
        c=[0]*n
        for i in sorted(range(n),key=lambda i:videos[i]['duration'],reverse=True)[:total]: c[i]=1
        return c
    c=[1]*n; rem=total-n; td=sum(v['duration'] for v in videos); rr=[]; used=0
    for i,v in enumerate(videos):
        s=rem*(v['duration']/td if td else 1/n); b=int(math.floor(s)); c[i]+=b; used+=b; rr.append((s-b,i))
    for _,i in sorted(rr,reverse=True)[:rem-used]: c[i]+=1
    return c

def score(frame,longside):
    h,w=frame.shape[:2]; scale=min(1.0,longside/max(h,w))
    if scale<1: frame=cv2.resize(frame,(max(1,int(w*scale)),max(1,int(h*scale))),interpolation=cv2.INTER_AREA)
    g=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY); full=float(cv2.Laplacian(g,cv2.CV_64F).var()); h,w=g.shape; crop=g[int(.15*h):int(.85*h),int(.15*w):int(.85*w)]; center=float(cv2.Laplacian(crop,cv2.CV_64F).var()); return full,center,.2*full+.8*center

def targets(v,count,win,ncand):
    out=[]; fps=v['fps']
    for i in range(count):
        a=v['duration']*(i+.5)/count; s=max(0,a-win); e=min(v['duration'],a+win); ts=[a] if ncand<=1 else [s+(e-s)*j/(ncand-1) for j in range(ncand)]; fis=[]
        for t in ts:
            fi=max(0,min(v['frames']-1,int(round(t*fps))))
            if fi not in fis:fis.append(fi)
        out.append(dict(anchor=a,fis=fis,best=None))
    return out

def scan(v,count,args):
    groups=targets(v,count,args.search_window,args.candidates); fmap={}
    for gi,g in enumerate(groups):
        for fi in g['fis']: fmap.setdefault(fi,[]).append(gi)
    if not fmap:return []
    needed=set(fmap); last=max(needed); cap=cv2.VideoCapture(str(v['path'])); fi=0
    while fi<=last:
        ok,frame=cap.read()
        if not ok:break
        if fi in needed:
            full,center,comb=score(frame,args.score_long_side)
            for gi in fmap[fi]:
                item=dict(frame=frame.copy(),frame_no=fi,time=fi/v['fps'],full_score=full,center_score=center,combined_score=comb)
                if groups[gi]['best'] is None or comb>groups[gi]['best']['combined_score']: groups[gi]['best']=item
        fi+=1
    cap.release(); out=[]
    for g in groups:
        if g['best']:
            g['best']['anchor']=g['anchor']; out.append(g['best'])
    return out

def copy_photos(paths,images_dir,name,limit):
    rows=[]
    for i,src in enumerate(paths[:limit],1):
        ext=src.suffix.lower(); fn=f"photo_{i:04d}__{safe(name)}__{safe(src.stem)}{ext}"; dst=images_dir/fn; shutil.copy2(src,dst)
        im=cv2.imread(str(src)); w=h=0; full=center=comb=0.0
        if im is not None:
            h,w=im.shape[:2]; full,center,comb=score(im,960)
        rows.append(dict(image=fn,source_type='original_photo',source_file=str(src),source_video='',video_index='',anchor_time_sec='',selected_time_sec='',frame_no='',width=w,height=h,full_sharpness=round(full,2),center_sharpness=round(center,2),combined_sharpness=round(comb,2),quality_pass=1))
    return rows

def process(name,src,pkg,args):
    vids=[]
    for p in sorted(src.rglob('*')):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            v=info(p)
            if v:vids.append(v)
    photos=find_static_images(src)
    outdir=args.output_root/name
    if outdir.exists():shutil.rmtree(outdir)
    images=outdir/'images'; images.mkdir(parents=True)
    names=load_names(pkg)
    if names:(outdir/'classes.txt').write_text('\n'.join(f"{k}: {names[k]}" for k in sorted(names,key=int))+'\n',encoding='utf-8')
    (outdir/'annotator.html').write_text(HTML.replace('__CFG__',json.dumps({'device_name':name,'names':names},ensure_ascii=False)),encoding='utf-8')
    photo_limit=min(len(photos),args.max_images); rows=copy_photos(photos,images,name,photo_limit); remaining=max(0,args.max_images-len(rows)); print(f"  原始照片: {len(photos)} | 加入: {len(rows)} | 视频帧剩余配额: {remaining}",flush=True)
    cand=[]
    if remaining>0 and vids:
        counts=alloc(vids,remaining)
        for vi,(v,cnt) in enumerate(zip(vids,counts),1):
            if cnt<=0:continue
            print(f"    video {vi:03d}/{len(vids):03d} {v['path'].name} | targets={cnt}",flush=True)
            for b in scan(v,cnt,args): b.update(video_idx=vi,video_path=v['path'],width=v['width'],height=v['height']); cand.append(b)
        good=[x for x in cand if x['center_score']>=args.min_center_score]; chosen=good[:remaining]
        if args.fill_low_quality and len(chosen)<remaining:
            bad=sorted((x for x in cand if x['center_score']<args.min_center_score),key=lambda x:x['combined_score'],reverse=True); chosen+=bad[:remaining-len(chosen)]
        chosen.sort(key=lambda x:(x['video_idx'],x['time']))
        for i,x in enumerate(chosen,1):
            fn=f"video_{i:04d}__{safe(name)}__video_{x['video_idx']:03d}__t{x['time']:09.3f}__f{x['frame_no']:08d}.png"; cv2.imwrite(str(images/fn),x['frame'],[int(cv2.IMWRITE_PNG_COMPRESSION),args.png_compression])
            rows.append(dict(image=fn,source_type='video_best_frame',source_file='',source_video=str(x['video_path']),video_index=x['video_idx'],anchor_time_sec=round(x['anchor'],3),selected_time_sec=round(x['time'],3),frame_no=x['frame_no'],width=x['width'],height=x['height'],full_sharpness=round(x['full_score'],2),center_sharpness=round(x['center_score'],2),combined_sharpness=round(x['combined_score'],2),quality_pass=int(x['center_score']>=args.min_center_score)))
    if rows:
        with (outdir/'frame_quality.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
    (outdir/'README_标注说明.txt').write_text(f"""设备：{name}\n\n数据来源：\n- 原始静态照片：直接复制，不重新编码\n- 视频抽帧：清晰帧优选后以 PNG 保存\n\n参数：\n- 总图片上限：{args.max_images}\n- 候选帧：{args.candidates}\n- 搜索窗口：±{args.search_window}s\n- 清晰度分析长边：{args.score_long_side}px\n- 中央清晰度阈值：{args.min_center_score}\n- PNG 压缩级别：{args.png_compression}\n\n标注网页：滚轮缩放，Space+左键平移，左键画框，F适应，R恢复100%。\n""",encoding='utf-8')
    zpath=args.output_root/f'{name}.zip'
    if zpath.exists():zpath.unlink()
    with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as z:
        for f in outdir.rglob('*'):
            if f.is_file():z.write(f,f.relative_to(args.output_root))
    vr=[r for r in rows if r['source_type']=='video_best_frame']; avg=round(sum(r['center_sharpness'] for r in vr)/len(vr),2) if vr else 0.0
    return dict(device_name=name,status='ok',original_photos_found=len(photos),original_photos_added=sum(1 for r in rows if r['source_type']=='original_photo'),videos=len(vids),video_candidates=len(cand),video_frames_added=len(vr),total_images=len(rows),avg_video_center_sharpness=avg,zip=str(zpath))

def main():
    a=argparse.ArgumentParser(); a.add_argument('--source-root',type=Path,default=SRC); a.add_argument('--old-packages',type=Path,default=OLD); a.add_argument('--output-root',type=Path,default=OUT); a.add_argument('--max-images',type=int,default=200); a.add_argument('--search-window',type=float,default=.30); a.add_argument('--candidates',type=int,default=5); a.add_argument('--score-long-side',type=int,default=960); a.add_argument('--min-center-score',type=float,default=80.0); a.add_argument('--png-compression',type=int,default=3,choices=range(10)); a.add_argument('--fill-low-quality',action='store_true'); a.add_argument('--device'); args=a.parse_args()
    if not args.source_root.exists():raise SystemExit(f'原始数据目录不存在: {args.source_root}')
    if not args.old_packages.exists():raise SystemExit(f'旧标注包目录不存在: {args.old_packages}')
    args.output_root.mkdir(parents=True,exist_ok=True); pkgs=sorted(p for p in args.old_packages.iterdir() if p.is_dir())
    if args.device:
        pkgs=[p for p in pkgs if p.name==args.device]
        if not pkgs:raise SystemExit(f'找不到设备标注包: {args.device}')
    results=[]; print(f'输出目录: {args.output_root}',flush=True)
    for i,pkg in enumerate(pkgs,1):
        print(f'\n[{i}/{len(pkgs)}] {pkg.name}',flush=True); src=find_device(args.source_root,pkg.name)
        if not src:
            print('  [SKIP] 无法匹配原始设备目录',flush=True); results.append(dict(device_name=pkg.name,status='source_folder_not_found')); continue
        r=process(pkg.name,src,pkg,args)
        if r:
            results.append(r); print(f"  [OK] photos={r['original_photos_added']} video={r['video_frames_added']} total={r['total_images']} avg_video_center={r['avg_video_center_sharpness']}",flush=True)
    fields=['device_name','status','original_photos_found','original_photos_added','videos','video_candidates','video_frames_added','total_images','avg_video_center_sharpness','zip']; summary=args.output_root/'rebuild_summary.csv'
    with summary.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(results)
    print(f'\n完成。汇总: {summary}')

if __name__=='__main__':main()

