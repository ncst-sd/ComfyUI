#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, math, re, subprocess, unicodedata
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np

PROJECT = Path.home() / 'yolo_device_monitor'
DEFAULT_SOURCE_ROOT = Path('/home/crscs/桌面/数字孪生机房巡检识别模型测试视频')
DEFAULT_PACKAGE_ROOT = PROJECT / 'annotation_packages_best_frames_fast'
IMAGE_EXTS = {'.jpg','.jpeg','.png','.bmp','.webp','.tif','.tiff'}
MANUAL_EXTS = {'.pdf','.docx','.txt','.md'}

COMMON_LIGHT_WORDS = [
    'PWR','POWER','RUN','ALM','ALARM','ACT','ACTIVE','LINK','LNK','LOS','FAIL','FAULT','STATUS','STA',
    'COM','SYS','SYSTEM','NET','LAN','WAN','RX','TX','ONLINE','BYPASS','BAT','BATTERY',
    '电源','运行','告警','故障','状态','业务','通信','网口','链路','同步','工作','正常','指示灯','灯'
]

def norm_name(s: str) -> str:
    s = unicodedata.normalize('NFKC', s).lower()
    return ''.join(ch for ch in s if ch.isalnum() or '\u4e00' <= ch <= '\u9fff')

def find_device_dir_by_name(root: Path, name: str):
    if not root.exists(): return None
    key = norm_name(name)
    matches = [p for p in root.iterdir() if p.is_dir() and norm_name(p.name) == key]
    return matches[0] if len(matches) == 1 else None

def read_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
        return '\n'.join((p.extract_text() or '') for p in PdfReader(str(path)).pages)
    except Exception:
        pass
    try:
        cp = subprocess.run(['pdftotext','-layout',str(path),'-'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
        if cp.returncode == 0: return cp.stdout
    except Exception:
        pass
    return ''

def read_docx_text(path: Path) -> str:
    try:
        from docx import Document
        return '\n'.join(p.text for p in Document(str(path)).paragraphs)
    except Exception:
        return ''

def read_manuals(manual_dir: Path) -> Tuple[str,List[str]]:
    texts, sources = [], []
    if not manual_dir.exists(): return '', []
    for p in sorted(manual_dir.rglob('*')):
        if not p.is_file() or p.suffix.lower() not in MANUAL_EXTS: continue
        if p.suffix.lower()=='.pdf': txt = read_pdf_text(p)
        elif p.suffix.lower()=='.docx': txt = read_docx_text(p)
        else:
            try: txt = p.read_text(encoding='utf-8', errors='ignore')
            except Exception: txt = ''
        if txt.strip(): texts.append(txt); sources.append(str(p))
    return '\n'.join(texts), sources

def extract_indicator_terms(text: str) -> List[Dict]:
    lines = [re.sub(r'\s+',' ',x).strip() for x in text.splitlines()]
    rx = re.compile(r'(PWR|POWER|RUN|ALM|ALARM|ACT|ACTIVE|LINK|LNK|LOS|FAIL|FAULT|STATUS|STA\d*|COM|SYS|SYSTEM|NET|LAN|WAN|RX|TX|ONLINE|BYPASS|BAT|BATTERY|电源|运行|告警|故障|状态|业务|通信|网口|链路|同步|工作|正常)', re.I)
    hits, seen = [], set()
    for line in lines:
        if not line or len(line)>240: continue
        labels = rx.findall(line)
        if '指示灯' not in line and not labels: continue
        keys = [x.upper() if x.isascii() else x for x in labels]
        for key in keys:
            sig=(key,line[:120])
            if sig in seen: continue
            seen.add(sig); hits.append({'keyword':key,'manual_text':line[:120]})
    upper = text.upper()
    for w in COMMON_LIGHT_WORDS:
        if w.isascii() and w in upper and not any(x['keyword']==w for x in hits):
            hits.append({'keyword':w,'manual_text':w})
    return hits

def try_ocr(image: np.ndarray) -> str:
    try:
        import pytesseract
        return re.sub(r'\s+',' ',pytesseract.image_to_string(image, config='--psm 6')).strip()
    except Exception:
        return ''

def candidate_boxes(img: np.ndarray):
    h,w = img.shape[:2]; area_img = h*w
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV); V = hsv[:,:,2]
    mask1 = cv2.inRange(V,190,255)
    mask2 = cv2.inRange(hsv,(0,85,90),(179,255,255))
    mask = cv2.bitwise_or(mask1,mask2)
    k=np.ones((3,3),np.uint8)
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,k); mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,k)
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    out=[]
    for c in contours:
        x,y,bw,bh=cv2.boundingRect(c); a=bw*bh; ratio=bw/max(1.0,bh)
        if a<9 or a>area_img*0.008 or bw>w*0.12 or bh>h*0.12 or ratio<0.15 or ratio>6.5: continue
        pad=max(2,int(max(bw,bh)*0.35)); x1=max(0,x-pad); y1=max(0,y-pad); x2=min(w,x+bw+pad); y2=min(h,y+bh+pad)
        roi=V[y:y+bh,x:x+bw]; score=float(np.mean(roi))/255.0 if roi.size else 0.0
        out.append((x1,y1,x2-x1,y2-y1,score))
    out.sort(key=lambda z:z[4], reverse=True)
    keep=[]
    def iou(a,b):
        ax,ay,aw,ah,_=a; bx,by,bw,bh,_=b; ax2,ay2=ax+aw,ay+ah; bx2,by2=bx+bw,by+bh
        ix1,iy1=max(ax,bx),max(ay,by); ix2,iy2=min(ax2,bx2),min(ay2,by2)
        iw,ih=max(0,ix2-ix1),max(0,iy2-iy1); inter=iw*ih; union=aw*ah+bw*bh-inter
        return inter/union if union>0 else 0
    for b in out:
        if all(iou(b,k)<0.45 for k in keep): keep.append(b)
    return keep[:80]

def best_manual_match(ocr_text: str, terms: List[Dict]):
    if not ocr_text: return None,0.0
    upper=ocr_text.upper(); hits=[]
    for t in terms:
        k=str(t['keyword']); ku=k.upper()
        if k and ku in upper: hits.append(t)
    if len(hits)==1: return hits[0],0.90
    return None,0.0

def prelabel_image(img_path: Path, terms: List[Dict]):
    img=cv2.imread(str(img_path))
    if img is None: return []
    h,w=img.shape[:2]; anns=[]
    for x,y,bw,bh,visual_score in candidate_boxes(img):
        px=max(20,int(bw*4)); py=max(15,int(bh*2.5))
        x1=max(0,x-px); y1=max(0,y-py); x2=min(w,x+bw+px); y2=min(h,y+bh+py)
        ocr=try_ocr(img[y1:y2,x1:x2]); match,_=best_manual_match(ocr,terms)
        if match is None:
            label='unknown_indicator'; evidence=''; conf=min(0.55,0.25+0.30*visual_score)
        else:
            label=match['keyword']; evidence=match['manual_text']; conf=min(0.95,0.55+0.35*visual_score)
        anns.append({'label':label,'bbox':[int(x),int(y),int(bw),int(bh)],'confidence':round(float(conf),4),'needs_review':True,'source':'manual_prelabel','ocr_text':ocr[:120],'manual_evidence':evidence})
    return anns

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--device', required=True, help='annotation_packages_best_frames_fast 中的设备目录名')
    ap.add_argument('--source-root', type=Path, default=DEFAULT_SOURCE_ROOT)
    ap.add_argument('--package-root', type=Path, default=DEFAULT_PACKAGE_ROOT)
    args=ap.parse_args()
    package_dir=args.package_root.expanduser()/args.device
    if not package_dir.exists(): raise SystemExit(f'待标注设备目录不存在: {package_dir}')
    final_ann=package_dir/'indicator_annotations.json'
    if final_ann.exists(): raise SystemExit(f'安全停止：{args.device} 已存在 indicator_annotations.json，脚本不会修改。')
    src=find_device_dir_by_name(args.source_root.expanduser(), args.device)
    if src is None: raise SystemExit(f'无法安全匹配原始设备目录: {args.device}')
    manual_dir=src/'说明书'; text,sources=read_manuals(manual_dir)
    if not text.strip(): raise SystemExit(f'没有从说明书中提取到可用文本: {manual_dir}')
    terms=extract_indicator_terms(text)
    images_dir=package_dir/'images'
    images=sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not images: raise SystemExit(f'没有待标注图片: {images_dir}')
    print(f'设备: {args.device}\n原始目录: {src}\n说明书文件数: {len(sources)}\n说明书候选灯术语: {len(terms)}')
    for t in terms[:30]: print(f"  - {t['keyword']}: {t['manual_text'][:80]}")
    annotations={}; total=0
    for i,p in enumerate(images,1):
        anns=prelabel_image(p,terms); annotations[p.name]=anns; total+=len(anns)
        print(f'[{i}/{len(images)}] {p.name}: {len(anns)} 个候选')
    result={'format':'manual_prelabel_v1','device_name':args.device,'needs_review':True,'manual_sources':sources,'manual_terms':terms,'annotations':annotations,'summary':{'images':len(images),'candidate_boxes':total}}
    out=package_dir/'pre_annotations.json'; out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'\n预标注完成\n输出: {out}\n图片数: {len(images)}\n候选框: {total}\n正式 indicator_annotations.json 未创建、未修改。')

if __name__=='__main__': main()

