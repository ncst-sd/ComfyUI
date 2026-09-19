#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse,json
from pathlib import Path
import cv2,yaml

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--json",type=Path,required=True)
    ap.add_argument("--dataset-root",type=Path,required=True)
    a=ap.parse_args()
    root=a.dataset_root.expanduser().resolve()
    images=root/"images"/"to_label"
    labels=root/"labels"/"to_label"
    labels.mkdir(parents=True,exist_ok=True)
    data=json.loads(a.json.expanduser().read_text(encoding="utf-8"))
    ann=data.get("annotations",{})
    names=data.get("names",{})
    image_map={p.name:p for p in images.iterdir() if p.is_file()}
    matched=boxes_total=0
    for name,boxes in ann.items():
        ip=image_map.get(name)
        if ip is None: continue
        img=cv2.imread(str(ip))
        if img is None: continue
        h,w=img.shape[:2]
        lines=[]
        for b in boxes:
            cls=int(b["class_id"])
            x,y,bw,bh=map(float,[b["x"],b["y"],b["w"],b["h"]])
            x1=max(0,min(w,x)); y1=max(0,min(h,y))
            x2=max(0,min(w,x+bw)); y2=max(0,min(h,y+bh))
            if x2<=x1 or y2<=y1: continue
            xc=((x1+x2)/2)/w; yc=((y1+y2)/2)/h
            nw=(x2-x1)/w; nh=(y2-y1)/h
            lines.append(f"{cls} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
            boxes_total+=1
        (labels/(Path(name).stem+".txt")).write_text("\n".join(lines)+("\n" if lines else ""),encoding="utf-8")
        matched+=1
    if names:
        cfg={"path":str(root),"train":"images/train","val":"images/val","test":"images/test","names":{int(k):str(v) for k,v in names.items()}}
        (root/"data.yaml").write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False),encoding="utf-8")
    print("成功匹配图片:",matched)
    print("总框数:",boxes_total)
    print("标签目录:",labels)

if __name__=="__main__":
    main()

