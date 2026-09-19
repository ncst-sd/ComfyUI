#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
update_annotator_html_shortcuts.py

批量更新所有 annotator.html，为每个类别增加“自定义快捷键”。

默认目录：
~/yolo_device_monitor/annotation_packages_best_frames_fast

效果：
- 每个类别右侧增加“快捷键”输入框
- 支持 A-Z、0-9、F1-F12、标点等
- 超过 9 的类别不再需要鼠标点类别
- 输入框内打字不会误触原来的 A/D/Z/C/F/R 等快捷键
- 自定义类别快捷键优先于原全局快捷键
- 同一快捷键只绑定一个类别
- 快捷键保存在浏览器 localStorage
- 不创建任何 HTML 备份文件
"""

import argparse
from pathlib import Path

ROOT_DEFAULT = Path.home() / "yolo_device_monitor" / "annotation_packages_best_frames_fast"
MARK = "CUSTOM_CLASS_SHORTCUTS_V4"

CSS = r"""
/* CUSTOM_CLASS_SHORTCUTS_V4 */
.shortcut-v4-row{
  display:flex !important;
  align-items:center !important;
  gap:6px !important;
}
.shortcut-v4-key{
  margin-left:auto !important;
  width:58px !important;
  min-width:58px !important;
  height:24px !important;
  padding:1px 4px !important;
  box-sizing:border-box !important;
  text-align:center !important;
  font-family:monospace !important;
  font-size:12px !important;
  background:#fff !important;
  color:#111 !important;
  border:1px solid #aaa !important;
  border-radius:3px !important;
}
.shortcut-v4-key::placeholder{
  color:#777 !important;
  opacity:1 !important;
}
"""

JS = r"""
// CUSTOM_CLASS_SHORTCUTS_V4
(function(){
  const STORE="annotator_class_shortcuts_v4";

  function isTyping(el){
    if(!el) return false;
    const tag=(el.tagName||"").toLowerCase();
    return tag==="input" || tag==="textarea" || tag==="select" || el.isContentEditable;
  }

  function keyName(e){
    if(!e || !e.key) return "";
    if(e.key===" ") return "SPACE";
    if(e.key==="Escape") return "ESC";
    if(e.key==="Enter") return "ENTER";
    if(e.key==="Delete") return "DELETE";
    if(e.key==="Backspace") return "BACKSPACE";
    if(/^F([1-9]|1[0-2])$/.test(e.key)) return e.key.toUpperCase();
    if(e.key.length===1) return e.key.toUpperCase();
    return e.key.toUpperCase();
  }

  function load(){
    try{
      return JSON.parse(localStorage.getItem(STORE)||"{}");
    }catch(_){
      return {};
    }
  }

  function save(m){
    localStorage.setItem(STORE,JSON.stringify(m));
  }

  function cleanText(el){
    if(!el) return "";
    let s=(el.innerText||el.textContent||"").trim();
    s=s.replace(/\s+/g," ");
    return s;
  }

  function parseClassRow(el){
    const txt=cleanText(el);
    const m=txt.match(/^(\d+)\s*:\s*(.+)$/);
    if(!m) return null;

    const id=m[1];
    let name=m[2].trim();
    name=name.replace(/\s+快捷键.*$/,"").trim();

    if(!name) return null;
    return {id,name};
  }

  function findRows(){
    const selectors=[
      "[data-class-id]",
      ".class-row",
      ".class-item",
      "li",
      "div",
      "label",
      "button",
      "td"
    ];

    const all=[...document.querySelectorAll(selectors.join(","))];
    const candidates=[];

    for(const el of all){
      if(el.classList && el.classList.contains("shortcut-v4-key")) continue;

      const info=parseClassRow(el);
      if(!info) continue;

      const txt=cleanText(el);
      const matches=txt.match(/\d+\s*:/g)||[];
      if(matches.length!==1) continue;

      candidates.push({el,info});
    }

    const byId=new Map();

    for(const item of candidates){
      const old=byId.get(item.info.id);

      if(!old){
        byId.set(item.info.id,item);
        continue;
      }

      if(old.el.contains(item.el)){
        byId.set(item.info.id,item);
      }
    }

    return [...byId.values()]
      .sort((a,b)=>Number(a.info.id)-Number(b.info.id));
  }

  function activateRow(row){
    const radio=row.querySelector('input[type="radio"]');
    if(radio){
      radio.click();
      return;
    }

    const btn=row.querySelector("button");
    if(btn){
      btn.click();
      return;
    }

    row.click();
  }

  function refreshValues(){
    const map=load();

    for(const {el,info} of findRows()){
      const inp=el.querySelector(".shortcut-v4-key");
      if(inp) inp.value=map[info.id]||"";
    }
  }

  function decorate(){
    const map=load();

    for(const {el,info} of findRows()){
      if(el.querySelector(".shortcut-v4-key")) continue;

      el.classList.add("shortcut-v4-row");

      const inp=document.createElement("input");
      inp.type="text";
      inp.className="shortcut-v4-key";
      inp.placeholder="快捷键";
      inp.value=map[info.id]||"";
      inp.title=`类别 ${info.id}: ${info.name} 的自定义快捷键`;

      ["click","mousedown","mouseup","keyup"].forEach(ev=>{
        inp.addEventListener(ev,e=>e.stopPropagation());
      });

      inp.addEventListener("keydown",e=>{
        e.stopPropagation();

        if(e.key==="Tab") return;

        if(e.key==="Backspace" || e.key==="Delete"){
          e.preventDefault();
          const m=load();
          delete m[info.id];
          save(m);
          inp.value="";
          return;
        }

        const k=keyName(e);
        if(!k) return;

        e.preventDefault();

        const m=load();

        for(const cid of Object.keys(m)){
          if(m[cid]===k) delete m[cid];
        }

        m[info.id]=k;
        save(m);
        refreshValues();
      });

      el.appendChild(inp);
    }
  }

  function onGlobalKey(e){
    if(isTyping(e.target)) return;

    const k=keyName(e);
    if(!k) return;

    const map=load();

    for(const {el,info} of findRows()){
      if(map[info.id]===k){
        e.preventDefault();
        e.stopImmediatePropagation();
        activateRow(el);
        return;
      }
    }
  }

  window.addEventListener("keydown",onGlobalKey,true);

  function runDecorate(){
    try{
      decorate();
    }catch(e){
      console.error("shortcut-v4",e);
    }
  }

  if(document.readyState==="loading"){
    document.addEventListener("DOMContentLoaded",runDecorate);
  }else{
    runDecorate();
  }

  new MutationObserver(()=>{
    runDecorate();
  }).observe(document.documentElement,{
    childList:true,
    subtree:true
  });

  setInterval(runDecorate,1200);
})();
"""

def patch(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")

    if MARK in text:
        return "skip"

    if "</head>" in text:
        text = text.replace(
            "</head>",
            f"<style>\n{CSS}\n</style>\n</head>",
            1
        )
    else:
        text = f"<style>\n{CSS}\n</style>\n" + text

    if "</body>" in text:
        text = text.replace(
            "</body>",
            f"<script>\n{JS}\n</script>\n</body>",
            1
        )
    else:
        text += f"\n<script>\n{JS}\n</script>\n"

    path.write_text(text, encoding="utf-8")
    return "updated"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=Path,
        default=ROOT_DEFAULT,
        help="标注包根目录"
    )
    args = ap.parse_args()

    root = args.root.expanduser()

    if not root.exists():
        raise SystemExit(f"目录不存在: {root}")

    files = sorted(root.rglob("annotator.html"))

    if not files:
        raise SystemExit(f"没有找到 annotator.html: {root}")

    updated = 0
    skipped = 0

    for f in files:
        r = patch(f)

        if r == "updated":
            updated += 1
            print("[更新]", f)
        else:
            skipped += 1
            print("[跳过]", f)

    print()
    print(f"完成：找到 {len(files)} 个 annotator.html")
    print(f"更新：{updated}")
    print(f"跳过：{skipped}")
    print("请关闭原页面重新打开，或使用 Ctrl+Shift+R 强制刷新。")

if __name__ == "__main__":
    main()

