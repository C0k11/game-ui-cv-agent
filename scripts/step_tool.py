# -*- coding: utf-8 -*-
"""Step-mode walk helper for the live debug session (2026-06-13).
  py scripts/step_tool.py peek      → print the pending action (no YOLO)
  py scripts/step_tool.py go        → approve pending (POST /step/go)
  py scripts/step_tool.py inspect   → pending + ADB frame + v10 cls dump
  py scripts/step_tool.py shot NAME → just grab an ADB frame to data/NAME.png
"""
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
BASE = "http://127.0.0.1:8000"
ADB = r"C:\Program Files\Netease\MuMu\nx_main\adb.exe"
DEV = "127.0.0.1:7555"
REPO = Path(r"D:\Project\ai game secretary")
V10 = r"D:\Project\ml_cache\models\yolo\runs\ui_yolo26m_v10\weights\best.pt"
# structural cls worth listing per step (skip pure badges/currency noise)
SKIP = {"加号", "减号", "信用点", "体力", "青辉石"}


def _get(path):
    with urllib.request.urlopen(BASE + path, timeout=10) as r:
        return json.load(r)


def _post(path):
    req = urllib.request.Request(BASE + path, method="POST", data=b"{}",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.load(r)


def peek():
    d = _get("/api/v1/step/pending")
    p = d.get("pending")
    if not p:
        print("pending: NONE (waits auto-exec, or skill done/transitioning)")
        return None
    print(f"PENDING [{p.get('skill')}/{p.get('sub_state')}] {p.get('action')} "
          f"@ {p.get('target')}  tick={p.get('tick')}")
    print(f"  reason: {p.get('reason')}")
    return p


def shot(name="_step"):
    out = REPO / "data" / f"{name}.png"
    with open(out, "wb") as f:
        subprocess.run([ADB, "-s", DEV, "exec-out", "screencap", "-p"],
                       stdout=f, check=True)
    return out


def inspect():
    p = peek()
    out = shot("_step")
    from ultralytics import YOLO
    m = YOLO(V10)
    r = m.predict(str(out), conf=0.30, imgsz=960, verbose=False)[0]
    h, w = r.orig_shape
    rows = []
    for b in r.boxes:
        nm = m.names[int(b.cls[0])]
        if nm in SKIP:
            continue
        x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
        rows.append((float(b.conf[0]), nm, (x1 + x2) / 2 / w, (y1 + y2) / 2 / h))
    rows.sort(reverse=True)
    print(f"--- v10 cls on current frame ({len(rows)}) ---")
    for cf, nm, cx, cy in rows:
        print(f"  {cf:.2f}  {nm:<18} ({cx:.3f},{cy:.3f})")
    return p


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "peek"
    if cmd == "peek":
        peek()
    elif cmd == "go":
        d = _post("/api/v1/step/go")
        print("approved:", json.dumps(d, ensure_ascii=False)[:160])
    elif cmd == "inspect":
        inspect()
    elif cmd == "shot":
        print("saved", shot(sys.argv[2] if len(sys.argv) > 2 else "_step"))
    else:
        print("usage: peek | go | inspect | shot NAME")
