"""Interactive single-step probe — NO skills, NO pipeline state machine.

Purpose: walk a flow by hand to SEE the ground truth — "this screen → YOLO
detects THESE cls → I tap one → what screen comes next?" — so skills can be
rewritten against reality instead of guessing.

Each run does ONE of:
  look                 capture 1 frame (WGC) → run active ui YOLO → print every
                       detected cls with center + conf, sorted; save an
                       annotated image to data/_probe_look.jpg
  tap <cls>            tap the center of the highest-conf box of that cls (ADB),
                       then look again
  tap <x> <y>          tap a normalized point, then look again
  back                 ADB back, then look
  swipe <x1> <y1> <x2> <y2>   ADB swipe (normalized), then look

State (the MuMu hwnd + android size) is cached in data/_probe_state.json so
each invocation is independent (no long-running process to manage).

Usage:
  py scripts/probe.py look
  py scripts/probe.py tap 咖啡厅入口
  py scripts/probe.py tap 0.5 0.9
  py scripts/probe.py back
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

REPO = Path(__file__).resolve().parents[1]
STATE = REPO / "data" / "_probe_state.json"
LOOK_IMG = REPO / "data" / "_probe_look.jpg"
_LOOK_HEADPAT = False  # set True to also run happy_face/emoticon headpat detect


def _load_state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_state(s: dict) -> None:
    STATE.write_text(json.dumps(s), encoding="utf-8")


def _capture():
    """Grab one BGR frame via WGC (background-safe), return (frame, hwnd, andsize)."""
    import numpy as np  # noqa
    from scripts.win_capture import find_window_by_title_substring, find_largest_visible_child
    from scripts.wgc_capture import WgcCapture
    hwnd = find_window_by_title_substring("mumu")
    render = find_largest_visible_child(hwnd) or hwnd
    wgc = WgcCapture(render, capture_hwnd=hwnd)
    wgc.wait_first_frame(3.0)
    frame = None
    for _ in range(30):
        frame = wgc.grab()
        if frame is not None:
            break
        time.sleep(0.05)
    wgc.stop()
    return frame, hwnd


def _run_ui_yolo(frame):
    """Run the ACTIVE ui model on the frame → list of (cls, conf, cx, cy, x1,y1,x2,y2)."""
    import json as _j
    from ultralytics import YOLO
    reg = _j.loads((REPO / "data" / "model_registry.json").read_text(encoding="utf-8"))
    ui = reg["ui"]; wpath = ui["versions"][ui["active"]]["path"]
    model = YOLO(wpath)
    h, w = frame.shape[:2]
    res = model.predict(frame, imgsz=960, conf=0.20, verbose=False, device=0)
    out = []
    for r in res:
        if r.boxes is None:
            continue
        names = r.names
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            cls = names.get(int(b.cls[0]), str(int(b.cls[0])))
            conf = float(b.conf[0])
            out.append((cls, conf,
                        (x1 + x2) / 2 / w, (y1 + y2) / 2 / h,
                        x1 / w, y1 / h, x2 / w, y2 / h))
    return out, (w, h), ui["active"]


def _run_headpat_detect(frame):
    """Headpat markers the way cafe._headpat sees them: happy_face template
    (primary) + Emoticon_Action YOLO (fallback). Returns list of
    (label, conf, cx, cy)."""
    out = []
    h, w = frame.shape[:2]
    # 1. happy_face / headpat templates (cafe play area, exclude UI bars)
    try:
        from vision.template_matcher import find_headpat_bubbles
        for hit in find_headpat_bubbles(frame, threshold=0.70,
                                        region=(0.12, 0.18, 0.98, 0.82)):
            out.append((f"tmpl:{hit.label}", float(hit.confidence),
                        (hit.x1 + hit.x2) / 2, (hit.y1 + hit.y2) / 2))
    except Exception as e:
        print(f"[probe] template_matcher err: {e}")
    # 2. emoticon YOLO model
    try:
        import json as _j
        from ultralytics import YOLO
        reg = _j.loads((REPO / "data" / "model_registry.json").read_text(encoding="utf-8"))
        em = reg["emoticon"]; wp = em["versions"][em["active"]]["path"]
        m = YOLO(wp)
        for r in m.predict(frame, imgsz=640, conf=0.15, verbose=False, device=0):
            if r.boxes is None:
                continue
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                out.append((f"emo:{r.names.get(int(b.cls[0]),'?')}", float(b.conf[0]),
                            (x1 + x2) / 2 / w, (y1 + y2) / 2 / h))
    except Exception as e:
        print(f"[probe] emoticon model err: {e}")
    return out


def _annotate(frame, dets, pats=None):
    import cv2
    img = frame.copy()
    h, w = img.shape[:2]
    for cls, conf, cx, cy, x1, y1, x2, y2 in dets:
        p1 = (int(x1 * w), int(y1 * h)); p2 = (int(x2 * w), int(y2 * h))
        cv2.rectangle(img, p1, p2, (0, 255, 0), 2)
        cv2.putText(img, f"{cls} {conf:.2f}", (p1[0], max(12, p1[1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    for lbl, conf, cx, cy in (pats or []):
        c = (int(cx * w), int(cy * h))
        cv2.circle(img, c, 18, (0, 0, 255), 3)
        cv2.putText(img, f"{lbl} {conf:.2f}", (c[0] - 20, c[1] - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imencode(".jpg", img)[1].tofile(str(LOOK_IMG))


def _look():
    frame, hwnd = _capture()
    if frame is None:
        print("[probe] capture FAILED (frame None) — is MuMu running, not minimized?")
        return
    dets, (w, h), ver = _run_ui_yolo(frame)
    pats = _run_headpat_detect(frame) if _LOOK_HEADPAT else []
    _annotate(frame, dets, pats)
    st = _load_state(); st["hwnd"] = hwnd; _save_state(st)
    dets.sort(key=lambda d: -d[1])
    print(f"=== LOOK: ui={ver} frame={w}x{h} — {len(dets)} ui detections (conf>=0.20) ===")
    for cls, conf, cx, cy, *_ in dets:
        print(f"  {conf:.2f}  ({cx:.3f},{cy:.3f})  {cls}")
    if _LOOK_HEADPAT:
        pats.sort(key=lambda d: -d[1])
        print(f"--- headpat markers ({len(pats)}): happy_face template + emoticon model ---")
        for lbl, conf, cx, cy in pats:
            print(f"  {conf:.2f}  ({cx:.3f},{cy:.3f})  {lbl}")
    print(f"[probe] annotated image: {LOOK_IMG}")


def _adb():
    from mumu_runner import AdbInput
    a = AdbInput()
    a.connect()
    aw, ah = a.screen_size()
    return a, aw, ah


def _tap(args):
    a, aw, ah = _adb()
    if len(args) == 1:
        # tap by cls name — find its box first
        frame, _ = _capture()
        dets, (w, h), _ = _run_ui_yolo(frame)
        cands = [d for d in dets if d[0] == args[0]]
        if not cands:
            print(f"[probe] cls {args[0]!r} NOT detected — nothing tapped. (run look)")
            return
        cands.sort(key=lambda d: -d[1])
        _, conf, cx, cy, *_ = cands[0]
        print(f"[probe] tap cls {args[0]!r} @ ({cx:.3f},{cy:.3f}) conf {conf:.2f}")
    else:
        cx, cy = float(args[0]), float(args[1])
        print(f"[probe] tap point ({cx:.3f},{cy:.3f})")
    a.tap(int(cx * aw), int(cy * ah))
    time.sleep(1.2)
    _look()


def _back():
    a, aw, ah = _adb()
    a.back(); time.sleep(1.2); _look()


def _swipe(args):
    a, aw, ah = _adb()
    x1, y1, x2, y2 = map(float, args[:4])
    a.swipe(int(x1 * aw), int(y1 * ah), int(x2 * aw), int(y2 * ah), 800)
    time.sleep(1.2); _look()


def main():
    argv = sys.argv[1:]
    if not argv:
        print(__doc__); return
    cmd = argv[0]
    if cmd == "look":
        global _LOOK_HEADPAT
        if "headpat" in argv[1:]:
            _LOOK_HEADPAT = True
        _look()
    elif cmd == "tap":
        _tap(argv[1:])
    elif cmd == "back":
        _back()
    elif cmd == "swipe":
        _swipe(argv[1:])
    else:
        print(f"unknown command {cmd!r}"); print(__doc__)


if __name__ == "__main__":
    main()
