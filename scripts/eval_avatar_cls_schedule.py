"""Human eval: extract every avatar from real schedule-popup frames, run
avatar_cls v2 on each, output side-by-side HTML for visual verification.

Why: the auto-generated val sets (harvest CN-named) have labeling errors —
some refs are mis-named so the model gets penalized for being correct.
Schedule-popup frames have NO label ambiguity: the user can visually
identify each character themselves.

Pipeline:
  1. Pick recent BA schedule-popup frames from trajectories
  2. Use static_ui v2 to detect `房间区域` bboxes (mAP 0.978 ✓)
  3. For each `房间区域`, slice bottom 35% horizontally into 3 cells
  4. Run avatar_cls v2 on each cell (with TTA optional)
  5. Render an HTML with: full frame thumbnail + each crop +
     top-3 predictions + confidence
  6. User scrolls through, marks ❌ if model wrong

Output: data/yolo_datasets/avatar_cls_schedule_eval/REPORT.html
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[1]
STATIC_UI_MODEL = Path(r"D:\Project\ml_cache\models\yolo\runs\static_ui_v3_yolo26n\weights\best.pt")
AVATAR_CLS_MODEL = Path(r"D:\Project\ml_cache\models\yolo\runs\avatar_cls_v2_yolo26n\weights\best.pt")
# Single-class 角色头像 detector — replaces the sliding-window approach.
# If this model doesn't exist yet, falls back to sliding window.
HEAD_DETECTOR_MODEL = Path(r"D:\Project\ml_cache\models\yolo\runs\head_detector_yolo26n\weights\best.pt")


def img_to_b64(img: np.ndarray, max_side: int = 200, quality: int = 80) -> str:
    h, w = img.shape[:2]
    s = max(h, w)
    if s > max_side:
        scale = max_side / s
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def find_schedule_frames(limit: int) -> List[Path]:
    """Walk trajectories to find schedule-popup-open frames."""
    ROOM_NAMES = ("視聽室", "體育館", "圖書館", "教室", "實驗室", "射擊場", "載具庫")
    out: List[Path] = []
    for run in sorted((REPO / "data" / "trajectories").iterdir(), reverse=True)[:50]:
        if not run.is_dir() or not run.name.startswith("run_"):
            continue
        for tj in sorted(run.glob("tick_*.json")):
            try:
                d = json.loads(tj.read_text(encoding="utf-8"))
            except Exception:
                continue
            if d.get("skill") != "Schedule":
                continue
            has_header = False
            has_room = False
            for b in d.get("ocr_boxes", []) or []:
                t = b.get("text") or ""
                if "全體課程表" in t:
                    has_header = True
                if any(rn in t for rn in ROOM_NAMES):
                    has_room = True
                if has_header and has_room:
                    break
            if has_header and has_room:
                jpg = tj.with_suffix(".jpg")
                if jpg.exists() and jpg.stat().st_size > 1000:
                    out.append(jpg)
                    if len(out) >= limit:
                        return out
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=5, help="How many schedule frames to evaluate")
    ap.add_argument("--out", default=str(REPO / "data" / "yolo_datasets" / "avatar_cls_schedule_eval" / "REPORT.html"))
    ap.add_argument("--detect-conf", type=float, default=0.15,
                    help="Lower default (0.15) so all 7 rooms get detected — high "
                         "v2 mAP for 房间区域 lets us recall more at low threshold")
    ap.add_argument("--cls-conf", type=float, default=0.5,
                    help="Avatar classifier confidence floor; tiles below this are dropped")
    ap.add_argument("--tta", action="store_true", help="Use test-time augmentation on classifier")
    ap.add_argument(
        "--mode",
        default="geom",
        choices=["geom", "head_det"],
        help="geom = static_ui 房间区域 + precise 3-slot geometric slicing (current best, "
             "head_detector v1 produced sloppy bboxes). head_det = use head_detector model.",
    )
    args = ap.parse_args()

    bilingual = json.loads((REPO / "data" / "avatar_cls_names_bilingual.json").read_text(encoding="utf-8"))
    en_to_cn = {v["en"]: v.get("cn") for v in bilingual.values()}

    def lbl(en: str) -> str:
        cn = en_to_cn.get(en)
        return f"{en}({cn})" if cn else en

    print(f"Loading models...")
    detect = YOLO(str(STATIC_UI_MODEL))
    detect_names = detect.names
    room_idx = next((k for k, v in detect_names.items() if v == "房间区域"), None)
    if room_idx is None:
        print(f"  ERR: static_ui has no '房间区域' class")
        return 1
    print(f"  static_ui: {len(detect_names)} classes; 房间区域 idx={room_idx}")
    clf = YOLO(str(AVATAR_CLS_MODEL))
    clf_names = [clf.names[i] for i in sorted(clf.names.keys())]
    print(f"  avatar_cls: {len(clf_names)} classes")
    # Head detector (loaded only if --mode head_det)
    head_det = None
    if args.mode == "head_det" and HEAD_DETECTOR_MODEL.is_file():
        head_det = YOLO(str(HEAD_DETECTOR_MODEL))
        print(f"  head_detector: loaded ({HEAD_DETECTOR_MODEL.name})")
    else:
        print(f"  mode: GEOMETRIC SLOT SLICING (static_ui 房间区域 → 3 precise cells)")

    # 房间区域 卡片下半部分是头像行，再等分 3 个 cell
    AVATAR_STRIP_Y_FRAC = (0.55, 1.0)  # bottom 45% of room card
    CELLS_PER_ROOM = 3

    frames = find_schedule_frames(args.frames)
    print(f"Frames to eval: {len(frames)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    html_parts = [f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>schedule popup eval</title>
<style>
body {{ font-family: system-ui, sans-serif; background:#0f1115; color:#e0e0e0; padding:20px; max-width:1600px; margin:0 auto }}
h1, h2 {{ color:#fff }}
h2 {{ margin-top:40px; padding:8px 12px; background:#1a1d24; border-radius:6px }}
.frame-img {{ display:block; max-width:600px; border-radius:6px; margin:10px 0 }}
table {{ border-collapse: collapse; width:100%; margin:10px 0 }}
th, td {{ padding:6px 10px; border-bottom:1px solid #333; text-align:left; vertical-align:middle }}
th {{ background:#1a1d24; color:#aaa; font-size:11px; text-transform:uppercase }}
.crop {{ width:96px; height:96px; image-rendering:pixelated; border-radius:4px; display:block }}
.pred {{ font-family:monospace; font-size:13px }}
.conf {{ font-family:monospace }}
.conf.high {{ color:#86efac }}
.conf.low {{ color:#fbbf24 }}
.conf.veryLow {{ color:#fca5a5 }}
.alts {{ font-family:monospace; font-size:11px; color:#999; line-height:1.6 }}
.pos {{ font-family:monospace; font-size:10px; color:#888 }}
.review {{ width:30px; text-align:center; font-size:18px }}
.intro {{ background:#1a1d24; padding:12px 16px; border-radius:6px; line-height:1.6 }}
</style></head><body>
<h1>schedule popup 头像识别 - 人工审核</h1>
<div class="intro">
模型: <code>avatar_cls_v2_yolo26n</code> (267 类) + <code>static_ui_yolo26n</code> 检测<br>
看图，对照预测，自己判断对错。<br>
高 conf (≥0.7) 绿色，中 (0.5-0.7) 黄色，低 (&lt;0.5) 红色。
</div>
"""]

    total_cells = 0
    for fi, jpg in enumerate(frames):
        img = cv2.imdecode(np.fromfile(str(jpg), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        frame_tag = f"{jpg.parent.name}/{jpg.stem}"

        # Step 1: detect 房间区域 cards (low conf for recall, then IoU-dedupe)
        det_results = detect(img, conf=args.detect_conf, verbose=False)
        raw_rooms = []
        for b in det_results[0].boxes:
            if int(b.cls[0]) == room_idx:
                raw_rooms.append((b.xyxy[0].tolist(), float(b.conf[0])))
        # IoU-NMS: with low conf threshold YOLO may emit duplicate boxes for
        # the same card.  Greedy drop: keep highest-conf box, suppress any
        # later box with IoU > 0.5 against it.
        raw_rooms.sort(key=lambda r: -r[1])
        rooms = []
        for (x1, y1, x2, y2), c in raw_rooms:
            keep = True
            for (kx1, ky1, kx2, ky2), _ in rooms:
                ix1 = max(x1, kx1); iy1 = max(y1, ky1)
                ix2 = min(x2, kx2); iy2 = min(y2, ky2)
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2-ix1)*(iy2-iy1)
                    a = (x2-x1)*(y2-y1)
                    b = (kx2-kx1)*(ky2-ky1)
                    if inter / max(1, a+b-inter) > 0.5:
                        keep = False; break
            if keep:
                rooms.append(((x1,y1,x2,y2), c))
        # Sort by (cy, cx) row-major for display
        rooms.sort(key=lambda r: ((r[0][1]+r[0][3])/2, (r[0][0]+r[0][2])/2))

        # Step 2: find avatars.  Two paths:
        #   A. head_detector loaded → use its bboxes directly (true YOLO,
        #      not sliding window).  Filter by which room each bbox falls in.
        #   B. fallback → sliding window across each room's bottom strip,
        #      keep only windows where avatar_cls is confident >= --cls-conf.
        head_crops = []  # (crop, cx_n, cy_n, room_idx, head_idx, room_conf, top1, top1_conf, probs)
        annotated = img.copy()
        for ri, ((rx1, ry1, rx2, ry2), rconf) in enumerate(rooms):
            rx1, ry1, rx2, ry2 = int(rx1), int(ry1), int(rx2), int(ry2)
            cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), (255, 200, 0), 2)
            cv2.putText(annotated, f'R{ri}', (rx1, ry1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

        # ── PATH A: head detector (only when --mode head_det) ──
        if head_det is not None:
            hd_results = head_det(img, conf=0.25, verbose=False)[0]
            for hi, b in enumerate(hd_results.boxes):
                fx1, fy1, fx2, fy2 = [int(v) for v in b.xyxy[0].tolist()]
                crop = img[fy1:fy2, fx1:fx2]
                if crop.size == 0:
                    continue
                head_cx = (fx1+fx2)/2
                head_cy = (fy1+fy2)/2
                ri_assigned = -1
                for ri, ((rx1, ry1, rx2, ry2), _) in enumerate(rooms):
                    if rx1 <= head_cx <= rx2 and ry1 <= head_cy <= ry2:
                        ri_assigned = ri
                        break
                r2 = clf.predict(crop, verbose=False, imgsz=224, augment=args.tta)[0]
                probs = r2.probs.data.cpu().numpy()
                top1_idx = int(np.argmax(probs))
                top1_name = clf_names[top1_idx]
                top1_conf = float(probs[top1_idx])
                cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2), (255, 255, 0), 2)
                cx_n = head_cx / w
                cy_n = head_cy / h
                head_crops.append((crop, cx_n, cy_n, ri_assigned, hi, float(b.conf[0]),
                                   top1_name, top1_conf, probs))
        elif args.mode == "geom":
            # ── PATH B (default): PRECISE geometric slicing ──
            # Within each detected 房间区域 (room card), the avatar strip lives
            # at bottom 30-45% of card height.  Slice strip into 3 even cells,
            # crop a SQUARE centered on each cell (side = strip height).
            # Empty slots → avatar_cls returns low conf → we drop them.
            STRIP_TOP = 0.55   # strip starts at 55% down the card
            STRIP_BOT = 1.00   # strip ends at card bottom
            for ri, ((rx1, ry1, rx2, ry2), rconf) in enumerate(rooms):
                rx1i, ry1i, rx2i, ry2i = int(rx1), int(ry1), int(rx2), int(ry2)
                rh = ry2i - ry1i
                rw = rx2i - rx1i
                sy1 = ry1i + int(STRIP_TOP * rh)
                sy2 = ry1i + int(STRIP_BOT * rh)
                strip_h = sy2 - sy1
                if strip_h < 20:
                    continue
                # 3 even slots horizontally
                cell_w_full = rw / 3.0
                side = min(int(cell_w_full), strip_h)  # square crop
                for ci in range(3):
                    cell_cx = rx1i + int((ci + 0.5) * cell_w_full)
                    cell_cy = sy1 + strip_h // 2
                    fx1 = cell_cx - side // 2
                    fy1 = cell_cy - side // 2
                    fx2 = fx1 + side
                    fy2 = fy1 + side
                    crop = img[max(0, fy1):min(h, fy2), max(0, fx1):min(w, fx2)]
                    if crop.size == 0:
                        continue
                    r2 = clf.predict(crop, verbose=False, imgsz=224, augment=args.tta)[0]
                    probs = r2.probs.data.cpu().numpy()
                    top1_idx = int(np.argmax(probs))
                    top1_conf = float(probs[top1_idx])
                    top1_name = clf_names[top1_idx]
                    # Skip empty slots: low avatar_cls conf means no avatar there
                    if top1_conf < args.cls_conf:
                        # Draw light gray placeholder
                        cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2), (120, 120, 120), 1)
                        continue
                    cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)
                    cx_n = (fx1+fx2)/2/w
                    cy_n = (fy1+fy2)/2/h
                    head_crops.append((crop, cx_n, cy_n, ri, ci, rconf,
                                       top1_name, top1_conf, probs))
        else:
            # ── PATH C: legacy sliding window (kept for reference) ──
            for ri, ((rx1, ry1, rx2, ry2), rconf) in enumerate(rooms):
                rx1, ry1, rx2, ry2 = int(rx1), int(ry1), int(rx2), int(ry2)
                rh = ry2 - ry1
                rw = rx2 - rx1
                sy1 = ry1 + int(0.55 * rh)
                sy2 = ry2
                strip_h = sy2 - sy1
                if strip_h < 20:
                    continue
                side = strip_h
                stride = max(8, side // 3)
                candidates = []
                x_cursor = rx1
                while x_cursor + side <= rx2:
                    fx1, fy1 = x_cursor, sy1
                    fx2, fy2 = x_cursor + side, sy2
                    crop = img[fy1:fy2, fx1:fx2]
                    if crop.size > 0:
                        r2 = clf.predict(crop, verbose=False, imgsz=224, augment=args.tta)[0]
                        probs = r2.probs.data.cpu().numpy()
                        top1_idx = int(np.argmax(probs))
                        top1_conf = float(probs[top1_idx])
                        if top1_conf >= args.cls_conf:
                            candidates.append({
                                "crop": crop, "top1": clf_names[top1_idx], "top1_conf": top1_conf,
                                "fx1": fx1, "fy1": fy1, "fx2": fx2, "fy2": fy2, "probs": probs,
                            })
                    x_cursor += stride
                candidates.sort(key=lambda c: -c["top1_conf"])
                kept = []
                for c in candidates:
                    drop = False
                    for k in kept:
                        ix1 = max(c["fx1"], k["fx1"]); iy1 = max(c["fy1"], k["fy1"])
                        ix2 = min(c["fx2"], k["fx2"]); iy2 = min(c["fy2"], k["fy2"])
                        if ix2 > ix1 and iy2 > iy1:
                            inter = (ix2-ix1)*(iy2-iy1)
                            a = (c["fx2"]-c["fx1"])*(c["fy2"]-c["fy1"])
                            b = (k["fx2"]-k["fx1"])*(k["fy2"]-k["fy1"])
                            if inter / max(1, a+b-inter) > 0.3:
                                drop = True; break
                    if not drop:
                        kept.append(c)
                kept.sort(key=lambda c: c["fx1"])
                for win_idx, c in enumerate(kept):
                    cv2.rectangle(annotated, (c["fx1"], c["fy1"]), (c["fx2"], c["fy2"]), (0, 255, 0), 1)
                    cx_n = (c["fx1"]+c["fx2"])/2/w
                    cy_n = (c["fy1"]+c["fy2"])/2/h
                    head_crops.append((c["crop"], cx_n, cy_n, ri, win_idx, rconf,
                                       c["top1"], c["top1_conf"], c["probs"]))

        html_parts.append(f'<h2>Frame {fi+1}: {frame_tag} ({w}×{h}, {len(rooms)} 房间 → {len(head_crops)} 头像 AI 锁定到)</h2>')
        html_parts.append(f'<img class="frame-img" src="data:image/jpeg;base64,{img_to_b64(annotated, max_side=900)}">')
        html_parts.append('<table><tr><th>对/错</th><th>crop</th><th>房间.idx</th><th>位置 (cx,cy)</th><th>top-1 预测</th><th>conf</th><th>top-3 候选</th><th>room conf</th></tr>')

        for crop, cx_n, cy_n, ri, win_idx, det_conf, top1_name, top1_conf, probs in head_crops:
            total_cells += 1
            top5_idx = np.argsort(probs)[-5:][::-1]
            top5 = [(clf_names[int(i)], float(probs[int(i)])) for i in top5_idx]
            conf_cls = "high" if top1_conf >= 0.7 else "low" if top1_conf >= 0.5 else "veryLow"
            top3_str = " | ".join(f'{lbl(n)}={p:.2f}' for n, p in top5[:3])

            html_parts.append(
                f'<tr>'
                f'<td class="review"></td>'
                f'<td><img class="crop" src="data:image/jpeg;base64,{img_to_b64(crop, max_side=96)}"></td>'
                f'<td class="pos">R{ri}.W{win_idx}</td>'
                f'<td class="pos">({cx_n:.2f}, {cy_n:.2f})</td>'
                f'<td class="pred">{lbl(top1_name)}</td>'
                f'<td class="conf {conf_cls}">{top1_conf:.3f}</td>'
                f'<td class="alts">{top3_str}</td>'
                f'<td class="pos">{det_conf:.2f}</td>'
                f'</tr>'
            )
        html_parts.append('</table>')

    html_parts.append(f'<h2>合计: {total_cells} 个头像 across {len(frames)} 帧</h2>')
    html_parts.append('</body></html>')

    out_path.write_text("".join(html_parts), encoding="utf-8")
    print(f"\nREPORT: {out_path}")
    print(f"Open in browser: file:///{out_path.as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
