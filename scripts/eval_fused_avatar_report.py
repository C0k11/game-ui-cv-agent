"""Human eval for fused_avatar detector: render truth vs prediction overlay HTML.

Same pattern as eval_static_ui_report.py but targets the fused multi-class avatar
detector.  For each frame in the val split (or any source dir of labeled .jpg):
  - Load truth bboxes from .txt
  - Run fused_avatar_yolo26m inference
  - Overlay BOTH on the image (truth = green, pred = red w/ class name)
  - Compute IoU matches → categorize:
      ✓ MATCHED  (truth bbox has overlapping pred with same class, IoU >= 0.5)
      ✗ MISS     (truth bbox has no matching pred → false negative)
      ⚠ FP       (pred bbox with no matching truth → false positive)
      ⚠ WRONG    (matched bbox but wrong class — character ID error)
  - Render summary stats per frame

Sort: frames with lowest recall first so the worst cases are visible at the top.

Usage:
    py scripts/eval_fused_avatar_report.py                  # all 49 val frames
    py scripts/eval_fused_avatar_report.py --conf 0.15      # lower threshold
    py scripts/eval_fused_avatar_report.py --frames 20      # cap render count
"""
from __future__ import annotations

import argparse
import base64
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(r"D:\Project\ml_cache\models\yolo\runs\fused_avatar_yolo26x\weights\best.pt")
DATA_YAML = Path(r"D:\Project\ml_cache\models\yolo\dataset\fused_avatar_v1\data.yaml")


def img_to_b64(img: np.ndarray, max_w: int = 1100, quality: int = 75) -> str:
    h, w = img.shape[:2]
    if w > max_w:
        scale = max_w / w
        img = cv2.resize(img, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""


def iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    return inter / max(1e-6, (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter)


def load_truth_boxes(label_path: Path, img_w: int, img_h: int, names: List[str]) -> List[Dict]:
    if not label_path.exists():
        return []
    try:
        text = label_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            text = label_path.read_text(encoding="utf-16")
        except Exception:
            return []
    out = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 5 or not parts[0].lstrip("-").isdigit():
            continue
        try:
            ci = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
        except ValueError:
            continue
        if not (0 <= ci < len(names)):
            continue
        x1 = (cx - w/2) * img_w; y1 = (cy - h/2) * img_h
        x2 = (cx + w/2) * img_w; y2 = (cy + h/2) * img_h
        out.append({"cls_idx": ci, "name": names[ci], "xyxy": (x1, y1, x2, y2)})
    return out


def match_truth_to_pred(truths: List[Dict], preds: List[Dict], iou_thr: float = 0.5):
    """Match GT boxes to predictions with conf-aware priority.

    Algorithm:
      1. Sort preds by confidence desc (high-conf preds get first pick)
      2. For each pred (in conf order), find best unmatched GT with IoU>=thr
      3. If found, match pred→GT; mark both as used
      4. Remaining unmatched GTs = misses, remaining unmatched preds = FPs

    This avoids the "low-conf duplicate steals match from high-conf real
    prediction" quirk when NMS-free models output near-identical boxes.
    """
    used_pred = set()
    used_gt = set()
    matches, wrong_cls = [], []
    # Sort pred indices by confidence desc
    pred_order = sorted(range(len(preds)), key=lambda i: -preds[i].get("conf", 0))
    for pi in pred_order:
        if pi in used_pred:
            continue
        best_t, best_iou = -1, 0.0
        for ti, t in enumerate(truths):
            if ti in used_gt:
                continue
            v = iou(t["xyxy"], preds[pi]["xyxy"])
            if v > best_iou:
                best_iou, best_t = v, ti
        if best_iou >= iou_thr and best_t >= 0:
            used_pred.add(pi)
            used_gt.add(best_t)
            if preds[pi]["name"] == truths[best_t]["name"]:
                matches.append((best_t, pi))
            else:
                wrong_cls.append((best_t, pi))
    misses = [ti for ti in range(len(truths)) if ti not in used_gt]
    fps = [pi for pi in range(len(preds)) if pi not in used_pred]
    return matches, misses, fps, wrong_cls


def draw_overlay(img, truths, preds, matches, misses, fps, wrong_cls):
    out = img.copy()
    matched_truth = {ti for ti, _ in matches}
    wrong_truth = {ti for ti, _ in wrong_cls}
    for ti, t in enumerate(truths):
        x1, y1, x2, y2 = map(int, t["xyxy"])
        if ti in matched_truth:
            color, lw = (0, 255, 0), 1   # green = correct
        elif ti in wrong_truth:
            color, lw = (0, 165, 255), 2  # orange = wrong class
        else:
            color, lw = (0, 0, 255), 3   # red = miss
        cv2.rectangle(out, (x1, y1), (x2, y2), color, lw)
        if ti in matched_truth:
            cv2.putText(out, t["name"], (x1, max(y1-3, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    for pi in fps:
        p = preds[pi]
        x1, y1, x2, y2 = map(int, p["xyxy"])
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(out, f"FP {p['name']} {p['conf']:.2f}",
                    (x1, max(y1-3, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)
    for _, pi in wrong_cls:
        p = preds[pi]
        x1, y1, x2, y2 = map(int, p["xyxy"])
        cv2.putText(out, f"got: {p['name']} {p['conf']:.2f}",
                    (x1, max(y1-3, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(MODEL_PATH))
    ap.add_argument("--data", default=str(DATA_YAML))
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.is_file():
        print(f"[err] model not found: {model_path}")
        return 1

    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))
    cfg = yaml.safe_load(Path(args.data).read_text(encoding="utf-8"))
    names = [cfg["names"][i] for i in sorted(cfg["names"].keys())]
    print(f"classes: {len(names)}")

    img_dir = Path(cfg["path"]) / "images" / args.split
    lbl_dir = Path(cfg["path"]) / "labels" / args.split
    images = sorted(img_dir.glob("*.jpg"))
    if args.frames:
        images = images[:args.frames]
    print(f"frames: {len(images)} from {args.split} split")

    out_path = Path(args.out) if args.out else (REPO / "data" / "yolo_datasets" / "fused_avatar_eval.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tot_truth = tot_match = tot_miss = tot_fp = tot_wrong = 0
    per_class_truth: Dict[str, int] = defaultdict(int)
    per_class_match: Dict[str, int] = defaultdict(int)
    per_class_fp: Dict[str, int] = defaultdict(int)
    per_class_wrong_pred: Dict[str, int] = defaultdict(int)  # predicted as this when actually other
    frame_blocks = []

    for fi, img_path in enumerate(images):
        img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]

        truths = load_truth_boxes(lbl_dir / (img_path.stem + ".txt"), w, h, names)
        det = model(img, conf=args.conf, verbose=False)[0]
        preds = []
        for b in det.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            preds.append({
                "cls_idx": int(b.cls[0]),
                "name": names[int(b.cls[0])],
                "xyxy": (x1, y1, x2, y2),
                "conf": float(b.conf[0]),
            })

        matches, misses, fps, wrong = match_truth_to_pred(truths, preds, args.iou_thr)
        tot_truth += len(truths); tot_match += len(matches)
        tot_miss += len(misses); tot_fp += len(fps); tot_wrong += len(wrong)
        for t in truths:
            per_class_truth[t["name"]] += 1
        for ti, _ in matches:
            per_class_match[truths[ti]["name"]] += 1
        for pi in fps:
            per_class_fp[preds[pi]["name"]] += 1
        for ti, pi in wrong:
            per_class_wrong_pred[preds[pi]["name"]] += 1

        annotated = draw_overlay(img, truths, preds, matches, misses, fps, wrong)
        ann_b64 = img_to_b64(annotated)
        miss_names = sorted({truths[ti]["name"] for ti in misses})
        fp_list = sorted([f'{preds[pi]["name"]}({preds[pi]["conf"]:.2f})' for pi in fps])
        wrong_list = [f'{truths[ti]["name"]}→{preds[pi]["name"]}({preds[pi]["conf"]:.2f})'
                      for ti, pi in wrong]
        frame_blocks.append({
            "name": img_path.name,
            "img_b64": ann_b64,
            "n_truth": len(truths),
            "n_match": len(matches),
            "n_miss": len(misses),
            "n_fp": len(fps),
            "n_wrong": len(wrong),
            "miss_names": miss_names,
            "fp_list": fp_list,
            "wrong_list": wrong_list,
        })

    # Sort: worst-recall frames first
    frame_blocks.sort(key=lambda b: (b["n_match"] / max(1, b["n_truth"])))

    overall_recall = tot_match / max(1, tot_truth)
    fp_rate = tot_fp / max(1, tot_match + tot_fp)

    html = [f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>fused_avatar eval</title>
<style>
body {{ font-family: system-ui, sans-serif; background:#0f1115; color:#e0e0e0; padding:20px; max-width:1400px; margin:0 auto }}
h1, h2 {{ color:#fff }}
h2 {{ margin-top:30px; padding:6px 10px; background:#1a1d24; border-radius:6px; font-size:15px }}
.summary {{ display:flex; gap:16px; flex-wrap:wrap; margin:15px 0 }}
.summary div {{ padding:10px 16px; background:#1a1d24; border-radius:6px; font-family:monospace }}
.legend {{ background:#1a1d24; padding:12px 16px; border-radius:6px; margin:10px 0; line-height:1.8 }}
.legend span {{ font-family:monospace; padding:1px 6px; border-radius:3px; margin-right:8px }}
.frame-img {{ display:block; max-width:100%; border-radius:6px; margin:6px 0 }}
.stats {{ font-family:monospace; font-size:13px; color:#bbb; line-height:1.6 }}
.miss {{ color:#fca5a5 }} .fp {{ color:#ee82ee }} .wrong {{ color:#fb923c }} .good {{ color:#86efac }}
table {{ border-collapse:collapse; font-size:13px }} th, td {{ padding:4px 10px }}
th {{ background:#1a1d24; text-align:left; cursor:pointer }}
tr:nth-child(even) {{ background:#161922 }}
</style></head><body>
<h1>fused_avatar_26m 检测 — 人工审核</h1>
<div class="legend">
<span style="background:#22c55e;color:#000">绿 = 正确 (匹配 + 类对)</span>
<span style="background:#fb923c;color:#000">橙 = 类错 (IoU 对但类别错)</span>
<span style="background:#ef4444;color:#fff">红 = 漏检 (truth 有, pred 没有)</span>
<span style="background:#d946ef;color:#fff">紫 = 误检 (pred 但无 truth)</span>
</div>
<div class="summary">
<div><b>整体</b> truth={tot_truth} | match={tot_match} | miss={tot_miss} | fp={tot_fp} | wrong-cls={tot_wrong}</div>
<div><b>Recall</b> {100*overall_recall:.2f}% (match / truth)</div>
<div><b>FP rate</b> {100*fp_rate:.2f}%</div>
<div><b>Frames</b> {len(images)}</div>
<div><b>Conf</b> {args.conf}, IoU thr {args.iou_thr}</div>
</div>
<p style="color:#888">按"召回率从低到高"排序 (最糟糕的帧最上面)</p>
"""]
    for fb in frame_blocks:
        recall = fb['n_match'] / max(1, fb['n_truth'])
        html.append(f'<h2>{fb["name"]} — recall {100*recall:.0f}% '
                    f'(match {fb["n_match"]}/{fb["n_truth"]}, miss {fb["n_miss"]}, fp {fb["n_fp"]}, wrong-cls {fb["n_wrong"]})</h2>')
        html.append(f'<img class="frame-img" src="data:image/jpeg;base64,{fb["img_b64"]}">')
        stats = []
        if fb["miss_names"]:
            stats.append(f'<div class="stats miss">漏检 ({fb["n_miss"]}): {", ".join(fb["miss_names"])}</div>')
        if fb["wrong_list"]:
            stats.append(f'<div class="stats wrong">类错: {", ".join(fb["wrong_list"][:15])}</div>')
        if fb["fp_list"]:
            stats.append(f'<div class="stats fp">误检 (top 10): {", ".join(fb["fp_list"][:10])}</div>')
        if not stats:
            stats.append('<div class="stats good">完美 ✓</div>')
        html.extend(stats)

    # Per-class table (sorted by recall ascending — worst first)
    html.append('<h2>Per-class recall (worst → best)</h2>')
    html.append('<table><tr><th>class</th><th>truth</th><th>matched</th><th>recall</th>'
                '<th>FP as this</th><th>wrong-pred-here</th></tr>')
    for cls in sorted(per_class_truth.keys(),
                       key=lambda c: per_class_match[c] / max(1, per_class_truth[c])):
        t = per_class_truth[cls]
        m = per_class_match[cls]
        rec = m / max(1, t)
        bg = '#22c55e' if rec >= 0.7 else ('#fbbf24' if rec >= 0.4 else '#ef4444')
        html.append(f'<tr><td>{cls}</td>'
                    f'<td style="text-align:right">{t}</td>'
                    f'<td style="text-align:right">{m}</td>'
                    f'<td style="background:{bg};color:#000;text-align:right">{100*rec:.0f}%</td>'
                    f'<td style="text-align:right">{per_class_fp.get(cls, 0)}</td>'
                    f'<td style="text-align:right">{per_class_wrong_pred.get(cls, 0)}</td></tr>')
    html.append('</table></body></html>')

    out_path.write_text("".join(html), encoding="utf-8")
    print(f"\nREPORT: {out_path}")
    print(f"Open: file:///{out_path.as_posix()}")
    print()
    print(f"Overall recall: {100*overall_recall:.2f}%  ({tot_match}/{tot_truth})")
    print(f"FP rate: {100*fp_rate:.2f}%")
    print(f"Wrong-class: {tot_wrong}/{tot_truth} = {100*tot_wrong/max(1,tot_truth):.1f}%")
    print(f"Per-class with truth > 0: {len(per_class_truth)} (out of {len(names)} model classes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
