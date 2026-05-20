"""Data flywheel: mine hard examples from trajectory using a trained fused_avatar model.

Runs the trained fused_avatar detector on every trajectory frame and identifies
predictions that are "interesting for the next training round" — typically:

  * Medium-conf detections (0.10-0.45): model knows something is there but isn't
    sure which character → classifier needs more samples to disambiguate.
  * Confidence-split detections: same bbox region has >=2 predictions of different
    classes, none above 0.5 → classifier is choosing between similar-looking
    characters; user review resolves ambiguity.
  * Rare-class predictions: model predicts a class with very few train samples,
    even at high conf → useful to verify and add to train.

For each hard example, the frame is copied to
  data/raw_images/_hard_examples_<tag>/frames/<runname>__<tick>.jpg
and a .txt label is PRE-FILLED with model predictions (user just reviews and
corrects in dashboard).  These frames then naturally feed into the next
build_fused_avatar_dataset.py run as train data.

Usage:
    py scripts/mine_hard_examples.py                       # default v1 mining
    py scripts/mine_hard_examples.py --tag v2 --max 200
    py scripts/mine_hard_examples.py --min-conf 0.15 --max-conf 0.40
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[1]
TRAJECTORIES = REPO / "data" / "trajectories"
RAW_IMAGES = REPO / "data" / "raw_images"
DEFAULT_MODEL = Path(r"D:\Project\ml_cache\models\yolo\runs\fused_avatar_yolo26m\weights\best.pt")
DEFAULT_DATA_YAML = Path(r"D:\Project\ml_cache\models\yolo\dataset\fused_avatar_v1\data.yaml")
MASTER_FILE = RAW_IMAGES / "_classes.txt"
MASTER_UI_BOUNDARY = 143


def imread_u(p: Path):
    try:
        buf = np.fromfile(str(p), dtype=np.uint8)
        if buf.size == 0:
            return None
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


def find_avatar_context_ticks(limit: int) -> List[Path]:
    """Walk trajectory, return JPGs from contexts that likely contain avatars.

    Mining only makes sense in frames the trained detector should fire on:
    schedule popup, cafe invite, arena squad, bounty squad, momotalk.
    """
    AVATAR_CONTEXTS = {
        ("Schedule", "check_roster"),
        ("Schedule", "execute"),
        ("Cafe", "invite"),
        ("Arena", "fight"),
        ("Arena", "select_opponent"),
        ("Bounty", "sweep"),
        ("Bounty", "select_stage"),
    }
    out: List[Path] = []
    if not TRAJECTORIES.is_dir():
        return out
    runs = sorted(
        [r for r in TRAJECTORIES.iterdir() if r.is_dir() and r.name.startswith("run_")],
        reverse=True,
    )
    for run in runs:
        if len(out) >= limit:
            break
        for tj in sorted(run.glob("tick_*.json")):
            if len(out) >= limit:
                break
            try:
                d = json.loads(tj.read_text(encoding="utf-8"))
            except Exception:
                continue
            key = (d.get("skill") or "", d.get("sub_state") or "")
            if key not in AVATAR_CONTEXTS:
                continue
            jpg = tj.with_suffix(".jpg")
            if jpg.exists() and jpg.stat().st_size > 5000:
                out.append(jpg)
    return out


def is_confidence_split(boxes_xyxy, classes, confs, iou_thr: float = 0.5) -> bool:
    """True if any pair of boxes overlap (IoU >= thr) but have different classes
    and both conf < 0.5 — classic 'classifier confused' pattern."""
    n = len(boxes_xyxy)
    for i in range(n):
        if confs[i] >= 0.5:
            continue
        for j in range(i + 1, n):
            if confs[j] >= 0.5:
                continue
            if classes[i] == classes[j]:
                continue
            ax1, ay1, ax2, ay2 = boxes_xyxy[i]
            bx1, by1, bx2, by2 = boxes_xyxy[j]
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = (ix2 - ix1) * (iy2 - iy1)
            union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
            if inter / max(1, union) >= iou_thr:
                return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--data", default=str(DEFAULT_DATA_YAML))
    ap.add_argument("--tag", default="v1", help="Output dir tag → _hard_examples_<tag>/")
    ap.add_argument("--scan-limit", type=int, default=2000,
                    help="Max trajectory frames to scan for hard examples")
    ap.add_argument("--max", type=int, default=200,
                    help="Max hard-example frames to output")
    ap.add_argument("--min-conf", type=float, default=0.10,
                    help="Predictions below this conf are ignored (too noisy)")
    ap.add_argument("--max-conf", type=float, default=0.45,
                    help="Predictions above this conf are auto-accepted (skip from hard)")
    ap.add_argument("--rare-class-thr", type=int, default=5,
                    help="Class is 'rare' if <= this many train samples")
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.is_file():
        print(f"[err] model not found: {model_path}")
        return 1

    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))
    cfg = yaml.safe_load(Path(args.data).read_text(encoding="utf-8"))
    names = [cfg["names"][i] for i in sorted(cfg["names"].keys())]
    name_to_idx = {n: i for i, n in enumerate(names)}

    # Count train samples per class for rare-class flagging
    master = [c.strip() for c in MASTER_FILE.read_text(encoding="utf-8").splitlines()
              if c.strip()]
    train_count: Dict[str, int] = defaultdict(int)
    for run in RAW_IMAGES.iterdir():
        if not run.is_dir() or run.name.startswith("_"):
            continue
        sub = run / "frames" if (run / "frames").is_dir() else run
        for txt in sub.glob("*.txt"):
            if txt.name == "classes.txt":
                continue
            try:
                c = txt.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                try:
                    c = txt.read_text(encoding="utf-16")
                except Exception:
                    continue
            for line in c.splitlines():
                p = line.strip().split()
                if len(p) >= 5 and p[0].isdigit():
                    idx = int(p[0])
                    if MASTER_UI_BOUNDARY <= idx < len(master):
                        train_count[master[idx]] += 1
    rare_classes = {n for n in names if train_count.get(n, 0) <= args.rare_class_thr}
    print(f"[init] {len(rare_classes)} rare classes (<= {args.rare_class_thr} train samples)")

    # Walk trajectory
    print(f"Scanning up to {args.scan_limit} trajectory frames...")
    candidate_jpgs = find_avatar_context_ticks(args.scan_limit)
    print(f"Found {len(candidate_jpgs)} avatar-context frames")

    out_dir = RAW_IMAGES / f"_hard_examples_{args.tag}" / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    scored: List[Tuple[float, Path, str, List[str]]] = []  # (score, jpg, reason, yolo_lines)

    for i, jpg in enumerate(candidate_jpgs):
        img = imread_u(jpg)
        if img is None:
            continue
        H, W = img.shape[:2]
        det = model(img, conf=args.min_conf, verbose=False)[0]
        if len(det.boxes) == 0:
            continue

        boxes_xyxy = [b.xyxy[0].tolist() for b in det.boxes]
        classes = [int(b.cls[0]) for b in det.boxes]
        confs = [float(b.conf[0]) for b in det.boxes]
        cls_names_pred = [names[c] for c in classes]

        # Decide if this frame is a "hard example"
        score = 0.0
        reasons = []
        # 1. Has medium-conf detection
        med_count = sum(1 for c in confs if args.min_conf <= c < args.max_conf)
        if med_count > 0:
            score += med_count * 1.0
            reasons.append(f"medium-conf x{med_count}")
        # 2. Confidence split (classifier confused)
        if is_confidence_split(boxes_xyxy, classes, confs):
            score += 3.0
            reasons.append("conf-split")
        # 3. Predicts a rare class
        for cls_name, conf in zip(cls_names_pred, confs):
            if cls_name in rare_classes and conf >= 0.20:
                score += 2.0
                reasons.append(f"rare:{cls_name}")
                break  # only count once per frame

        if score < 1.0:
            continue

        # Compose yolo label lines (model predictions as starting point)
        lines = []
        for (x1, y1, x2, y2), cls_idx, conf in zip(boxes_xyxy, classes, confs):
            cx = (x1 + x2) / 2 / W
            cy = (y1 + y2) / 2 / H
            w = (x2 - x1) / W
            h = (y2 - y1) / H
            # Look up the GLOBAL master index for this class name (not the fused idx)
            cls_name = names[cls_idx]
            try:
                master_idx = master.index(cls_name)
            except ValueError:
                continue
            lines.append(f"{master_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        if not lines:
            continue
        scored.append((score, jpg, " | ".join(reasons), lines))

        if (i + 1) % 200 == 0:
            print(f"  scanned {i+1}/{len(candidate_jpgs)} | hard so far: {len(scored)}")

    # Rank by score, take top-N
    scored.sort(key=lambda x: -x[0])
    selected = scored[:args.max]
    print(f"\nSelected {len(selected)} hard examples (from {len(scored)} candidates)")

    # Emit
    emit_count = 0
    audit_entries = []
    for score, jpg, reason, lines in selected:
        stem = f"{jpg.parent.name}__{jpg.stem}"
        out_jpg = out_dir / f"{stem}.jpg"
        out_txt = out_dir / f"{stem}.txt"
        if out_jpg.exists():
            continue
        shutil.copy2(jpg, out_jpg)
        out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
        audit_entries.append({
            "frame": out_jpg.name,
            "score": round(score, 2),
            "reason": reason,
            "n_preds": len(lines),
        })
        emit_count += 1

    # Write audit
    audit_path = out_dir.parent / "audit.json"
    audit_path.write_text(json.dumps(audit_entries, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    print(f"\nEmitted {emit_count} hard examples → {out_dir}")
    print(f"Audit:    {audit_path}")
    print()
    print(f"Next: open dashboard, select '_hard_examples_{args.tag}' dataset, review")
    print(f"and correct labels (model predictions are pre-filled).  Frames you")
    print(f"keep automatically join train for next build_fused_avatar_dataset.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
