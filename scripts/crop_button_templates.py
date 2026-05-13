"""Auto-crop UI button templates from trajectories using OCR anchors.

For each target button label (e.g. "任務開始"), this script:
  1. Scans all `data/trajectories/run_*/tick_*.json` for OCR boxes matching
     the label at high confidence
  2. Picks the highest-confidence sample
  3. Crops the corresponding `tick_*.jpg` using the OCR bbox with a small
     padding so the template captures the full button (not just glyphs)
  4. Saves to the target captures directory with a descriptive name

Usage:
    py scripts/crop_button_templates.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
TRAJ_DIR = REPO / "data" / "trajectories"
CAP_DIR = REPO / "data" / "captures"


# (label, output_path, padding_norm, min_conf, prefer_region)
# `padding_norm` is fraction of the OCR box (x,y); BA buttons typically have
# 15-25% padding around the text so the gradient/border is captured.
# `prefer_region` (x1,y1,x2,y2) narrows the candidate selection.
TARGETS = [
    {
        # Sweep start button: cyan bg with "掃蕩開始" — OCR sometimes misses
        # the 掃 character.  Accept the most stable partial first.
        "label_variants": ["掃蕩開始", "扫荡开始", "掃荡開始", "扫蕩开始",
                           "捅荡開始", "捅荡开始", "捅蕩開始", "捅蕩开始",
                           "荡開始", "荡开始"],
        "out": CAP_DIR / "activity" / "common" / "sweep-start-button.png",
        "pad_x": 0.30, "pad_y": 0.50,
        "min_conf": 0.80,
        "prefer_region": (0.55, 0.50, 0.95, 0.75),
    },
    {
        # Formation sortie (出擊) — bottom-right yellow button after team set
        "label_variants": ["出擊", "出击", "出撃", "出擎"],
        "out": CAP_DIR / "normal_task" / "sortie-button.png",
        "pad_x": 0.40, "pad_y": 0.50,
        "min_conf": 0.85,
        "prefer_region": (0.70, 0.75, 1.00, 0.99),
    },
]


def find_candidates(target: dict) -> list:
    """Return list of (conf, jpg_path, ocr_box) for matching OCR detections."""
    hits = []
    json_files = sorted(glob.glob(str(TRAJ_DIR / "run_*" / "tick_*.json")))
    print(f"  scanning {len(json_files)} tick JSONs for {target['label_variants']!r}",
          flush=True)
    for f in json_files:
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        for b in d.get("ocr_boxes", []):
            if b.get("confidence", b.get("conf", 0)) < target["min_conf"]:
                continue
            if b["text"].strip() not in target["label_variants"]:
                continue
            r = target["prefer_region"]
            cx = (b["x1"] + b["x2"]) / 2
            cy = (b["y1"] + b["y2"]) / 2
            if not (r[0] <= cx <= r[2] and r[1] <= cy <= r[3]):
                continue
            jpg = f.replace(".json", ".jpg")
            if not os.path.exists(jpg):
                continue
            hits.append((b.get("conf", b.get("confidence", 0)), jpg, b))
    hits.sort(key=lambda t: -t[0])
    return hits


def crop_button(jpg: str, box: dict, pad_x: float, pad_y: float) -> np.ndarray:
    """Crop a button-sized region centered on the OCR box, with padding."""
    img = cv2.imdecode(np.fromfile(jpg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    H, W = img.shape[:2]
    # OCR box in normalized coords
    x1n, y1n, x2n, y2n = box["x1"], box["y1"], box["x2"], box["y2"]
    bw = (x2n - x1n)
    bh = (y2n - y1n)
    # Expand by pad fraction on each side
    nx1 = max(0.0, x1n - bw * pad_x)
    ny1 = max(0.0, y1n - bh * pad_y)
    nx2 = min(1.0, x2n + bw * pad_x)
    ny2 = min(1.0, y2n + bh * pad_y)
    px1, py1, px2, py2 = int(nx1 * W), int(ny1 * H), int(nx2 * W), int(ny2 * H)
    crop = img[py1:py2, px1:px2]
    return crop


def main() -> None:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass
    for tgt in TARGETS:
        print(f"\n=== {tgt['label_variants'][0]} → {tgt['out'].name} ===", flush=True)
        cands = find_candidates(tgt)
        print(f"  {len(cands)} candidate OCR hits")
        if not cands:
            print(f"  ⚠ no hits — skip")
            continue
        best_conf, best_jpg, best_box = cands[0]
        print(f"  best: conf={best_conf:.3f}  frame={Path(best_jpg).parent.name}/{Path(best_jpg).name}")
        crop = crop_button(best_jpg, best_box, tgt["pad_x"], tgt["pad_y"])
        if crop is None or crop.size == 0:
            print(f"  ⚠ crop failed")
            continue
        tgt["out"].parent.mkdir(parents=True, exist_ok=True)
        # cv2 doesn't like unicode paths on Windows — use imencode + open
        ok, buf = cv2.imencode(".png", crop)
        if not ok:
            print(f"  ⚠ encode failed")
            continue
        with open(tgt["out"], "wb") as f:
            f.write(buf.tobytes())
        h, w = crop.shape[:2]
        print(f"  ✓ saved {w}×{h} → {tgt['out']}")


if __name__ == "__main__":
    main()
