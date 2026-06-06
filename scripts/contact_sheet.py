"""Per-source-run contact sheet for visually auditing an imported dataset.

Why visual: the live YoloOverlay can be burned into older trajectory captures
(boxes + dark label bars composited into the pixels). It lags one tick, so no
pixel/coord heuristic detects it reliably — but it's per-run and unmistakable
to the eye. One thumbnail per source run classifies every run at a glance
(burned frames are dense with colored boxes + text). See
import_traj_weak_cls.py for the full caveat.

Usage:
    py -3 scripts/contact_sheet.py run_v6weak_20260603 [--cols 5 --tw 285]

Reads <dataset>/_source_manifest.jsonl (written by the importer) to group
frames by src_run; falls back to all frames if no manifest. Writes
data/_contact_<dataset>.jpg.
"""
from __future__ import annotations
import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw_images"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="raw_images dataset name, e.g. run_v6weak_20260603")
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--tw", type=int, default=285, help="thumb width px")
    args = ap.parse_args()

    run_dir = RAW_DIR / os.path.basename(args.dataset)
    if not run_dir.is_dir():
        raise SystemExit(f"not found: {run_dir}")

    man_path = run_dir / "_source_manifest.jsonl"
    by_run: dict = defaultdict(list)
    if man_path.exists():
        for line in man_path.read_text(encoding="utf-8").splitlines():
            m = json.loads(line)
            by_run[m.get("src_run", "?")].append(m["frame"])
    else:
        by_run["(all)"] = [os.path.basename(p) for p in glob.glob(str(run_dir / "frame_*.jpg"))]

    picks = []
    for run in sorted(by_run):
        fs = sorted(by_run[run])
        if fs:
            picks.append((run, fs[len(fs) // 2]))

    cols, tw = args.cols, args.tw
    th = tw * 9 // 16
    pad, lbl = 6, 16
    rows = (len(picks) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (tw + pad) + pad, rows * (th + lbl + pad) + pad), (20, 20, 20))
    dr = ImageDraw.Draw(sheet)
    for i, (run, f) in enumerate(picks):
        r, c = divmod(i, cols)
        x, y = pad + c * (tw + pad), pad + r * (th + lbl + pad)
        try:
            sheet.paste(Image.open(run_dir / f).convert("RGB").resize((tw, th)), (x, y + lbl))
        except Exception:
            pass
        dr.text((x + 2, y + 2), f"{run[-6:]} {f[6:12]}", fill=(0, 255, 0))
    out = REPO_ROOT / "data" / f"_contact_{os.path.basename(args.dataset)}.jpg"
    sheet.save(out, quality=78)
    print(f"{len(picks)} thumbs from {len(by_run)} runs -> {out}  ({sheet.size})")


if __name__ == "__main__":
    main()
