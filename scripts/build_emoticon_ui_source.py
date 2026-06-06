"""Stage emoticon_v2 frames as a clean multi-label ui-dataset source.

WHY (negative-label hazard):
  The emoticon_v2 detector dataset (single class Emoticon_Action = cafe headpat
  bubble) was captured on FULL cafe frames that ALSO show cafe UI chrome —
  移动至2号点 / 咖啡厅收益 / 咖啡厅邀请卷 and the 体力/信用点/青辉石 topbar, all of
  which ARE master UI classes but go UNLABELED in the emoticon labels. Dropping
  these ~200 frames into the ui training set as-is would teach the ui model those
  visible-but-unlabeled buttons as background — degrading exactly the weak
  (12-20f) cafe UI classes the Cafe skill depends on.

FIX (teacher auto-label — same convention as head_detector / the flywheel):
  Run the live ui model (model_registry active = v5) over each emoticon frame to
  recover the UI-chrome boxes, then MERGE the emoticon box(es) remapped to the
  new master index for Emoticon_Action. Avatar classes (143..394) are excluded
  from teacher labels — those are the fused_avatar model's domain and the ui
  model would only ever spuriously fire them on cafe student sprites.

  The emoticon bubble itself is a stable yellow marker unchanged across game
  versions, so the 2026-03 frames remain valid positives for that class even
  though their surrounding cafe layout is older.

OUTPUT: data/raw_images/_emoticon_v2/<stem>.jpg + <stem>.txt  (+ classes.txt)
  build_ui_v2.py picks this up via REAL_SOURCES (md5-dedup keeps every frame;
  emoticon frames are 2026-03 captures, disjoint from the 2026-05 ui sources, so
  no dedup collision can drop an emoticon label).

Usage: py scripts/build_emoticon_ui_source.py [--conf 0.30] [--clean]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw_images"
MASTER = RAW / "_classes.txt"
EMO_DS = Path("D:/Project/ml_cache/models/yolo/dataset/emoticon_v2")
OUT = RAW / "_emoticon_v2"
REGISTRY = REPO / "data" / "model_registry.json"

# Student-avatar class span in the master schema — these belong to the
# fused_avatar detector, not the ui model; never let the teacher label them.
AVATAR_LO, AVATAR_HI = 143, 394


def ui_weights() -> str:
    reg = json.loads(REGISTRY.read_text(encoding="utf-8"))
    sec = reg["ui"]
    return sec["versions"][sec["active"]]["path"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.30,
                    help="teacher conf floor (recall-biased: a missed UI box "
                         "becomes a training negative, so keep it low)")
    ap.add_argument("--imgsz", type=int, default=960,
                    help="ui model MUST infer at 960 (training size)")
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()

    master = MASTER.read_text(encoding="utf-8").splitlines()
    try:
        emo_idx = master.index("Emoticon_Action")
    except ValueError:
        print("[!] 'Emoticon_Action' not in master _classes.txt — append it first")
        return 1
    nc = len(master)
    print(f"[schema] master={nc} classes, Emoticon_Action=idx {emo_idx}")

    if args.clean and OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO
    w = ui_weights()
    print(f"[teacher] ui model: {w}  (conf={args.conf}, imgsz={args.imgsz})")
    model = YOLO(w)

    frames = []
    for sp in ("train", "val"):
        idir = EMO_DS / "images" / sp
        ldir = EMO_DS / "labels" / sp
        if not idir.is_dir():
            continue
        for jpg in sorted(idir.glob("*.jpg")):
            frames.append((jpg, ldir / (jpg.stem + ".txt")))
    print(f"[scan] {len(frames)} emoticon frames (train+val)")
    if not frames:
        print("[!] no emoticon frames found")
        return 1

    n_emo_boxes = n_ui_boxes = n_avatar_skipped = n_out = 0
    for jpg, txt in frames:
        # emoticon boxes → remap cls 0 → emo_idx
        emo_lines = []
        if txt.exists():
            for ln in txt.read_text(encoding="utf-8").splitlines():
                p = ln.split()
                if len(p) >= 5:
                    emo_lines.append(
                        f"{emo_idx} {float(p[1]):.6f} {float(p[2]):.6f} "
                        f"{float(p[3]):.6f} {float(p[4]):.6f}"
                    )
        n_emo_boxes += len(emo_lines)

        # teacher UI-chrome boxes from the live ui model
        ui_lines = []
        res = model(str(jpg), conf=args.conf, imgsz=args.imgsz, verbose=False)
        for r in res:
            for b in r.boxes:
                cid = int(b.cls[0])
                if AVATAR_LO <= cid <= AVATAR_HI:
                    n_avatar_skipped += 1
                    continue
                if cid >= nc or cid == emo_idx:
                    continue
                cx, cy, bw, bh = b.xywhn[0].tolist()
                ui_lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        n_ui_boxes += len(ui_lines)

        lines = ui_lines + emo_lines
        shutil.copy2(jpg, OUT / jpg.name)
        (OUT / (jpg.stem + ".txt")).write_text(
            ("\n".join(lines) + "\n") if lines else "", encoding="utf-8"
        )
        n_out += 1

    # classes.txt = full master (prefix-matches build_ui_v2's schema-drift check)
    (OUT / "classes.txt").write_text("\n".join(master) + "\n", encoding="utf-8")

    print(f"[done] {OUT}")
    print(f"  frames={n_out}  emoticon_boxes={n_emo_boxes}  "
          f"teacher_ui_boxes={n_ui_boxes}  avatar_boxes_skipped={n_avatar_skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
