"""Batch YOLO-prefill an ENTIRE run's frames with the active ui model.

Writes a YOLO label .txt (`cls cx cy w h`, master-class index) next to every
image so the human only has to CORRECT in the dashboard (fix mislabels via
select-box→click-class, ADD the icons the model misses on new layouts, DELETE
wrong ones) instead of drawing from scratch.

Dedup: any-cls IoU>0.6, keep higher conf → never two boxes crowding one thing
(the "不能让 bbox 挤一个东西" rule). Skips frames that already have a non-empty
label (unless overwrite) so it never clobbers human edits.

Reused by the dashboard endpoint /api/v1/datasets/yolo_prefill_run.

Usage:
  py scripts/yolo_prefill_run.py run_20260529_123209
  py scripts/yolo_prefill_run.py traj/run_xxx --conf 0.25 --overwrite
"""
from __future__ import annotations
import sys, glob, json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw_images"
TRAJ = REPO / "data" / "trajectories"


def resolve_img_dir(dataset: str) -> Path:
    d = (TRAJ / dataset[5:]) if dataset.startswith("traj/") else (RAW / dataset)
    return d / "frames" if (d / "frames").is_dir() else d


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    aa = (ax2 - ax1) * (ay2 - ay1); bb = (bx2 - bx1) * (by2 - by1)
    return inter / (aa + bb - inter + 1e-9)


_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        from ultralytics import YOLO
        reg = json.loads((REPO / "data" / "model_registry.json").read_text(encoding="utf-8"))
        ui = reg["ui"]
        _MODEL = YOLO(ui["versions"][ui["active"]]["path"])
    return _MODEL


def prefill_run(img_dir, *, conf: float = 0.25, imgsz: int = 960,
                overwrite: bool = False, progress=None) -> dict:
    """Prefill every *.jpg under img_dir. Returns {written, skipped, total}."""
    img_dir = Path(img_dir)
    m = get_model()
    imgs = sorted(glob.glob(str(img_dir / "*.jpg")))
    written = skipped = 0
    CHUNK = 64
    for ci in range(0, len(imgs), CHUNK):
        chunk = imgs[ci:ci + CHUNK]
        todo = []
        for p in chunk:
            lp = Path(p).with_suffix(".txt")
            if (not overwrite) and lp.exists():
                try:
                    if lp.read_text(encoding="utf-8").strip():
                        skipped += 1
                        continue
                except Exception:
                    pass
            todo.append(p)
        if not todo:
            continue
        for p, r in zip(todo, m.predict(todo, stream=True, imgsz=imgsz,
                                        conf=conf, device=0, verbose=False)):
            raw = []
            for b in r.boxes:
                cid = int(b.cls[0]); sc = float(b.conf[0])
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                if x2 <= x1 or y2 <= y1:
                    continue
                raw.append((sc, cid, [x1, y1, x2, y2]))
            raw.sort(key=lambda t: -t[0])
            kept = []
            for sc, cid, box in raw:
                if any(_iou(box, k[2]) > 0.6 for k in kept):
                    continue
                kept.append((sc, cid, box))
            h, w = r.orig_shape  # (height, width)
            lines = []
            for sc, cid, (x1, y1, x2, y2) in kept:
                cx = ((x1 + x2) / 2) / w; cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w; bh = (y2 - y1) / h
                lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            Path(p).with_suffix(".txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            written += 1
            if progress and written % 50 == 0:
                progress(written, len(imgs))
    return {"written": written, "skipped": skipped, "total": len(imgs)}


def main():
    argv = sys.argv[1:]
    if not argv:
        print("usage: yolo_prefill_run.py <dataset> [--conf 0.25] [--overwrite]")
        return
    dataset = argv[0]; conf = 0.25; overwrite = False
    i = 1
    while i < len(argv):
        if argv[i] == "--conf":
            conf = float(argv[i + 1]); i += 2; continue
        if argv[i] == "--overwrite":
            overwrite = True; i += 1; continue
        i += 1
    img_dir = resolve_img_dir(dataset)
    if not img_dir.is_dir():
        print(f"dataset dir not found: {img_dir}")
        return
    print(f"prefill {img_dir}  conf={conf}  overwrite={overwrite}")
    res = prefill_run(img_dir, conf=conf, overwrite=overwrite,
                      progress=lambda w, t: print(f"  ...{w}/{t}"))
    print(res)


if __name__ == "__main__":
    main()
