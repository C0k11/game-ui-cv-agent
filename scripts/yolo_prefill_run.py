"""Batch YOLO-prefill a run's frames — pick WHICH detector, and ACCUMULATE
across passes so one frame can carry boxes from several models.

Each pass runs ONE model (ui / fused_avatar / emoticon / battle), remaps its
local class ids to the shared master `_classes.txt` index BY NAME, keeps only
the boxes inside that model's authoritative master-class span, and MERGES them
into the per-image label .txt — boxes already written by other passes are
preserved (only same-class IoU>0.6 duplicates are dropped). This is exactly the
cross-teacher labeling the unified 26x model needs: the ui pass stamps UI boxes,
the avatar pass adds head boxes on the SAME frame without erasing the UI ones.

Modes:
  merge      (default) append this model's boxes, never touch other classes
  overwrite  per-model replace: drop ONLY this model's own-span boxes and write
             fresh ones; OTHER models' boxes (ui/avatar) stay. Use to re-run one
             teacher at a higher conf to fix ITS mistakes (e.g. emoticon
             false-positives outside cafe).
  skip       leave frames that already have a non-empty label untouched

Recommended flow for a fresh dataset (build the unified training set):
  ui pass → avatar pass → emoticon pass  (all merge, same frames)  → human
  精修 ONCE in the dashboard → train. Run the teacher passes BEFORE精修, not
  after (merge would re-add boxes a human deliberately deleted).

Label format: `cls cx cy w h` normalized, cls = 0-based MASTER index — the
format server/app.py:list_dataset_images parses (a 6th column is read as OBB
angle there, so we never emit one).

Usage:
  py scripts/yolo_prefill_run.py run_xxx                        # ui, merge
  py scripts/yolo_prefill_run.py run_xxx --model fused_avatar   # +head boxes
  py scripts/yolo_prefill_run.py run_xxx --model emoticon
  py scripts/yolo_prefill_run.py traj/run_xxx --conf 0.3 --mode overwrite

Reused by the dashboard endpoint /api/v1/datasets/yolo_prefill_run.
"""
from __future__ import annotations
import sys, glob, json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw_images"
TRAJ = REPO / "data" / "trajectories"
MASTER_FILE = RAW / "_classes.txt"

# Per-tag inference imgsz — mirrors brain/pipeline.py _IMGSZ_BY_TAG. Wrong imgsz
# silently yields 0 detections (ui @1920 = nothing), so pin it per model.
_IMGSZ_BY_TAG = {"ui": 960, "avatar": 960, "battle": 960, "cafe": 640}

# registry model-key -> pipeline tag (for imgsz + span lookup)
_KEY_TO_TAG = {"ui": "ui", "fused_avatar": "avatar",
               "emoticon": "cafe", "battle_heads": "battle"}

# Authoritative master-class span per detector (0-based master indices).
# master layout: [0,142]=UI-A, [143,393]=avatars(251), [394,450]=UI-B,
# 451=Emoticon_Action. A pass keeps ONLY boxes inside its span so the ui model
# can't stamp a spurious avatar class onto a cafe sprite (and vice-versa).
def _ui_span(i: int) -> bool:       return (0 <= i <= 142) or (395 <= i <= 450)
def _avatar_span(i: int) -> bool:   return 143 <= i <= 394   # 含柚子战斗(394, fused 第252角色)
def _emoticon_span(i: int) -> bool: return i == 451
_OWNS = {"ui": _ui_span, "avatar": _avatar_span,
         "cafe": _emoticon_span, "battle": (lambda i: False)}


def resolve_img_dir(dataset: str) -> Path:
    d = (TRAJ / dataset[5:]) if dataset.startswith("traj/") else (RAW / dataset)
    return d / "frames" if (d / "frames").is_dir() else d


_MASTER_IDX = None
def master_idx() -> dict:
    global _MASTER_IDX
    if _MASTER_IDX is None:
        names = [c.strip() for c in MASTER_FILE.read_text(encoding="utf-8").splitlines() if c.strip()]
        _MASTER_IDX = {n: i for i, n in enumerate(names)}
    return _MASTER_IDX


_MODELS: dict = {}
def get_model(model_key: str = "ui"):
    """Return (YOLO, remap{local->master}, tag). remap goes BY NAME so a model
    trained with its own local ordering (fused_avatar, emoticon) lands on the
    correct master index; names absent from master are simply dropped."""
    if model_key not in _MODELS:
        from ultralytics import YOLO
        reg = json.loads((REPO / "data" / "model_registry.json").read_text(encoding="utf-8"))
        if model_key not in reg:
            raise SystemExit(f"unknown model {model_key!r}; choices: {list(reg)}")
        node = reg[model_key]
        m = YOLO(node["versions"][node["active"]]["path"])
        midx = master_idx()
        remap = {}
        for li, nm in m.names.items():
            mi = midx.get(nm)
            if mi is not None:
                remap[int(li)] = mi
        tag = _KEY_TO_TAG.get(model_key, "ui")
        _MODELS[model_key] = (m, remap, tag)
    return _MODELS[model_key]


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    aa = (ax2 - ax1) * (ay2 - ay1); bb = (bx2 - bx1) * (by2 - by1)
    return inter / (aa + bb - inter + 1e-9)


def _read_existing(lp: Path, w: float, h: float):
    """Existing label rows -> [(master_id, [x1,y1,x2,y2] px)]. 5-col master
    format; a stray 6th col (OBB angle) is ignored."""
    out = []
    if not lp.exists():
        return out
    for ln in lp.read_text(encoding="utf-8").splitlines():
        p = ln.split()
        if len(p) < 5:
            continue
        try:
            cid = int(float(p[0])); cx, cy, bw, bh = (float(x) for x in p[1:5])
        except ValueError:
            continue
        out.append((cid, [(cx - bw / 2) * w, (cy - bh / 2) * h,
                          (cx + bw / 2) * w, (cy + bh / 2) * h]))
    return out


def prefill_run(img_dir, *, model_key: str = "ui", conf: float = 0.25,
                imgsz: "int | None" = None, mode: str = "merge",
                overwrite: bool = False, target_classes=None, progress=None) -> dict:
    """Prefill every *.jpg under img_dir with ONE model. See module docstring
    for modes. `overwrite=True` is a back-compat alias for mode='overwrite'.
    Returns {written, skipped, total, model, mode}."""
    if overwrite:
        mode = "overwrite"
    img_dir = Path(img_dir)
    m, remap, tag = get_model(model_key)
    owns = _OWNS.get(tag, lambda i: True)
    # 只标目标 cls (飞轮补单个/几个弱类时, 不用标全部, 大幅提效)。空=标该模型全 span。
    tgt = set(int(t) for t in target_classes) if target_classes else None
    if imgsz is None:
        imgsz = _IMGSZ_BY_TAG.get(tag, 960)
    imgs = sorted(glob.glob(str(img_dir / "*.jpg")))
    written = skipped = 0
    CHUNK = 64
    for ci in range(0, len(imgs), CHUNK):
        chunk = imgs[ci:ci + CHUNK]
        todo = []
        for p in chunk:
            lp = Path(p).with_suffix(".txt")
            if mode == "skip" and lp.exists():
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
            h, w = r.orig_shape  # (height, width)
            # this model's detections -> master id, span-filtered, conf-sorted
            new = []
            for b in r.boxes:
                mi = remap.get(int(b.cls[0]))
                if mi is None or not owns(mi):
                    continue
                if tgt is not None and mi not in tgt:
                    continue
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                if x2 <= x1 or y2 <= y1:
                    continue
                new.append((float(b.conf[0]), mi, [x1, y1, x2, y2]))
            new.sort(key=lambda t: -t[0])
            lp = Path(p).with_suffix(".txt")
            existing = _read_existing(lp, w, h)
            if mode == "overwrite":
                # per-model replace: drop ONLY this model's own-span boxes (we're
                # re-running it to FIX them, e.g. raise emoticon conf); every
                # other model's boxes (ui / avatar) are kept untouched.
                kept = [(mid, box) for mid, box in existing if not owns(mid)]
            else:
                kept = existing  # merge: keep all, append new (same-cls dedup)
            for _sc, mi, box in new:
                if any(k[0] == mi and _iou(box, k[1]) > 0.6 for k in kept):
                    continue  # same-class dup (re-run / overlap) — keep first
                kept.append((mi, box))
            lines = []
            for mid, (x1, y1, x2, y2) in kept:
                cx = ((x1 + x2) / 2) / w; cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w; bh = (y2 - y1) / h
                lines.append(f"{mid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            lp.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            written += 1
            if progress and written % 50 == 0:
                progress(written, len(imgs))
    return {"written": written, "skipped": skipped, "total": len(imgs),
            "model": model_key, "mode": mode}


def main():
    argv = sys.argv[1:]
    if not argv:
        print("usage: yolo_prefill_run.py <dataset> "
              "[--model ui|fused_avatar|emoticon|battle_heads] "
              "[--conf 0.25] [--mode merge|overwrite|skip]")
        return
    dataset = argv[0]
    model_key = "ui"; conf = 0.25; mode = "merge"
    i = 1
    while i < len(argv):
        a = argv[i]
        if a == "--model":
            model_key = argv[i + 1]; i += 2; continue
        if a == "--conf":
            conf = float(argv[i + 1]); i += 2; continue
        if a == "--mode":
            mode = argv[i + 1]; i += 2; continue
        if a == "--overwrite":
            mode = "overwrite"; i += 1; continue
        i += 1
    img_dir = resolve_img_dir(dataset)
    if not img_dir.is_dir():
        print(f"dataset dir not found: {img_dir}")
        return
    print(f"prefill {img_dir}  model={model_key}  conf={conf}  mode={mode}")
    res = prefill_run(img_dir, model_key=model_key, conf=conf, mode=mode,
                      progress=lambda w, t: print(f"  ...{w}/{t}"))
    print(res)


if __name__ == "__main__":
    main()
