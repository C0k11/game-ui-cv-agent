"""Auto-label trajectory frames using reference UI capture templates.

Loads every PNG in `data/captures/<context>/**.png` as a template.
Each unique (context, filename) tuple becomes one YOLO class:
    cafe__invite-student-button
    arena__battle-win
    main_page__bus
    ...

Runs cv2.matchTemplate against every frame, emits YOLO labels.
Small templates (area < 1000 px) auto-bumped to a stricter threshold to
avoid noise floods on tiny patterns (11x17 arrows etc.).

Outputs:
    data/yolo_datasets/ui_v1_auto/
        images/<frame_name>.jpg          (copied or symlinked)
        labels/<frame_name>.txt          (YOLO format)
        classes.txt                       (class_id → name)
        class_map.json                    (extra metadata)
        hit_stats.json                    (per-class hit count, etc.)

Usage:
    py scripts/auto_annotate_ref.py --sample 100       # spot-check 100 frames
    py scripts/auto_annotate_ref.py                     # full run (all trajectory frames)
    py scripts/auto_annotate_ref.py --threshold 0.78    # tune sensitivity
"""
from __future__ import annotations
import argparse
import json
import os
import random
import shutil
import sys
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# Single-thread cv2 inside each worker — we get parallelism from the Pool,
# not from cv2's internal threading. Otherwise 16 workers × ~8 threads each
# fights for the same cores and slows down 3-5x.
cv2.setNumThreads(1)

REPO = Path(__file__).resolve().parents[1]
CAPTURES_DIR = REPO / "data" / "captures"
TRAJECTORIES_DIR = REPO / "data" / "trajectories"
OUT_DIR = REPO / "data" / "yolo_datasets" / "ui_v1_auto"

# Skip these subdirs (not relevant UI templates)
SKIP_CTX_DIRS = {"角色头像", "角色头像_crop", "角色头像_crop_harvested_named"}

# Templates smaller than this area get a stricter threshold (noise floor).
SMALL_AREA_PX = 1000
# Templates smaller than this are skipped entirely (cv2 + YOLO both struggle).
TINY_AREA_PX = 250
# Templates with RGB std below this are "near-uniform color" — they match every
# similarly-colored patch across the screen. cv2 noise floor is here:
#   - shop/list-type-name-feature.png is literally a pure white square (std=0.2)
#     that matched 97/100 random frames at avg conf 0.99 → garbage.
LOW_VARIANCE_STD = 15.0

DEFAULT_THRESHOLD = 0.80
SMALL_TEMPLATE_THRESHOLD = 0.92
MAX_INSTANCES = 3


def load_templates(verbose: bool = True) -> list[dict]:
    """Walk data/captures/, return list of template dicts."""
    templates = []
    skipped_tiny = 0
    skipped_load = 0
    skipped_uniform = 0
    skipped_uniform_names = []
    for ctx_dir in sorted(CAPTURES_DIR.iterdir()):
        if not ctx_dir.is_dir():
            continue
        if ctx_dir.name in SKIP_CTX_DIRS:
            continue
        for png_path in sorted(ctx_dir.rglob("*.png")):
            raw = cv2.imdecode(np.fromfile(str(png_path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if raw is None:
                skipped_load += 1
                continue
            if raw.ndim == 3 and raw.shape[2] == 4:
                bgr = raw[:, :, :3]
                mask = raw[:, :, 3]
                # mask must be 8-bit single-channel for cv2.matchTemplate
                if mask.dtype != np.uint8:
                    mask = mask.astype(np.uint8)
            else:
                bgr = raw
                mask = None
            h, w = bgr.shape[:2]
            area = w * h
            if area < TINY_AREA_PX:
                skipped_tiny += 1
                continue
            # Reject near-uniform color templates (they false-positive everywhere).
            std = float(bgr.std())
            if std < LOW_VARIANCE_STD:
                skipped_uniform += 1
                skipped_uniform_names.append(
                    f"{ctx_dir.name}/{png_path.name} (std={std:.1f})"
                )
                continue
            # Class name: ctx + filename (handles common/<...> subdirs by flattening)
            # e.g. activity/common/menu.png → activity__common__menu
            rel = png_path.relative_to(ctx_dir).with_suffix("")
            class_name = f"{ctx_dir.name}__{str(rel).replace(chr(92), '__').replace('/', '__')}"
            templates.append({
                "class_name": class_name,
                "ctx": ctx_dir.name,
                "path": png_path,
                "bgr": bgr,
                "mask": mask,
                "w": w, "h": h,
                "area": area,
                "threshold": SMALL_TEMPLATE_THRESHOLD if area < SMALL_AREA_PX else DEFAULT_THRESHOLD,
            })
    if verbose:
        print(f"[load] {len(templates)} templates loaded "
              f"(skipped {skipped_tiny} tiny < {TINY_AREA_PX}px, "
              f"{skipped_uniform} uniform-color, {skipped_load} unreadable)")
        if skipped_uniform_names:
            print(f"[load] uniform-color blacklist (std < {LOW_VARIANCE_STD}):")
            for n in skipped_uniform_names:
                print(f"         {n}")
        small = sum(1 for t in templates if t["area"] < SMALL_AREA_PX)
        print(f"[load] {small} small templates (area < {SMALL_AREA_PX}) use threshold {SMALL_TEMPLATE_THRESHOLD}")
    return templates


def match_one(frame_bgr: np.ndarray, t: dict, max_instances: int) -> list[dict]:
    """Return list of {x, y, w, h, score} for one template."""
    if t["h"] > frame_bgr.shape[0] or t["w"] > frame_bgr.shape[1]:
        return []
    if t["mask"] is not None:
        # Some reference PNGs have alpha=0 outside the actual icon; ensure mask
        # has at least some non-zero pixels or matchTemplate explodes.
        if int(t["mask"].max()) == 0:
            res = cv2.matchTemplate(frame_bgr, t["bgr"], cv2.TM_CCOEFF_NORMED)
        else:
            try:
                res = cv2.matchTemplate(frame_bgr, t["bgr"], cv2.TM_CCOEFF_NORMED, mask=t["mask"])
            except cv2.error:
                # Fall back without mask if cv2 chokes
                res = cv2.matchTemplate(frame_bgr, t["bgr"], cv2.TM_CCOEFF_NORMED)
    else:
        res = cv2.matchTemplate(frame_bgr, t["bgr"], cv2.TM_CCOEFF_NORMED)
    matches = []
    thr = t["threshold"]
    for _ in range(max_instances):
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if not np.isfinite(max_val) or max_val < thr:
            break
        x, y = max_loc
        matches.append({"x": x, "y": y, "w": t["w"], "h": t["h"], "score": float(max_val)})
        # Suppress matched region
        x1 = max(0, x - t["w"] // 2)
        y1 = max(0, y - t["h"] // 2)
        x2 = min(res.shape[1], x + t["w"] // 2 + 1)
        y2 = min(res.shape[0], y + t["h"] // 2 + 1)
        res[y1:y2, x1:x2] = -1.0
    return matches


def gather_frames(sample: int | None = None, stride: int = 1,
                  runs_pattern: str | None = None,
                  root_dir: Path | None = None) -> list[Path]:
    """Find all *.jpg under <root_dir>/run_*. Optionally subsample
    via stride (every Nth frame, evenly across runs) OR random sample (--sample).
    If runs_pattern given (e.g. "run_20260521_*"), only those runs are processed.
    Default root is data/trajectories/; dashboard captures live in data/raw_images/.
    """
    all_frames = []
    root = root_dir if root_dir is not None else TRAJECTORIES_DIR
    pattern = runs_pattern if runs_pattern else "run_*"
    runs = sorted(root.glob(pattern))
    for run_dir in runs:
        if not run_dir.is_dir():
            continue
        run_frames = sorted(run_dir.glob("*.jpg"))
        if stride > 1:
            run_frames = run_frames[::stride]
        all_frames.extend(run_frames)
    print(f"[scan] found {len(all_frames)} frames across {len(runs)} runs "
          f"(pattern={pattern!r}, stride={stride})")
    if sample and sample < len(all_frames):
        random.seed(42)
        all_frames = random.sample(all_frames, sample)
        all_frames.sort()
        print(f"[scan] randomly sampled {len(all_frames)} (seed=42)")
    return all_frames


# ── Worker-side globals (each subprocess gets its own copy) ────────────────
_WORKER_TEMPLATES: list[dict] | None = None
_WORKER_MAX_INSTANCES: int = MAX_INSTANCES


def _worker_init(max_instances: int):
    """Each Pool worker loads its own template set once at startup."""
    global _WORKER_TEMPLATES, _WORKER_MAX_INSTANCES
    cv2.setNumThreads(1)
    _WORKER_TEMPLATES = load_templates(verbose=False)
    _WORKER_MAX_INSTANCES = max_instances


def _worker_process_frame(img_path_str: str):
    """Match one frame against all templates. Return (img_path, lines, hits)."""
    global _WORKER_TEMPLATES, _WORKER_MAX_INSTANCES
    img = cv2.imread(img_path_str)
    if img is None:
        return img_path_str, [], {}
    img_h, img_w = img.shape[:2]
    lines: list[str] = []
    hits: dict[int, list[float]] = {}  # class_id → list of scores
    for class_id, t in enumerate(_WORKER_TEMPLATES):
        ms = match_one(img, t, _WORKER_MAX_INSTANCES)
        if not ms:
            continue
        hits[class_id] = [m["score"] for m in ms]
        for m in ms:
            cx = (m["x"] + m["w"] / 2) / img_w
            cy = (m["y"] + m["h"] / 2) / img_h
            nw = m["w"] / img_w
            nh = m["h"] / img_h
            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return img_path_str, lines, hits


def process(frames: list[Path], templates: list[dict], out_dir: Path,
            copy_images: bool = False, max_instances: int = MAX_INSTANCES,
            workers: int = 0) -> dict:
    """Run matching across the (frame × template) grid via a multiprocessing
    Pool. Each worker holds its own copy of all templates in memory.
    """
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)

    # classes.txt + class_map.json
    class_names = [t["class_name"] for t in templates]
    (out_dir / "classes.txt").write_text("\n".join(class_names), encoding="utf-8")
    (out_dir / "class_map.json").write_text(
        json.dumps({i: {"name": t["class_name"], "ctx": t["ctx"],
                        "w": t["w"], "h": t["h"], "area": t["area"],
                        "threshold": t["threshold"],
                        "source": str(t["path"].relative_to(REPO)).replace("\\", "/")}
                    for i, t in enumerate(templates)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    hit_stats = defaultdict(lambda: {"hits": 0, "frames": 0, "scores": []})
    total_boxes = 0
    frames_with_any = 0

    n_workers = workers if workers > 0 else max(1, cpu_count() - 2)
    print(f"[mp] starting Pool with {n_workers} workers "
          f"(each loads {len(templates)} templates, ~{len(templates)*0.01:.0f}MB / worker)")
    print(f"[mp] cpu_count() reports {cpu_count()}")

    img_paths_str = [str(p) for p in frames]
    with Pool(processes=n_workers,
              initializer=_worker_init, initargs=(max_instances,)) as pool:
        for img_path_str, lines, hits in tqdm(
            pool.imap_unordered(_worker_process_frame, img_paths_str, chunksize=4),
            total=len(img_paths_str), desc="frames",
        ):
            img_path = Path(img_path_str)
            for cid, scores in hits.items():
                hit_stats[cid]["hits"] += len(scores)
                hit_stats[cid]["scores"].extend(scores)
                hit_stats[cid]["frames"] += 1
            total_boxes += len(lines)

            if lines:
                frames_with_any += 1
                stem = f"{img_path.parent.name}__{img_path.stem}"
                (out_dir / "labels" / f"{stem}.txt").write_text(
                    "\n".join(lines), encoding="utf-8"
                )
                if copy_images:
                    shutil.copy2(img_path, out_dir / "images" / f"{stem}.jpg")
                else:
                    link = out_dir / "images" / f"{stem}.jpg"
                    if link.exists():
                        link.unlink()
                    try:
                        link.symlink_to(img_path)
                    except OSError:
                        shutil.copy2(img_path, link)

    # Persist stats
    stats_out = {}
    for cid, s in hit_stats.items():
        scores = s["scores"]
        stats_out[cid] = {
            "name": templates[cid]["class_name"],
            "ctx": templates[cid]["ctx"],
            "hits": s["hits"],
            "frames_with_hit": s["frames"],
            "avg_score": float(np.mean(scores)) if scores else 0.0,
            "max_score": float(np.max(scores)) if scores else 0.0,
            "min_score": float(np.min(scores)) if scores else 0.0,
        }
    # Sort by hits desc
    stats_out = dict(sorted(stats_out.items(), key=lambda x: -x[1]["hits"]))
    (out_dir / "hit_stats.json").write_text(
        json.dumps(stats_out, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    print()
    print(f"[done] processed {len(frames)} frames")
    print(f"       frames with >=1 hit: {frames_with_any} ({100*frames_with_any/max(len(frames),1):.1f}%)")
    print(f"       total boxes: {total_boxes}")
    print(f"       classes with >=1 hit: {len(stats_out)} / {len(templates)}")
    return {
        "frames_total": len(frames),
        "frames_with_any": frames_with_any,
        "total_boxes": total_boxes,
        "classes_hit": len(stats_out),
        "classes_total": len(templates),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None,
                    help="randomly sample N frames (default: all)")
    ap.add_argument("--stride", type=int, default=1,
                    help="take every Nth frame within each run (default: 1 = all)")
    ap.add_argument("--runs", type=str, default=None,
                    help="glob pattern under root to limit which runs are processed, "
                         "e.g. 'run_20260521_*' (default: all runs)")
    ap.add_argument("--root", type=Path, default=None,
                    help="root dir to scan for run_* subdirs (default: data/trajectories/, "
                         "dashboard captures live in data/raw_images/)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="override default threshold for large templates")
    ap.add_argument("--max-instances", type=int, default=MAX_INSTANCES)
    ap.add_argument("--copy-images", action="store_true",
                    help="copy frames instead of symlinking (slower, more disk)")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--workers", type=int, default=0,
                    help="multiprocess workers (0 = cpu_count - 2)")
    args = ap.parse_args()

    if args.threshold is not None:
        global DEFAULT_THRESHOLD
        DEFAULT_THRESHOLD = args.threshold

    print(f"[cfg] threshold large={DEFAULT_THRESHOLD}, small={SMALL_TEMPLATE_THRESHOLD}")
    print(f"[cfg] max instances per template per frame = {args.max_instances}")
    print(f"[cfg] output dir = {args.out}")
    print()

    templates = load_templates()
    if not templates:
        print("[!] no templates loaded, aborting", file=sys.stderr)
        return 1

    frames = gather_frames(sample=args.sample, stride=args.stride,
                           runs_pattern=args.runs, root_dir=args.root)
    if not frames:
        print("[!] no frames found", file=sys.stderr)
        return 1

    process(frames, templates, args.out,
            copy_images=args.copy_images,
            max_instances=args.max_instances,
            workers=args.workers)
    print(f"\n[output] {args.out}")
    print(f"         labels/   YOLO format per-frame .txt")
    print(f"         images/   symlinks (or copies) to source frames")
    print(f"         classes.txt + class_map.json + hit_stats.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
