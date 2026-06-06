"""Import flywheel trajectory frames containing weak UI classes into a
raw_images labeling run, pre-filled with the model's detections so the
dashboard reviewer only has to confirm / correct / add the misses.

Why
---
v6 iteration needs more samples of UI classes the live model keeps missing
(the momo "学生发送信息中 / 学生信息回复选项" 漏检 bug, schedule 票数, story
节点图标, 免费/制造入口, 绿勾, region tiles...). The flywheel already
captured 66k frames under data/trajectories/ — every tick stores the model's
`yolo_boxes` as a pre-annotation. This tool mines the frames that show those
weak classes and lands them as a dashboard-ready dataset.

Output (one new run under data/raw_images/<out>/):
    frame_000000.jpg          screenshot copy (from tick_NNNN.jpg)
    frame_000000.txt          YOLO labels — 5 cols `cls xc yc w h`, NORMALIZED
                              center, cls = 0-based MASTER index. Exactly the
                              format server/app.py:list_dataset_images parses
                              (a 6th column there is read as OBB *angle*, NOT
                              conf — so we never emit one).
    classes.txt               verbatim copy of master _classes.txt, so the
                              dashboard's _ensure_dataset_migrated() fast-paths
                              (local == master) and indices already align.
    _source_manifest.jsonl    provenance: out frame -> src run/tick, skill,
                              sub_state, which buckets matched, conf per box.

Selection
---------
Two modes, unioned:
  A) cls-match    — frame's yolo_boxes contain >=1 class in a weak bucket.
                    Clean positives; reviewer confirms + fills nearby misses.
  B) skill-context — frame is from a skill where a weak class SHOULD appear
                    but the model returned none of them (the 漏检 case). Lets
                    the reviewer add the missed box from scratch. Capped +
                    stride-deduped so we don't flood with near-identical ticks.

Boxes whose cls name is not in the master 451-class UI registry (avatar heads,
emoticons, battle c0-c3 — they belong to *other* detectors) are dropped from
the pre-labels.

⚠ OVERLAY-BURN HISTORY — some runs captured with --game-overlay (while the
capture backend grabbed the screen) had the live YoloOverlay (boxes + dark
"classname conf" label bars) composited INTO the screenshot pixels — useless as
training data. No pixel heuristic detects it (the overlay lags one tick); the
reliable tell is the dark labels on the top HUD currency bar — see
scripts/audit_overlay_burn.py. This was isolated to the 2026-05-28 overlay-test
session; those 11 burned runs were DELETED 2026-06-03 and every other trajectory
date audited clean — so NO date filter is applied by default. If you capture new
--game-overlay runs, audit them before importing. (The dashboard label step is a
final safety net: a burned frame is obvious on screen.)

Usage
-----
    py -3 scripts/import_traj_weak_cls.py --dry                 # report only
    py -3 scripts/import_traj_weak_cls.py --out run_v6weak_20260603
    py -3 scripts/import_traj_weak_cls.py --out X --since 20260602  # date filter
    flags: --max-per-class N --context-cap N --stride N --min-conf F
           --no-context --force
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAJ_DIR = REPO_ROOT / "data" / "trajectories"
RAW_DIR = REPO_ROOT / "data" / "raw_images"
MASTER_FILE = RAW_DIR / "_classes.txt"

# ── Weak-class buckets (mode A) — master class names, validated present ──────
# Counts in comments = frames containing the class across all trajectories
# (from the 2026-06-03 distribution scan).
WEAK_BUCKETS: Dict[str, List[str]] = {
    "momo": [
        "学生发送信息中",            # 438  365f  (train had 28 → the 漏检 bug)
        "学生信息回复选项",          # 440  219f  (train had 32)
        "学生momotalk信息未读",      # 439  1478f
        "momotalk学生聊天区域按钮",  # 448  12f
        "momotalk学生聊天区域已进入",# 449  2f
    ],
    "schedule": [
        "课程表票",                  # 35   912f
        "课程表开始",                # 400  27f
        "课程表入口",                # 10   377f
    ],
    "story_node": [
        "剧情图标未完成",            # 430  150f
        "剧情图标已完成",            # 427  150f
        "战斗图标已完成",            # 447  0f   (pure miss — needs context mode)
        "入场键",                    # 79   386f
        "完成",                      # 426  157f
        "剧情new",                   # 428  0f   (pure miss)
        "剧情menu",                  # 431  78f
        "剧情观看",                  # 434  0f   (pure miss)
        "剧情中断退出",              # 433  0f   (pure miss)
        "进入章节",                  # 139  8f
    ],
    "craft_free": [
        "免费",                      # 446  1f   (very rare — needs context mode)
        "制造入口",                  # 14   127f
        "快速制造",                  # 443  48f
        "开始制造",                  # 444  2f
    ],
    "green_check": [
        "绿勾",                      # 403  12f
    ],
    "region_tile": [
        # world-map / lesson region tiles + raid region selectors
        "夏莱办公室", "夏莱居住区", "格黑娜学院中央区", "阿拜多斯高中",
        "千年研究所", "三一广场", "赤冬联邦学院", "百鬼夜行中心部",
        "D.U. 白鸟区", "山海经中央特区", "春叶原", "狂猎综合艺术区",
        "房间区域", "房间区域未解锁", "三一", "千年", "格黑娜",
    ],
}

# Per-class collection quota (mode A). Selection is per-CLASS, not per-bucket,
# so a common class (学生momotalk信息未读, 1478f) can't crowd out the rare
# critical ones inside the same bucket. A frame is kept iff it advances some
# under-quota target class. Criticals (the 漏检 bug classes) get high quotas;
# the abundant common class is throttled so it doesn't bloat labeling.
QUOTA_DEFAULT = 130
QUOTA_OVERRIDE: Dict[str, int] = {
    "学生发送信息中": 240,        # 438 — MOMO_SENDING, take ~all 365 avail
    "学生信息回复选项": 240,      # 440 — MOMO_REPLY_OPT, take ~all 219 avail
    "学生momotalk信息未读": 90,   # 439 — abundant, throttle
    "课程表票": 160,             # 35
    "入场键": 160,               # 79
    "制造入口": 120,             # 14
}

# ── Skill-context (mode B) — collect frames from these skills even when the
# model detected NONE of the bucket's classes, to surface 漏检 misses.
# skill name (exact, matches trajectory json "skill") -> bucket.
CONTEXT_SKILLS: Dict[str, str] = {
    "MomoTalk": "momo",
    "StoryMining": "story_node",
    "StoryCleanup": "story_node",
    "Craft": "craft_free",
}


def load_master() -> Tuple[List[str], Dict[str, int]]:
    names = [c.strip() for c in MASTER_FILE.read_text(encoding="utf-8").splitlines() if c.strip()]
    return names, {n: i for i, n in enumerate(names)}


def to_xywh(b: dict, img_w: float, img_h: float) -> Optional[Tuple[float, float, float, float]]:
    """corner (x1,y1,x2,y2) -> normalized center (xc,yc,w,h). Handles pixel
    coords (any side > 1.5) by dividing by image dims. Returns None if degenerate."""
    x1, y1, x2, y2 = (float(b.get(k, 0)) for k in ("x1", "y1", "x2", "y2"))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    pixel = max(x2, y2, x1, y1) > 1.5
    if pixel:
        if img_w <= 0 or img_h <= 0:
            return None
        x1, x2 = x1 / img_w, x2 / img_w
        y1, y2 = y1 / img_h, y2 / img_h
    xc, yc = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    w, h = x2 - x1, y2 - y1
    xc = min(max(xc, 0.0), 1.0); yc = min(max(yc, 0.0), 1.0)
    w = min(max(w, 0.0), 1.0);  h = min(max(h, 0.0), 1.0)
    if w <= 0.0005 or h <= 0.0005:
        return None
    return xc, yc, w, h


def _iou_xywh(a, b) -> float:
    ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2
    ax2, ay2 = a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2
    bx2, by2 = b[0] + b[2] / 2, b[1] + b[3] / 2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    ua = a[2] * a[3] + b[2] * b[3] - inter
    return inter / ua if ua > 0 else 0.0


def prelabels_for_frame(boxes: list, name2idx: Dict[str, int], img_w: float,
                        img_h: float, min_conf: float):
    """Build deduped pre-label rows for one frame.
    Returns (rows, dropped_non_master) where rows = list of
    (cls_idx, xc, yc, w, h, conf, name). Drops non-master classes and runs a
    light per-class NMS (the ensemble union is not cross-model NMS'd, so the
    same UI element can appear several times at different conf)."""
    cand = []  # (conf, idx, name, (xc,yc,w,h))
    dropped = 0
    for b in boxes:
        nm = b.get("cls")
        idx = name2idx.get(nm)
        if idx is None:
            dropped += 1
            continue
        conf = float(b.get("conf", 0.0))
        if conf < min_conf:
            continue
        xywh = to_xywh(b, img_w, img_h)
        if xywh is None:
            continue
        cand.append((conf, idx, nm, xywh))
    cand.sort(key=lambda c: -c[0])
    kept = []
    for conf, idx, nm, xywh in cand:
        if any(k[1] == idx and _iou_xywh(k[3], xywh) > 0.5 for k in kept):
            continue  # duplicate of a higher-conf box of the same class
        kept.append((conf, idx, nm, xywh))
    rows = [(idx, xy[0], xy[1], xy[2], xy[3], conf, nm)
            for conf, idx, nm, xy in kept]
    return rows, dropped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="", help="output run name under data/raw_images/")
    ap.add_argument("--dry", action="store_true", help="report only, no writes")
    ap.add_argument("--since", default="",
                    help="only runs whose date >= YYYYMMDD (default: all — the "
                         "burned 05-28 runs were deleted, rest audited clean).")
    ap.add_argument("--runs-glob", default="run_*")
    ap.add_argument("--max-per-class", type=int, default=QUOTA_DEFAULT,
                    help="default per-class (mode A) quota; overridden by QUOTA_OVERRIDE")
    ap.add_argument("--context-cap", type=int, default=70,
                    help="cap on skill-context (mode B) miss frames per skill")
    ap.add_argument("--stride", type=int, default=3,
                    help="min ticks between kept frames of same signature in a run")
    ap.add_argument("--min-conf", type=float, default=0.25,
                    help="drop pre-label boxes below this conf")
    ap.add_argument("--no-context", action="store_true", help="disable mode B")
    ap.add_argument("--force", action="store_true", help="overwrite existing out dir")
    args = ap.parse_args()

    if not args.dry and not args.out:
        ap.error("--out is required unless --dry")

    master, name2idx = load_master()
    bucket_of: Dict[str, str] = {}
    quota: Dict[str, int] = {}
    for bk, names in WEAK_BUCKETS.items():
        for nm in names:
            if nm not in name2idx:
                print(f"[warn] bucket {bk!r} class not in master: {nm!r}")
            bucket_of[nm] = bk
            quota[nm] = QUOTA_OVERRIDE.get(nm, args.max_per_class)
    targets: Set[str] = set(bucket_of)

    run_dirs = sorted(d for d in glob.glob(str(TRAJ_DIR / args.runs_glob)) if os.path.isdir(d))

    # accumulators
    selected: List[dict] = []                 # chosen frame records
    cls_count: Counter = Counter()            # target cls -> frames selected showing it (quota)
    cls_avail: Counter = Counter()            # target cls -> frames available (uncapped)
    ctx_count: Counter = Counter()            # mode-B frames per skill
    avail_ctx: Counter = Counter()
    scanned = 0

    for rd in run_dirs:
        run = os.path.basename(rd)
        date = run.replace("run_", "")[:8]
        if args.since and date < args.since:
            continue
        last_kept_tick: Dict[str, int] = {}   # signature -> last kept tick (per run)
        jfs = sorted(glob.glob(os.path.join(rd, "*.json")))
        for jf in jfs:
            try:
                d = json.load(open(jf, encoding="utf-8"))
            except Exception:
                continue
            if "yolo_boxes" not in d:
                continue
            scanned += 1
            tick = int(d.get("tick", 0))
            skill = d.get("skill") or ""
            sub = d.get("sub_state") or ""
            boxes = d.get("yolo_boxes") or []
            present = {b.get("cls") for b in boxes}
            hit_cls = present & targets
            for n in hit_cls:
                cls_avail[n] += 1

            mode = None
            sig = None
            if hit_cls:
                # keep only if it advances some still-under-quota target class
                if any(cls_count[n] < quota[n] for n in hit_cls):
                    mode = "A"
                    sig = "A:" + ",".join(sorted(hit_cls)) + "|" + sub
            if mode is None and (not args.no_context) and skill in CONTEXT_SKILLS:
                avail_ctx[skill] += 1
                if ctx_count[skill] < args.context_cap:
                    mode = "B"
                    sig = "B:" + skill
            if mode is None:
                continue

            # stride dedup within run: skip if same signature seen too recently
            if tick - last_kept_tick.get(sig, -10_000) < args.stride:
                continue

            jpg = os.path.join(rd, os.path.basename(jf).replace(".json", ".jpg"))
            if not os.path.isfile(jpg):
                continue

            last_kept_tick[sig] = tick
            buckets = sorted({bucket_of[n] for n in hit_cls}) if hit_cls else [skill + "(ctx)"]
            selected.append({
                "src_run": run, "src_tick": os.path.basename(jf)[:-5],
                "jpg": jpg, "skill": skill, "sub_state": sub,
                "mode": mode, "buckets": buckets,
                "boxes": boxes, "img_w": float(d.get("image_w", 0) or 0),
                "img_h": float(d.get("image_h", 0) or 0),
            })
            if mode == "A":
                for n in hit_cls:
                    cls_count[n] += 1
            else:
                ctx_count[skill] += 1

    # ── report ──
    print(f"scanned {scanned} frames across {len(run_dirs)} runs"
          + (f" (since {args.since})" if args.since else ""))
    print(f"\nselected {len(selected)} frames")
    print("\n  mode A — per-class quota   [selected / available  (quota)]:")
    for bk, names in WEAK_BUCKETS.items():
        print(f"   [{bk}]")
        for nm in names:
            print(f"    [{name2idx.get(nm,-1):>3}] {nm:<22} {cls_count[nm]:>4} / {cls_avail[nm]:<5} (q{quota[nm]})")
    print("\n  mode B (skill-context misses)   [selected / available]:")
    for sk in CONTEXT_SKILLS:
        print(f"    {sk:<14} {ctx_count[sk]:>4} / {avail_ctx[sk]}")

    if args.dry:
        print("\n[dry] no files written. add --out <name> to import.")
        return

    out_dir = RAW_DIR / os.path.basename(args.out)
    if out_dir.exists():
        if not args.force:
            raise SystemExit(f"[abort] {out_dir} exists; pass --force to overwrite")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    (out_dir / "classes.txt").write_text("\n".join(master) + "\n", encoding="utf-8")
    manifest = (out_dir / "_source_manifest.jsonl").open("w", encoding="utf-8")
    n_written = 0
    total_prelabels = 0
    total_dropped = 0
    for i, rec in enumerate(selected):
        stem = f"frame_{i:06d}"
        shutil.copy2(rec["jpg"], out_dir / f"{stem}.jpg")
        rows, dropped = prelabels_for_frame(
            rec["boxes"], name2idx, rec["img_w"], rec["img_h"], args.min_conf)
        total_dropped += dropped
        lines = [f"{idx} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
                 for idx, xc, yc, w, h, _conf, _nm in rows]
        (out_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""),
                                             encoding="utf-8")
        total_prelabels += len(rows)
        manifest.write(json.dumps({
            "frame": f"{stem}.jpg",
            "src_run": rec["src_run"], "src_tick": rec["src_tick"],
            "skill": rec["skill"], "sub_state": rec["sub_state"],
            "mode": rec["mode"], "buckets": rec["buckets"],
            "n_prelabels": len(rows),
            "prelabels": [{"cls": nm, "idx": idx, "conf": round(conf, 3)}
                          for idx, _xc, _yc, _w, _h, conf, nm in rows],
        }, ensure_ascii=False) + "\n")
        n_written += 1
    manifest.close()

    print(f"\n[done] wrote {n_written} frames -> {out_dir}")
    print(f"       {total_prelabels} pre-label boxes ({total_dropped} non-master boxes dropped)")
    print(f"       dashboard dataset name: {os.path.basename(args.out)}")


if __name__ == "__main__":
    main()
