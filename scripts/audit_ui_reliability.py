"""Audit ui-detector cls reliability across ALL trajectory frames.

User goal (2026-05-29): "很多 bbox 训练后仍检不出。用 ui 模型扫一遍所有
trajectory，统计哪些 bbox 不可靠，要完整统计，再统一修复。也要查 bbox
是否挤在同一个东西上(导致低 conf)。"

What it does:
  1. Runs the active ui model over every data/trajectories/run_*/*.jpg
     at production imgsz=960 on GPU (conf=0.20 to capture the low tail).
  2. Per-cls aggregates: #frames detected in, conf percentiles (p10/p50/p90),
     max conf, and how many detections fall in the borderline 0.20-0.45 band.
  3. Cross-references each cls's TRAIN instance count (from the dataset labels)
     so we can flag "trained a lot but never/weakly detected" = unreliable.
  4. Detects bbox CROWDING: per frame, pairs of DIFFERENT-cls boxes with high
     IoU (>=0.55) — the model stacking several cls on one object indicates
     label confusion (the user's "不能让 bbox 挤一个东西"). Aggregates the
     worst-offending cls pairs.
  5. Writes a sorted xlsx (+ csv fallback) and prints a BROKEN/WEAK summary.

Usage:
  py scripts/audit_ui_reliability.py                 # all frames, active model
  py scripts/audit_ui_reliability.py --limit 20000   # sample first N
  py scripts/audit_ui_reliability.py --weights <pt> --imgsz 960 --conf 0.20
Output: D:\\Project\\ai game secretary\\data\\_ui_reliability_audit.xlsx
"""
from __future__ import annotations
import sys, os, glob, json
from collections import defaultdict


def parse_args(argv):
    weights = None
    imgsz, conf, limit, every = 960, 0.20, 0, 1
    out = r"D:\Project\ai game secretary\data\_ui_reliability_audit.xlsx"
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--weights": weights = argv[i+1]; i += 2; continue
        if a == "--imgsz": imgsz = int(argv[i+1]); i += 2; continue
        if a == "--conf": conf = float(argv[i+1]); i += 2; continue
        if a == "--limit": limit = int(argv[i+1]); i += 2; continue
        if a == "--every": every = int(argv[i+1]); i += 2; continue
        if a == "--out": out = argv[i+1]; i += 2; continue
        i += 1
    return weights, imgsz, conf, limit, every, out


def _resolve_active_weights():
    reg = json.load(open(r"D:\Project\ai game secretary\data\model_registry.json", encoding="utf-8"))
    ui = reg["ui"]; v = ui["active"]
    return ui["versions"][v]["path"]


def _train_counts(names):
    """instances per cls id in the training labels."""
    import yaml
    dy = r"D:\Project\ml_cache\models\yolo\dataset\ui_v1\data.yaml"
    cnt = defaultdict(int)
    try:
        lbl = r"D:\Project\ml_cache\models\yolo\dataset\ui_v1\labels\train"
        for f in glob.glob(lbl + r"\*.txt"):
            try:
                for ln in open(f, encoding="utf-8"):
                    p = ln.split()
                    if p:
                        cnt[int(p[0])] += 1
            except Exception:
                pass
    except Exception:
        pass
    return cnt


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0: return 0.0
    aa = (ax2-ax1)*(ay2-ay1); bb = (bx2-bx1)*(by2-by1)
    return inter / (aa + bb - inter + 1e-9)


def main():
    weights, imgsz, conf, limit, every, out = parse_args(sys.argv[1:])
    if not weights:
        weights = _resolve_active_weights()
    from ultralytics import YOLO
    import numpy as np
    m = YOLO(weights)
    names = m.names
    n_cls = (max(names) + 1) if isinstance(names, dict) else len(names)
    def nm(i): return names.get(i, str(i)) if isinstance(names, dict) else (names[i] if i < len(names) else str(i))

    frames = sorted(glob.glob(r"D:\Project\ai game secretary\data\trajectories\run_*\*.jpg"))
    if every > 1: frames = frames[::every]
    if limit > 0: frames = frames[:limit]
    print(f"weights={weights}")
    print(f"frames={len(frames)} imgsz={imgsz} conf={conf}")

    confs = defaultdict(list)         # cls_id -> [conf...]
    det_frames = defaultdict(int)     # cls_id -> #frames it appears in
    crowd_pairs = defaultdict(int)    # frozenset({a,b}) -> count of high-IoU overlaps
    crowd_frames = 0
    total = 0

    # Process in CHUNKS — passing all 61080 paths to predict() at once opens
    # every file to validate the source list → OSError "Too many open files".
    # Per chunk use stream=True so each frame is a batch=1 forward (a batched
    # forward of 64 × 960px imgs at conf=0.05 keeps a huge [B, anchors, 448]
    # score tensor → CUDA OOM). 64 open handles is safe; batch=1 is light.
    CHUNK = 64
    CROWD_CONF = 0.30
    chunk_errors = 0
    for _ci in range(0, len(frames), CHUNK):
        _chunk = frames[_ci:_ci + CHUNK]
        try:
            _chunk_results = list(m.predict(_chunk, stream=True, imgsz=imgsz,
                                            conf=conf, device=0, verbose=False))
        except Exception as _e:
            # Skip a bad chunk (transient CUDA hiccup / corrupt frame) instead
            # of killing the whole 61080-frame run.
            chunk_errors += 1
            print(f"  [chunk @{_ci} skipped: {type(_e).__name__}: {_e}]")
            continue
        for r in _chunk_results:
            total += 1
            if total % 5000 == 0:
                print(f"  ...{total}/{len(frames)}")
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue
            seen_this = set()
            cls_ids = boxes.cls.tolist()
            cfs = boxes.conf.tolist()
            xyxy = boxes.xyxy.tolist()
            for cid, cf in zip(cls_ids, cfs):
                cid = int(cid)
                confs[cid].append(float(cf))
                seen_this.add(cid)
            for cid in seen_this:
                det_frames[cid] += 1
            # crowding: different-cls boxes overlapping heavily, only among
            # REAL-conf (>=CROWD_CONF) boxes so the ultra-low 0.05 detection
            # threshold (used to surface 邮件箱-grade hits) doesn't flood it.
            had_crowd = False
            N = len(xyxy)
            for ii in range(N):
                for jj in range(ii + 1, N):
                    if int(cls_ids[ii]) == int(cls_ids[jj]):
                        continue
                    if cfs[ii] < CROWD_CONF or cfs[jj] < CROWD_CONF:
                        continue
                    if _iou(xyxy[ii], xyxy[jj]) >= 0.55:
                        pair = frozenset((int(cls_ids[ii]), int(cls_ids[jj])))
                        crowd_pairs[pair] += 1
                        had_crowd = True
            if had_crowd:
                crowd_frames += 1

    tc = _train_counts(names)

    # build per-cls rows for cls that were TRAINED (tc>0) or detected
    rows = []
    for cid in range(n_cls):
        tcount = tc.get(cid, 0)
        cs = confs.get(cid, [])
        dframes = det_frames.get(cid, 0)
        if tcount == 0 and dframes == 0:
            continue
        if cs:
            arr = np.array(cs)
            p10, p50, p90, mx = (float(np.percentile(arr, 10)), float(np.percentile(arr, 50)),
                                 float(np.percentile(arr, 90)), float(arr.max()))
            borderline = int((arr < 0.45).sum())
        else:
            p10 = p50 = p90 = mx = 0.0; borderline = 0
        det_rate = dframes / max(1, total)
        # Reliability flag — judged by HOW WELL a cls detects WHEN it appears
        # (max + p90 conf), which is screen-frequency-independent. p50/det_rate
        # are skewed by how often a cls's screen shows up + by 0.15-threshold
        # noise (e.g. 青辉石 has max 0.95 but a low p50 from partial edge hits),
        # so they're reported but NOT used for the verdict.
        TRAINED = tcount >= 40
        if TRAINED and dframes == 0:
            flag = "BROKEN(trained,never-detected)"
        elif TRAINED and mx < 0.45:
            flag = "BROKEN(never-confident<0.45)"   # detects only at junk conf
        elif TRAINED and p90 < 0.55:
            flag = "WEAK(rarely-confident)"
        elif cs and p90 < 0.70:
            flag = "MEH"
        elif dframes == 0:
            flag = "untriggered(maybe rare screen)"
        else:
            flag = "ok"
        rows.append((cid, nm(cid), tcount, dframes, round(det_rate, 4),
                     round(p10, 3), round(p50, 3), round(p90, 3), round(mx, 3),
                     borderline, flag))

    # sort: BROKEN first, then WEAK, then by det_rate asc
    order = {"BROKEN(trained,never-detected)": 0, "BROKEN(never-confident<0.45)": 1,
             "WEAK(rarely-confident)": 2, "MEH": 3, "untriggered(maybe rare screen)": 4, "ok": 5}
    rows.sort(key=lambda r: (order.get(r[10], 9), r[4]))

    crowd_top = sorted(crowd_pairs.items(), key=lambda kv: -kv[1])[:30]

    # console summary
    print(f"\n=== scanned {total} frames ({chunk_errors} chunks skipped); {crowd_frames} had bbox crowding (IoU>=0.55 diff-cls) ===")
    print("\n--- BROKEN / WEAK cls (trained but unreliable) ---")
    print(f"{'cls':>4} {'name':<16} {'train':>6} {'detF':>6} {'rate':>7} {'p50':>5} {'max':>5}  flag")
    for r in rows:
        if r[10].startswith("BROKEN") or r[10].startswith("WEAK"):
            print(f"{r[0]:>4} {r[1][:16]:<16} {r[2]:>6} {r[3]:>6} {r[4]:>7} {r[6]:>5} {r[8]:>5}  {r[10]}")
    print("\n--- top bbox-crowding cls pairs (label-confusion suspects) ---")
    for pair, c in crowd_top:
        ids = list(pair);
        if len(ids) == 2:
            print(f"  {c:>6}x  {nm(ids[0])} <-> {nm(ids[1])}")

    # write xlsx
    header = ["cls","name","train_inst","det_frames","det_rate","conf_p10","conf_p50","conf_p90","conf_max","n_below_0.45","flag"]
    try:
        from openpyxl import Workbook
        wb = Workbook(); ws = wb.active; ws.title = "cls_reliability"
        ws.append(header)
        for r in rows: ws.append(list(r))
        ws2 = wb.create_sheet("crowding_pairs")
        ws2.append(["count","clsA","clsB"])
        for pair, c in crowd_top:
            ids = list(pair)
            if len(ids) == 2: ws2.append([c, nm(ids[0]), nm(ids[1])])
        wb.save(out)
        print(f"\n[xlsx] {out}")
    except ImportError:
        import csv
        co = out.replace(".xlsx", ".csv")
        with open(co, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f); w.writerow(header); w.writerows(rows)
        print(f"\n[csv] {co}")


if __name__ == "__main__":
    main()
