"""Watch a YOLO training run and emit a human-friendly progress
file every minute.  Pure-python loop, no Claude/agent involvement.

Usage:
    py scripts/monitor_train.py                       # default = v4
    py scripts/monitor_train.py --run fused_avatar_yolo26x --epochs 200
"""
from __future__ import annotations
import argparse
import csv
import datetime as _dt
from pathlib import Path
import time

REPO = Path(__file__).resolve().parents[1]
RUNS_ROOT = Path("D:/Project/ml_cache/models/yolo/runs")
OUT_MD = REPO / "_TRAIN_PROGRESS.md"
OUT_CSV = REPO / "_TRAIN_PROGRESS.csv"
STOP = REPO / "_TRAIN_STOP"

# Baselines (mAP50)
V1_MAP50 = 0.597  # 26m epoch 168 (v1)
V2_MAP50 = 0.617  # 26x epoch 134 (v2)
V3_MAP50 = 0.680  # 26x epoch 124 best (v3, heavy-aug overfit)

# Will be filled by argparse
RUN_NAME = "fused_avatar_yolo26x_v4"
TARGET_EPOCHS = 100
RESULTS = RUNS_ROOT / RUN_NAME / "results.csv"


def read_results():
    if not RESULTS.exists():
        return None
    rows = []
    with RESULTS.open(encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                rows.append({
                    "epoch": int(row["epoch"]),
                    "time": float(row["time"]),
                    "box_loss": float(row["train/box_loss"]),
                    "cls_loss": float(row["train/cls_loss"]),
                    "precision": float(row["metrics/precision(B)"]),
                    "recall": float(row["metrics/recall(B)"]),
                    "mAP50": float(row["metrics/mAP50(B)"]),
                    "mAP50_95": float(row["metrics/mAP50-95(B)"]),
                    "val_box_loss": float(row.get("val/box_loss", 0)),
                    "val_cls_loss": float(row.get("val/cls_loss", 0)),
                })
            except (KeyError, ValueError):
                continue
    return rows


def fmt_eta(cur_epoch, last_time_s):
    if cur_epoch == 0:
        return "n/a"
    per_epoch = last_time_s / cur_epoch
    remaining = (TARGET_EPOCHS - cur_epoch) * per_epoch
    hours = int(remaining // 3600)
    mins = int((remaining % 3600) // 60)
    return f"{hours}h {mins:02d}m"


def _safe_write(path, text):
    """Skip silently if file is locked (Excel open etc). Retry next loop."""
    try:
        path.write_text(text, encoding="utf-8")
    except PermissionError:
        pass  # Excel locked it, skip this round


def emit_md(rows):
    if not rows:
        _safe_write(OUT_MD,
            f"# Training Progress\n\n_(waiting for first epoch to complete...)_\n\n"
            f"Last refresh: {_dt.datetime.now().strftime('%H:%M:%S')}\n",
        )
        return

    last = rows[-1]
    best_idx = max(range(len(rows)), key=lambda i: rows[i]["mAP50"])
    best = rows[best_idx]

    # Progress bar
    progress_pct = last["epoch"] / TARGET_EPOCHS * 100
    bar_len = 40
    filled = int(bar_len * progress_pct / 100)
    bar = "█" * filled + "░" * (bar_len - filled)

    # Status vs baselines
    def _stat(name, base):
        if last["mAP50"] >= base:
            return f"✅ 超过 {name} (+{(last['mAP50']-base)*100:.2f}pp)"
        return f"⏳ 差 {(base-last['mAP50'])*100:.2f}pp vs {name}"
    status_v1 = _stat("v1", V1_MAP50)
    status_v2 = _stat("v2", V2_MAP50)
    status_v3 = _stat("v3", V3_MAP50)

    eta = fmt_eta(last["epoch"], last["time"])

    md = []
    md.append(f"# 🏋️ {RUN_NAME} Training Progress")
    md.append("")
    md.append(f"**Last refresh**: `{_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}` · "
              f"**Epoch**: `{last['epoch']}/{TARGET_EPOCHS}` · **ETA**: `{eta}`")
    md.append("")
    md.append(f"```")
    md.append(f"{bar}  {progress_pct:.1f}%")
    md.append(f"```")
    md.append("")
    md.append("## Current vs Baselines")
    md.append("")
    md.append("| Metric | Current | v1 (26m) | v2 (26x) | v3 (26x heavy-aug) | Status |")
    md.append("|---|---|---|---|---|---|")
    md.append(f"| **mAP50** | **{last['mAP50']:.4f}** | 0.5970 | 0.6170 | 0.6803 | {status_v3} |")
    md.append(f"| **mAP50-95** | **{last['mAP50_95']:.4f}** | ~0.30 | ~0.40 | 0.6229 | (stricter IoU) |")
    md.append(f"| Precision | {last['precision']:.4f} | | | | |")
    md.append(f"| Recall | {last['recall']:.4f} | | | | |")
    md.append(f"| Train cls_loss | {last['cls_loss']:.4f} | | | | (lower=better) |")
    md.append(f"| Val cls_loss | {last['val_cls_loss']:.4f} | | | | (climbing = overfit) |")
    md.append("")
    md.append(f"**🏆 Best epoch so far**: `#{best['epoch']}` with mAP50 = **{best['mAP50']:.4f}** "
              f"(mAP50-95 = {best['mAP50_95']:.4f})")
    md.append("")
    md.append(f"_v1/v2/v3 status_: {status_v1} · {status_v2} · {status_v3}")
    md.append("")
    md.append("## Last 15 Epochs")
    md.append("")
    md.append("| Epoch | elapsed | cls_loss | box_loss | mAP50 | mAP50-95 | Δ mAP50 |")
    md.append("|---:|---:|---:|---:|---:|---:|---:|")
    show_rows = rows[-15:]
    prev_map = None
    for r in show_rows:
        elap = f"{r['time']/60:.1f}m"
        delta = (r["mAP50"] - prev_map) if prev_map is not None else 0
        delta_str = f"{delta:+.4f}" if prev_map is not None else "—"
        md.append(f"| {r['epoch']} | {elap} | {r['cls_loss']:.4f} | {r['box_loss']:.4f} | "
                  f"{r['mAP50']:.4f} | {r['mAP50_95']:.4f} | {delta_str} |")
        prev_map = r["mAP50"]
    md.append("")
    md.append("## Auto-refresh tip")
    md.append("")
    md.append("- Open this file in VSCode → preview pane refreshes on save")
    md.append("- Or open `_TRAIN_PROGRESS.csv` in Excel/LibreOffice")
    md.append("- This file updates every 60 seconds while monitor_train.py is running")
    md.append("")
    md.append(f"*Training output: `D:/Project/ml_cache/models/yolo/runs/{RUN_NAME}/`*")

    _safe_write(OUT_MD, "\n".join(md))


def emit_csv(rows):
    """Mirror results.csv with friendlier column names, latest 50 epochs."""
    if not rows:
        _safe_write(OUT_CSV, "epoch,elapsed_min,cls_loss,box_loss,precision,recall,mAP50,mAP50_95,delta_mAP50\n")
        return
    last_50 = rows[-50:]
    lines = ["epoch,elapsed_min,cls_loss,box_loss,precision,recall,mAP50,mAP50_95,delta_mAP50"]
    prev = None
    for r in last_50:
        delta = (r["mAP50"] - prev) if prev is not None else 0
        lines.append(
            f"{r['epoch']},{r['time']/60:.2f},{r['cls_loss']:.4f},{r['box_loss']:.4f},"
            f"{r['precision']:.4f},{r['recall']:.4f},{r['mAP50']:.4f},{r['mAP50_95']:.4f},{delta:+.4f}"
        )
        prev = r["mAP50"]
    _safe_write(OUT_CSV, "\n".join(lines) + "\n")


def main() -> int:
    global RUN_NAME, TARGET_EPOCHS, RESULTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=RUN_NAME, help="run dir name under runs/")
    ap.add_argument("--epochs", type=int, default=TARGET_EPOCHS, help="target epoch count")
    args = ap.parse_args()
    RUN_NAME = args.run
    TARGET_EPOCHS = args.epochs
    RESULTS = RUNS_ROOT / RUN_NAME / "results.csv"

    print(f"[monitor] watching {RESULTS}")
    print(f"[monitor] writing  {OUT_MD}")
    print(f"[monitor] writing  {OUT_CSV}")
    print(f"[monitor] target epochs: {TARGET_EPOCHS}")
    print(f"[monitor] stop file: touch {STOP} to exit")
    last_epoch = -1
    while True:
        rows = read_results()
        if rows is None:
            emit_md([])
        else:
            emit_md(rows)
            emit_csv(rows)
            cur = rows[-1]["epoch"] if rows else 0
            if cur != last_epoch:
                print(f"[monitor] epoch {cur} mAP50={rows[-1]['mAP50']:.4f}")
                last_epoch = cur
            if cur >= TARGET_EPOCHS:
                print(f"[monitor] training complete (epoch {TARGET_EPOCHS} reached)")
                return 0
        if STOP.exists():
            print("[monitor] STOP file detected, exiting")
            STOP.unlink()
            return 0
        time.sleep(60)


if __name__ == "__main__":
    raise SystemExit(main())
