"""Watch a YOLO training run, evaluate each new epoch on the MANUAL val
subset, and keep a separate `best_manual.pt` snapshot whose fitness
metric ignores synth val.

Why: ultralytics default fitness = nominal mAP50 (manual + synth + neg).
When val is dominated by synth (~90%), best.pt gets locked early to a
checkpoint that's synth-optimal but real-world-mediocre. This watcher
fixes that without monkey-patching ultralytics.

How:
  1. Polls results.csv every 30s.
  2. On every new epoch row, copies last.pt aside, runs eval_manual_val
     on it, and if manual mAP50 > previous manual_best, replaces
     `<run>/weights/best_manual.pt`.
  3. Writes a `manual_fitness_log.csv` alongside results.csv with the
     per-epoch manual metrics.

Usage:
    py scripts/manual_fitness_watcher.py
    py scripts/manual_fitness_watcher.py --run fused_avatar_yolo26x_v5
"""
from __future__ import annotations
import argparse
import csv
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS_ROOT = Path("D:/Project/ml_cache/models/yolo/runs")
DATASET = Path("D:/Project/ml_cache/models/yolo/dataset/fused_avatar_v1")


def make_manual_yaml() -> Path:
    """Reuse the yaml that eval_manual_val.py creates."""
    yaml_path = DATASET / "data_manual_val.yaml"
    if not yaml_path.exists():
        # Trigger the eval script once to generate it
        subprocess.run(
            [sys.executable, str(REPO / "scripts" / "eval_manual_val.py"),
             "--weights", "doesnt-matter-just-want-yaml-side-effect",
             "--name", "_init_yaml"],
            capture_output=True,
        )
        if not yaml_path.exists():
            print(f"[!] {yaml_path} still missing", file=sys.stderr)
            sys.exit(2)
    return yaml_path


def eval_manual_mAP50(weights: Path, yaml_path: Path) -> tuple[float, float, float, float]:
    """Returns (mAP50, mAP50_95, P, R) on manual val. Tiny batch + half=True
    to coexist with the running trainer (which is using ~16GB VRAM).
    Trainer batch=8 imgsz=960 → ~16GB. Eval batch=1 imgsz=960 → ~2GB. Safe.
    """
    from ultralytics import YOLO  # imported lazily — module is heavy
    model = YOLO(str(weights))
    metrics = model.val(
        data=str(yaml_path),
        imgsz=960,
        batch=1,        # tiny — coexist with trainer
        half=True,      # FP16 — half VRAM
        conf=0.001,
        iou=0.7,
        plots=False,
        verbose=False,
        name="_watcher_eval",
        exist_ok=True,
        device=0,
    )
    b = metrics.box
    return float(b.map50), float(b.map), float(b.mp), float(b.mr)


def read_completed_epochs(results_csv: Path) -> list[int]:
    if not results_csv.exists():
        return []
    epochs = []
    with results_csv.open(encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                epochs.append(int(row["epoch"]))
            except (KeyError, ValueError):
                continue
    return epochs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="fused_avatar_yolo26x_v4")
    ap.add_argument("--poll-sec", type=int, default=30)
    ap.add_argument("--skip-history", action="store_true",
                    help="mark all existing epochs as seen, only eval new ones")
    ap.add_argument("--seed-best", type=float, default=0.0,
                    help="initial best_manual_mAP50 (so trivial new epochs don't overwrite)")
    args = ap.parse_args()

    run_dir = RUNS_ROOT / args.run
    results_csv = run_dir / "results.csv"
    weights_dir = run_dir / "weights"
    last_pt = weights_dir / "last.pt"
    best_manual_pt = weights_dir / "best_manual.pt"
    log_path = run_dir / "manual_fitness_log.csv"

    yaml_path = make_manual_yaml()
    print(f"[watcher] watching {results_csv}")
    print(f"[watcher] manual val yaml: {yaml_path}")
    print(f"[watcher] writing best_manual.pt + manual_fitness_log.csv")

    # Bootstrap log
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        log_path.write_text("epoch,manual_mAP50,manual_mAP50_95,manual_P,manual_R,is_new_best\n", encoding="utf-8")

    best_manual = float(args.seed_best)
    seen_epochs: set[int] = set()
    # Replay history from log so we don't re-eval
    if log_path.exists():
        with log_path.open(encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    seen_epochs.add(int(row["epoch"]))
                    best_manual = max(best_manual, float(row["manual_mAP50"]))
                except (KeyError, ValueError):
                    continue
        if seen_epochs:
            print(f"[watcher] resume: best_manual_mAP50 = {best_manual:.4f}, last seen ep = {max(seen_epochs)}")
    if args.skip_history:
        prior = read_completed_epochs(results_csv)
        seen_epochs.update(prior)
        print(f"[watcher] --skip-history: marked {len(prior)} existing epochs as seen "
              f"(max ep = {max(prior) if prior else 0}). Only new epochs will be evaluated.")

    while True:
        try:
            epochs = read_completed_epochs(results_csv)
        except Exception as e:
            print(f"[watcher] read fail: {e}")
            time.sleep(args.poll_sec)
            continue

        new = [e for e in epochs if e not in seen_epochs]
        if new:
            for ep in new:
                if not last_pt.exists():
                    print(f"[watcher] ep{ep}: last.pt missing, skip")
                    seen_epochs.add(ep)
                    continue

                # Snapshot last.pt to a stable path so val doesn't fight the trainer
                tmp_pt = weights_dir / "_watcher_last.pt"
                try:
                    shutil.copyfile(last_pt, tmp_pt)
                except (PermissionError, OSError) as e:
                    print(f"[watcher] ep{ep}: copy fail, retry next loop: {e}")
                    break  # don't mark seen; retry

                try:
                    m50, m5095, p, r_ = eval_manual_mAP50(tmp_pt, yaml_path)
                except Exception as e:
                    print(f"[watcher] ep{ep}: eval crash, skip: {e}")
                    seen_epochs.add(ep)
                    if tmp_pt.exists():
                        tmp_pt.unlink()
                    continue

                is_best = m50 > best_manual + 1e-6
                if is_best:
                    best_manual = m50
                    shutil.copyfile(tmp_pt, best_manual_pt)
                    print(f"[watcher] ep{ep}: 🏆 NEW BEST manual mAP50 = {m50:.4f} (vs prior {best_manual - m50 + m50:.4f}) → saved best_manual.pt")
                else:
                    print(f"[watcher] ep{ep}: manual mAP50 = {m50:.4f} (best = {best_manual:.4f})")

                # Append to log
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(f"{ep},{m50:.4f},{m5095:.4f},{p:.4f},{r_:.4f},{int(is_best)}\n")

                seen_epochs.add(ep)
                if tmp_pt.exists():
                    tmp_pt.unlink()

        time.sleep(args.poll_sec)


if __name__ == "__main__":
    raise SystemExit(main())
