# -*- coding: utf-8 -*-
"""Block until v10 training finishes, then dump final metrics.

Filesystem-based liveness only (NO os.kill — on Windows os.kill(pid,0) goes
through TerminateProcess and is both unreliable as a probe and dangerous).
Training is considered DONE when the train log stops growing for STALL_S, or
results.csv reaches the final epoch. Run with: py scripts/watch_v10.py
"""
import csv
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

RUN = r"D:\Project\ml_cache\models\yolo\runs\ui_yolo26m_v10"
CSVP = os.path.join(RUN, "results.csv")
BESTP = os.path.join(RUN, "weights", "best.pt")
LOG = r"D:\Project\ai game secretary\logs\train_v10.log"
TOTAL_EP = 70
STALL_S = 360          # log untouched this long ⇒ training ended (done or died)
POLL_S = 120


def log_mtime():
    try:
        return os.path.getmtime(LOG)
    except OSError:
        return 0.0


def epochs_done():
    if not os.path.exists(CSVP):
        return 0
    try:
        return sum(1 for _ in open(CSVP)) - 1   # minus header
    except OSError:
        return 0


while True:
    time.sleep(POLL_S)
    if epochs_done() >= TOTAL_EP:
        break
    # stall check: log file not modified for STALL_S
    age = time.time() - log_mtime()
    if age > STALL_S:
        break

print("=== v10 TRAINING ENDED ===", flush=True)
if os.path.exists(CSVP):
    rows = list(csv.DictReader(open(CSVP)))
    print(f"epochs completed: {len(rows)}")
    if rows:
        def m95(r):
            try:
                return float(r.get("metrics/mAP50-95(B)", 0) or 0)
            except ValueError:
                return 0.0
        last = rows[-1]
        best = max(rows, key=m95)
        print(f"last ep{last['epoch']}: mAP50={last.get('metrics/mAP50(B)')}  "
              f"mAP50-95={last.get('metrics/mAP50-95(B)')}")
        print(f"BEST ep{best['epoch']}: mAP50={best.get('metrics/mAP50(B)')}  "
              f"mAP50-95={best.get('metrics/mAP50-95(B)')}")
else:
    print("NO results.csv — training may have crashed")
print(f"best.pt exists: {os.path.exists(BESTP)}")
