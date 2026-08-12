# -*- coding: utf-8 -*-
"""就地预标飞轮池 (run_*_clean), 零 copy 不占盘.

curate_flywheel.py 会把帧复制进一个新队列目录 — 对 27K 帧的全池盘点来说
那是 ~7GB 无谓拷贝。本脚本改为**就地**在原 run_* 目录写 5 列 YOLO txt:
  - 近重复过滤 (与 curate 同一 48x27 灰度 MAD 判据, 默认 3.0)
  - 跳过已有 txt 的帧 (可能是人审过的, 绝不覆盖)
  - v14 预标, MASTER 类索引, 5 列 (第 6 列会被当 OBB angle)
产出 data/_flywheel_prelabel_manifest.csv 记录本轮标了哪些帧, 供后续
「cls×场景覆盖矩阵」挑缺口帧用。

Usage:
  py -X utf8 scripts/prelabel_flywheel_inplace.py            # 全池
  py -X utf8 scripts/prelabel_flywheel_inplace.py --dry      # 只统计不写
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw_images"
MASTER = RAW / "_classes.txt"
UI_WEIGHTS = Path(r"D:\Project\ml_cache\models\yolo\runs\ui_yolo26m_v14\weights\best.pt")
MANIFEST = ROOT / "data" / "_flywheel_prelabel_manifest.csv"

sys.path.insert(0, str(ROOT))


# ⛔战斗域目录: 用 UI 模型预标战斗帧 = 灌垃圾。名字带这些词的一律不碰,
# 不依赖 build_battle_v9.SRCS (那张表只收已并入训练的, botplay 新录不在里面 —
# 2026-08-02 实测 7 个 run_*_botplay_clean 全在飞轮口径内, 当时只是碰巧
# 每张都已有 txt 才没被写脏)。战斗侧标注走 battle 模型自己的管线。
BATTLE_MARKERS = ("botplay", "battle", "combat")


def _flywheel_dirs() -> list[Path]:
    """dashboard 的飞轮分组口径: run_* 且未纳入 build_ui_v2 / battle sources。"""
    from scripts.build_ui_v2 import REAL_SOURCES, SYNTH_SOURCES, VAL_SOURCES
    used = set(REAL_SOURCES) | set(SYNTH_SOURCES) | set(VAL_SOURCES)
    try:
        from scripts.build_battle_v9 import SRCS as battle_srcs
        used |= {p.name for p in battle_srcs}
    except Exception:
        pass
    out, skipped = [], []
    for d in sorted(RAW.iterdir()):
        if not (d.is_dir() and d.name.startswith("run_") and d.name not in used
                and not d.name.startswith("_val_") and any(d.glob("*.jpg"))):
            continue
        if any(m in d.name.lower() for m in BATTLE_MARKERS):
            skipped.append(d.name)
            continue
        out.append(d)
    if skipped:
        print(f"⛔battle-domain dirs excluded ({len(skipped)}): {skipped[:3]}"
              + (" ..." if len(skipped) > 3 else ""))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dedup", type=float, default=3.0)
    ap.add_argument("--conf", type=float, default=0.20)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    from scripts.curate_flywheel import _dedup_dir

    srcs = _flywheel_dirs()
    print(f"flywheel dirs: {len(srcs)}")

    t0 = time.time()
    keep: list[str] = []
    total_in = 0
    with ProcessPoolExecutor(max_workers=24) as ex:
        for _src, n_in, kept in ex.map(_dedup_dir, [(str(d), args.dedup) for d in srcs]):
            total_in += n_in
            keep.extend(kept)
    print(f"dedup {args.dedup}: {total_in} -> {len(keep)}  ({time.time() - t0:.0f}s)")

    # 已有 txt 的帧绝不覆盖 (可能是人审过的标注)
    todo = [p for p in keep if not Path(p).with_suffix(".txt").exists()]
    print(f"already labeled (skipped): {len(keep) - len(todo)}   to pre-label: {len(todo)}")
    if args.dry:
        return

    master = [c.strip() for c in MASTER.read_text(encoding="utf-8").splitlines() if c.strip()]
    name2idx = {n: i for i, n in enumerate(master)}

    from ultralytics import YOLO
    model = YOLO(str(UI_WEIGHTS))
    rows, boxes_written, skipped_names = [], 0, set()
    BATCH = 16
    t1 = time.time()
    for s in range(0, len(todo), BATCH):
        chunk = todo[s:s + BATCH]
        for path, res in zip(chunk, model.predict(chunk, conf=args.conf, imgsz=960, verbose=False)):
            h, w = res.orig_shape
            lines = []
            for b in res.boxes:
                name = model.names[int(b.cls[0])]
                idx = name2idx.get(name)
                if idx is None:
                    skipped_names.add(name)
                    continue
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
                lines.append(
                    f"{idx} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                    f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
            Path(path).with_suffix(".txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            boxes_written += len(lines)
            rows.append((str(Path(path).relative_to(RAW)), len(lines)))
        if (s // BATCH) % 20 == 0:
            done = min(s + BATCH, len(todo))
            print(f"  {done}/{len(todo)}  {time.time() - t1:.0f}s", flush=True)

    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([("frame", "n_boxes")] + rows)
    print(f"pre-labeled {len(rows)} frames, {boxes_written} boxes in {time.time() - t1:.0f}s"
          + (f", skipped cls: {skipped_names}" if skipped_names else ""))
    print(f"manifest: {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
