# -*- coding: utf-8 -*-
"""用 可購買N次(游戏印在像素里的语义不变量) 裁决 购买(103)/购买灰色(489)。

任务书 cursor_task_v17_dataset_repair P0-2:
  - 212 个同坐标 103+489 冲突(自动预标没人审直接进池)
  - _traj_val_0807 那屏 售罄按钮被标成可点的 购买, 184 份近重复放大了它
真值: 可購買0次 -> 489, 可購買>0次 -> 103。
判据校准: run_20260807_205039_tick_0001 上 9/9 与肉眼真值一致
  (OCR 带 = 按钮顶上方 1.65~2.45 个按钮高, x2 平铺 64px, 双份读数一致才算)。
读不出 = 不动 + 列人审清单(一般商店没有 可購買N次, 天然 fail-closed 跳过)。

用法: py -X utf8 scripts/fix_buy_state_ocr.py [--apply]
不带 --apply 只报告; --apply 先整池备份到 _backups/<stamp>_kgmfix/ 并断言完整。
"""
from __future__ import annotations

import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from routing_v2.percept import read as R  # noqa: E402

import cv2  # noqa: E402

RAW = ROOT / "data" / "raw_images"
POOLS = ["_traj_val_0807", "_val_unlab_20260807", "_val_unlab_20260730"]
BUY, BUY_GREY = 103, 489
_KGM = re.compile(r"買\s*(\d+)\s*次")


def read_n(frame, cx, cy, w, h):
    """OCR 可購買N次 -> N; 读不出/两份不一致 -> None(fail-closed)。"""
    H, W = frame.shape[:2]
    x1 = max(0, int((cx - 1.1 * w) * W))
    x2 = min(W, int((cx + 1.1 * w) * W))
    y1 = max(0, int((cy - h / 2 - 2.45 * h) * H))
    y2 = min(H, int((cy - h / 2 - 1.65 * h) * H))
    if x2 - x1 < 8 or y2 - y1 < 4:
        return None
    crop = frame[y1:y2, x1:x2]
    s = 64.0 / crop.shape[0]
    crop = cv2.resize(crop, (int(crop.shape[1] * s), 64),
                      interpolation=cv2.INTER_CUBIC)
    try:
        res, _ = R._engine()(cv2.hconcat([crop, crop]))
    except Exception:
        return None
    txt = "".join(str(t[1]) for t in (res or []))
    hits = _KGM.findall(txt)
    if len(hits) >= 2 and len(set(hits)) == 1:
        return int(hits[0])
    return None


def main() -> int:
    apply = "--apply" in sys.argv
    stamp = time.strftime("%Y%m%d_%H%M%S")
    stat = Counter()
    unread = []
    changes = {}          # txt_path -> new_lines

    for pool in POOLS:
        pd = RAW / pool
        if not pd.is_dir():
            print(f"WARN 池缺失 {pool}")
            continue
        for txt in sorted(pd.glob("*.txt")):
            if txt.name == "classes.txt":
                continue
            lines = txt.read_text(encoding="utf-8").splitlines()
            tgt = [(i, ln) for i, ln in enumerate(lines)
                   if ln.split() and ln.split()[0] in ("103", "489")]
            if not tgt:
                continue
            jpg = txt.with_suffix(".jpg")
            if not jpg.exists():
                stat["no_img"] += len(tgt)
                continue
            frame = cv2.imdecode(np.fromfile(str(jpg), dtype=np.uint8),
                                 cv2.IMREAD_COLOR)
            if frame is None:
                stat["bad_img"] += len(tgt)
                continue
            # 按坐标分组(同坐标 103+489 = 冲突对, 一次 OCR 裁决整组)
            groups = {}
            for i, ln in tgt:
                p = ln.split()
                groups.setdefault(tuple(p[1:5]), []).append((i, int(p[0])))
            drop_idx, flip = set(), {}
            for xy, members in groups.items():
                cx, cy, w, h = (float(v) for v in xy)
                n = read_n(frame, cx, cy, w, h)
                stat["boxes"] += len(members)
                if n is None:
                    stat["unreadable"] += len(members)
                    unread.append(f"{pool}/{txt.stem} @{xy[0]},{xy[1]}"
                                  f" cls={[c for _, c in members]}")
                    continue
                want = BUY_GREY if n == 0 else BUY
                keep_done = False
                for i, c in sorted(members, key=lambda t: t[1] != want):
                    if c == want and not keep_done:
                        keep_done = True
                        stat["kept_ok"] += 1
                    elif c == want and keep_done:
                        drop_idx.add(i)          # 同类重复行
                        stat["dup_dropped"] += 1
                    elif len(members) > 1:
                        drop_idx.add(i)          # 冲突对里错的那条
                        stat["conflict_dropped"] += 1
                    else:
                        flip[i] = want           # 单条标错: 翻类
                        stat[f"flip_{c}to{want}"] += 1
            if drop_idx or flip:
                new = []
                for i, ln in enumerate(lines):
                    if i in drop_idx:
                        continue
                    if i in flip:
                        p = ln.split()
                        p[0] = str(flip[i])
                        ln = " ".join(p)
                    new.append(ln)
                changes[txt] = new

    print(f"检查框数 {stat['boxes']} | 判对保留 {stat['kept_ok']} | "
          f"冲突删错行 {stat['conflict_dropped']} | "
          f"翻 103->489 {stat['flip_103to489']} | "
          f"翻 489->103 {stat['flip_489to103']} | "
          f"同类重复删 {stat['dup_dropped']} | 读不出不动 {stat['unreadable']}")
    print(f"要改的文件: {len(changes)}")
    if unread:
        print(f"人审清单({len(unread)} 条, 前10):")
        for u in unread[:10]:
            print("  " + u)

    if not apply:
        print("(未带 --apply, 只报告)")
        return 0

    # 备份: 整池 txt 快照, 断言数量一致才继续(半备份半覆盖是最坏情况)
    for pool in POOLS:
        src = RAW / pool
        dst = RAW / "_backups" / f"{stamp}_kgmfix" / pool
        dst.mkdir(parents=True, exist_ok=True)
        n = 0
        for t in src.glob("*.txt"):
            shutil.copy2(t, dst / t.name)
            n += 1
        assert n == len(list(src.glob("*.txt"))), f"备份不完整 {pool}"
        print(f"[backup] {pool}: {n} txt -> {dst}")
    for txt, new in changes.items():
        txt.write_text("\n".join(new) + ("\n" if new else ""),
                       encoding="utf-8")
    print(f"已写回 {len(changes)} 个文件")
    return 0


if __name__ == "__main__":
    sys.exit(main())
