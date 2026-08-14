# -*- coding: utf-8 -*-
"""ui_v2 数据集卫生常设自检(cursor_task_v17_dataset_repair 交付物 3)。

三道检查(rc=0 干净 / rc=1 有病):
  1. val 无泄漏: val 帧到最近 train 帧 dHash >= 14(用 val_leak_purge 的
     哈希缓存, 缓存齐全时秒级; 缓存缺失会现算, 全量约 90s)。
  2. 同一 label 文件内无同坐标异类框: val 违例 = FAIL;
     train 违例暂 WARN(剩余 25 框等 P1 看图裁决, 裁完把 _TRAIN_STRICT 翻 True)。
  3. 单实例类(每日领奖7/点击继续字样142)无同帧多标(IoU>0.5 视为同物):
     train+val 一律 FAIL(dedup_label_boxes 已清零, 再出现=回归)。

用法: py -X utf8 scripts/audit_dataset_hygiene.py [--fast]
--fast 跳过检查1的哈希现算(缓存缺失时直接 WARN 而不算, 给 test_offline 用)。
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

DS = Path("D:/Project/ml_cache/models/yolo/dataset/ui_v2")
_TRAIN_STRICT = False        # P1 裁完 train 侧同坐标冲突后翻 True
SINGLETON = {7, 142}
FAILS: list = []
WARNS: list = []


def fail(m):
    FAILS.append(m)
    print("FAIL " + m)


def warn(m):
    WARNS.append(m)
    print("WARN " + m)


def ok(m):
    print("  ok " + m)


def check_leak(fast: bool) -> None:
    import numpy as np
    cache_t = DS / "_dhash_cache_train.npz"
    cache_v = DS / "_dhash_cache_val.npz"
    if fast and not (cache_t.exists() and cache_v.exists()):
        warn("dHash 缓存缺失, --fast 跳过泄漏检查(跑一次 val_leak_purge 生成)")
        return
    import val_leak_purge as V
    tr_n, tr_h, _ = V.hash_split("train")
    va_n, va_h, _ = V.hash_split("val")
    if not len(va_h):
        fail("val 为空")
        return
    d = V.nearest_hamming(va_h, tr_h)
    n = int((d < V.THRESH).sum())
    if n:
        fail(f"val 有 {n} 帧与 train dHash<{V.THRESH}(泄漏) -- "
             f"跑 val_leak_purge --apply")
    else:
        ok(f"val {len(va_h)} 帧全部与 train 距离 >= {V.THRESH}")


def _boxes(txt: Path):
    for ln in txt.read_text(encoding="utf-8").splitlines():
        p = ln.split()
        if len(p) >= 5 and p[0].isdigit():
            try:
                yield int(p[0]), tuple(p[1:5]), tuple(float(v) for v in p[1:5])
            except ValueError:
                continue


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    ua = a[2] * a[3] + b[2] * b[3] - inter
    return inter / ua if ua > 0 else 0.0


def check_labels(fast: bool = False) -> None:
    for split in ("val", "train"):
        n_conf = n_single = 0
        samples = []
        files = sorted((DS / "labels" / split).glob("*.txt"))
        if fast and split == "train" and len(files) > 3000:
            files = files[:3000]      # 套件模式抽前3000(确定性); 全量去掉--fast
        for txt in files:
            coord_cls = defaultdict(set)
            sing = defaultdict(list)
            for c, key, geo in _boxes(txt):
                coord_cls[key].add(c)
                if c in SINGLETON:
                    sing[c].append(geo)
            for key, cs in coord_cls.items():
                if len(cs) > 1:
                    n_conf += 1
                    if len(samples) < 3:
                        samples.append(f"{txt.stem} {sorted(cs)}@{key[0]}")
            for c, gs in sing.items():
                if len(gs) > 1 and any(
                        _iou(a, b) > 0.5 for i, a in enumerate(gs)
                        for b in gs[i + 1:]):
                    n_single += 1
        if n_conf:
            msg = (f"{split} 同坐标异类 {n_conf} 处: {samples}")
            if split == "val" or _TRAIN_STRICT:
                fail(msg)
            else:
                warn(msg + " (等 P1 裁决, 裁完翻 _TRAIN_STRICT)")
        else:
            ok(f"{split} 无同坐标异类框")
        if n_single:
            fail(f"{split} 单实例类同帧多标 {n_single} 处 -- "
                 f"跑 dedup_label_boxes --apply")
        else:
            ok(f"{split} 单实例类无同帧多标")


def main() -> int:
    fast = "--fast" in sys.argv
    check_leak(fast)
    check_labels(fast)
    print()
    if FAILS:
        print(f"{len(FAILS)} 项不卫生, {len(WARNS)} 项待办 WARN")
        return 1
    print(f"数据集卫生 ({len(WARNS)} 项待办 WARN)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
