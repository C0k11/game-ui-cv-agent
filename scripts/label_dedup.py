# -*- coding: utf-8 -*-
"""源池标注去重(P2, cursor_task_v17_dataset_repair)。

两条规则(顺序执行, 全部机械可复现):
  R1 同文件同类 IoU>0.95 = 同一物件标了两次: 保留文件序第一条, 删其余。
  R2 单实例类(每日领奖7 / 点击继续字样142)残余仍多框且两两 IoU>0.5:
     保留面积最大的一条(横幅/按钮类偏松的框更可能盖全字形), 删其余。
     两两 IoU<=0.5 的组**不动**, 列人审清单(fail-closed)。
实测依据(scripts/_dup_scan_tmp.py): 7 双标 846 帧 / 142 双标 60 帧,
组内最大 IoU 全部 >0.5(中位 0.926/0.863); R1 命中 611 对 50 类。

用法: py -X utf8 scripts/dedup_label_boxes.py [--apply] [--root <labels根>]
--apply 前把要改的 txt 逐个备份到 <根>/_backups/<stamp>_dedup/ 并断言数量。
--root 默认 data/raw_images(源池); 传数据集 labels 根可镜像清洗已建集
  (build_ui_v2 重建会再引入泄漏, 已建集只能外科同步, 不重建)。
"""
from __future__ import annotations

import shutil
import sys
import time
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent))
import cls_domains as D  # noqa: E402

RAW = Path(__file__).resolve().parents[1] / "data" / "raw_images"
SINGLETON = {7, 142}
IOU_DUP = 0.95
IOU_SINGLE = 0.5


def iou(a, b):
    ax1, ay1 = a[1] - a[3] / 2, a[2] - a[4] / 2
    ax2, ay2 = a[1] + a[3] / 2, a[2] + a[4] / 2
    bx1, by1 = b[1] - b[3] / 2, b[2] - b[4] / 2
    bx2, by2 = b[1] + b[3] / 2, b[2] + b[4] / 2
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    ua = a[3] * a[4] + b[3] * b[4] - inter
    return inter / ua if ua > 0 else 0.0


def main() -> int:
    apply = "--apply" in sys.argv
    global RAW
    if "--root" in sys.argv:
        RAW = Path(sys.argv[sys.argv.index("--root") + 1])
    master = D.load_master()
    removed_r1: Counter = Counter()
    removed_r2: Counter = Counter()
    manual = []
    changes = {}

    for txt in RAW.rglob("*.txt"):
        rel = txt.relative_to(RAW).as_posix()
        if (txt.name in ("classes.txt", "_classes.txt", "_classes_next.txt")
                or rel.startswith("_backups/") or "/_ann/" in rel
                or "_labels_bak" in rel):
            continue
        try:
            lines = txt.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue
        boxes = []          # (line_idx, cls, cx, cy, w, h)
        for i, ln in enumerate(lines):
            p = ln.split()
            if len(p) >= 5 and p[0].isdigit():
                try:
                    boxes.append((i, int(p[0]), *(float(v) for v in p[1:5])))
                except ValueError:
                    continue
        drop = set()
        # R1: 同类 IoU>0.95 保留第一条
        by_cls = {}
        for b in boxes:
            by_cls.setdefault(b[1], []).append(b)
        for c, bs in by_cls.items():
            kept = []
            for b in bs:
                if any(iou(b[1:], k[1:]) > IOU_DUP for k in kept):
                    drop.add(b[0])
                    removed_r1[c] += 1
                else:
                    kept.append(b)
            # R2: 单实例类残余多框
            if c in SINGLETON and len(kept) > 1:
                pair_ok = all(iou(a[1:], b2[1:]) > IOU_SINGLE
                              for x, a in enumerate(kept)
                              for b2 in kept[x + 1:])
                if not pair_ok:
                    manual.append(f"{rel}: cls{c} x{len(kept)} 两两IoU<=0.5")
                    continue
                best = max(kept, key=lambda b: b[4] * b[5])
                for b in kept:
                    if b[0] != best[0]:
                        drop.add(b[0])
                        removed_r2[c] += 1
        if drop:
            changes[txt] = [ln for i, ln in enumerate(lines) if i not in drop]

    print(f"要改文件 {len(changes)} | R1 删 {sum(removed_r1.values())} 框"
          f"({len(removed_r1)} 类) | R2 删 {sum(removed_r2.values())} 框 | "
          f"人审 {len(manual)}")
    print("R1 top10:")
    for c, n in removed_r1.most_common(10):
        print(f"  {c:>4} {master[c][:22]:<24} {n}")
    print("R2:")
    for c, n in removed_r2.most_common():
        print(f"  {c:>4} {master[c][:22]:<24} {n}")
    for m in manual[:10]:
        print("  人审 " + m)

    if not apply:
        print("(未带 --apply, 只报告)")
        return 0
    stamp = time.strftime("%Y%m%d_%H%M%S")
    n_bak = 0
    for txt in changes:
        rel = txt.relative_to(RAW)
        dst = RAW / "_backups" / f"{stamp}_dedup" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(txt, dst)
        n_bak += 1
    assert n_bak == len(changes), "备份数量对不上"
    print(f"[backup] {n_bak} txt -> _backups/{stamp}_dedup/")
    for txt, new in changes.items():
        txt.write_text("\n".join(new) + ("\n" if new else ""),
                       encoding="utf-8")
    print(f"已写回 {len(changes)} 个文件")
    return 0


if __name__ == "__main__":
    sys.exit(main())
