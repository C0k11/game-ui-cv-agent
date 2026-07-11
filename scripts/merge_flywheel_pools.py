# -*- coding: utf-8 -*-
"""合并零散飞轮 _clean 池 → 单一 merged 池 (2026-07-11 用户: 不要太零散).

用法: py scripts/merge_flywheel_pools.py 20260711
  → data/raw_images/run_20260711_*_clean (排除已是 merged 的) 全部帧
    以 <时刻>_<原名> 前缀拷入 run_<date>_merged_clean, 校验计数后删源池。
已有标签(若有)一并带走; classes.txt 写 master 副本。
"""
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
RAW = Path(r"D:\Project\ai game secretary\data\raw_images")


def main() -> None:
    date = sys.argv[1] if len(sys.argv) > 1 else "20260711"
    dst = RAW / f"run_{date}_merged_clean"
    srcs = sorted(p for p in RAW.iterdir()
                  if p.is_dir() and p.name.startswith(f"run_{date}_")
                  and p.name.endswith("_clean") and "merged" not in p.name)
    if not srcs:
        print("no source pools"); return
    dst.mkdir(exist_ok=True)
    shutil.copy2(RAW / "_classes.txt", dst / "classes.txt")
    total = 0
    for src in srcs:
        tag = src.name.replace(f"run_{date}_", "").replace("_clean", "")
        n = 0
        for jpg in sorted(src.glob("*.jpg")):
            new = dst / f"{tag}_{jpg.name}"
            shutil.copy2(jpg, new)
            txt = jpg.with_suffix(".txt")
            if txt.exists():
                shutil.copy2(txt, new.with_suffix(".txt"))
            n += 1
        total += n
        print(f"  {src.name}: {n}帧 → 前缀 {tag}_")
    # 校验后删源
    got = len(list(dst.glob("*.jpg")))
    assert got >= total, f"merged {got} < copied {total}"
    for src in srcs:
        shutil.rmtree(src)
    print(f"== merged {total}帧 from {len(srcs)}池 → {dst.name}; 源池已删")


if __name__ == "__main__":
    main()
