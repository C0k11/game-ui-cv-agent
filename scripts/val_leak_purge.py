# -*- coding: utf-8 -*-
"""ui_v2 val 去泄漏: 测量 / 模拟 / 执行 三档(P0-1, cursor_task_v17_dataset_repair)。

泄漏定义(仓库自有口径, build_ui_v2 注释定的线, 禁止放宽): val 帧到最近
train 帧的 8x8 dHash 汉明距离 < 14。build 现有 dedup 用 jpg 字节 md5,
只逮字节级重复 -- 重编码副本(_arrow_boost 族)和近邻帧全部放行, 这就是
实测 3,407 张逐像素同图只被逮到 32 张的原因。

用法:
  py -X utf8 scripts/val_leak_purge.py               测量: 直方图+按源池泄漏表
  py -X utf8 scripts/val_leak_purge.py --simulate    模拟清洗: 逐类 val 前后对比
  py -X utf8 scripts/val_leak_purge.py --apply       真清洗: 隔离区+清单+断言完整
哈希缓存在数据集根 _dhash_cache.npz(按文件名+mtime 失效), 32 进程并行。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import cls_domains as D  # noqa: E402

DS = Path("D:/Project/ml_cache/models/yolo/dataset/ui_v2")
THRESH = 14


def dhash64(path: str):
    """8x8 dHash -> uint64。读失败返回 None(调用方必须计数, 不许静默)。"""
    import cv2
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        small = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA)
        bits = (small[:, 1:] > small[:, :-1]).flatten()
        return int(np.packbits(bits).view(">u8")[0])
    except Exception:
        return None


def _hash_many(paths):
    return [dhash64(p) for p in paths]


def hash_split(split: str) -> tuple:
    """返回 (names, hashes uint64 数组, 读失败数)。带 mtime 缓存。"""
    imgs = sorted((DS / "images" / split).glob("*.jpg"))
    cache_f = DS / f"_dhash_cache_{split}.npz"
    cached = {}
    if cache_f.exists():
        z = np.load(cache_f, allow_pickle=True)
        cached = {n: (h, m) for n, h, m in
                  zip(z["names"], z["hashes"], z["mtimes"])}
    names, hashes, mtimes, todo = [], [], [], []
    for p in imgs:
        mt = p.stat().st_mtime_ns
        c = cached.get(p.name)
        names.append(p.name)
        mtimes.append(mt)
        if c is not None and int(c[1]) == mt:
            hashes.append(int(c[0]))
        else:
            hashes.append(-1)
            todo.append((len(hashes) - 1, str(p)))
    fails = 0
    if todo:
        print(f"[{split}] 哈希 {len(todo)}/{len(imgs)} 帧(其余走缓存)...")
        chunk = 256
        batches = [todo[i:i + chunk] for i in range(0, len(todo), chunk)]
        with ProcessPoolExecutor() as ex:
            for batch, out in zip(batches, ex.map(
                    _hash_many, [[p for _, p in b] for b in batches])):
                for (idx, _), h in zip(batch, out):
                    if h is None:
                        fails += 1
                        h = 0
                    hashes[idx] = h
        np.savez(cache_f, names=np.array(names),
                 hashes=np.array(hashes, dtype=np.uint64),
                 mtimes=np.array(mtimes, dtype=np.int64))
    if fails:
        print(f"WARN [{split}] {fails} 帧读不出(哈希置 0, 会被判泄漏保守处理)")
    return names, np.array(hashes, dtype=np.uint64), fails


_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def nearest_hamming(va: np.ndarray, tr: np.ndarray) -> np.ndarray:
    """每个 val 哈希到 train 哈希集的最小汉明距离。"""
    out = np.empty(len(va), dtype=np.int32)
    step = 512
    for i in range(0, len(va), step):
        x = np.bitwise_xor.outer(va[i:i + step], tr)
        d = _POP[x.view(np.uint8)].reshape(*x.shape, 8).sum(-1, dtype=np.uint16)
        out[i:i + step] = d.min(axis=1)
        del x, d
    return out


def pool_of(stem: str) -> str:
    return stem.split("__", 1)[0] if "__" in stem else "(无池前缀)"


def val_class_counts(skip: set = frozenset()) -> Counter:
    c: Counter = Counter()
    for txt in (DS / "labels" / "val").glob("*.txt"):
        if txt.stem in skip:
            continue
        for ln in txt.read_text(encoding="utf-8").splitlines():
            p = ln.split()
            if len(p) >= 5:
                c[int(p[0])] += 1
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    tr_names, tr_h, _ = hash_split("train")
    va_names, va_h, _ = hash_split("val")
    print(f"train {len(tr_h)} / val {len(va_h)} 哈希就绪 "
          f"({time.time() - t0:.0f}s)")
    dist = nearest_hamming(va_h, tr_h)

    n0 = int((dist == 0).sum())
    n8 = int((dist < 8).sum())
    n14 = int((dist < THRESH).sum())
    tot = len(dist)
    print(f"\n== 泄漏测量(val {tot} 帧, 线={THRESH}) ==")
    print(f"  距离=0(逐像素同图): {n0}  ({n0 / tot * 100:.1f}%)")
    print(f"  距离<8:            {n8}  ({n8 / tot * 100:.1f}%)")
    print(f"  距离<{THRESH}(泄漏):     {n14}  ({n14 / tot * 100:.1f}%)")
    print(f"  幸存:              {tot - n14}")

    leak_stems = {Path(va_names[i]).stem for i in range(tot)
                  if dist[i] < THRESH}
    per_pool: Counter = Counter()
    pool_tot: Counter = Counter()
    for i, n in enumerate(va_names):
        p = pool_of(Path(n).stem)
        pool_tot[p] += 1
        if dist[i] < THRESH:
            per_pool[p] += 1
    print("\n按源池(泄漏/总量):")
    for p, c in per_pool.most_common():
        print(f"  {p[:44]:<46} {c:>6}/{pool_tot[p]}")
    clean_pools = [p for p in pool_tot if p not in per_pool]
    if clean_pools:
        print("零泄漏源池: " + ", ".join(
            f"{p}({pool_tot[p]})" for p in clean_pools))

    if not (args.simulate or args.apply):
        return 0

    master = D.load_master()
    before = val_class_counts()
    after = val_class_counts(skip=leak_stems)
    print(f"\n== {'模拟' if not args.apply else '执行'}清洗后逐类 val ==")
    print(f"清洗前 val 有实例的类: {len(before)}, 清洗后: "
          f"{len([c for c in after.values() if c])}")
    died = sorted(set(before) - set(after))
    print(f"掉到 val=0 的类: {len(died)} 个")
    for i in died:
        nm = master[i] if i < len(master) else "?"
        print(f"  {i:>4} {nm[:30]:<32} val {before[i]} -> 0")
    kept_small = sorted((i for i in after if 0 < after[i] < 8),
                        key=lambda i: after[i])
    print(f"清洗后 0<val<8 的薄类: {len(kept_small)} 个")
    for i in kept_small[:20]:
        nm = master[i] if i < len(master) else "?"
        print(f"  {i:>4} {nm[:30]:<32} val {before[i]} -> {after[i]}")

    if args.apply:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        q = DS / f"_val_quarantine_leak{THRESH}_{stamp}"
        (q / "images").mkdir(parents=True)
        (q / "labels").mkdir(parents=True)
        manifest = []
        moved = 0
        for i, n in enumerate(va_names):
            if dist[i] >= THRESH:
                continue
            stem = Path(n).stem
            src_i = DS / "images" / "val" / n
            src_l = DS / "labels" / "val" / f"{stem}.txt"
            src_i.rename(q / "images" / n)
            if src_l.exists():
                src_l.rename(q / "labels" / f"{stem}.txt")
            manifest.append({"stem": stem, "min_dist": int(dist[i])})
            moved += 1
        (q / "manifest.jsonl").write_text(
            "\n".join(json.dumps(m, ensure_ascii=False) for m in manifest)
            + "\n", encoding="utf-8")
        assert moved == len(leak_stems), f"移动 {moved} != 泄漏 {len(leak_stems)}"
        left = len(list((DS / "images" / "val").glob("*.jpg")))
        print(f"\n已隔离 {moved} 帧 -> {q.name}, val 剩 {left} 帧"
              f"(清单 manifest.jsonl, 可整目录搬回还原)")
        # 清哈希缓存(val 变了)
        (DS / "_dhash_cache_val.npz").unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
