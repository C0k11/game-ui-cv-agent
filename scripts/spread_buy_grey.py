# -*- coding: utf-8 -*-
"""`购买灰色` 模板标注 —— **按版面分组**，拿亮态帧的框位置去标买完态帧。

⛔为什么不能靠预标：现役模型在买完态对 `购买灰色` **0 检出**
   （day3 三处实证：免費包已领 0/1、信用点商店 8 件全灰 **0/8**、大赛商店 0/2）。
⛔为什么不能用像素判据：[[template_label_assets]]「亮/灰用亮度或饱和度区分」
   已被否掉五次（锁定 tile 那次实测**锁着的比没锁的还亮**）。
⛔⛔**为什么必须按版面分组**（2026-08-12 第一版就栽在这）：不分组直接聚类，
   23 个"槽位"互相打架 —— 同一位置出现 w=0.035 和 w=0.036 两个模板、
   cy 有 0.555/0.695/0.825/0.830/0.845/0.910 六种，因为池子里混着
   **信用点商店 4×2 / 组合包页 3 列 / 战术大赛 2 瓶** 好几种版面。
   无差别套 = 给每张帧塞一堆位置错误的框，正是
   [[template_label_assets]]「**一条错标能让整族判据失效**」。

判据（全结构性，不碰像素）:
  · 版面身份 = 同帧内互斥的页面锚（信用点商店_已选中 / 战术大赛商店已选择 / 组合包*）
  · 亮态帧 → 按版面聚 `购买` 的槽位；**只取出现 ≥3 帧**的
  · ⛔**槽位数必须和该版面的已知商品数吻合**，否则说明亮态样本不全 → 跳过该版面
  · 买完态帧 = 同版面 + 有货架锚 + 一个 `购买` 都没有

用法: py scripts/spread_buy_grey.py [池名] [--go]
"""
import io
import os
import sys
from collections import Counter, defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data", "raw_images")
# 版面 → 该版面**已知**有几个购买位。槽位数对不上就不套（宁可不标）。
EXPECT_SLOTS = {"大赛货架": 2, "组合包页": 3}
MARK = {"信用点商店_已选中": "信用点货架", "战术大赛商店已选择": "大赛货架",
        "组合包已选择": "组合包页", "组合包未选择": "组合包页"}


def main():
    pool = next((a for a in sys.argv[1:] if not a.startswith("--")),
                "flywheel_v16_20260812")
    go = "--go" in sys.argv
    d = os.path.join(RAW, pool)
    M = [l.strip() for l in open(os.path.join(RAW, "_classes.txt"),
                                 encoding="utf-8") if l.strip()]
    IDX = {n: i for i, n in enumerate(M)}
    BUY, GREY = IDX["购买"], IDX["购买灰色"]
    MARKI = {IDX[k]: v for k, v in MARK.items() if k in IDX}
    ANCH = {IDX[n] for n in ("全部选择", "已全部选择", "全部选择灰", "选择购买")
            if n in IDX}

    tpl = defaultdict(Counter)
    bought = defaultdict(list)
    for f in sorted(os.listdir(d)):
        if not f.endswith(".txt") or f == "classes.txt":
            continue
        p = os.path.join(d, f)
        rows = [l.split() for l in open(p, encoding="utf-8") if l.split()]
        cs = {int(r[0]) for r in rows}
        lay = {MARKI[c] for c in cs if c in MARKI}
        if len(lay) != 1:              # 版面认不准 → 一律不碰
            continue
        L = lay.pop()
        if BUY in cs:
            for r in rows:
                if int(r[0]) == BUY:
                    tpl[L][(round(float(r[1]), 2), round(float(r[2]), 2),
                            round(float(r[3]), 3), round(float(r[4]), 3))] += 1
        elif (cs & ANCH) and GREY not in cs:
            bought[L].append(p)

    total = 0
    for L in sorted(set(list(tpl) + list(bought))):
        slots = [(k, v) for k, v in tpl[L].items() if v >= 3]
        need = EXPECT_SLOTS.get(L)
        ok = need is not None and len(slots) == need
        print(f"\n【{L}】亮态槽位 {len(slots)} 个 / 已知商品位 {need}"
              f" / 买完态帧 {len(bought[L])}   → {'✅套' if ok else '⛔跳过'}")
        for (cx, cy, w, h), n in sorted(slots, key=lambda x: x[0][0]):
            print(f"     cx={cx:.2f} cy={cy:.2f} w={w:.3f} h={h:.3f} ×{n}")
        if not ok:
            print("     （槽位数和已知商品位对不上 = 亮态样本不全，不标）")
            continue
        if not go:
            total += len(bought[L]) * len(slots)
            continue
        for p in bought[L]:
            rows = [l.rstrip("\n") for l in open(p, encoding="utf-8") if l.strip()]
            for (cx, cy, w, h), _ in slots:
                rows.append("%d %.6f %.6f %.6f %.6f" % (GREY, cx, cy, w, h))
            open(p, "w", encoding="utf-8").write("\n".join(rows) + "\n")
            total += len(slots)
    print(f"\n{'✅ 已补' if go else '（预演）将补'} {total} 个 `购买灰色` 框"
          f"  ⛔仍需前端人审")


if __name__ == "__main__":
    main()
