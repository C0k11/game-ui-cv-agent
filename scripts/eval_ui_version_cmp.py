# -*- coding: utf-8 -*-
"""ui v15 vs v16 在**同一个 val** 上的逐类验收。

为什么不看 mAP:
  v15 的 val 是 2,192 帧、v16 的是 10,830 帧 —— 两个 mAP 量的是**不同的题**，
  跨 val 版本不可比。验收只认**固定阈值下的 TP/FP/FN**，因为线上就是拿固定
  阈值在判（pages.py 的页面签名用 CONF=0.45，金钱判据用 0.40）。

为什么 FP 必须拆开:
  DUP  同一个目标被框了两次     —— 无害，`find()` 取 conf argmax
  WRONG 框对了位置但认成了别的类 —— **真正会让 flow 点错东西的那种**
  BG   凭空框出一个             —— 会让页面签名误命中
  v14 -> v15 那次就是靠 WRONG 从 63 掉到 4 才认定是真赢的。

索引对齐（已实测）:
  v15 nc=492 / v16 nc=528，前 492 个索引里有 31 处名字不同，**全部**是
  `_废弃{同一索引}_...` 形式的改名 —— 索引一一对应，所以按**索引**比，
  绝不按名字比（按名字比这 31 个会全部变成"v15 有 v16 没有"）。
  492..527 是 v16 新增的类，v15 结构上不可能检出，单独列，不算它退步。

跑法（训练跑完之后再跑，别和训练抢 GPU）:
  py scripts/eval_ui_version_cmp.py
  py scripts/eval_ui_version_cmp.py --limit 1500        # 快速抽样
  py scripts/eval_ui_version_cmp.py --b D:/.../epoch55.pt --btag v16ep55
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

DATA_YAML = r"D:\Project\ml_cache\models\yolo\dataset\ui_v2\data.yaml"
RUNS = r"D:\Project\ml_cache\models\yolo\runs"
A_DEF = os.path.join(RUNS, "ui_yolo26m_v15", "weights", "best_real.pt")
B_DEF = os.path.join(RUNS, "ui_yolo26m_v16", "weights", "best.pt")

IOU_MATCH = 0.50
SCAN_CONF = 0.20          # 推理时保留到这个 conf，之后在内存里换阈值评估
THRESHOLDS = (0.25, 0.35, 0.45)
MAIN_CONF = 0.45          # 主口径 = pages.py 的页面签名阈值
# val GT 少于这个数的类，recall 只能取到几个离散值，别当结论用
#   （[[v16_dataset_integration]] 那条「小样本陷阱」: 2 帧扫出"零错读"
#    扩到 14 帧就变成 8 对 4 错）。
MIN_GT_FOR_VERDICT = 20

# 用户点名要核的族（idx -> 说明）。训练完先看这几族，任何一族退步就不上 registry。
FAMILIES = {
    "任务页签(v16新增)": list(range(511, 521)),
    "关卡弹窗两页签": [421, 422, 495, 496],
    "每日领奖(cls7 迁位后)": [7],
    "走格子(v16新增)": list(range(497, 511)),
    "批量扫荡方案": [493, 494],
    "制造槽/等待时间": [492, 526],
}

# 金钱相关的类：**必须按 region 拆**（v14->v15 那次，弹窗体内的青辉石从
#   0.061 涨到 0.429 才是真的修好，而合并看是 0.935 -> 0.944，完全看不出）。
MONEY_NAMES = ("青辉石", "购买青辉石", "确认键", "取消键")
TOPBAR_CY = 0.12          # cy < 0.12 是顶栏余额；>= 是弹窗体内的价签


def load_names():
    import yaml
    d = yaml.safe_load(open(DATA_YAML, encoding="utf-8"))
    return d["names"], d["path"], d.get("val", "images/val")


def val_items(root, val_rel, limit=0):
    img_dir = os.path.join(root, val_rel.replace("/", os.sep))
    lbl_dir = img_dir.replace(os.sep + "images" + os.sep,
                              os.sep + "labels" + os.sep)
    if not os.path.isdir(lbl_dir):
        lbl_dir = re.sub(r"images", "labels", img_dir, count=1)
    files = sorted(f for f in os.listdir(img_dir)
                   if f.lower().endswith((".jpg", ".png", ".jpeg")))
    if limit and limit < len(files):
        step = len(files) / float(limit)
        files = [files[int(i * step)] for i in range(limit)]
    out = []
    for f in files:
        stem = f.rsplit(".", 1)[0]
        out.append((os.path.join(img_dir, f),
                    os.path.join(lbl_dir, stem + ".txt")))
    return out


def read_gt(path):
    """YOLO 标签 -> [(cls, x1,y1,x2,y2)]，归一化坐标。

    第 6 列是 angle 不是 conf（历史坑），一律只取前 5 列。
    """
    if not os.path.isfile(path):
        return []
    out = []
    for line in open(path, encoding="utf-8"):
        p = line.split()
        if len(p) < 5:
            continue
        c = int(float(p[0]))
        cx, cy, w, h = (float(v) for v in p[1:5])
        out.append((c, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return out


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / ua if ua > 0 else 0.0


def infer_all(weights, items, tag, device=0):
    """返回 [[(cls, conf, x1,y1,x2,y2), ...], ...]，与 items 同序（归一化坐标）。"""
    from ultralytics import YOLO
    m = YOLO(weights)
    preds = []
    B = 16
    for i in range(0, len(items), B):
        chunk = [p for p, _ in items[i:i + B]]
        res = m.predict(chunk, imgsz=960, conf=SCAN_CONF, iou=0.6,
                        verbose=False, device=device)
        for r in res:
            h, w = r.orig_shape
            one = []
            for b in r.boxes:
                x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
                one.append((int(b.cls.item()), float(b.conf.item()),
                            x1 / w, y1 / h, x2 / w, y2 / h))
            preds.append(one)
        if (i // B) % 20 == 0:
            print(f"  [{tag}] {min(i + B, len(items))}/{len(items)}", flush=True)
    return preds


def score(items, gts, preds, conf, nc):
    """固定阈值下逐类统计。FP 拆 DUP / WRONG / BG。"""
    st = {c: dict(tp=0, fn=0, dup=0, wrong=0, bg=0, gt=0, pred=0)
          for c in range(nc)}
    confuse = defaultdict(int)                 # (pred_cls, gt_cls) -> n
    money = defaultdict(lambda: dict(tp=0, fn=0, gt=0))   # (name, 区) -> ...
    for (ipath, _), gt, pr in zip(items, gts, preds):
        pr = [p for p in pr if p[1] >= conf]
        pr.sort(key=lambda p: -p[1])
        used = [False] * len(gt)
        for c, *_ in gt:
            if c < nc:
                st[c]["gt"] += 1
        for pc, pconf, *box in pr:
            if pc >= nc:
                continue
            st[pc]["pred"] += 1
            best_i, best_v = -1, 0.0
            for gi, (gc, *gb) in enumerate(gt):
                if gc != pc or used[gi]:
                    continue
                v = iou(box, gb)
                if v > best_v:
                    best_i, best_v = gi, v
            if best_i >= 0 and best_v >= IOU_MATCH:
                used[best_i] = True
                st[pc]["tp"] += 1
                continue
            # 同类但那个 GT 已经被更高 conf 的框认领了 = 重复框，无害
            dup = any(gc == pc and used[gi] and iou(box, gb) >= IOU_MATCH
                      for gi, (gc, *gb) in enumerate(gt))
            if dup:
                st[pc]["dup"] += 1
                continue
            # 位置对得上但类判错了 = WRONG，这是会让 flow 点错东西的那种
            wi, wv = -1, 0.0
            for gi, (gc, *gb) in enumerate(gt):
                v = iou(box, gb)
                if v > wv:
                    wi, wv = gi, v
            if wi >= 0 and wv >= IOU_MATCH:
                st[pc]["wrong"] += 1
                confuse[(pc, gt[wi][0])] += 1
            else:
                st[pc]["bg"] += 1
        for gi, (gc, *gb) in enumerate(gt):
            if gc < nc and not used[gi]:
                st[gc]["fn"] += 1
    return st, confuse


def money_region_stats(items, gts, preds, conf, names):
    """金钱类按 region 拆。合并看会把真正的缺口盖住（v14->v15 实证）。"""
    idx = {i for i, n in names.items() if str(n) in MONEY_NAMES}
    out = defaultdict(lambda: dict(tp=0, fn=0, gt=0))
    for (_, _), gt, pr in zip(items, gts, preds):
        pr = [p for p in pr if p[1] >= conf]
        used = [False] * len(gt)
        for pc, pconf, *box in sorted(pr, key=lambda p: -p[1]):
            if pc not in idx:
                continue
            for gi, (gc, *gb) in enumerate(gt):
                if gc == pc and not used[gi] and iou(box, gb) >= IOU_MATCH:
                    used[gi] = True
                    cy = (gb[1] + gb[3]) / 2
                    key = (str(names[pc]), "顶栏" if cy < TOPBAR_CY else "弹窗体内")
                    out[key]["tp"] += 1
                    break
        for gi, (gc, *gb) in enumerate(gt):
            if gc not in idx:
                continue
            cy = (gb[1] + gb[3]) / 2
            key = (str(names[gc]), "顶栏" if cy < TOPBAR_CY else "弹窗体内")
            out[key]["gt"] += 1
            if not used[gi]:
                out[key]["fn"] += 1
    return out


def rec(s):
    return s["tp"] / s["gt"] if s["gt"] else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default=A_DEF)
    ap.add_argument("--b", default=B_DEF)
    ap.add_argument("--atag", default="v15")
    ap.add_argument("--btag", default="v16")
    ap.add_argument("--limit", type=int, default=0)
    # 训练还在跑时用 --device cpu 冒烟, 别和训练抢显存
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default=os.path.join(RUNS, "cmp_v15_v16"))
    args = ap.parse_args()

    for p in (args.a, args.b):
        if not os.path.isfile(p):
            print(f"权重不存在: {p}")
            return 1

    names, root, val_rel = load_names()
    nc15 = 492                                   # v15 的类数，超出的它检不出
    items = val_items(root, val_rel, args.limit)
    print(f"val {len(items)} 帧 / 类表 {len(names)} 类 / "
          f"匹配 IoU>={IOU_MATCH} / 主阈值 conf={MAIN_CONF}")
    gts = [read_gt(l) for _, l in items]
    ngt = sum(len(g) for g in gts)
    print(f"GT 共 {ngt} 框")

    dev = args.device if args.device == "cpu" else int(args.device)
    pa = infer_all(args.a, items, args.atag, dev)
    pb = infer_all(args.b, items, args.btag, dev)

    os.makedirs(args.out, exist_ok=True)
    report = []

    def say(s=""):
        print(s)
        report.append(s)

    for conf in THRESHOLDS:
        sa, ca = score(items, gts, pa, conf, len(names))
        sb, cb = score(items, gts, pb, conf, len(names))
        tot = lambda s, k: sum(s[c][k] for c in s)
        say()
        say(f"== conf >= {conf} ==")
        say(f"  {'':6s} {'TP':>7s} {'FN':>7s} {'DUP':>6s} {'WRONG':>6s} {'BG':>6s}")
        for tag, s in ((args.atag, sa), (args.btag, sb)):
            say(f"  {tag:6s} {tot(s,'tp'):7d} {tot(s,'fn'):7d} "
                f"{tot(s,'dup'):6d} {tot(s,'wrong'):6d} {tot(s,'bg'):6d}")
        if conf != MAIN_CONF:
            continue

        # 只在主阈值上做逐类判决
        say()
        say("-- 逐族（用户点名要核的）--")
        for fam, idxs in FAMILIES.items():
            say(f"  [{fam}]")
            for i in idxs:
                if i >= len(names):
                    continue
                n = str(names[i])
                g = sb[i]["gt"]
                if g == 0:
                    say(f"    idx{i:3d} {n:26s} val 无 GT，测不了")
                    continue
                r_a = rec(sa[i]) if i < nc15 else None
                r_b = rec(sb[i])
                a_txt = f"{r_a:.3f}" if r_a is not None else "  n/a"
                weak = "  [样本太少,不可当结论]" if g < MIN_GT_FOR_VERDICT else ""
                say(f"    idx{i:3d} {n:26s} GT{g:6d}  "
                    f"{args.atag} R={a_txt}  {args.btag} R={r_b:.3f}  "
                    f"WRONG {sa[i]['wrong'] if i < nc15 else 0}->{sb[i]['wrong']}"
                    f"{weak}")

        # 退步清单：只看 v15 也有能力的那 492 个类，且 val 里真有 GT
        say()
        say("-- 退步清单（v16 比 v15 差；只算 idx<492 且 val 有 GT 的类）--")
        bad = []
        for i in range(nc15):
            if sb[i]["gt"] == 0:
                continue
            ra, rb = rec(sa[i]), rec(sb[i])
            dw = sb[i]["wrong"] - sa[i]["wrong"]
            if sb[i]["gt"] < MIN_GT_FOR_VERDICT:
                continue                 # 小样本不进退步判决, 单独列
            if (rb - ra) < -0.02 or dw > 0:
                bad.append((rb - ra, dw, i))
        bad.sort()
        if not bad:
            say("  无。v16 在每一个 v15 也认得的类上都不差于 v15。")
        for d, dw, i in bad[:40]:
            say(f"  idx{i:3d} {str(names[i]):26s} GT{sb[i]['gt']:6d}  "
                f"R {rec(sa[i]):.3f}->{rec(sb[i]):.3f} ({d:+.3f})  "
                f"WRONG {sa[i]['wrong']}->{sb[i]['wrong']} ({dw:+d})")
        if len(bad) > 40:
            say(f"  ... 还有 {len(bad)-40} 类，全量看 json")

        say()
        say("-- v16 新增能力（idx>=492，v15 结构上检不出）--")
        for i in range(nc15, len(names)):
            if sb[i]["gt"] == 0:
                say(f"  idx{i:3d} {str(names[i]):26s} val 无 GT，**这一版测不了**")
                continue
            say(f"  idx{i:3d} {str(names[i]):26s} GT{sb[i]['gt']:6d}  "
                f"R={rec(sb[i]):.3f}  WRONG={sb[i]['wrong']}  BG={sb[i]['bg']}")

        say()
        say("-- 金钱类按 region 拆（合并看会把真缺口盖住）--")
        ma = money_region_stats(items, gts, pa, conf, names)
        mb = money_region_stats(items, gts, pb, conf, names)
        for key in sorted(set(ma) | set(mb)):
            a, b = ma.get(key, {}), mb.get(key, {})
            g = b.get("gt") or a.get("gt") or 0
            if not g:
                continue
            ra = (a.get("tp", 0) / a["gt"]) if a.get("gt") else 0.0
            rb = (b.get("tp", 0) / b["gt"]) if b.get("gt") else 0.0
            say(f"  {key[0]:12s} {key[1]:6s} GT{g:6d}  "
                f"{args.atag} R={ra:.3f}  {args.btag} R={rb:.3f}")

        say()
        say("-- v16 最常见的 WRONG（框对位置认错类，flow 会点错东西）--")
        for (pc, gc), n in sorted(cb.items(), key=lambda kv: -kv[1])[:15]:
            say(f"  {n:5d} 次  预测 {str(names[pc]):24s} <- 实为 {names[gc]}")

        with open(os.path.join(args.out, "per_class.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"names": {str(k): str(v) for k, v in names.items()},
                       "conf": conf, "iou": IOU_MATCH, "frames": len(items),
                       args.atag: {str(k): v for k, v in sa.items()},
                       args.btag: {str(k): v for k, v in sb.items()}},
                      f, ensure_ascii=False, indent=1)

    with open(os.path.join(args.out, "report.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"\n报告: {os.path.join(args.out, 'report.txt')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
