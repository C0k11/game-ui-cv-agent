"""Build the ui_v2 dataset for ui_yolo26m_v5 (warm-start fine-tune).

v5 plan (vs ui_v1 which v4 trained on):
  • Schema = MASTER _classes.txt (451 cls, incl 450 选择购买). ui_v1 was 450.
  • REAL sources are DEDUPED by content md5 — run_20260521_103956_distinct was
    628 unique frames blown to 12003 via 200x oversample copies (95% dup). We
    keep the 628 uniques. Heavy dup was the v1/v2/v3 overfit driver; v4 only
    survived it via aug+synth. Cut it.
  • NEW real data merged: run_20260531_110516 (1052) + run_20260531_143201 (37
    shop frames — fills 选择购买/已选中/绿勾).
  • cls92 战术大赛对战选择区域 KEPT (no DROP_CLASSES).
  • MODERATE re-oversample: classes with 8..TARGET unique frames → duplicate to
    TARGET (~30) so dedup doesn't starve the old-only event/battle classes that
    aren't in the new captures. Classes with <8 unique are NOT duped (that's the
    overfit trap — they rely on synth / future real captures instead).
  • 450 选择购买: explicit higher target (50 ≈ ×3 of its 17 real frames) per user.
  • SYNTH kept (441/442/449 bond): _synth_bond{,_goto,_enter} — proven in v4.
  • Val = _ui_val_pool (blind to weak cls; only for early-stop mechanics — real
    judgement is dashboard visual on the shipped model).

Output: D:/Project/ml_cache/models/yolo/dataset/ui_v2/
Usage: py scripts/build_ui_v2.py [--clean]
"""
from __future__ import annotations
import argparse
import hashlib
import shutil
import sys
from collections import defaultdict
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw_images"
MASTER = RAW / "_classes.txt"
OUT_ROOT = Path("D:/Project/ml_cache/models/yolo/dataset/ui_v2")

sys.path.insert(0, str(REPO / "scripts"))
from build_ui_dataset import sanitize_label_text  # noqa: E402

REAL_SOURCES = [
    "run_20260521_103956_distinct",  # 628 unique (dedup from 12003)
    "run_20260527_094158",
    "run_20260527_101545",
    "run_20260518_002646",
    "run_20260529_000756",
    "run_20260529_123209",
    "run_20260531_110516",           # NEW 1052
    "run_20260531_143201",           # NEW 37 shop
    "run_20260518_163513",           # +335 daily-skill UI (06-03 清理后加白名单)
    "run_20260531_173326",           # +706
    "run_20260531_174456",           # +285
    "run_20260531_175038",           # +36
    "run_20260603_134626",           # +27 (06-03 补录批, 进 train)
    "run_20260603_134649",           # +31
    "run_20260603_140153",           # +31
    "run_20260603_170116",           # +31
    "run_20260603_175151",           # +30
    "run_20260603_175217",           # +30
    "run_20260603_175238",           # +30
    "run_20260603_175257",           # +24
    "run_v6weak_20260603",           # +1130 飞轮弱cls (⚠ 开训前补头像/摸头 + 确认烧录帧已清)
    # emoticon merge (v6): 200 cafe frames, 592 Emoticon_Action(451) bubbles +
    # teacher-relabeled cafe UI (咖啡厅收益/邀请卷 etc) via build_emoticon_ui_source.py.
    # Folds the standalone emoticon_yolo26n into the ui model — pipeline then runs
    # one fewer YOLO per cafe tick. 2026-03 captures, md5-disjoint from above.
    "_emoticon_v2",
    "run_20260607_193003",           # v7 飞轮: 女仆背景 lobby/cafe 弱类(制造入口等) 783标
    "run_20260607_140123",           # v7 飞轮: 银发背景 lobby/cafe 弱类 1409标
    # v8 飞轮 (2026-06-09/10 全技能 live 素材, curate_flywheel 去重 + 多 teacher
    # add-only 补标 + 红黄点 HSV 仲裁清洗 + hub badge 模板锚定补 452):
    # 制造入口544 / 任务大厅入口542 / 双倍三倍617+76hub / 短篇网格 / 剧情战斗结算。
    # 两个 06-10 val run 已抽离进 _val_v8flywheel(整 run 抽, 防同 session 泄漏)。
    "run_20260610_v8queue",
    "run_20260610_024533",           # 用户手标批量扫荡 dialog 全套(127帧, 新类 455-468 主源)
    # v8b: 旧款箭头增压 — v8 训到 ep20 左右切换在旧风格(lobby/任务屏白chevron)上
    # 灾难遗忘(momo新款47/47 vs 旧款15/238), 旧款帧×重编码副本拉回锚点。
    "_arrow_boost",
    # v9 飞轮 (2026-06-11 全天 live 干净帧, 用户全手标/人审, review_v9_pools 复查
    # 标签零问题): 完整日常编排全技能素材 — 455/456 第二session / 450×110 /
    # 452 hub ribbon×576 (v8 val 盲区解药) / 格黑娜vs阿拜多斯 87帧 (v8 混淆解药) /
    # dialog 调暗大厅 / schedule Location Select 滚动多态。
    "run_20260611_044844_clean",
    "run_20260611_050507_clean",
    "run_20260611_051200_clean",
    "run_20260611_052955_clean",
    "run_20260611_053359_clean",
    "run_20260611_053709_clean",
    "run_20260611_055934_clean",
    "run_20260611_061637_clean",
    "run_20260611_061938_clean",
    "run_20260611_064804_clean",
    "run_20260611_071526_clean",
    "run_20260611_072242_clean",
    "run_20260611_073139_clean",
    "run_20260611_073341_clean",
    "run_20260611_074607_clean",
    # v9 晚间专录: 战术大赛商店(新类469-473 主源, 含472/473能量饮料+471货币) +
    # cafe emoticon 高帧×2 (451×747 — 摸头折叠进 ui 的底气)。
    "run_20260611_205439",
    "run_20260611_205540",
    "run_20260611_212919",
    # v9 防遗忘考古回收 (2026-06-11 全历史盘点, 用户"都有用的, 免得遗忘很多cls"):
    # 171121 = 06-03 被弃 ex-val(513帧6,809框 UI 大池, 弃因=与v6c train同session泄漏
    # → 进 train 反而合法; 入库前 HSV 修复 29 处黄点→红点 — 它没吃过 v8 那轮清洗)。
    # _ui_val_pool = 05-27 老风格旧 val(51帧1,010框 红黄点/货币/邀请键 rehearsal)。
    # 考古明确排除: 021030(与v8queue同内容已重编码, 双标签互污) / 173604+183022(纯
    # 头像归fused域) / _synth_*(用户06-06砍的毒) / expanded·full·static(古schema不兼容)。
    "run_20260603_171121",
    "_ui_val_pool",
    # ── v10 待加 (2026-06-12, ⚠ DASHBOARD 人审后才解注释训练) ──────────────
    # 两批今日素材, v9 预标 + 星标/sweep 清洗, 但 v9 预标=teacher, 必须人审过
    # 才进 train(否则重演 关卡得星 teacher 污染). 审完去掉下面两行的注释再 build:
    #   run_20260612_191319   = 313帧 (星修110 + 455补128, commit fd40b59)
    #   run_20260612_chainlive = 452帧 (今日整链+商店, 6059框 v9预标)
    # "run_20260612_191319",
    # "run_20260612_chainlive",
]
SYNTH_SOURCES = []   # v7: 砍头像 synth(头像归 fused v6 专精; rehearsal 仅 unified 才需要) → ui v7 纯 UI+emoticon 真实帧
                 # ⚠️ v6c (2026-06-06 用户决策): 砍 _synth_ui_swap — UI 只用真实帧根治 synth 过拟合
                 #    (v6b 实锤: UI val 0.892 高 / live 崩, 咖啡厅入口 val>0.9 / live 仅 0.25)。头像 synth
                 #    影响小保留。UI 弱类(咖啡厅入口/开始制造/CAFE_EARNINGS)暂靠 skill 兜底(cafe/craft 外推),
                 #    v7 再上飞轮真实帧(run_20260606_flywheel 519帧, 标注后)补。
                 # 删: _synth_bond/goto/enter — 假阴性毒 (2026-06-04)
VAL_SOURCES = [
    "run_20260606_flywheel",  # v7 主 val: 06-06 飞轮 477帧(独立 session 防泄漏, 含 UI 弱类靶子). 旧 06-03 val 弃用(171121 与 v6c train 同 session 泄漏 / 183022 头像 / _ui_val_pool 旧盲)
    # v8 增补 val: 从 v8queue 按整 run 抽的 38 帧(06-09/10 新界面覆盖)。稀有类保护:
    # 任何类被抽走 >40% 或全局剩 <20 实例的 run 不许抽(momo/剧情场景类各 session
    # 垄断, 实测仅 2 run 可安全抽出 — 别强抽, v9 等场景跨天重复后再扩)。
    "_val_v8flywheel",
]

TARGET = 30            # moderate oversample floor (was 200 — the overfit driver)
TARGET_OVERRIDE = {450: 50}   # 选择购买: ~x3 its 17 real frames
MIN_UNIQUE = 8         # don't oversample classes thinner than this (overfit trap)


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


# v7: ui = 纯 UI+emoticon — drop 头像段(143-394, 归 fused v6 专精)。flywheel / cafe / momo
# 真实帧由 v6c(nc455)预填含头像框, 对 ui v7 多余(否则 val 被头像 GT 干扰 + train 学多余头像)。
# ⚠️ 原始 raw_images 标注不动(保留头像给未来 unified), 仅 build 输出 ui_v2 时过滤。
HEAD_LO, HEAD_HI = 143, 394
def _keep_ui_lines(cleaned: str, nc: int):
    out = []
    for ln in cleaned.splitlines():
        if not ln.strip():
            continue
        c = int(ln.split()[0])
        if c >= nc or HEAD_LO <= c <= HEAD_HI:   # 越界 或 头像段 → drop
            continue
        out.append(ln)
    return out


def label_classes(cleaned: str):
    cs = set()
    for ln in cleaned.splitlines():
        ln = ln.strip()
        if ln:
            cs.add(int(ln.split()[0]))
    return cs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()

    master = MASTER.read_text(encoding="utf-8").splitlines()
    nc = len(master)
    print(f"[schema] master = {nc} classes (last: {master[-1]!r})")

    # Verify each source's classes.txt is a PREFIX of master (no reorder drift).
    for s in REAL_SOURCES + SYNTH_SOURCES + VAL_SOURCES:
        cf = RAW / s / "classes.txt"
        if not cf.exists():
            continue
        sc = cf.read_text(encoding="utf-8").splitlines()
        if sc != master[:len(sc)]:
            for i in range(min(len(sc), nc)):
                if sc[i] != master[i]:
                    print(f"[!] SCHEMA DRIFT in {s}: idx {i} = {sc[i]!r} vs master {master[i]!r}")
                    return 1
    print("[schema] all sources prefix-match master ✓")

    if args.clean and OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    img_tr = OUT_ROOT / "images" / "train"
    lbl_tr = OUT_ROOT / "labels" / "train"
    img_va = OUT_ROOT / "images" / "val"
    lbl_va = OUT_ROOT / "labels" / "val"
    for d in (img_tr, lbl_tr, img_va, lbl_va):
        d.mkdir(parents=True, exist_ok=True)

    # ── gather REAL frames, dedup by content md5 ──
    seen_md5 = {}
    uniques = []   # (src, stem, jpg, cleaned_txt, classes)
    dropped_overflow = 0
    n_raw = 0
    for s in REAL_SOURCES:
        sd = RAW / s
        if not sd.is_dir():
            print(f"[!] missing real source {sd}")
            continue
        for jpg in sorted(sd.glob("*.jpg")):
            txt = sd / (jpg.stem + ".txt")
            if not txt.exists():
                continue
            n_raw += 1
            h = md5(jpg)
            if h in seen_md5:
                continue
            seen_md5[h] = True
            cleaned = sanitize_label_text(txt.read_text(encoding="utf-8"))
            # drop any out-of-range cls (>= nc)
            keep = _keep_ui_lines(cleaned, nc)
            dropped_overflow += len(cleaned.splitlines()) - len(keep)
            cleaned = ("\n".join(keep) + "\n") if keep else ""
            uniques.append((s, jpg.stem, jpg, cleaned, label_classes(cleaned)))
    print(f"[real] {n_raw} raw frames → {len(uniques)} unique (dedup removed "
          f"{n_raw - len(uniques)} dup copies); dropped {dropped_overflow} out-of-range boxes")

    # ── synth frames (no dedup) ──
    synth = []
    for s in SYNTH_SOURCES:
        sd = RAW / s
        if not sd.is_dir():
            continue
        for jpg in sorted(sd.glob("*.jpg")):
            txt = sd / (jpg.stem + ".txt")
            if not txt.exists():
                continue
            cleaned = sanitize_label_text(txt.read_text(encoding="utf-8"))
            keep = _keep_ui_lines(cleaned, nc)
            cleaned = ("\n".join(keep) + "\n") if keep else ""
            synth.append((s, jpg.stem, jpg, cleaned, label_classes(cleaned)))
    print(f"[synth] {len(synth)} frames")

    base = uniques + synth   # one entry each (the de-duped real + synth)

    # ── moderate oversample: bring 8..TARGET classes up to TARGET ──
    cls_to_frames = defaultdict(list)
    for i, (_, _, _, _, cs) in enumerate(base):
        for c in cs:
            cls_to_frames[c].append(i)
    dup_entries = []   # indices into base, duplicated
    for c, idxs in cls_to_frames.items():
        n = len(idxs)
        tgt = TARGET_OVERRIDE.get(c, TARGET)
        if n < MIN_UNIQUE or n >= tgt:
            continue
        need = tgt - n
        for k in range(need):
            dup_entries.append(idxs[k % n])
    print(f"[oversample] +{len(dup_entries)} dup entries (target {TARGET}, "
          f"450→{TARGET_OVERRIDE[450]}, min_unique {MIN_UNIQUE})")

    # ── write train (uniques + synth once, then dups with __dN suffix) ──
    written_tr = set()   # dst stems this build produced — stray purge below
    def write_entry(entry, dst_stem):
        s, stem, jpg, cleaned, _ = entry
        # Source may vanish between scan and write (live dashboard labeling
        # session deletes frames while a build runs — 2026-06-10). Skip, don't die.
        if not jpg.exists():
            print(f"[skip] source vanished mid-build: {jpg.name}")
            return
        dj = img_tr / f"{dst_stem}.jpg"
        if dj.exists():
            dj.unlink()
        try:
            dj.symlink_to(jpg.resolve())
        except OSError:
            shutil.copy2(jpg, dj)
        (lbl_tr / f"{dst_stem}.txt").write_text(cleaned, encoding="utf-8")
        written_tr.add(dst_stem)

    neg = 0
    for entry in base:
        s, stem = entry[0], entry[1]
        write_entry(entry, f"{s}__{stem}")
        if not entry[3].strip():
            neg += 1
    dup_count = defaultdict(int)
    for bi in dup_entries:
        entry = base[bi]
        s, stem = entry[0], entry[1]
        dup_count[bi] += 1
        write_entry(entry, f"{s}__{stem}__d{dup_count[bi]}")
    n_train = len(base) + len(dup_entries)
    print(f"[train] {n_train} frames ({len(base)} unique+synth, {len(dup_entries)} dups, "
          f"{neg} negatives)")

    # ── val from VAL_SOURCES (held-out 多域: ui+头像+摸头, 跨多个 run) ──
    written_va = set()
    n_val = 0
    for vsrc in VAL_SOURCES:
        vd = RAW / vsrc
        for jpg in sorted(vd.glob("*.jpg")):
            txt = vd / (jpg.stem + ".txt")
            if not txt.exists():
                continue
            cleaned = sanitize_label_text(txt.read_text(encoding="utf-8"))
            keep = _keep_ui_lines(cleaned, nc)
            cleaned = ("\n".join(keep) + "\n") if keep else ""
            stem = f"{vsrc}__{jpg.stem}"   # source-prefix → 防跨 run 同名(frame_000000)互相覆盖
            dj = img_va / f"{stem}.jpg"
            if dj.exists():
                dj.unlink()
            try:
                dj.symlink_to(jpg.resolve())
            except OSError:
                shutil.copy2(jpg, dj)
            (lbl_va / f"{stem}.txt").write_text(cleaned, encoding="utf-8")
            written_va.add(stem)
            n_val += 1
    print(f"[val] {n_val} frames from {len(VAL_SOURCES)} sources: {VAL_SOURCES}")

    # ── hygiene: purge strays + caches (2026-06-11 实锤双病根) ──────────────
    # Build is incremental (no rmtree without --clean), so entries dropped from
    # sources (deleted frames / removed pools / renamed stems) linger as orphan
    # jpg+txt — ultralytics globs the dir, so STRAYS GET TRAINED with stale
    # labels (131 found tonight). And cache='disk' .npy never get reclaimed
    # (190.9GB of v8-era cache found). Purge anything this build didn't write.
    n_stray = n_npy = 0
    for d, keep in ((img_tr, written_tr), (lbl_tr, written_tr),
                    (img_va, written_va), (lbl_va, written_va)):
        for f in d.iterdir():
            if f.suffix == ".npy":
                f.unlink(); n_npy += 1
            elif f.suffix in (".jpg", ".txt") and f.stem not in keep:
                f.unlink(); n_stray += 1
    for c in OUT_ROOT.rglob("*.cache"):   # ultralytics scan caches — stale lists
        c.unlink()
    print(f"[hygiene] purged {n_stray} stray files + {n_npy} npy caches")

    # ── data.yaml ──
    yaml = [f"path: {OUT_ROOT.as_posix()}", "train: images/train", "val: images/val",
            f"nc: {nc}", "names:"]
    for i, n in enumerate(master):
        yaml.append(f"  {i}: '{n.replace(chr(39), chr(92) + chr(39))}'")
    (OUT_ROOT / "data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")
    print(f"[done] {OUT_ROOT}  (nc={nc}, train={n_train}, val={n_val})")

    # ── report key class counts (instances in train) ──
    inst = defaultdict(int)
    for entry in base:
        for ln in entry[3].splitlines():
            if ln.strip():
                inst[int(ln.split()[0])] += 1
    for bi in dup_entries:
        for ln in base[bi][3].splitlines():
            if ln.strip():
                inst[int(ln.split()[0])] += 1
    print("[key cls instances in train]")
    for c in (92, 450, 54, 403, 55, 441, 442, 449):
        print(f"  {c} {master[c][:22]:<22} {inst.get(c,0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
