"""Batch YOLO-prefill a run's frames — pick WHICH detector, and ACCUMULATE
across passes so one frame can carry boxes from several models.

Each pass runs ONE model (ui / fused_avatar / emoticon / battle), remaps its
local class ids to the shared master `_classes.txt` index BY NAME, keeps only
the boxes inside that model's authoritative master-class span, and MERGES them
into the per-image label .txt — boxes already written by other passes are
preserved (only same-class IoU>0.6 duplicates are dropped). This is exactly the
cross-teacher labeling the unified 26x model needs: the ui pass stamps UI boxes,
the avatar pass adds head boxes on the SAME frame without erasing the UI ones.

Modes:
  merge      (default) append this model's boxes, never touch other classes
  overwrite  per-model replace: drop ONLY this model's own-span boxes and write
             fresh ones; OTHER models' boxes (ui/avatar) stay. Use to re-run one
             teacher at a higher conf to fix ITS mistakes (e.g. emoticon
             false-positives outside cafe).
  skip       leave frames that already have a non-empty label untouched

Recommended flow for a fresh dataset (build the unified training set):
  ui pass → avatar pass → emoticon pass  (all merge, same frames)  → human
  精修 ONCE in the dashboard → train. Run the teacher passes BEFORE精修, not
  after (merge would re-add boxes a human deliberately deleted).

Label format: `cls cx cy w h` normalized, cls = 0-based MASTER index — the
format server/app.py:list_dataset_images parses (a 6th column is read as OBB
angle there, so we never emit one).

Usage:
  py scripts/yolo_prefill_run.py run_xxx                        # ui, merge
  py scripts/yolo_prefill_run.py run_xxx --model fused_avatar   # +head boxes
  py scripts/yolo_prefill_run.py run_xxx --model emoticon
  py scripts/yolo_prefill_run.py traj/run_xxx --conf 0.3 --mode overwrite

Reused by the dashboard endpoint /api/v1/datasets/yolo_prefill_run.
"""
from __future__ import annotations
import sys, glob, json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw_images"
TRAJ = REPO / "data" / "trajectories"
MASTER_FILE = RAW / "_classes.txt"

# ⛔人审资产保护注册表(2026-07-25 建)。data/raw_images/ 在 .gitignore:40 —— 人审
# 标注**全盘唯一副本, 零 VCS 兜底**。而预标是整池覆盖: 实测若对已人审的
# run_20260715_025638_botplay_clean 跑一次 overwrite, 559 个人审 battle 框会被
# 361 个模型框顶替。这里登记的池, prefill 默认**直接拒绝**, 除非显式
# allow_audited=True(API 端点永不传, 只有人在命令行才能给)。
AUDIT_FILE = REPO / "data" / "label_audit.json"


def audited_pools() -> dict:
    try:
        return json.loads(AUDIT_FILE.read_text(encoding="utf-8")).get("pools", {})
    except Exception:
        return {}


def _guard_audited(img_dir: Path, mode: str, allow_audited: bool) -> None:
    """fail-closed: 人审池默认不许被预标覆盖。"""
    if mode == "skip" or allow_audited:
        return
    info = audited_pools().get(img_dir.name)
    if info:
        raise SystemExit(
            f"⛔ {img_dir.name} 已登记为**人审池**({info.get('reviewed_at','?')}, "
            f"{info.get('note','')}) — 拒绝 mode={mode} 覆盖。\n"
            f"   人审标注在 .gitignore 内, 覆盖=永久丢失。确实要重标请显式传 "
            f"allow_audited=True(CLI: --allow-audited), 并先自行备份。")


def _backup_labels(img_dir: Path, stamp: str) -> int:
    """写盘前快照将被改写的 label, **保池名目录结构**(7 个 botplay 池的文件名
    全是 frame_00000.txt, 扁平备份无法区分来源 —— 磁盘上那两个 0 文件的
    *_trackprefill 备份目录就是缺这条的实证)。返回备份文件数。"""
    import shutil
    srcs = sorted(img_dir.glob("*.txt"))
    srcs = [p for p in srcs if p.name != "classes.txt"]
    if not srcs:
        return 0
    dst = RAW / "_backups" / f"{stamp}_prefill" / img_dir.name
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in srcs:
        shutil.copy2(p, dst / p.name)
        n += 1
    got = len(list(dst.glob("*.txt")))
    if got != len(srcs):     # ⛔断言非空且完整, 半备份半覆盖是最坏情况
        raise SystemExit(f"⛔ 备份不完整 {got}/{len(srcs)} → {dst}, 中止预标")
    return n


# ⭐cx 位置规则(2026-07-25 实测): 「模型判我方 且 cx>=阈值 → 改判敌方」。
# ⛔**严格池级门控, 绝不能默认开**: battle v10 训练集 我方:敌方=4.7:1 且 10 个
# 源池里 6 个敌方框=0 → 模型先验"战场小人=我方", 实测敌方→我方 22.5%/反向 0%。
# 该规则在**固定横版镜头**的活动关(botplay)上把类错 43→4; 但在**大决战**上是
# 灾难: 耶罗尼姆斯池人审 GT 里 我方 2483 框中 1689 个(68.0%)cx>=0.50、白_黑
# 27.5% —— 那边镜头跟随平移, "我方在左"根本不成立, 无门控套用会灌 1682 条毒标签。
# 阈值取 0.55 而非 0.50: 人审池上扫阈值 0.45→8误伤 / 0.50→3 / 0.55→3 / 0.60→4,
# 底部平坦, 0.55 给"推进到中线的学生"留余量(实测边界误伤集中在 0.50-0.55)。
# ⚠这条规则**永远只能活在后处理里**: 实测把整帧水平平移 ±25% 帧宽(外观一个像素
# 没动), v10 判定保持 98.4% —— 全卷积 end2end head 架构上就学不到绝对 x 坐标,
# 指望"训练进去"是没有的事。
CX_RULE_THRESHOLD = 0.55
CX_RULE_ALLOWED_POOLS = ("botplay_clean",)   # 池名子串白名单


def cx_rule_applies(pool_name: str) -> bool:
    return any(k in pool_name for k in CX_RULE_ALLOWED_POOLS)


_CLS_ALLY = _CLS_ENEMY = -1      # 按名字解析, 别硬编码 476/477(master 会增删)


def _resolve_side_ids() -> None:
    global _CLS_ALLY, _CLS_ENEMY
    idx = master_idx()
    _CLS_ALLY, _CLS_ENEMY = idx.get("我方", -1), idx.get("敌方", -1)
    if _CLS_ALLY < 0 or _CLS_ENEMY < 0:
        raise SystemExit("⛔ master 词表里找不到 我方/敌方 — cx 规则无法启用")

# Per-tag inference imgsz — mirrors brain/pipeline.py _IMGSZ_BY_TAG. Wrong imgsz
# silently yields 0 detections (ui @1920 = nothing), so pin it per model.
_IMGSZ_BY_TAG = {"ui": 960, "avatar": 960, "battle": 960, "cafe": 640}

# registry model-key -> pipeline tag (for imgsz + span lookup)
_KEY_TO_TAG = {"ui": "ui", "fused_avatar": "avatar",
               "emoticon": "cafe", "battle_heads": "battle"}

# Authoritative master-class span per detector (0-based master indices).
# master layout: [0,142]=UI-A, [143,393]=avatars(251), [394,450]=UI-B,
# 451=Emoticon_Action. A pass keeps ONLY boxes inside its span so the ui model
# can't stamp a spurious avatar class onto a cafe sprite (and vice-versa).
# 2026-07-17: ui 域含 451 Emoticon_Action — ui v6+ 已兼职摸头(live 管线
# fold-in 正是靠它, emoticon 独立模型已退役), 旧表把 451 划给 emoticon
# teacher 导致 ui 预标漏摸头框(用户抓)。⚠451 同时在 emoticon teacher 域
# (dashboard 补标仍可用): 两域重叠, overwrite 模式二者都会重写 451 框。
def _ui_span(i: int) -> bool:       return not (143 <= i <= 394) and i < 476  # UI=非头像非战斗身份段(476+), 含451摸头
def _avatar_span(i: int) -> bool:   return 143 <= i <= 394   # 含柚子战斗(394, fused 第252角色)
def _emoticon_span(i: int) -> bool: return i == 451
# battle 静态 span — 只供 UI datalist 展示/校验; **写路径一律用
# owns_for(tag, remap) 的词表动态 span**(见下)。
# ⚠2026-07-14 教训: 曾枚举 476-479, 主教480/球481/黑白482 加类后静默漏
# (用户"只标cls"输黑白被拒实锤) → 改开放规则: HUD 段 + 476 起全部身份类
# (新 boss 类永远往 master 尾部追加, 自动纳入)。
def _battle_span(i: int) -> bool:   return i in (set(range(128, 137)) | {412}) or i >= 476
_OWNS = {"ui": _ui_span, "avatar": _avatar_span,
         "cafe": _emoticon_span, "battle": _battle_span}


def owns_for(tag: str, remap: dict):
    """写路径(prefill/suggest)用的 owns 判定。battle 域 = 该权重词表∩master
    (remap.values(), v5 加新类自动跟随, 无枚举漏类)。
    ⚠2026-07-11 审计教训: 曾把 _OWNS['battle'] 设 lambda True — merge 模式
    没事(检出本来只有战斗类), 但 **overwrite 模式 kept=[e if not owns] 起始
    变空 = 整池 UI/头像/emoticon 手标全被清掉**, owns 在 overwrite 里是
    "保留其他模型框"的过滤器, 双重用途缺一不可。"""
    if tag == "battle":
        owned = set(remap.values())
        return owned.__contains__
    return _OWNS.get(tag, lambda i: True)


def resolve_img_dir(dataset: str) -> Path:
    d = (TRAJ / dataset[5:]) if dataset.startswith("traj/") else (RAW / dataset)
    return d / "frames" if (d / "frames").is_dir() else d


_MASTER_IDX = None
def master_idx() -> dict:
    global _MASTER_IDX
    if _MASTER_IDX is None:
        names = [c.strip() for c in MASTER_FILE.read_text(encoding="utf-8").splitlines() if c.strip()]
        _MASTER_IDX = {n: i for i, n in enumerate(names)}
    return _MASTER_IDX


_MODELS: dict = {}
def get_model(model_key: str = "ui", version: "str | None" = None):
    """Return (YOLO, remap{local->master}, tag). remap goes BY NAME so a model
    trained with its own local ordering (fused_avatar, emoticon) lands on the
    correct master index; names absent from master are simply dropped.

    version=None → registry active 版本。指定 version → 用该版本权重, 但 span/tag
    仍由 model_key 决定 (只换权重、不换域)。关键用法: model_key='ui' + version='v6c'
    → 借 unified v6c(nc455) 当 teacher, 经 _ui_span 过滤只吐 UI 域框 (头像/emoticon
    丢弃); model_key='ui' + version='v5' → 旧特化补 cafe 弱类(咖啡厅收益/邀请卷)。"""
    cache_key = (model_key, version)
    if cache_key not in _MODELS:
        from ultralytics import YOLO
        reg = json.loads((REPO / "data" / "model_registry.json").read_text(encoding="utf-8"))
        if model_key not in reg:
            raise SystemExit(f"unknown model {model_key!r}; choices: {list(reg)}")
        node = reg[model_key]
        ver = version or node["active"]
        versions = node.get("versions", {})
        if ver not in versions:
            raise SystemExit(f"unknown version {ver!r} for {model_key!r}; "
                             f"choices: {list(versions)}")
        m = YOLO(versions[ver]["path"])
        midx = master_idx()
        remap = {}
        for li, nm in m.names.items():
            mi = midx.get(nm)
            if mi is not None:
                remap[int(li)] = mi
        if not remap:
            # 词表与 master 零命中 = 权重错配(如 battle legacy 的 c0/c1/c2/c3
            # 占位名) — 静默跑会把未标帧全写成空 label 还报成功, 必须炸。
            raise ValueError(
                f"{model_key}/{ver} 权重词表与 master 零命中"
                f"(names={list(m.names.values())[:6]}...) — 选错版本? "
                f"battle 预标请用 v4。")
        tag = _KEY_TO_TAG.get(model_key, "ui")
        _MODELS[cache_key] = (m, remap, tag)
    return _MODELS[cache_key]


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    aa = (ax2 - ax1) * (ay2 - ay1); bb = (bx2 - bx1) * (by2 - by1)
    return inter / (aa + bb - inter + 1e-9)


def _containment(a, b) -> float:
    """交集 / 较小框面积 — 包含关系强度(大小框对的 IoU 天然低, 用这个补判)。"""
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    amin = min((ax2 - ax1) * (ay2 - ay1), (bx2 - bx1) * (by2 - by1))
    return inter / (amin + 1e-9)


def is_dup_box(box, mi, kept) -> bool:
    """预标去重: 任意类 IoU>0.6(同位歧义双标: 红点/黄点、确认/灰确认), 或
    **同类**包含型(交集/小框>0.75) — 2026-07-12 凹轴池实锤14对: 训练口径不
    统一让模型对同一学生出「含血条大框」+「本体小框」, IoU 仅~0.5 溜过 0.6
    线。跨类包含不删(Boss 大框套我方小框/角色框套 UI 按钮 = 合法结构)。
    kept: iterable of (master_id, [x1,y1,x2,y2])。"""
    for kmid, kbox in kept:
        if _iou(box, kbox) > 0.6:
            return True
        if kmid == mi and _containment(box, kbox) > 0.75:
            return True
    return False


def _read_existing(lp: Path, w: float, h: float):
    """Existing label rows -> [(master_id, [x1,y1,x2,y2] px)]. 5-col master
    format; a stray 6th col (OBB angle) is ignored."""
    out = []
    if not lp.exists():
        return out
    for ln in lp.read_text(encoding="utf-8").splitlines():
        p = ln.split()
        if len(p) < 5:
            continue
        try:
            cid = int(float(p[0])); cx, cy, bw, bh = (float(x) for x in p[1:5])
        except ValueError:
            continue
        out.append((cid, [(cx - bw / 2) * w, (cy - bh / 2) * h,
                          (cx + bw / 2) * w, (cy + bh / 2) * h]))
    return out


def prefill_run(img_dir, *, model_key: str = "ui", version: "str | None" = None,
                conf: float = 0.25, imgsz: "int | None" = None, mode: str = "merge",
                overwrite: bool = False, target_classes=None, progress=None,
                allow_audited: bool = False, cx_rule: bool = False) -> dict:
    """Prefill every *.jpg under img_dir with ONE model. See module docstring
    for modes. `overwrite=True` is a back-compat alias for mode='overwrite'.
    version=None → registry active; 指定版本可选不同 teacher 权重 (见 get_model)。

    allow_audited: 突破人审池保护闸(API 端点**永不传**, 只给命令行的人)。
    cx_rule: 开启 cx 位置规则改判(见 CX_RULE_* 注释), 仅白名单池生效。
    Returns {written, skipped, total, model, version, mode, backed_up, cx_flipped}."""
    if overwrite:
        mode = "overwrite"
    img_dir = Path(img_dir)
    _guard_audited(img_dir, mode, allow_audited)
    n_bak = 0
    if mode != "skip":
        import time as _t
        n_bak = _backup_labels(img_dir, _t.strftime("%Y%m%d_%H%M%S"))
    _cx_on = cx_rule and cx_rule_applies(img_dir.name)
    if _cx_on:
        _resolve_side_ids()
    if cx_rule and not _cx_on:
        print(f"⚠cx 规则已请求但 {img_dir.name} 不在白名单 "
              f"{CX_RULE_ALLOWED_POOLS} — **不启用**(大决战等跟随镜头域套用会灌毒标签)")
    n_flip = 0
    m, remap, tag = get_model(model_key, version)
    owns = owns_for(tag, remap)
    # 只标目标 cls (飞轮补单个/几个弱类时, 不用标全部, 大幅提效)。空=标该模型全 span。
    tgt = set(int(t) for t in target_classes) if target_classes else None
    if imgsz is None:
        imgsz = _IMGSZ_BY_TAG.get(tag, 960)
    imgs = sorted(glob.glob(str(img_dir / "*.jpg")))
    written = skipped = 0
    CHUNK = 64
    for ci in range(0, len(imgs), CHUNK):
        chunk = imgs[ci:ci + CHUNK]
        todo = []
        for p in chunk:
            lp = Path(p).with_suffix(".txt")
            if mode == "skip" and lp.exists():
                try:
                    if lp.read_text(encoding="utf-8").strip():
                        skipped += 1
                        continue
                except Exception:
                    pass
            todo.append(p)
        if not todo:
            continue
        for p, r in zip(todo, m.predict(todo, stream=True, imgsz=imgsz,
                                        conf=conf, device=0, verbose=False)):
            h, w = r.orig_shape  # (height, width)
            # this model's detections -> master id, span-filtered, conf-sorted
            new = []
            for b in r.boxes:
                mi = remap.get(int(b.cls[0]))
                if mi is None or not owns(mi):
                    continue
                if tgt is not None and mi not in tgt:
                    continue
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                if x2 <= x1 or y2 <= y1:
                    continue
                if _cx_on and mi == _CLS_ALLY:
                    if ((x1 + x2) / 2) / w >= CX_RULE_THRESHOLD:
                        mi = _CLS_ENEMY      # 见 CX_RULE_* 注释
                        n_flip += 1
                new.append((float(b.conf[0]), mi, [x1, y1, x2, y2]))
            new.sort(key=lambda t: -t[0])
            lp = Path(p).with_suffix(".txt")
            existing = _read_existing(lp, w, h)
            if mode == "overwrite":
                # per-model replace: drop ONLY this model's own-span boxes (we're
                # re-running it to FIX them, e.g. raise emoticon conf); every
                # other model's boxes (ui / avatar) are kept untouched.
                kept = [(mid, box) for mid, box in existing if not owns(mid)]
            else:
                kept = existing  # merge: keep all, append new (same-cls dedup)
            for _sc, mi, box in new:
                if is_dup_box(box, mi, kept):
                    continue  # IoU>0.6 任意类 + 同类包含型 — 见 is_dup_box 注释
                kept.append((mi, box))
            lines = []
            for mid, (x1, y1, x2, y2) in kept:
                cx = ((x1 + x2) / 2) / w; cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w; bh = (y2 - y1) / h
                lines.append(f"{mid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            lp.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            written += 1
            if progress and written % 50 == 0:
                progress(written, len(imgs))
    if _cx_on:
        print(f"⭐cx 规则(>={CX_RULE_THRESHOLD}) 把 {n_flip} 个「我方」改判敌方 "
              f"— 人审时优先复查这些框(阈值边界 0.50-0.55 有真学生误伤)")
    return {"written": written, "skipped": skipped, "total": len(imgs),
            "model": model_key, "version": version, "mode": mode,
            "backed_up": n_bak, "cx_flipped": n_flip}


def main():
    argv = sys.argv[1:]
    if not argv:
        print("usage: yolo_prefill_run.py <dataset> "
              "[--model ui|fused_avatar|emoticon|battle_heads] "
              "[--conf 0.25] [--mode merge|overwrite|skip] [--imgsz 960]\n"
              "  [--cx-rule]        battle 敌我位置规则(仅 botplay 白名单池生效)\n"
              "  [--allow-audited]  突破人审池保护闸(会覆盖人工标注, 想清楚再用)")
        return
    dataset = argv[0]
    model_key = "ui"; conf = 0.25; mode = "merge"; version = None
    imgsz = None; cx_rule = False; allow_audited = False
    i = 1
    while i < len(argv):
        a = argv[i]
        if a == "--model":
            model_key = argv[i + 1]; i += 2; continue
        if a == "--version":
            version = argv[i + 1]; i += 2; continue
        if a == "--conf":
            conf = float(argv[i + 1]); i += 2; continue
        if a == "--imgsz":
            imgsz = int(argv[i + 1]); i += 2; continue
        if a == "--mode":
            mode = argv[i + 1]; i += 2; continue
        if a == "--overwrite":
            mode = "overwrite"; i += 1; continue
        if a == "--cx-rule":
            cx_rule = True; i += 1; continue
        if a == "--allow-audited":
            allow_audited = True; i += 1; continue
        i += 1
    img_dir = resolve_img_dir(dataset)
    if not img_dir.is_dir():
        print(f"dataset dir not found: {img_dir}")
        return
    print(f"prefill {img_dir}  model={model_key}  version={version or 'active'}  "
          f"conf={conf}  mode={mode}  imgsz={imgsz or 'by-tag'}  "
          f"cx_rule={cx_rule}")
    res = prefill_run(img_dir, model_key=model_key, version=version, conf=conf,
                      imgsz=imgsz, mode=mode, cx_rule=cx_rule,
                      allow_audited=allow_audited,
                      progress=lambda w, t: print(f"  ...{w}/{t}"))
    print(res)


if __name__ == "__main__":
    main()
