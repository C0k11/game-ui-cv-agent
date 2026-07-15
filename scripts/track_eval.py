# -*- coding: utf-8 -*-
"""tracker 离线评测: 检测缓存一次 → ByteTrack/BoT-SORT 参数网格 → GT 链指标.

combat AI 2.0 tracker Phase 1 的地基: 拿人审过的凹轴池当 GT, 量化
"锁定稳定性"(id 不换人)——这是按轴放技能的前提, 检测 AP 高≠锁得住。

流程:
  1. --cache   v8 对池逐帧检测(conf0.05, 身份类), 存 _detcache/*.npz — GPU 只跑这一次
  2. (默认)    用缓存喂手动 BYTETracker/BOTSORT 实例扫参数网格 — 纯 CPU, 秒级/配置
  3. GT 链     人审 label 的身份类框按 IoU 匈牙利串成链(gap≤2 保活) = "真实的人"
  4. 指标      每帧 GT↔track 框匈牙利(IoU≥0.45):
               purity   链上主 tid 占比(=锁定不换人的程度, 关键指标)
               idsw     链上相邻匹配帧 tid 变化次数
               cover    GT 框被 track 框覆盖比例(检测+轨迹召回)
               cls_acc  匹配框类别正确率(投票前的原始输出)
  5. --slots   站位绑定 demo: 开场我方链按 cx 排序=编成序, 报告各 slot 全程锁定

用法:
  py scripts/track_eval.py 白_黑 --cache
  py scripts/track_eval.py 白_黑
  py scripts/track_eval.py 白_黑 --slots
只喂身份类(实战设计: HUD 静态类走单帧读, 身份类走 tracker)。
"""
import glob
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"D:\Project\ai game secretary")
from vision.io_utils import imread_any  # noqa: E402

RAW = Path(r"D:\Project\ai game secretary\data\raw_images")
CACHE_DIR = RAW / "_detcache"
IDENTITY_MASTER = {476, 477, 478, 479, 480, 481, 482}  # 我方/敌方/塞特/Boss/主教/球/黑白
GT_MIN_CHAIN = 5          # 短于此的 GT 链不评(开场淡入/结算残帧)
GT_LINK_IOU = 0.30        # GT 帧间串链阈值(3fps 位移大, 放宽)
GT_LINK_GAP = 2           # GT 链断档保活帧数(遮挡/漏标容忍)
MATCH_IOU = 0.45          # GT↔track 评测匹配阈值


def find_pool(pat: str) -> Path:
    return Path([p for p in glob.glob(str(RAW / "axis_*")) if pat in p][0])


def frames_of(pool: Path):
    return sorted(pool.glob("*.jpg"),
                  key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)))


# ────────────────────────── 检测缓存 ──────────────────────────

def build_cache(pool: Path) -> Path:
    from track_prefill import _latest_battle_weights  # 权重同源 registry 最新 vN
    from ultralytics import YOLO
    master = [l.strip() for l in
              open(RAW / "_classes.txt", encoding="utf-8") if l.strip()]
    n2i = {n: i for i, n in enumerate(master)}
    model = YOLO(_latest_battle_weights())
    m2master = {i: n2i[n] for i, n in model.names.items() if n in n2i}

    jpgs = frames_of(pool)
    rows = []                     # fi, master_cls, conf, cx,cy,w,h (像素)
    W = H = 0
    for fi, p in enumerate(jpgs):
        img = imread_any(str(p))
        H, W = img.shape[:2]
        r = model.predict(img, conf=0.05, iou=0.5, imgsz=960,
                          verbose=False)[0]
        if r.boxes is None:
            continue
        for b in r.boxes:
            mi = m2master.get(int(b.cls[0]))
            if mi not in IDENTITY_MASTER:
                continue
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            rows.append((fi, mi, float(b.conf[0]),
                         (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
        if fi % 100 == 0:
            print(f"  detect {fi}/{len(jpgs)}", flush=True)
    CACHE_DIR.mkdir(exist_ok=True)
    out = CACHE_DIR / f"{pool.name[:60]}.npz"
    np.savez_compressed(out, rows=np.array(rows, dtype=np.float32),
                        n_frames=len(jpgs), w=W, h=H)
    print(f"[cache] {len(rows)} identity dets / {len(jpgs)} frames → {out.name}")
    return out


def load_cache(pool: Path):
    f = CACHE_DIR / f"{pool.name[:60]}.npz"
    if not f.exists():
        sys.exit(f"缓存不存在, 先跑 --cache: {f}")
    z = np.load(f)
    per_frame = defaultdict(list)
    for row in z["rows"]:
        per_frame[int(row[0])].append(row[1:])
    return per_frame, int(z["n_frames"]), float(z["w"]), float(z["h"])


# ────────────────────────── GT 链 ──────────────────────────

def iou_mat(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a:(N,4) b:(M,4) xyxy → (N,M) IoU。"""
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)


def hungarian(cost: np.ndarray, thresh: float):
    """代价=−IoU 的匈牙利, 只保留 IoU≥thresh 的对。返回 [(i,j)]。"""
    from scipy.optimize import linear_sum_assignment
    if cost.size == 0:
        return []
    ri, ci = linear_sum_assignment(-cost)
    return [(i, j) for i, j in zip(ri, ci) if cost[i, j] >= thresh]


def build_gt_chains(pool: Path, n_frames: int, W: float, H: float):
    """人审 label → 身份类 GT 链。返回 chains: [ {fi: (cls, xyxy)} ]。"""
    jpgs = frames_of(pool)
    gt_frames = []                          # fi -> [(cls, xyxy)]
    for p in jpgs:
        lbl, boxes = p.with_suffix(".txt"), []
        if lbl.exists():
            for l in lbl.read_text(encoding="utf-8").splitlines():
                q = l.split()
                if len(q) >= 5 and int(q[0]) in IDENTITY_MASTER:
                    c, xc, yc, w, h = int(q[0]), *map(float, q[1:5])
                    boxes.append((c, np.array([
                        (xc - w / 2) * W, (yc - h / 2) * H,
                        (xc + w / 2) * W, (yc + h / 2) * H])))
        gt_frames.append(boxes)

    chains, active = [], []                 # active: (chain_dict, last_fi)
    for fi, boxes in enumerate(gt_frames):
        pool_boxes = np.array([b for _, b in boxes]) if boxes else np.zeros((0, 4))
        last = [np.array(ch[l][1]) for ch, l in active]
        pairs = hungarian(iou_mat(np.array(last) if last else np.zeros((0, 4)),
                                  pool_boxes), GT_LINK_IOU)
        used_j = set()
        for i, j in pairs:
            ch, _ = active[i]
            ch[fi] = boxes[j]
            active[i] = (ch, fi)
            used_j.add(j)
        for j, box in enumerate(boxes):
            if j not in used_j:
                active.append(({fi: box}, fi))
        still = []
        for ch, l in active:
            if fi - l > GT_LINK_GAP:
                chains.append(ch)
            else:
                still.append((ch, l))
        active = still
    chains += [ch for ch, _ in active]
    chains = [c for c in chains if len(c) >= GT_MIN_CHAIN]
    return chains


# ────────────────────────── tracker 离线跑 ──────────────────────────

class _Dets:
    """BYTETracker.update 的输入 shim(.conf/.cls/.xywh/.xyxy + mask 索引)。"""

    def __init__(self, arr: np.ndarray):
        self._a = arr                        # (N,6): cls,conf,cx,cy,w,h

    def __len__(self):
        return len(self._a)

    def __getitem__(self, m):
        return _Dets(self._a[m])

    @property
    def conf(self):
        return self._a[:, 1]

    @property
    def cls(self):
        return self._a[:, 0]

    @property
    def xywh(self):
        return self._a[:, 2:6]

    @property
    def xyxy(self):
        c = self._a[:, 2:6]
        return np.stack([c[:, 0] - c[:, 2] / 2, c[:, 1] - c[:, 3] / 2,
                         c[:, 0] + c[:, 2] / 2, c[:, 1] + c[:, 3] / 2], 1)


def run_tracker(per_frame, n_frames, cfg: dict, botsort=False, pool=None):
    from ultralytics.trackers.bot_sort import BOTSORT
    from ultralytics.trackers.byte_tracker import BYTETracker
    base = dict(
        tracker_type="botsort" if botsort else "bytetrack",
        track_high_thresh=0.25, track_low_thresh=0.10,
        new_track_thresh=0.25, track_buffer=30, match_thresh=0.8,
        fuse_score=True, gmc_method="none", proximity_thresh=0.5,
        appearance_thresh=0.8, with_reid=False, model="auto")
    base.update(cfg)
    args = SimpleNamespace(**base)
    # frame_rate=30 → buffer_size=track_buffer 原值(帧数语义, 不做二次缩放)
    tr = (BOTSORT if botsort else BYTETracker)(args, frame_rate=30)
    jpgs = frames_of(pool) if botsort and args.gmc_method != "none" else None

    out = defaultdict(list)                 # fi -> [(tid, cls, xyxy)]
    for fi in range(n_frames):
        arr = np.array(per_frame.get(fi, np.zeros((0, 6))), dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(0, 6)
        img = imread_any(str(jpgs[fi])) if jpgs is not None else None
        res = tr.update(_Dets(arr), img)
        for row in res:                     # x1,y1,x2,y2,tid,score,cls,idx
            out[fi].append((int(row[4]), int(row[6]), row[0:4].astype(float)))
    return out


# ────────────────────────── 评测 ──────────────────────────

def evaluate(track_frames, chains):
    tot_gt = tot_match = tot_cls_ok = tot_idsw = 0
    purities, ally_pur, ally_idsw = [], [], 0
    for ch in chains:
        is_ally = Counter(
            cls for cls, _ in ch.values()).most_common(1)[0][0] == 476
        tid_seq = []
        for fi, (cls, gbox) in sorted(ch.items()):
            tot_gt += 1
            cand = track_frames.get(fi, [])
            if not cand:
                continue
            m = iou_mat(gbox[None, :], np.array([b for _, _, b in cand]))[0]
            j = int(np.argmax(m))
            if m[j] >= MATCH_IOU:
                tid, tcls, _ = cand[j]
                tot_match += 1
                tot_cls_ok += int(tcls == cls)
                tid_seq.append(tid)
        if tid_seq:
            pur = Counter(tid_seq).most_common(1)[0][1] / len(tid_seq)
            sw = sum(1 for a, b in zip(tid_seq, tid_seq[1:]) if a != b)
            purities.append(pur)
            tot_idsw += sw
            if is_ally:
                ally_pur.append(pur)
                ally_idsw += sw
    return {
        "chains": len(chains),
        "cover": tot_match / max(tot_gt, 1),
        "purity": float(np.mean(purities)) if purities else 0.0,
        "idsw": tot_idsw,
        "a_pur": float(np.mean(ally_pur)) if ally_pur else 0.0,
        "a_sw": ally_idsw,
        "cls_acc": tot_cls_ok / max(tot_match, 1),
    }


# ────────────────────────── 站位绑定 demo ──────────────────────────

def slots_demo(track_frames, chains, n_frames):
    """开场站位=编成序(左→右)。GT 我方链按首现帧+cx 排序发 slot,
    报告各 slot 的锁定质量 — 这就是实战"第3个位置放EX"要依赖的映射。"""
    ally = [c for c in chains
            if Counter(cls for cls, _ in c.values()).most_common(1)[0][0] == 476]
    ally.sort(key=lambda c: min(c))
    first_fi = min(min(c) for c in ally) if ally else 0
    opening = [c for c in ally if min(c) <= first_fi + 6]   # 开场2s内进场
    opening.sort(key=lambda c: (c[min(c)][1][0] + c[min(c)][1][2]) / 2)              # 首帧 x 左→右
    print(f"\n== 站位绑定(开场帧 {first_fi}, 我方链 {len(ally)} 条, "
          f"开场进场 {len(opening)} 条) ==")
    for si, ch in enumerate(opening, 1):
        tids = []
        for fi, (cls, gbox) in sorted(ch.items()):
            cand = track_frames.get(fi, [])
            if not cand:
                continue
            m = iou_mat(gbox[None, :], np.array([b for _, _, b in cand]))[0]
            j = int(np.argmax(m))
            if m[j] >= MATCH_IOU:
                tids.append(cand[j][0])
        c = Counter(tids)
        main_tid, n_main = c.most_common(1)[0] if c else (-1, 0)
        sw = sum(1 for a, b in zip(tids, tids[1:]) if a != b)
        print(f"  slot{si}: 链{min(ch)}~{max(ch)}帧({len(ch)}) "
              f"主tid={main_tid} 纯度{n_main / max(len(tids), 1):.2f} "
              f"switch={sw} 匹配{len(tids)}/{len(ch)}")


# ────────────────────────── main ──────────────────────────

GRID = [  # (名字, botsort?, 覆写)
    ("byte默认(30fps假设)", False, {}),
    ("byte buf=3(1s)", False, {"track_buffer": 3}),
    ("byte buf=9(3s)", False, {"track_buffer": 9}),
    ("byte buf=9 match.9", False, {"track_buffer": 9, "match_thresh": 0.9}),
    ("byte buf=15 match.9", False, {"track_buffer": 15, "match_thresh": 0.9}),
    ("byte buf=9 new.4", False, {"track_buffer": 9, "new_track_thresh": 0.4}),
    ("botsort buf=9(无gmc)", True, {"track_buffer": 9}),
    ("botsort buf=9 光流gmc", True, {"track_buffer": 9,
                                      "gmc_method": "sparseOptFlow"}),
]


def main():
    pat = sys.argv[1]
    pool = find_pool(pat)
    print(f"pool = {pool.name[:70]}")
    if "--cache" in sys.argv:
        build_cache(pool)
        return
    per_frame, n_frames, W, H = load_cache(pool)
    chains = build_gt_chains(pool, n_frames, W, H)
    n_ally = sum(1 for c in chains if Counter(
        cls for cls, _ in c.values()).most_common(1)[0][0] == 476)
    print(f"GT 链: {len(chains)} 条(我方 {n_ally}), 帧 {n_frames}")

    if "--slots" in sys.argv:
        tf = run_tracker(per_frame, n_frames,
                         {"track_buffer": 9, "match_thresh": 0.9}, pool=pool)
        slots_demo(tf, chains, n_frames)
        return

    print(f"\n{'配置':<24} {'cover':>6} {'purity':>7} {'idsw':>5} "
          f"{'我方pur':>7} {'我方sw':>6} {'cls':>6}")
    for name, botsort, cfg in GRID:
        tf = run_tracker(per_frame, n_frames, cfg, botsort=botsort, pool=pool)
        m = evaluate(tf, chains)
        print(f"{name:<24} {m['cover']:>6.3f} {m['purity']:>7.3f} "
              f"{m['idsw']:>5} {m['a_pur']:>7.3f} {m['a_sw']:>6} "
              f"{m['cls_acc']:>6.3f}")


if __name__ == "__main__":
    main()
