# 正式 bbox-swap UI synth (用户思路): 真实帧的 UI box 替换成大小相近的别 cls sprite
# → 元素真实位置 + 背景自然 + 位置泛化. 替代乱飞的 _synth_overlay.
# 输出 multi-domain 帧: 头像/摸头 box 原样保留, UI box 部分被换(label 改成新 cls).
# 性能: 多线程 (cv2/numpy 释放 GIL, lib 只读共享) + numpy 向量化候选筛选, 吃满全核.
import sys, random, shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2, numpy as np

cv2.setNumThreads(1)  # 每个 cv2 调用单线程, 并行靠 ThreadPoolExecutor (避免 32*N 过度订阅)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
RAW = REPO / "data" / "raw_images"
OUT = RAW / "_synth_ui_swap"
MAX_FRAMES = 6000
WORKERS = 32

try:
    from scripts.build_ui_v2 import REAL_SOURCES, VAL_SOURCES
except Exception:
    REAL_SOURCES, VAL_SOURCES = [], []

def imread_u(p): return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
def is_ui(c): return not (143 <= c <= 394) and c != 451

# ---- 写帧线程只读共享的全局 (main 里填充) ----
FLAT_SPR = []          # [spr ndarray, ...]
CCS = SWS = SHS = ARS = None   # numpy 标量数组, 与 FLAT_SPR 同序

def load_frames():
    val = set(VAL_SOURCES)
    frames = []
    for src in REAL_SOURCES:
        if src in val:
            continue
        d = RAW / src
        if not d.is_dir():
            continue
        for txt in sorted(d.glob("*.txt")):
            if txt.name == "classes.txt":
                continue
            boxes = []
            for ln in txt.read_text(encoding="utf-8", errors="ignore").splitlines():
                p = ln.split()
                if len(p) < 5:
                    continue
                try:
                    c = int(p[0]); cx, cy, w, h = map(float, p[1:5])
                except ValueError:
                    continue
                boxes.append((c, cx, cy, w, h))
            jpg = txt.with_suffix(".jpg")
            if jpg.exists() and sum(1 for b in boxes if is_ui(b[0])) >= 3:
                frames.append((jpg, boxes))
    return frames

def crop_sprites(args):
    # 建 lib 的并行单元: 读一帧, 裁出所有 UI sprite
    jpg, boxes = args
    out = []
    img = imread_u(jpg)
    if img is None:
        return out
    H, W = img.shape[:2]
    for c, cx, cy, w, h in boxes:
        if not is_ui(c):
            continue
        x1, y1 = max(0, int((cx-w/2)*W)), max(0, int((cy-h/2)*H))
        x2, y2 = min(W, int((cx+w/2)*W)), min(H, int((cy+h/2)*H))
        spr = img[y1:y2, x1:x2]
        if spr.size and spr.shape[0] > 8 and spr.shape[1] > 8:
            out.append((c, spr.copy(), w, h))
    return out

def make_frame(args):
    # 写帧的并行单元: 读真实帧, 50% 概率把每个 UI box 换成大小相近的别 cls sprite
    idx, jpg, boxes = args
    rng = random.Random(1000 + idx)
    img = imread_u(jpg)
    if img is None:
        return None
    H, W = img.shape[:2]
    comp = img.copy()
    labels = []
    for c, cx, cy, w, h in boxes:
        if is_ui(c) and rng.random() < 0.5:
            box_ar = w / max(h, 1e-6)
            r = ARS / box_ar
            # 长宽比相近 ±33% (防方图标贴长条→拉伸) + w/h 各自相近 ±55% + 别同 cls
            mask = ((r > 0.75) & (r < 1.33)
                    & (SWS > 0.65*w) & (SWS < 1.55*w)
                    & (SHS > 0.65*h) & (SHS < 1.55*h)
                    & (CCS != c))
            idxs = np.nonzero(mask)[0]
            if len(idxs):
                k = int(idxs[rng.randrange(len(idxs))])
                spr = FLAT_SPR[k]; cc = int(CCS[k])
                x1, y1 = max(0, int((cx-w/2)*W)), max(0, int((cy-h/2)*H))
                x2, y2 = min(W, int((cx+w/2)*W)), min(H, int((cy+h/2)*H))
                bw, bh = x2-x1, y2-y1
                if bw > 4 and bh > 4:
                    comp[y1:y2, x1:x2] = cv2.resize(spr, (bw, bh), interpolation=cv2.INTER_AREA)
                    labels.append(f"{cc} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                    continue
        labels.append(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")  # 原样保留(头像/摸头/未换UI)
    stem = f"swap_{idx:06d}"
    cv2.imwrite(str(OUT/f"{stem}.jpg"), comp, [cv2.IMWRITE_JPEG_QUALITY, 88])
    (OUT/f"{stem}.txt").write_text("\n".join(labels)+"\n", encoding="utf-8")
    return idx

def main():
    global FLAT_SPR, CCS, SWS, SHS, ARS
    frames = load_frames()
    print(f"real frames with >=3 UI boxes: {len(frames)}", flush=True)

    # 建 lib: 并行 imread + crop
    lib_flat = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for res in ex.map(crop_sprites, frames):
            lib_flat.extend(res)
    ncls = len(set(x[0] for x in lib_flat))
    print(f"sprite lib: {ncls} cls, {len(lib_flat)} sprites", flush=True)

    FLAT_SPR = [x[1] for x in lib_flat]
    CCS = np.array([x[0] for x in lib_flat], dtype=np.int32)
    SWS = np.array([x[2] for x in lib_flat], dtype=np.float32)
    SHS = np.array([x[3] for x in lib_flat], dtype=np.float32)
    ARS = SWS / np.maximum(SHS, 1e-6)

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    master = [c.strip() for c in (RAW/"_classes.txt").read_text(encoding="utf-8").splitlines() if c.strip()]
    (OUT/"classes.txt").write_text("\n".join(master)+"\n", encoding="utf-8")

    random.Random(0).shuffle(frames)
    sel = [(i, jpg, boxes) for i, (jpg, boxes) in enumerate(frames[:MAX_FRAMES])]
    written = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for r in ex.map(make_frame, sel):
            if r is not None:
                written += 1
                if written % 500 == 0:
                    print(f"  written {written}/{len(sel)}", flush=True)
    print(f"DONE: wrote {written} bbox-swap frames → {OUT}", flush=True)

if __name__ == "__main__":
    main()
