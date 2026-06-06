# 验证 "bbox 大小相近替换" UI synth 思路 (用户提的): 真实帧的 UI box 替换成
# 大小相近的别的 cls sprite -> 元素落在真实位置, 背景自然, 又有位置泛化.
# 产出 6 张 overlay montage: 绿框=替换进的新cls, 橙框=保留的原cls.
import sys, random
from pathlib import Path
from collections import defaultdict
import cv2, numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
RAW = REPO / "data" / "raw_images"
try:
    from scripts.build_ui_v2 import REAL_SOURCES
except Exception:
    REAL_SOURCES = ["run_20260531_173326", "run_20260531_174456", "run_20260603_134626"]

def imread_u(p):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)

def is_ui(c):  # 非头像(143-394)、非 emoticon(451) = UI
    return not (143 <= c <= 394) and c != 451

def area(w, h):
    return max(w * h, 1e-9)

def main():
    rng = random.Random(0)
    frames = []
    for src in REAL_SOURCES:
        d = RAW / src
        if not d.is_dir():
            continue
        for txt in sorted(d.glob("*.txt")):
            if txt.name == "classes.txt":
                continue
            boxes = []
            for ln in txt.read_text(encoding="utf-8", errors="ignore").splitlines():
                pp = ln.split()
                if len(pp) < 5:
                    continue
                try:
                    c = int(pp[0]); cx, cy, w, h = map(float, pp[1:5])
                except ValueError:
                    continue
                if is_ui(c):
                    boxes.append((c, cx, cy, w, h))
            jpg = txt.with_suffix(".jpg")
            if jpg.exists() and len(boxes) >= 4:
                frames.append((jpg, boxes))
        if len(frames) >= 60:
            break
    print(f"collected {len(frames)} real frames with UI boxes")

    # sprite 库: 从真实帧裁 UI box 当各 cls 的真实外观
    lib = defaultdict(list)
    for jpg, boxes in frames:
        img = imread_u(jpg)
        if img is None:
            continue
        H, W = img.shape[:2]
        for c, cx, cy, w, h in boxes:
            x1, y1 = int((cx - w/2)*W), int((cy - h/2)*H)
            x2, y2 = int((cx + w/2)*W), int((cy + h/2)*H)
            spr = img[max(0, y1):y2, max(0, x1):x2]
            if spr.size and spr.shape[0] > 8 and spr.shape[1] > 8:
                lib[c].append((spr.copy(), w, h))
    print(f"sprite lib: {len(lib)} cls, {sum(len(v) for v in lib.values())} sprites")

    tiles = []
    for jpg, boxes in frames[:6]:
        img = imread_u(jpg)
        if img is None:
            continue
        H, W = img.shape[:2]
        comp = img.copy()
        vis_boxes = []
        for c, cx, cy, w, h in boxes:
            swapped = False
            if rng.random() < 0.55:
                # 找大小相近(面积比 0.6-1.7)的别的 cls sprite
                cands = [(cc, spr) for cc, lst in lib.items() if cc != c
                         for spr, sw, sh in lst if 0.6 < area(sw, sh)/area(w, h) < 1.7]
                if cands:
                    cc, spr = rng.choice(cands)
                    x1, y1 = int((cx - w/2)*W), int((cy - h/2)*H)
                    x2, y2 = int((cx + w/2)*W), int((cy + h/2)*H)
                    bw, bh = x2 - x1, y2 - y1
                    if bw > 4 and bh > 4:
                        comp[y1:y2, x1:x2] = cv2.resize(spr, (bw, bh), interpolation=cv2.INTER_AREA)
                        vis_boxes.append((cx, cy, w, h, True))
                        swapped = True
            if not swapped:
                vis_boxes.append((cx, cy, w, h, False))
        vis = comp.copy()
        for cx, cy, w, h, sw in vis_boxes:
            x1, y1 = int((cx - w/2)*W), int((cy - h/2)*H)
            x2, y2 = int((cx + w/2)*W), int((cy + h/2)*H)
            col = (0, 255, 0) if sw else (0, 170, 255)
            cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
        tiles.append(cv2.resize(vis, (640, 360)))

    rows = []
    for i in range(0, len(tiles), 2):
        row = tiles[i:i+2]
        while len(row) < 2:
            row.append(np.zeros((360, 640, 3), np.uint8))
        rows.append(np.hstack(row))
    m = np.vstack(rows) if rows else np.zeros((360, 640, 3), np.uint8)
    out = r"D:\Project\ai game secretary\ui_swap_preview.jpg"
    cv2.imwrite(out, m)
    print("SAVED:", out, m.shape, "tiles", len(tiles))

if __name__ == "__main__":
    main()
