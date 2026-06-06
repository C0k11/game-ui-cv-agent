# 通用 synth sample 检查: py synth_sample.py <raw_images子目录> [输出名]
# 随机取 6 帧画 bbox montage (头像绿/UI橙/摸头粉), 存项目根目录.
import sys, random
from pathlib import Path
import cv2, numpy as np

RAW = Path(r"D:\Project\ai game secretary\data\raw_images")

def imread_u(p):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)

def color(ci):
    if 143 <= ci <= 394: return (0, 220, 0)      # 头像 绿
    if ci == 451:        return (200, 0, 200)    # emoticon 粉
    return (0, 170, 255)                          # UI 橙

def main():
    sub = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else f"sample_{sub.strip('_')}.jpg"
    d = RAW / sub
    jpgs = sorted(d.glob("*.jpg"))
    print(f"{sub}: {len(jpgs)} frames")
    random.Random(1).shuffle(jpgs)
    tiles = []
    for jpg in jpgs[:6]:
        img = imread_u(jpg)
        if img is None:
            continue
        h, w = img.shape[:2]
        lp = jpg.with_suffix(".txt")
        nb = {"av": 0, "ui": 0, "emo": 0}
        if lp.exists():
            for ln in lp.read_text(encoding="utf-8").splitlines():
                p = ln.split()
                if len(p) < 5:
                    continue
                ci = int(float(p[0])); cx, cy, bw, bh = map(float, p[1:5])
                x1, y1 = int((cx-bw/2)*w), int((cy-bh/2)*h)
                x2, y2 = int((cx+bw/2)*w), int((cy+bh/2)*h)
                cv2.rectangle(img, (x1, y1), (x2, y2), color(ci), 5)
                nb["av" if 143 <= ci <= 394 else ("emo" if ci == 451 else "ui")] += 1
        cv2.putText(img, f"av{nb['av']} ui{nb['ui']} emo{nb['emo']}", (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
        tiles.append(cv2.resize(img, (960, 540)))
    rows = []
    for i in range(0, len(tiles), 2):
        row = tiles[i:i+2]
        while len(row) < 2:
            row.append(np.zeros((540, 960, 3), np.uint8))
        rows.append(np.hstack(row))
    m = np.vstack(rows) if rows else np.zeros((360, 640, 3), np.uint8)
    outp = Path(r"D:\Project\ai game secretary") / out
    cv2.imwrite(str(outp), m)
    print("SAVED:", outp, m.shape, "tiles", len(tiles))

if __name__ == "__main__":
    main()
