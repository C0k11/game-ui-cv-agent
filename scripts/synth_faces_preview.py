# 把 _synth_avatar_md 里的头像 box 单独裁出来放大拼成 grid, 专看截脸准不准
import cv2, numpy as np
from pathlib import Path

RAW = Path(r"D:\Project\ai game secretary\data\raw_images\_synth_avatar_md")
frames = ["frame_000050", "frame_000150", "frame_000300", "frame_000450",
          "frame_000600", "frame_000750", "frame_000900", "frame_000950"]
tiles = []
for fn in frames:
    img = cv2.imread(str(RAW / f"{fn}.jpg"))
    lp = RAW / f"{fn}.txt"
    if img is None or not lp.exists():
        continue
    h, w = img.shape[:2]
    for ln in lp.read_text(encoding="utf-8").splitlines():
        p = ln.split()
        if len(p) < 5:
            continue
        ci = int(float(p[0]))
        if not (143 <= ci <= 394):   # 只裁头像
            continue
        cx, cy, bw, bh = map(float, p[1:5])
        x1, y1 = max(0, int((cx - bw / 2) * w)), max(0, int((cy - bh / 2) * h))
        x2, y2 = min(w, int((cx + bw / 2) * w)), min(h, int((cy + bh / 2) * h))
        crop = img[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
            continue
        tiles.append(cv2.resize(crop, (180, 180), interpolation=cv2.INTER_NEAREST))
        if len(tiles) >= 30:
            break
    if len(tiles) >= 30:
        break

COLS = 6
rows = []
for i in range(0, len(tiles), COLS):
    row = tiles[i:i + COLS]
    while len(row) < COLS:
        row.append(np.zeros((180, 180, 3), np.uint8))
    rows.append(np.hstack(row))
m = np.vstack(rows) if rows else np.zeros((180, 180, 3), np.uint8)
out = r"D:\Project\ai game secretary\synth_preview_faces.jpg"
cv2.imwrite(out, m)
print("SAVED:", out, "shape", m.shape, "faces", len(tiles))
