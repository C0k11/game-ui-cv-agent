# 把 synth 样本 + bbox 框拼成一张 montage, 存盘给用户肉眼看
# 头像框=绿, UI框=橙, emoticon框=粉
import cv2, numpy as np
from pathlib import Path

RAW = Path(r"D:\Project\ai game secretary\data\raw_images")
picks = [
    ("_synth_avatar_md", "frame_000050"), ("_synth_avatar_md", "frame_000200"), ("_synth_avatar_md", "frame_000400"),
    ("_synth_avatar_md", "frame_000600"), ("_synth_avatar_md", "frame_000800"), ("_synth_avatar_md", "frame_000950"),
]

def color(ci):
    if 143 <= ci <= 394: return (0, 220, 0)      # 头像 绿
    if ci == 451:        return (200, 0, 200)    # emoticon 粉
    return (0, 170, 255)                          # UI 橙

tiles = []
for d, fn in picks:
    img = cv2.imread(str(RAW / d / f"{fn}.jpg"))
    if img is None:
        print("MISS", d, fn); continue
    h, w = img.shape[:2]
    lp = RAW / d / f"{fn}.txt"
    nbox = 0
    if lp.exists():
        for ln in lp.read_text(encoding="utf-8").splitlines():
            p = ln.split()
            if len(p) < 5: continue
            ci = int(float(p[0])); cx, cy, bw, bh = map(float, p[1:5])
            x1, y1 = int((cx-bw/2)*w), int((cy-bh/2)*h)
            x2, y2 = int((cx+bw/2)*w), int((cy+bh/2)*h)
            cv2.rectangle(img, (x1, y1), (x2, y2), color(ci), 3)
            nbox += 1
    cv2.putText(img, f"{d}/{fn}  {nbox}box", (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
    tiles.append(cv2.resize(img, (960, 540)))

rows = []
for i in range(0, len(tiles), 2):
    row = tiles[i:i+2]
    while len(row) < 2: row.append(np.zeros_like(tiles[0]))
    rows.append(np.hstack(row))
montage = np.vstack(rows)
out = r"D:\Project\ai game secretary\synth_preview_avatar_fixed.jpg"
cv2.imwrite(out, montage)
print("SAVED:", out, "shape", montage.shape, "tiles", len(tiles))
