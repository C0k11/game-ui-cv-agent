# 验证 cost 组件画法: 取几个角色 ref, 左上角画 cost 圆圈数字 (模拟战斗牌遮挡), montage 看效果
import cv2, numpy as np, random
from pathlib import Path

REF = Path(r"D:\Project\ai game secretary\data\captures\角色头像")

def imread_u(p):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)

def draw_cost(img, cx, cy, r, num):
    """战斗牌 cost 圈: 深半透明圆 + 浅边 + 白数字 (照 fused star/heart 风格)."""
    ov = img.copy()
    cv2.circle(ov, (cx, cy), r, (45, 45, 45), -1)
    cv2.addWeighted(ov, 0.82, img, 0.18, 0, img)          # 半透明深底圆
    cv2.circle(img, (cx, cy), r, (215, 215, 215), max(2, r // 12))  # 浅灰边
    txt = str(num)
    fs = r / 13.0
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, fs, 2)
    cv2.putText(img, txt, (cx - tw // 2, cy + th // 2), cv2.FONT_HERSHEY_DUPLEX, fs, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, txt, (cx - tw // 2, cy + th // 2), cv2.FONT_HERSHEY_DUPLEX, fs, (255, 255, 255), 2, cv2.LINE_AA)

refs = (sorted(REF.glob("*.png")) + sorted(REF.glob("*.jpg")))[:6]
print(f"refs found: {len(list(REF.glob('*.png')) + list(REF.glob('*.jpg')))}, using {len(refs)}")
tiles = []
for p in refs:
    img = imread_u(p)
    if img is None:
        continue
    img = cv2.resize(img, (240, 240))
    h, w = img.shape[:2]
    r = int(w * 0.15); cx = int(w * 0.19); cy = int(h * 0.17)   # 左上角 cost 圈
    draw_cost(img, cx, cy, r, random.randint(1, 9))
    tiles.append(img)

rows = []
for i in range(0, len(tiles), 3):
    row = tiles[i:i + 3]
    while len(row) < 3:
        row.append(np.zeros((240, 240, 3), np.uint8))
    rows.append(np.hstack(row))
m = np.vstack(rows) if rows else np.zeros((240, 240, 3), np.uint8)
out = r"D:\Project\ai game secretary\cost_preview.jpg"
cv2.imwrite(out, m)
print("SAVED:", out, m.shape, "tiles", len(tiles))
