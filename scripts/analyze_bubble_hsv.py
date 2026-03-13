"""Analyze HSV color profile of Emoticon_Action.png headpat bubble."""
import cv2
import numpy as np

img = cv2.imread(r"D:\Project\ai game secretary\data\captures\Emoticon_Action.png", cv2.IMREAD_UNCHANGED)
if img.shape[2] == 4:
    # Use alpha channel to mask out background
    alpha = img[:, :, 3]
    bgr = img[:, :, :3]
    mask = alpha > 128
else:
    bgr = img
    mask = np.ones(img.shape[:2], dtype=bool)

hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
h = hsv[:, :, 0][mask]
s = hsv[:, :, 1][mask]
v = hsv[:, :, 2][mask]

print(f"Pixel count: {mask.sum()}")
print(f"H: min={h.min()} max={h.max()} mean={h.mean():.1f} median={np.median(h):.0f}")
print(f"S: min={s.min()} max={s.max()} mean={s.mean():.1f} median={np.median(s):.0f}")
print(f"V: min={v.min()} max={v.max()} mean={v.mean():.1f} median={np.median(v):.0f}")
print()
# Suggest HSV range for detection
h_lo, h_hi = max(0, int(np.percentile(h, 5))), min(180, int(np.percentile(h, 95)))
s_lo, s_hi = max(0, int(np.percentile(s, 5))), min(255, int(np.percentile(s, 95)))
v_lo, v_hi = max(0, int(np.percentile(v, 5))), min(255, int(np.percentile(v, 95)))
print(f"Suggested HSV range (5th-95th percentile):")
print(f"  H: {h_lo}-{h_hi}")
print(f"  S: {s_lo}-{s_hi}")
print(f"  V: {v_lo}-{v_hi}")
