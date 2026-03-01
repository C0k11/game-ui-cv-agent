"""Extract frames from a video at a fixed interval."""
import cv2
import sys
from pathlib import Path

video_path = Path(r"data/raw_images/run_20260226-100214/1625929590-1-192.mp4")
output_dir = Path(r"data/raw_images/run_20260226-100214/frames")
output_dir.mkdir(exist_ok=True)

INTERVAL_SEC = 0.5

cap = cv2.VideoCapture(str(video_path))
fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
skip = int(fps * INTERVAL_SEC)

print(f"Video: {fps:.1f}fps, {total} frames, extracting every {skip} frames ({INTERVAL_SEC}s)")

count = 0
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_idx % skip == 0:
        out_path = output_dir / f"frame_{count:06d}.jpg"
        cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        count += 1
    frame_idx += 1

cap.release()
print(f"Extracted {count} frames to {output_dir}")
