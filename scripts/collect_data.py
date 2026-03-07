"""Record raw gameplay frames via ADB screencap (reliable, no black frames).

DXcam/Win32 capture returns black frames for GPU-rendered MuMu windows.
ADB screencap always works because it captures the Android framebuffer directly.

Usage:
    python scripts/collect_data.py [--interval 0.5] [--title MuMu]
"""
import time
import cv2
import numpy as np
import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

REPO_ROOT = Path(__file__).resolve().parents[1]
MUMU_ADB = Path(r"C:\Program Files\Netease\MuMu\nx_device\12.0\shell\adb.exe")


def _find_adb() -> str:
    if MUMU_ADB.exists():
        return str(MUMU_ADB)
    return "adb"


def _adb_screencap(adb: str, serial: str = "127.0.0.1:7555") -> np.ndarray:
    """Capture screen via ADB screencap -p (PNG) → decode → BGR numpy array."""
    result = subprocess.run(
        [adb, "-s", serial, "exec-out", "screencap", "-p"],
        capture_output=True, timeout=10,
    )
    if result.returncode != 0 or len(result.stdout) < 100:
        return None
    arr = np.frombuffer(result.stdout, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


class DataCollector:
    """Record raw gameplay frames using ADB screencap."""

    def __init__(self, output_dir: str = "data/raw_images", interval_sec: float = 0.5,
                 serial: str = "127.0.0.1:7555"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.interval_sec = interval_sec
        self.serial = serial
        self.adb = _find_adb()

        # Ensure ADB connected
        subprocess.run([self.adb, "connect", serial], capture_output=True, timeout=5)
        test = _adb_screencap(self.adb, serial)
        if test is None:
            print(f"[Error] ADB screencap failed for {serial}")
            self.ready = False
        else:
            print(f"[Info] ADB screencap OK: {test.shape[1]}x{test.shape[0]}")
            self.ready = True

    def run(self):
        if not self.ready:
            return

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.output_dir / f"run_{run_id}"
        run_dir.mkdir(exist_ok=True)

        print(f"[Info] Starting data collection...")
        print(f"[Info] Saving frames to {run_dir}")
        print(f"[Info] Interval: {self.interval_sec}s")
        print("[Info] Press Ctrl+C to stop recording.")

        frame_count = 0
        try:
            while True:
                start_t = time.time()
                frame = _adb_screencap(self.adb, self.serial)
                if frame is not None:
                    filepath = run_dir / f"frame_{frame_count:06d}.jpg"
                    cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    print(f"Saved {filepath.name} ({frame.shape[1]}x{frame.shape[0]})", end='\r')
                    frame_count += 1
                else:
                    print(f"[Warn] screencap returned None at frame {frame_count}")

                elapsed = time.time() - start_t
                sleep_time = max(0, self.interval_sec - elapsed)
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            print(f"\n[Info] Data collection stopped. Total frames: {frame_count}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--serial", default="127.0.0.1:7555")
    args = parser.parse_args()
    collector = DataCollector(interval_sec=args.interval, serial=args.serial)
    collector.run()
