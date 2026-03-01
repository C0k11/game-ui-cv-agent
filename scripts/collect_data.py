import time
import cv2
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from vision.window import GameWindow
import dxcam

class DataCollector:
    """
    Utility script to record raw gameplay frames using DXcam for YOLO26n and OCR training.
    """
    def __init__(self, output_dir: str = "data/raw_images", interval_sec: float = 0.5):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.interval_sec = interval_sec
        
        self.window = GameWindow("Blue Archive")
        if not self.window.find_window():
            print("[Error] 'Blue Archive' window not found. Please start the game first.")
            self.region = None
        else:
            self.region = self.window.get_region()
            print(f"[Info] Found game window at region: {self.region}")

        # Setup DXcam in simple grab mode (we don't need high FPS video mode just for dataset gathering)
        self.camera = dxcam.create(output_idx=0, output_color="BGR")

    def run(self):
        if not self.region:
            return

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.output_dir / f"run_{run_id}"
        run_dir.mkdir(exist_ok=True)

        print(f"[Info] Starting data collection...")
        print(f"[Info] Saving frames to {run_dir}")
        print(f"[Info] Taking a screenshot every {self.interval_sec} seconds.")
        print("[Info] Press Ctrl+C to stop recording.")

        frame_count = 0
        try:
            while True:
                start_t = time.time()
                frame = self.camera.grab(region=self.region)
                if frame is not None:
                    filepath = run_dir / f"frame_{frame_count:06d}.jpg"
                    # Using high quality JPEG to save disk space while preserving UI details
                    cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    print(f"Saved {filepath.name}", end='\r')
                    frame_count += 1

                elapsed = time.time() - start_t
                sleep_time = max(0, self.interval_sec - elapsed)
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            print(f"\n[Info] Data collection stopped. Total frames: {frame_count}")

if __name__ == "__main__":
    # Record at 2 FPS (every 0.5s) to get diverse but not overly redundant frames
    collector = DataCollector(interval_sec=0.5)
    collector.run()
