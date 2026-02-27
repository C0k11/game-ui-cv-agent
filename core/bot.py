import time
import pydirectinput
from typing import Optional, Tuple
from vision.engine import VisionEngine
from vision.window import GameWindow

# Optional: configure pydirectinput to be faster
pydirectinput.PAUSE = 0.05

class BABot:
    """
    Blue Archive Automation Bot using DXcam + YOLO26n + RapidOCR.
    """
    def __init__(self, yolo_model_path: str = None):
        self.window = GameWindow("Blue Archive")
        if not self.window.find_window():
            print("[Warning] 'Blue Archive' window not found. Please start the game.")
            self.region = None
        else:
            self.region = self.window.get_region()
            print(f"[Info] Found game window at region: {self.region}")

        # Initialize Vision Engine
        self.vision = VisionEngine(
            yolo_model_path=yolo_model_path,
            region=self.region,
            max_fps=120,
            video_mode=True
        )

        self.running = False

    def click(self, x: int, y: int):
        """
        Perform a direct mouse click at the specified screen coordinates.
        If a region is set, x and y are relative to the region (game window).
        """
        screen_x = x
        screen_y = y
        if self.region:
            screen_x += self.region[0]
            screen_y += self.region[1]
            
        pydirectinput.click(screen_x, screen_y)

    def run_loop(self):
        """
        Main execution loop.
        """
        self.running = True
        print("[Info] Bot started. Press Ctrl+C to stop.")
        try:
            while self.running:
                start_t = time.perf_context()
                
                # 1. Capture
                frame = self.vision.grab()
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                # 2. Detect
                results = self.vision.detect(frame, conf=0.65)
                
                # Example: process results here
                # if results is not None:
                #     for box in results.boxes:
                #         cls_id = int(box.cls[0])
                #         conf = float(box.conf[0])
                #         x1, y1, x2, y2 = box.xyxy[0].tolist()
                #         cx, cy = int((x1+x2)/2), int((y1+y2)/2)
                #         label = self.vision.yolo.names[cls_id]
                        
                #         if label == "start_btn":
                #             self.click(cx, cy)
                
                # ... Add OCR reading logic for specific regions if needed
                
                # FPS limiter / debug output
                # elapsed = time.perf_context() - start_t
                # print(f"Frame processed in {elapsed*1000:.1f}ms")
                time.sleep(0.05)
                
        except KeyboardInterrupt:
            print("[Info] Stopping bot...")
        finally:
            self.vision.stop()

if __name__ == "__main__":
    bot = BABot()
    bot.run_loop()
