import time
import cv2
import numpy as np
import dxcam
from rapidocr_onnxruntime import RapidOCR
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

class VisionEngine:
    """
    Core vision engine for Steam BA automation (4K + 4090).
    Combines DXcam for >240Hz zero-copy capture, YOLO for 2ms detection, and RapidOCR for <1ms text reading.
    """
    def __init__(self, yolo_model_path: str = None, region: tuple = None, max_fps: int = 120, video_mode: bool = False):
        # 1. DXcam: 眼睛 (High-speed screen capture)
        self.camera = dxcam.create(output_idx=0, output_color="BGR")
        self.region = region
        self.video_mode = video_mode
        
        if self.video_mode:
            self.camera.start(target_fps=max_fps, video_mode=True, region=self.region)
        
        # 2. YOLO: 大脑视觉 (Object detection)
        self.yolo = None
        if yolo_model_path and YOLO is not None:
            # We assume the user will export this to TensorRT (.engine) for optimal 4090 performance
            self.yolo = YOLO(yolo_model_path, task='detect')
            
        # 3. RapidOCR 3.6: 读书能力 (Text recognition on crops)
        from pathlib import Path
        custom_rec = Path(__file__).resolve().parent.parent / "data" / "ocr_model" / "ba_rec.onnx"
        if custom_rec.exists():
            self.ocr = RapidOCR(rec_model_path=str(custom_rec))
        else:
            self.ocr = RapidOCR()

    def grab(self) -> np.ndarray:
        """
        Grab the latest frame.
        If in video mode, gets the latest from the buffer.
        If in single capture mode, grabs a new frame.
        Returns a BGR numpy array.
        """
        if self.video_mode:
            return self.camera.get_latest_frame()
        else:
            return self.camera.grab(region=self.region)

    def detect(self, frame: np.ndarray, conf: float = 0.65, imgsz: int = 1280):
        """
        Run YOLO detection on the frame.
        """
        if self.yolo is None:
            return None
        # verbose=False to keep terminal clean during high FPS
        results = self.yolo(frame, conf=conf, imgsz=imgsz, verbose=False, half=True)
        return results[0]

    def read_text(self, image_crop: np.ndarray) -> str:
        """
        Run RapidOCR specifically on a cropped region to maximize speed.
        """
        if image_crop is None or image_crop.size == 0:
            return ""
            
        result, _ = self.ocr(image_crop)
        if result:
            # result is a list of [box, text, score]
            # Join all detected text lines
            texts = [line[1] for line in result]
            return " ".join(texts)
        return ""

    def stop(self):
        """Stop the camera capture."""
        self.camera.stop()

    def __del__(self):
        self.stop()
