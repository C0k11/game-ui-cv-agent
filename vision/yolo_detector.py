"""YOLO26-Nano object detector for Blue Archive UI elements.

Phase 1: headpat_bubble detection in cafe.
Future: skill_ui, boss_weakness, stun_gauge for Total Assault.

YOLO26n: 2.4M params, NMS-free end-to-end, ~39ms CPU ONNX.

Usage:
    from vision.yolo_detector import YoloDetector
    det = YoloDetector(model_path="D:/Project/ml_cache/models/yolo/best.pt")
    results = det.detect(screenshot_path, conf=0.5)
    # results = [{"label": "headpat_bubble", "bbox": [x1,y1,x2,y2], "center": (cx,cy), "conf": 0.92}, ...]
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from config import YOLO_MODEL_PATH as _CFG_YOLO_PATH
except ImportError:
    _CFG_YOLO_PATH = ""

_DEFAULT_MODEL_PATH = os.environ.get(
    "YOLO_MODEL_PATH",
    _CFG_YOLO_PATH or r"D:\Project\ml_cache\models\yolo\best.pt",
)

_lock = threading.Lock()
_cached_detector: Optional["YoloDetector"] = None
_cached_path: str = ""


class YoloDetector:
    """Thin wrapper around ultralytics YOLO for game UI detection."""

    def __init__(self, model_path: str = _DEFAULT_MODEL_PATH, device: str = "cuda"):
        self._model_path = model_path
        self._device = device
        self._model: Any = None

    @property
    def available(self) -> bool:
        """Check if the model file exists (training completed)."""
        return Path(self._model_path).is_file()

    def _ensure_loaded(self) -> bool:
        """Lazy-load model on first use. Returns True if model is ready."""
        if self._model is not None:
            return True
        if not self.available:
            return False
        try:
            from ultralytics import YOLO
            self._model = YOLO(self._model_path)
            # Warm up with a tiny dummy inference
            import numpy as np
            dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            self._model(dummy, verbose=False)
            print(f"[YoloDetector] loaded: {self._model_path}")
            return True
        except Exception as e:
            print(f"[YoloDetector] load failed: {type(e).__name__}: {e}")
            self._model = None
            return False

    def detect(self, image_path: str, conf: float = 0.5,
               classes: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """Run detection on an image. Returns list of detections.

        Each detection: {"label": str, "bbox": [x1,y1,x2,y2],
                         "center": (cx, cy), "conf": float, "cls": int}
        """
        if not self._ensure_loaded():
            return []
        try:
            kwargs: Dict[str, Any] = {"verbose": False, "conf": conf}
            if classes is not None:
                kwargs["classes"] = classes
            results = self._model(image_path, **kwargs)
            detections: List[Dict[str, Any]] = []
            for result in results:
                names = result.names  # {0: "headpat_bubble", ...}
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    detections.append({
                        "label": names.get(cls_id, f"class_{cls_id}"),
                        "bbox": [x1, y1, x2, y2],
                        "center": (int(cx), int(cy)),
                        "conf": float(box.conf[0]),
                        "cls": cls_id,
                    })
            return detections
        except Exception as e:
            print(f"[YoloDetector] detect error: {type(e).__name__}: {e}")
            return []

    def detect_headpat_bubbles(self, image_path: str, conf: float = 0.5
                               ) -> List[Tuple[int, int]]:
        """Convenience: detect headpat bubbles, return list of (cx, cy) centers."""
        dets = self.detect(image_path, conf=conf, classes=[0])
        return [d["center"] for d in dets]


def get_yolo_detector(model_path: str = _DEFAULT_MODEL_PATH,
                      device: str = "cuda") -> Optional[YoloDetector]:
    """Get or create a cached YoloDetector instance. Returns None if model not trained yet."""
    global _cached_detector, _cached_path
    with _lock:
        if _cached_detector is not None and _cached_path == model_path:
            return _cached_detector
        det = YoloDetector(model_path=model_path, device=device)
        if not det.available:
            return None
        _cached_detector = det
        _cached_path = model_path
        return det
