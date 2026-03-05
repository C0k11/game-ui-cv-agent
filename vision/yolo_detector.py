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

_lock = threading.Lock()
# Dictionary mapping skill name to its cached detector instance
_cached_detectors: Dict[str, 'YoloDetector'] = {}

# The base directory where YOLO models are stored
_ML_CACHE_DIR = Path(r"D:\Project\ml_cache\models\yolo")


class YoloDetector:
    """Thin wrapper around ultralytics YOLO for game UI detection."""

    def __init__(self, skill_name: str, device: str = "cuda"):
        self.skill_name = skill_name
        # Prefer TensorRT engine over PyTorch .pt
        engine_path = _ML_CACHE_DIR / f"{skill_name}.engine"
        pt_path = _ML_CACHE_DIR / f"{skill_name}.pt"
        self._model_path = str(engine_path) if engine_path.is_file() else str(pt_path)
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
            print(f"[YoloDetector] Loaded {self.skill_name} model from {self._model_path}")
            return True
        except Exception as e:
            print(f"[YoloDetector] Failed to load {self.skill_name} model: {e}")
            return False

    def detect(self, image_path: str, conf: float = 0.5, classes: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """
        Run inference and return list of detections:
        [ {"bbox": [x1,y1,x2,y2], "center": (cx,cy), "conf": 0.9, "cls": 0}, ... ]
        """
        if not self._ensure_loaded():
            return []
        
        try:
            results = self._model(image_path, conf=conf, classes=classes, verbose=False)
            detections = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    cls_id = int(box.cls[0])
                    detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "center": (int(cx), int(cy)),
                        "conf": float(box.conf[0]),
                        "cls": cls_id,
                    })
            return detections
        except Exception as e:
            print(f"[YoloDetector] detect error: {type(e).__name__}: {e}")
            return []

    def detect_headpat_bubbles(self, image_path: str, conf: float = 0.5) -> List[Tuple[int, int]]:
        """Convenience: detect headpat bubbles, return list of (cx, cy) centers."""
        if not self._ensure_loaded():
            return []
        cls_id = None
        for k, v in self._model.names.items():
            if v == "headpat_bubble":
                cls_id = k
                break
        
        # If the model doesn't have headpat_bubble yet, return empty list
        if cls_id is None:
            print(f"[YoloDetector] Warning: 'headpat_bubble' class not found in {self.skill_name} model.")
            return []
            
        return self.detect(image_path, conf=conf, classes=[cls_id])

    def detect_student_avatars(self, image_path: str, conf: float = 0.4) -> List[Dict[str, Any]]:
        """Convenience: detect student avatars in Schedule, return full detection objects."""
        if not self._ensure_loaded():
            return []
        cls_id = None
        for k, v in self._model.names.items():
            if v == "student_avatar":
                cls_id = k
                break
        
        # If the model doesn't have student_avatar yet, return empty list
        if cls_id is None:
            print(f"[YoloDetector] Warning: 'student_avatar' class not found in {self.skill_name} model.")
            return []
            
        return self.detect(image_path, conf=conf, classes=[cls_id])


def get_yolo_detector(skill_name: str, device: str = "cuda") -> Optional[YoloDetector]:
    """Get or create a cached YoloDetector instance for a specific skill. Returns None if model not trained yet."""
    global _cached_detectors
    with _lock:
        if skill_name in _cached_detectors:
            return _cached_detectors[skill_name]
        
        det = YoloDetector(skill_name=skill_name, device=device)
        if not det.available:
            return None
            
        _cached_detectors[skill_name] = det
        return det
