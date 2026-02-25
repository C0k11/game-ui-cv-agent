import cv2
import numpy as np
import os
from pathlib import Path
from typing import Tuple, Dict, List, Optional

class AvatarMatcher:
    """Matches YOLO-detected student avatar ROIs against known target avatars,
    using anti-occlusion masking and histogram/MSE similarity."""
    
    def __init__(self, avatars_dir: str):
        self.avatars_dir = Path(avatars_dir)
        self.cache: Dict[str, np.ndarray] = {}

    def _crop_bottom_right(self, img: np.ndarray, crop_ratio: float = 0.3) -> np.ndarray:
        """Mask out the bottom-right portion of the image to avoid occlusion
        from heart icons and relationship level numbers."""
        h, w = img.shape[:2]
        mask = np.ones((h, w), dtype=np.uint8) * 255
        
        # Black out bottom right corner
        cw = int(w * crop_ratio)
        ch = int(h * crop_ratio)
        mask[h - ch:, w - cw:] = 0
        
        return cv2.bitwise_and(img, img, mask=mask)

    def _get_avatar_img(self, name: str) -> Optional[np.ndarray]:
        if name in self.cache:
            return self.cache[name]
            
        path = self.avatars_dir / f"{name}.png"
        if not path.exists():
            path = self.avatars_dir / name
        if not path.exists():
            return None
            
        # imread with numpy to handle unicode paths in Windows
        img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            self.cache[name] = img
        return img

    def _calc_hist(self, img: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Calculate 2D histogram for Hue and Saturation
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist

    def match_avatar(self, target_roi: np.ndarray, candidate_names: List[str]) -> Tuple[Optional[str], float]:
        """
        Compare target_roi (cropped from screenshot) against a list of candidate avatar names.
        Returns the best matching name and its score (0.0 to 1.0).
        """
        if target_roi is None or target_roi.size == 0 or not candidate_names:
            return None, 0.0

        best_name = None
        best_score = -1.0

        th, tw = target_roi.shape[:2]
        target_masked = self._crop_bottom_right(target_roi)
        target_hist = self._calc_hist(target_masked)

        for name in candidate_names:
            cand_img = self._get_avatar_img(name)
            if cand_img is None:
                continue
            
            # Resize candidate to match the YOLO ROI size
            cand_resized = cv2.resize(cand_img, (tw, th))
            cand_masked = self._crop_bottom_right(cand_resized)
            cand_hist = self._calc_hist(cand_masked)

            # Compare histograms using correlation
            score = cv2.compareHist(target_hist, cand_hist, cv2.HISTCMP_CORREL)
            
            if score > best_score:
                best_score = score
                best_name = name

        return best_name, max(0.0, float(best_score))

