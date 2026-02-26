import cv2
import numpy as np
import os
from pathlib import Path
from typing import Tuple, Dict, List, Optional

class AvatarMatcher:
    """Matches YOLO-detected student avatar ROIs against known target avatars,
    using template matching with alpha masking for robustness."""

    def __init__(self, avatars_dir: str):
        self.avatars_dir = Path(avatars_dir)
        # Cache stores (bgr_image, alpha_mask) tuples
        self.cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def _get_avatar_data(self, name: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if name in self.cache:
            return self.cache[name]

        path = self.avatars_dir / f"{name}.png"
        if not path.exists():
            path = self.avatars_dir / name
        if not path.exists():
            return None

        # Load with alpha channel (IMREAD_UNCHANGED)
        img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None or img.shape[-1] != 4:
            return None

        bgr = img[:, :, :3]
        alpha = img[:, :, 3].copy()
        
        # Anti-occlusion: mask out the bottom-right 30% (hearts, relationship levels)
        h, w = alpha.shape
        cw = int(w * 0.3)
        ch = int(h * 0.3)
        alpha[h - ch:, w - cw:] = 0

        self.cache[name] = (bgr, alpha)
        return self.cache[name]

    def match_avatar(self, target_roi: np.ndarray, candidate_names: List[str]) -> Tuple[Optional[str], float]:
        """
        Compare target_roi (cropped from screenshot) against a list of candidate avatar names.
        Returns the best matching name and its score (-1.0 to 1.0).
        """
        if target_roi is None or target_roi.size == 0 or not candidate_names:
            return None, -1.0

        best_name = None
        best_score = -1.0
        th, tw = target_roi.shape[:2]

        for name in candidate_names:
            cand_data = self._get_avatar_data(name)
            if cand_data is None:
                continue

            cand_bgr, cand_alpha = cand_data

            # Resize candidate to match the YOLO ROI size
            cand_bgr_resized = cv2.resize(cand_bgr, (tw, th))
            cand_alpha_resized = cv2.resize(cand_alpha, (tw, th))

            # Compare using template matching with mask
            res = cv2.matchTemplate(target_roi, cand_bgr_resized, cv2.TM_CCOEFF_NORMED, mask=cand_alpha_resized)
            score = float(res[0][0])

            if score > best_score:
                best_score = score
                best_name = name

        return best_name, max(-1.0, float(best_score))
