import cv2
import numpy as np
import os
from pathlib import Path
from typing import Tuple, Dict, List, Optional

class AvatarMatcher:
    """Matches YOLO-detected student avatar ROIs against known target avatars,
    using template matching + histogram comparison for robustness."""

    def __init__(self, avatars_dir: str):
        self.avatars_dir = Path(avatars_dir)
        # Pre-cropped face directory (preferred, faster, no alpha needed)
        self.crop_dir = self.avatars_dir.parent / "角色头像_crop"
        # Cache stores (bgr_image, alpha_mask, hsv_hist) tuples at _REF_SIZE
        self.cache: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        # Per-size cache of resized candidate tuples: (name, size) -> (bgr, alpha)
        self._resize_cache: Dict[Tuple[str, int, int], Tuple[np.ndarray, np.ndarray]] = {}
        # Per-size circular mask cache for ROI
        self._mask_cache: Dict[Tuple[int, int], np.ndarray] = {}

    # Standard size for pre-cached reference avatars.
    # Roster thumbnails are ~50-80px; 96 gives good detail without being too large.
    _REF_SIZE = 96

    def _get_avatar_data(self, name: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if name in self.cache:
            return self.cache[name]

        # Prefer pre-cropped face images (fast, no alpha needed)
        crop_path = self.crop_dir / f"{name}.png" if self.crop_dir.is_dir() else None
        if crop_path and not crop_path.exists():
            # Try stripping .png suffix from name (config stores 'Wakamo.png')
            base = name[:-4] if name.lower().endswith(".png") else name
            crop_path = self.crop_dir / f"{base}.png"

        if crop_path and crop_path.exists():
            img = cv2.imdecode(np.fromfile(str(crop_path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                sz = self._REF_SIZE
                bgr = cv2.resize(img, (sz, sz), interpolation=cv2.INTER_AREA)
                alpha = self._circular_mask(sz, sz)
                hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsv], [0, 1], alpha, [30, 32], [0, 180, 0, 256])
                cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
                self.cache[name] = (bgr, alpha, hist)
                return self.cache[name]

        # Fallback: load full portrait with alpha and crop internally
        path = self.avatars_dir / f"{name}.png"
        if not path.exists():
            path = self.avatars_dir / name
        if not path.exists():
            return None

        img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        if len(img.shape) < 3 or img.shape[2] != 4:
            # No alpha — use as-is with circular mask
            bgr = img if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            sz = self._REF_SIZE
            bgr = cv2.resize(bgr, (sz, sz), interpolation=cv2.INTER_AREA)
            alpha = self._circular_mask(sz, sz)
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], alpha, [30, 32], [0, 180, 0, 256])
            cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
            self.cache[name] = (bgr, alpha, hist)
            return self.cache[name]

        h, w = img.shape[:2]

        # Wiki portraits are bust shots (456×404); roster thumbnails show the FACE.
        # Crop to the upper-center square (face area) before matching.
        crop_top = 0
        crop_bot = int(h * 0.65)
        crop_left = int(w * 0.125)
        crop_right = int(w * 0.875)
        img = img[crop_top:crop_bot, crop_left:crop_right]

        # Make square (crop to center square)
        ch, cw = img.shape[:2]
        if cw > ch:
            off = (cw - ch) // 2
            img = img[:, off:off + ch]
        elif ch > cw:
            off = (ch - cw) // 2
            img = img[off:off + cw, :]

        sz = self._REF_SIZE
        img = cv2.resize(img, (sz, sz), interpolation=cv2.INTER_AREA)

        bgr = img[:, :, :3]
        alpha = img[:, :, 3].copy()

        cy_c, cx_c = sz // 2, sz // 2
        radius = sz // 2
        yy, xx = np.ogrid[:sz, :sz]
        circ = ((xx - cx_c) ** 2 + (yy - cy_c) ** 2) <= radius ** 2
        alpha[~circ] = 0

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask8 = (alpha > 0).astype(np.uint8) * 255
        hist = cv2.calcHist([hsv], [0, 1], mask8, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

        self.cache[name] = (bgr, alpha, hist)
        return self.cache[name]

    @staticmethod
    def _circular_mask(h: int, w: int) -> np.ndarray:
        """Create a circular mask for the given dimensions."""
        cy_c, cx_c = h // 2, w // 2
        radius = min(h, w) // 2
        yy, xx = np.ogrid[:h, :w]
        return (((xx - cx_c) ** 2 + (yy - cy_c) ** 2) <= radius ** 2).astype(np.uint8) * 255

    def match_avatar(self, target_roi: np.ndarray, candidate_names: List[str]) -> Tuple[Optional[str], float]:
        """
        Compare target_roi (cropped from screenshot) against a list of candidate avatar names.
        Uses combined template matching (60%) + histogram comparison (40%).
        Returns the best matching name and its score (-1.0 to 1.0).
        """
        if target_roi is None or target_roi.size == 0 or not candidate_names:
            return None, -1.0

        best_name = None
        best_score = -1.0
        th, tw = target_roi.shape[:2]

        # Apply circular mask to ROI (roster thumbnails are circular)
        mask_key = (th, tw)
        roi_circ_mask = self._mask_cache.get(mask_key)
        if roi_circ_mask is None:
            roi_circ_mask = self._circular_mask(th, tw)
            self._mask_cache[mask_key] = roi_circ_mask

        # Compute ROI HSV histogram
        roi_hsv = cv2.cvtColor(target_roi, cv2.COLOR_BGR2HSV)
        roi_hist = cv2.calcHist([roi_hsv], [0, 1], roi_circ_mask, [30, 32], [0, 180, 0, 256])
        cv2.normalize(roi_hist, roi_hist, 0, 1, cv2.NORM_MINMAX)

        for name in candidate_names:
            cand_data = self._get_avatar_data(name)
            if cand_data is None:
                continue

            cand_bgr, cand_alpha, cand_hist = cand_data

            # Resize candidate to match ROI size (cached across calls for same size)
            rk = (name, th, tw)
            resized = self._resize_cache.get(rk)
            if resized is None:
                resized = (
                    cv2.resize(cand_bgr, (tw, th)),
                    cv2.resize(cand_alpha, (tw, th)),
                )
                self._resize_cache[rk] = resized
            cand_bgr_resized, cand_alpha_resized = resized

            # Template matching score
            res = cv2.matchTemplate(target_roi, cand_bgr_resized, cv2.TM_CCOEFF_NORMED, mask=cand_alpha_resized)
            tmpl_score = float(res[0][0])

            # Histogram comparison score (correlation, range -1 to 1)
            hist_score = float(cv2.compareHist(roi_hist, cand_hist, cv2.HISTCMP_CORREL))

            # Combined score: template matching + histogram
            score = 0.6 * tmpl_score + 0.4 * hist_score

            if score > best_score:
                best_score = score
                best_name = name

        return best_name, max(-1.0, float(best_score))
