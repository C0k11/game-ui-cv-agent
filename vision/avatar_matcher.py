import cv2
import numpy as np
import os
from pathlib import Path
from typing import Tuple, Dict, List, Optional

class AvatarMatcher:
    """Matches YOLO-detected student avatar ROIs against known target avatars,
    using template matching + histogram comparison for robustness."""

    def __init__(self, avatars_dir: str, crop_dir: Optional[str] = None):
        self.avatars_dir = Path(avatars_dir)
        # Pre-cropped face directory (preferred, faster, no alpha needed).
        # Override priority: explicit ctor arg > env var > sibling default.
        # Env var lets users swap template sets without touching skill code.
        if crop_dir:
            self.crop_dir = Path(crop_dir)
        else:
            env_dir = os.environ.get("AVATAR_CROP_DIR")
            if env_dir:
                self.crop_dir = Path(env_dir)
            else:
                self.crop_dir = self.avatars_dir.parent / "角色头像_crop"
        # Cache stores (bgr_image, alpha_mask, hsv_hist) tuples at _REF_SIZE
        self.cache: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        # Per-size cache of resized candidate tuples: (name, size) -> (bgr, alpha)
        self._resize_cache: Dict[Tuple[str, int, int], Tuple[np.ndarray, np.ndarray]] = {}
        # Per-size circular mask cache for ROI
        self._mask_cache: Dict[Tuple[int, int], np.ndarray] = {}

    # Standard size for pre-cached reference avatars.
    # Roster thumbnails are ~50-80px; 64 gives plenty of detail and ~2.25× the
    # matchTemplate throughput vs 96 (cost scales with size²).  Schedule's
    # roster scan is 28 cells × ~250 templates worth of work per pass; the
    # speed difference matters at tick-budget scale.
    _REF_SIZE = 64
    # Stage-2 matchTemplate shortlist size.  Tunable per-instance.
    # Empirically (run_20260428_175656 schedule, 168 cells, threshold 0.35):
    #   top_k=40: 22 hits, 25 ms/cell  (original)
    #   top_k=25: 22 hits, 19 ms/cell
    #   top_k=15: 26 hits, 16 ms/cell  ← default (more accurate AND faster)
    #   top_k=10: 26 hits, 14 ms/cell
    #   top_k≤6:  +false-positive risk (non-fav distractors removed too far)
    # The accuracy gain is because lower top_k strips look-alike non-fav
    # variants from Stage-2 → genuine favs win the open-set contest more often.
    STAGE2_TOP_K = 15

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

    def all_template_names(self) -> List[str]:
        """List every avatar template available (open-set candidate pool).

        Prefers the pre-cropped face directory; falls back to full portraits.
        Cached on first call.
        """
        if getattr(self, "_all_names_cache", None) is not None:
            return self._all_names_cache
        names: List[str] = []
        src = self.crop_dir if self.crop_dir.is_dir() else self.avatars_dir
        if src.is_dir():
            for p in sorted(src.glob("*.png")):
                names.append(p.stem)
        self._all_names_cache = names
        return names

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

    def match_avatar_open_set(
        self,
        target_roi: np.ndarray,
        favorite_names,
        all_names: Optional[List[str]] = None,
    ) -> Tuple[Optional[str], float, Optional[str], float]:
        """Open-set variant: return both best-favorite and best-overall match
        in a single pass over the full template pool.

        Returns a 4-tuple (best_fav_name, best_fav_score,
        best_overall_name, best_overall_score).  The caller should
        accept the favorite match only when best_overall_name is in
        `favorite_names` — otherwise the roster cell genuinely shows a
        non-favorite student that just happens to resemble a favorite.

        Parameters
        ----------
        target_roi :
            BGR crop of the roster cell (square).
        favorite_names :
            Any iterable of names the caller considers "favorites".
            Internally converted to a ``set`` for O(1) membership tests.
        all_names :
            Optional open-set pool.  Defaults to every template available
            under ``角色头像_crop`` (or the full portrait directory if
            crop is missing).  Pass a subset to scope the search.
        """
        if target_roi is None or target_roi.size == 0:
            return None, -1.0, None, -1.0

        fav_set = set(favorite_names or ())
        if all_names is None:
            all_names = self.all_template_names()
        if not all_names:
            return None, -1.0, None, -1.0

        th, tw = target_roi.shape[:2]
        mask_key = (th, tw)
        roi_circ_mask = self._mask_cache.get(mask_key)
        if roi_circ_mask is None:
            roi_circ_mask = self._circular_mask(th, tw)
            self._mask_cache[mask_key] = roi_circ_mask

        roi_hsv = cv2.cvtColor(target_roi, cv2.COLOR_BGR2HSV)
        roi_hist = cv2.calcHist([roi_hsv], [0, 1], roi_circ_mask,
                                [30, 32], [0, 180, 0, 256])
        cv2.normalize(roi_hist, roi_hist, 0, 1, cv2.NORM_MINMAX)

        best_fav_name: Optional[str] = None
        best_fav_score: float = -1.0
        best_all_name: Optional[str] = None
        best_all_score: float = -1.0

        # ── Stage 1: cheap histogram prefilter over ALL candidates ──
        # cv2.compareHist runs in ~1 µs whereas cv2.matchTemplate with a
        # mask costs ~300 µs per cell-template pair.  The prefilter keeps
        # the top-K candidates by histogram correlation and we run the
        # full combined score only on those.  K is chosen so every
        # favourite can still surface: a favorite that scores < the Kth
        # non-favorite by histogram alone is essentially a mismatch
        # anyway (color/tone is the coarse cue), so the shortlist is
        # safe to truncate.
        hist_prefilter: List[Tuple[float, str]] = []
        for name in all_names:
            cand_data = self._get_avatar_data(name)
            if cand_data is None:
                continue
            _, _, cand_hist = cand_data
            hs = float(cv2.compareHist(roi_hist, cand_hist, cv2.HISTCMP_CORREL))
            hist_prefilter.append((hs, name))
        if not hist_prefilter:
            return None, -1.0, None, -1.0
        hist_prefilter.sort(key=lambda t: t[0], reverse=True)

        # Keep top-K by histogram, plus every favorite regardless of
        # rank.  The favorite over-include is cheap (usually tens of
        # entries) and guarantees the closed-set fav match stays
        # correct even when a favorite has a weak histogram.
        # K=25 is the verified sweet spot (full bench: K=40/25/15/10 all
        # produce the same 22 ★FAV count on the 175656 schedule run, but
        # K<15 starts losing edge cases).  Stage-2 matchTemplate cost
        # scales linearly with K.
        top_k = self.STAGE2_TOP_K
        shortlist: List[Tuple[float, str]] = list(hist_prefilter[:top_k])
        seen = {name for _, name in shortlist}
        for hs, name in hist_prefilter[top_k:]:
            if name in fav_set and name not in seen:
                shortlist.append((hs, name))
                seen.add(name)

        # ── Stage 2: full combined score on the shortlist only ──
        for hist_score, name in shortlist:
            cand_data = self._get_avatar_data(name)
            if cand_data is None:
                continue
            cand_bgr, cand_alpha, _ = cand_data

            rk = (name, th, tw)
            resized = self._resize_cache.get(rk)
            if resized is None:
                resized = (
                    cv2.resize(cand_bgr, (tw, th)),
                    cv2.resize(cand_alpha, (tw, th)),
                )
                self._resize_cache[rk] = resized
            cand_bgr_resized, cand_alpha_resized = resized

            res = cv2.matchTemplate(target_roi, cand_bgr_resized,
                                    cv2.TM_CCOEFF_NORMED, mask=cand_alpha_resized)
            tmpl_score = float(res[0][0])
            score = 0.6 * tmpl_score + 0.4 * hist_score

            if score > best_all_score:
                best_all_score = score
                best_all_name = name
            if name in fav_set and score > best_fav_score:
                best_fav_score = score
                best_fav_name = name

        # ── Tie-break / near-tie bias toward favorites ──
        # Two cases we want to accept as favorite instead of rejecting:
        #
        #   (a) EXACT tie: fav_score == all_score (e.g. base-cloned variant
        #       templates have identical pixels → identical scores; alphabetical
        #       order arbitrarily put the base first, which isn't in favs).
        #   (b) NEAR tie with strong favorite: fav_score ≥ 0.50 and the gap
        #       to all_score is ≤ 0.05 — the top-1 is probably a look-alike
        #       non-favorite and we'd rather the user's preferred identity.
        #
        # Callers still see both names; skill's open-set rule can decide whether
        # to accept, but surfacing the favorite as best_all here flips the
        # "top must equal favorite" gate.
        if best_fav_name is not None and best_fav_score > 0:
            exact_tie = (
                best_all_name is not None
                and best_all_name != best_fav_name
                and abs(best_fav_score - best_all_score) < 1e-6
            )
            near_tie_strong = (
                best_fav_score >= 0.50
                and (best_all_score - best_fav_score) <= 0.05
            )
            if exact_tie or near_tie_strong:
                best_all_name = best_fav_name
                best_all_score = best_fav_score

        return (best_fav_name, max(-1.0, best_fav_score),
                best_all_name, max(-1.0, best_all_score))
