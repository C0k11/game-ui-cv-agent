"""YOLO26n-cls avatar classifier with 4-layer verdict pipeline.

Replaces AvatarMatcher (template + histogram) for BA character identification
across cafe-invite list, schedule grid, dispatch screens, etc.

Architecture (decided 2026-05-17 after avatar_cls_yolo26n hit val top1=86.55%,
top5=95.32%):

  layer 1 — FAST PATH (single frame, ~80% of cells)
      top1_conf >= 0.95  AND  (top1_conf - top2_conf) >= 0.30
      → emit immediately, no buffer touch
      Why both: 48% of WRONG predictions also have conf>0.90, but the
      margin between top1 and top2 collapses on those cases (Softmax
      gives ~[0.91, 0.08] when confused between two similar palettes).
      Requiring margin>=0.30 catches the over-confident mis-fires.

  layer 2 — TEMPORAL VOTING (cells that didn't pass layer 1)
      buffer keyed by (room_idx, cell_idx), maxlen=5
      need >=3 frames before emitting a verdict
      decision rule:
        a) if any class appears as top1 in >=2/3 of frames → accept
        b) else: stack all probs → mean → argmax (conf-weighted top5 union)

  layer 3 — SPATIAL MUTEX (post-vote, same frame)
      BA invite list shows distinct characters per slot — same class
      should not win two cells simultaneously.  Hungarian assignment
      (scipy.linear_sum_assignment) maximizes total log-prob over all
      (cell -> class) pairings, optionally restricted to each cell's
      top5 to keep the cost matrix small.

  layer 4 — UNKNOWN DROP (protect downstream)
      top1_conf < 0.50  OR  any verdict that fails layer 1+2+3
      → write crop to unknown_dir/<date>/<frame>_<r>_<c>.jpg
      Operator drags the crop into the right class folder, rebuild
      dataset, retrain.  Self-improving loop.

Public API:
    clf = AvatarClassifier()  # lazy-loads on first call
    verdicts = clf.classify_cells(
        cells=[(crop_bgr, room_idx, cell_idx), ...],
        frame_id=tick,
    )
    # verdicts: List[Verdict] in same order as cells

    clf.reset_buffer(reason="invite list scrolled")
        # call when UI state changes so old votes don't bleed
"""
from __future__ import annotations

import os
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---- model location ---------------------------------------------------------
DEFAULT_MODEL_PATH = Path(
    r"D:\Project\ml_cache\models\yolo\runs\avatar_cls_yolo26n\weights\best.pt"
)
DEFAULT_UNKNOWN_DIR = Path(
    r"D:\Project\ai game secretary\data\yolo_datasets\avatar_cls_unknown"
)

# ---- thresholds (class constants; override on instance for tuning) ---------
FAST_CONF = 0.95          # layer-1 confidence cutoff
FAST_MARGIN = 0.30        # layer-1 (top1 - top2) margin
VOTE_BUFFER_MAX = 5       # rolling window per (room, cell)
VOTE_MIN_FRAMES = 3       # minimum frames before vote emits
VOTE_MAJORITY = 2         # >= N/VOTE_MIN_FRAMES agreement for fast vote
UNKNOWN_CONF = 0.50       # below this → unknown bucket
MUTEX_TOPK = 5            # cost matrix uses each cell's top-K classes


@dataclass
class Verdict:
    """One classification result.

    name: class name, or None when still buffering, or "__unknown__".
    source: which layer produced this verdict.
    top5: [(class_name, prob), ...] for diagnostics / fallback.
    """
    name: Optional[str]
    conf: float
    source: str                       # "fast" | "vote" | "mutex" | "pending" | "unknown"
    top5: List[Tuple[str, float]] = field(default_factory=list)
    # debug fields (cheap, useful for log analysis)
    room_idx: int = -1
    cell_idx: int = -1
    frame_id: int = -1


# Module-level singleton (one model, many call sites)
_lock = threading.Lock()
_singleton: Optional["AvatarClassifier"] = None


def get_default() -> "AvatarClassifier":
    """Get the process-wide default classifier (lazy-init)."""
    global _singleton
    with _lock:
        if _singleton is None:
            _singleton = AvatarClassifier()
        return _singleton


class AvatarClassifier:
    """Stateful avatar classifier with multi-frame voting and spatial mutex.

    Stateful in the sense that vote buffers persist across `classify_cells`
    calls.  Call `reset_buffer()` when the UI context changes (scrolled
    invite list, left cafe, etc.) so old votes don't poison new decisions.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        unknown_dir: Optional[Path] = None,
        device: str = "cuda",
        imgsz: int = 224,
    ):
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.unknown_dir = Path(unknown_dir) if unknown_dir else DEFAULT_UNKNOWN_DIR
        self.device = device
        self.imgsz = imgsz

        self._model: Any = None
        self._class_names: List[str] = []
        self._name_to_idx: Dict[str, int] = {}
        # vote buffer: (room_idx, cell_idx) -> deque of (probs np.ndarray, frame_id)
        self._vote_buffer: Dict[Tuple[int, int], deque] = {}

    # ─── lifecycle ──────────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        return self.model_path.is_file()

    def _ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        if not self.available:
            print(f"[AvatarClassifier] model not found at {self.model_path}")
            return False
        try:
            from ultralytics import YOLO
            self._model = YOLO(str(self.model_path))
            # names: {0: 'Akari', 1: 'Akari_(New_Year)', ...}
            names_dict = self._model.names
            self._class_names = [names_dict[i] for i in sorted(names_dict.keys())]
            self._name_to_idx = {n: i for i, n in enumerate(self._class_names)}
            # warm up
            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            self._model.predict(dummy, verbose=False, imgsz=self.imgsz)
            print(f"[AvatarClassifier] loaded {len(self._class_names)} classes from {self.model_path}")
            return True
        except Exception as e:
            print(f"[AvatarClassifier] load failed: {e}")
            return False

    def reset_buffer(self, reason: str = "") -> None:
        """Flush all vote buffers.  Call when UI context changes."""
        n = len(self._vote_buffer)
        self._vote_buffer.clear()
        if reason and n > 0:
            print(f"[AvatarClassifier] buffer reset ({n} slots) — {reason}")

    def get_class_names(self) -> List[str]:
        if not self._ensure_loaded():
            return []
        return list(self._class_names)

    # ─── main entry ─────────────────────────────────────────────────────────

    def classify_cells(
        self,
        cells: List[Tuple[np.ndarray, int, int]],
        frame_id: int = -1,
    ) -> List[Verdict]:
        """Classify a list of avatar crops with the full 4-layer pipeline.

        cells: list of (crop_bgr, room_idx, cell_idx) tuples.  All cells in
            one call are assumed to be from the SAME frame, which is what
            enables layer-3 spatial mutex.
        frame_id: monotonic int (tick number) for buffer / log tracking.

        Returns a list of Verdict in the same order as `cells`.
        """
        if not cells:
            return []
        if not self._ensure_loaded():
            return [
                Verdict(None, 0.0, "no_model", [], r, c, frame_id)
                for _, r, c in cells
            ]

        # ── batch inference (one model call for all cells) ──
        crops = [c[0] for c in cells]
        # Filter blank/tiny crops
        valid_idx = [i for i, im in enumerate(crops) if im is not None and im.size > 0]
        if not valid_idx:
            return [
                Verdict("__unknown__", 0.0, "empty", [], r, c, frame_id)
                for _, r, c in cells
            ]

        valid_crops = [crops[i] for i in valid_idx]
        results = self._model.predict(
            valid_crops, verbose=False, imgsz=self.imgsz, device=self.device,
        )

        # Build per-cell probs (None for invalid)
        all_probs: List[Optional[np.ndarray]] = [None] * len(cells)
        for vi, r in zip(valid_idx, results):
            all_probs[vi] = r.probs.data.cpu().numpy()

        # ── layer 1 + layer 2 per cell ──
        verdicts: List[Verdict] = []
        # cells that need layer 3 (mutex) — must have a tentative name + probs
        mutex_inputs: List[int] = []
        for i, (crop, room_idx, cell_idx) in enumerate(cells):
            probs = all_probs[i]
            if probs is None:
                verdicts.append(Verdict("__unknown__", 0.0, "empty", [],
                                        room_idx, cell_idx, frame_id))
                continue

            top5 = self._top5(probs)
            v = self._single_cell_pipeline(
                probs, top5, room_idx, cell_idx, frame_id, crop,
            )
            verdicts.append(v)
            if v.source in ("fast", "vote") and v.name and v.name != "__unknown__":
                mutex_inputs.append(i)

        # ── layer 3: Hungarian assignment for same-frame mutex ──
        if len(mutex_inputs) >= 2:
            self._apply_spatial_mutex(verdicts, all_probs, mutex_inputs)

        return verdicts

    # ─── layer helpers ──────────────────────────────────────────────────────

    def _single_cell_pipeline(
        self,
        probs: np.ndarray,
        top5: List[Tuple[str, float]],
        room_idx: int,
        cell_idx: int,
        frame_id: int,
        crop: np.ndarray,
    ) -> Verdict:
        """Run layers 1, 2, 4 (mutex is later, batch-level)."""
        top1_name, top1_conf = top5[0]
        top2_conf = top5[1][1] if len(top5) > 1 else 0.0
        margin = top1_conf - top2_conf

        # layer 1: fast accept
        if top1_conf >= FAST_CONF and margin >= FAST_MARGIN:
            return Verdict(top1_name, top1_conf, "fast", top5,
                           room_idx, cell_idx, frame_id)

        # layer 4: unknown floor (drop crop for relabel)
        if top1_conf < UNKNOWN_CONF:
            self._dump_unknown(crop, room_idx, cell_idx, frame_id, top5)
            return Verdict("__unknown__", top1_conf, "unknown", top5,
                           room_idx, cell_idx, frame_id)

        # layer 2: buffer + vote
        key = (room_idx, cell_idx)
        buf = self._vote_buffer.setdefault(key, deque(maxlen=VOTE_BUFFER_MAX))
        buf.append((probs, frame_id))

        if len(buf) < VOTE_MIN_FRAMES:
            return Verdict(None, top1_conf, "pending", top5,
                           room_idx, cell_idx, frame_id)

        # Decision rule 2a: majority top1 across recent frames
        recent = list(buf)[-VOTE_MIN_FRAMES:]
        top1_counts: Dict[str, List[float]] = {}
        for p, _fid in recent:
            i = int(np.argmax(p))
            top1_counts.setdefault(self._class_names[i], []).append(float(p[i]))
        winner = max(top1_counts.items(), key=lambda kv: (len(kv[1]), max(kv[1])))
        if len(winner[1]) >= VOTE_MAJORITY:
            return Verdict(winner[0], float(np.mean(winner[1])), "vote", top5,
                           room_idx, cell_idx, frame_id)

        # Decision rule 2b: mean probs across window, then argmax
        stacked = np.stack([p for p, _ in recent], axis=0)
        mean_probs = stacked.mean(axis=0)
        wi = int(np.argmax(mean_probs))
        wconf = float(mean_probs[wi])
        if wconf < UNKNOWN_CONF:
            self._dump_unknown(crop, room_idx, cell_idx, frame_id, top5)
            return Verdict("__unknown__", wconf, "unknown", top5,
                           room_idx, cell_idx, frame_id)
        # Recompute top5 from mean
        mean_top5 = self._top5(mean_probs)
        return Verdict(self._class_names[wi], wconf, "vote", mean_top5,
                       room_idx, cell_idx, frame_id)

    def _apply_spatial_mutex(
        self,
        verdicts: List[Verdict],
        all_probs: List[Optional[np.ndarray]],
        active_idx: List[int],
    ) -> None:
        """Hungarian assignment to resolve duplicate top1 across cells.

        Only runs if 2+ cells share a top1 name AND have valid probs.
        Cost = -log(p) so maximizing assignment minimizes total cost.
        """
        # Quick check: do any two cells have the same name?
        names_seen: Dict[str, List[int]] = {}
        for i in active_idx:
            n = verdicts[i].name
            if n:
                names_seen.setdefault(n, []).append(i)
        dupes = {n: idxs for n, idxs in names_seen.items() if len(idxs) > 1}
        if not dupes:
            return

        # Build candidate class set: union of each cell's top-K
        candidate_classes: set = set()
        for i in active_idx:
            probs = all_probs[i]
            if probs is None:
                continue
            top_k = np.argsort(probs)[-MUTEX_TOPK:][::-1]
            for k in top_k:
                candidate_classes.add(int(k))
        cand_idx = sorted(candidate_classes)
        if len(cand_idx) < len(active_idx):
            # Fewer candidates than cells — can't make 1:1.  Leave verdicts as-is.
            return

        # Cost matrix: rows = cells, cols = candidate classes
        # cost[i][j] = -log(probs[cell_i][cand_idx[j]])
        # scipy minimizes total cost → equivalent to maximizing total log-prob
        cost = np.full((len(active_idx), len(cand_idx)), fill_value=20.0)
        for ri, ci in enumerate(active_idx):
            probs = all_probs[ci]
            if probs is None:
                continue
            for cj, cls_idx in enumerate(cand_idx):
                p = float(probs[cls_idx])
                # Clip floor so log doesn't explode on prob=0
                cost[ri, cj] = -np.log(max(p, 1e-6))

        try:
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(cost)
        except Exception as e:
            print(f"[AvatarClassifier] Hungarian failed ({e}) — leaving greedy result")
            return

        for ri, cj in zip(row_ind, col_ind):
            cell_i = active_idx[ri]
            cls_idx = cand_idx[cj]
            probs = all_probs[cell_i]
            new_name = self._class_names[cls_idx]
            new_conf = float(probs[cls_idx])
            old = verdicts[cell_i]
            if old.name != new_name:
                # Rebuild top5 (already correct since probs unchanged)
                verdicts[cell_i] = Verdict(
                    new_name, new_conf,
                    f"mutex({old.source})",
                    old.top5,
                    old.room_idx, old.cell_idx, old.frame_id,
                )

    # ─── utilities ──────────────────────────────────────────────────────────

    def _top5(self, probs: np.ndarray) -> List[Tuple[str, float]]:
        idxs = np.argsort(probs)[-5:][::-1]
        return [(self._class_names[int(i)], float(probs[int(i)])) for i in idxs]

    def _dump_unknown(
        self,
        crop: np.ndarray,
        room_idx: int,
        cell_idx: int,
        frame_id: int,
        top5: List[Tuple[str, float]],
    ) -> None:
        """Save a low-confidence crop for later human relabeling.

        Filename encodes the model's best guess so the operator can quickly
        verify obvious cases:
            <date>/f{frame}_r{room}_c{cell}_guess_{name}_{conf}.jpg
        """
        if crop is None or crop.size == 0:
            return
        try:
            today = date.today().isoformat()
            out_dir = self.unknown_dir / today
            out_dir.mkdir(parents=True, exist_ok=True)
            guess, conf = top5[0] if top5 else ("nil", 0.0)
            safe_guess = guess.replace("/", "_").replace("\\", "_")
            fn = f"f{frame_id}_r{room_idx}_c{cell_idx}_{safe_guess}_{conf:.2f}.jpg"
            out_path = out_dir / fn
            # Unicode-safe write
            import cv2
            ok, buf = cv2.imencode(".jpg", crop)
            if ok:
                with open(out_path, "wb") as f:
                    f.write(buf.tobytes())
        except Exception as e:
            print(f"[AvatarClassifier] unknown dump failed: {e}")


# ─── module smoke test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    """Quick sanity check against the val set."""
    import sys
    import cv2

    clf = AvatarClassifier()
    if not clf.available:
        print(f"NO MODEL at {clf.model_path}")
        sys.exit(1)
    clf._ensure_loaded()
    print(f"classes: {clf.get_class_names()[:5]} ... ({len(clf.get_class_names())} total)")

    # Run a tiny batch from val
    val_root = Path(r"D:\Project\ai game secretary\data\yolo_datasets\avatar_cls\val")
    samples: list = []
    for cls_dir in sorted(val_root.iterdir())[:3]:
        if not cls_dir.is_dir():
            continue
        for jpg in list(cls_dir.glob("*.jpg"))[:2]:
            img = cv2.imdecode(np.fromfile(str(jpg), dtype=np.uint8), cv2.IMREAD_COLOR)
            samples.append((img, 0, len(samples), cls_dir.name))

    cells = [(img, r, c) for img, r, c, _ in samples]
    truths = [t for _, _, _, t in samples]

    # Run 3 frames so vote can fire
    for frame in range(3):
        v = clf.classify_cells(cells, frame_id=frame)
        print(f"\nframe {frame}:")
        for vd, truth in zip(v, truths):
            ok = "✓" if vd.name == truth else "✗" if vd.name else "·"
            print(f"  {ok} truth={truth:30s} pred={str(vd.name):30s} "
                  f"conf={vd.conf:.3f} src={vd.source}")
