"""Mine trajectory OCR outputs for likely misreads → write data/ocr_corrections.json.

Strategy:
  1. Enumerate every unique OCR text across all `data/trajectories/run_*/tick_*.json`.
  2. For each token that appears frequently (>= MIN_FREQ) and has no exact
     match against the BA canonical vocabulary:
     a. Normalize it (fold Trad→Simp, apply existing corrections).
     b. Find the closest canonical vocab entry by edit distance.
     c. If edit distance ≤ 2 AND length-ratio is reasonable, propose
        correction `noisy_token → canonical_token`.
  3. Write merged corrections (new proposals + existing ba_vocab.CORRECTIONS)
     to `data/ocr_corrections.json`.

Intended as an offline tool — run after collecting new trajectory data.
Safe to re-run: produces idempotent output.

Usage:
    py -3 scripts/mine_ocr_corrections.py
    py -3 scripts/mine_ocr_corrections.py --min-freq 5 --max-distance 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "ocr_training"))

from vision.ocr_normalize import normalize  # noqa: E402

try:
    from ba_vocab import get_all_vocab, CORRECTIONS as BASE_CORRECTIONS  # type: ignore
except Exception:
    def get_all_vocab() -> List[str]:
        return []
    BASE_CORRECTIONS: Dict[str, str] = {}

TRAJ_DIR = REPO / "data" / "trajectories"
OUT_JSON = REPO / "data" / "ocr_corrections.json"


def _has_cjk(s: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in s)


def _edit_distance(a: str, b: str, cap: int = 3) -> int:
    """Levenshtein with early termination at `cap`."""
    la, lb = len(a), len(b)
    if abs(la - lb) > cap:
        return cap + 1
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        curr = [i] + [0] * lb
        row_min = curr[0]
        for j in range(1, lb + 1):
            ins = curr[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if a[i - 1] == b[j - 1] else 1)
            curr[j] = min(ins, dele, sub)
            if curr[j] < row_min:
                row_min = curr[j]
        if row_min > cap:
            return cap + 1
        prev = curr
    return prev[lb]


def scan_trajectories(min_conf: float = 0.55) -> Counter:
    """Return Counter of OCR text → frequency across all tick JSONs."""
    counter: Counter = Counter()
    files = 0
    if not TRAJ_DIR.exists():
        return counter
    for run_dir in sorted(TRAJ_DIR.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
            continue
        for jf in run_dir.glob("tick_*.json"):
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
            except Exception:
                continue
            for b in data.get("ocr_boxes") or []:
                t = (b.get("text") or "").strip()
                c = b.get("conf", 0) or 0
                if not t or c < min_conf or len(t) < 2:
                    continue
                counter[t] += 1
            files += 1
    print(f"  scanned {files} tick JSONs, {len(counter)} unique strings", flush=True)
    return counter


def build_canonical_index(vocab: Iterable[str]) -> Tuple[Dict[str, str], Dict[int, List[str]]]:
    """Normalize each canonical string; bucket by length for faster NN lookup."""
    norm_to_canonical: Dict[str, str] = {}
    by_len: Dict[int, List[str]] = {}
    for v in vocab:
        if not v or len(v) < 2:
            continue
        nv = normalize(v)
        norm_to_canonical.setdefault(nv, v)  # keep first canonical form per normalized key
        by_len.setdefault(len(nv), []).append(nv)
    return norm_to_canonical, by_len


def propose_corrections(
    counter: Counter,
    min_freq: int,
    max_distance: int,
) -> Dict[str, str]:
    vocab = get_all_vocab()
    if not vocab:
        print("[warn] ba_vocab empty; no canonical index", flush=True)
        return {}
    norm_to_canonical, by_len = build_canonical_index(vocab)

    proposals: Dict[str, str] = {}
    skipped_exact = 0
    skipped_short = 0
    skipped_ratio = 0
    for token, freq in counter.items():
        if freq < min_freq:
            continue
        is_cjk = _has_cjk(token)
        # Tighter length gates — short tokens cause false-positive cascades
        # (e.g. Art → AP, 1号店 → 商店). BA UI keywords worth correcting are
        # ≥4 CJK chars or ≥5 ASCII chars; shorter words have too many edit-
        # distance-2 neighbors to reliably pair.
        if is_cjk and len(token) < 4:
            skipped_short += 1
            continue
        if not is_cjk and len(token) < 5:
            skipped_short += 1
            continue
        nt = normalize(token)
        if nt in norm_to_canonical:
            skipped_exact += 1
            continue  # already matches canonical via existing normalization
        # For CJK require distance 1 (very conservative, typical "one char
        # misread"). For ASCII allow up to `max_distance` but enforce a
        # ratio cap so short garbles don't collapse onto short vocab.
        eff_max_d = 1 if is_cjk else max_distance
        best_d = eff_max_d + 1
        best_target = None
        for lenc in range(max(2, len(nt) - eff_max_d), len(nt) + eff_max_d + 1):
            for candidate in by_len.get(lenc, ()):
                d = _edit_distance(nt, candidate, cap=eff_max_d)
                if d < best_d:
                    best_d = d
                    best_target = candidate
                    if d == 1:
                        break
            if best_d == 1:
                break
        if best_target is None or best_d > eff_max_d:
            continue
        if abs(len(nt) - len(best_target)) > 2:
            continue
        # Distance/length ratio — distance must be < 33% of the longer
        # string length. Filters out cases like "Art" → "AP" (d=2, len=3).
        longer = max(len(nt), len(best_target))
        if best_d / longer > 0.33:
            skipped_ratio += 1
            continue
        # Character-set overlap sanity: src and target must share > 50%
        # of characters (order-independent). Prevents "Sports→Story",
        # "本DVD→UID", "Lv.6餐廳→Lv." — cases where edit distance is
        # within bounds but the words are semantically unrelated.
        overlap = len(set(nt) & set(best_target))
        if overlap / max(len(set(nt)), len(set(best_target))) < 0.6:
            skipped_ratio += 1
            continue
        # Digit-differ guard: if the only edits are digit substitutions,
        # the mapping is semantically wrong ("1号店" ≠ "2号店",
        # "Story05" ≠ "Story"). Require at least one non-digit diff.
        src_digits = [c for c in nt if c.isdigit()]
        tgt_digits = [c for c in best_target if c.isdigit()]
        if src_digits != tgt_digits:
            # Different digits — only accept if the non-digit skeletons
            # also differ (i.e. there's a real misread, not just a number).
            src_skel = "".join(c for c in nt if not c.isdigit())
            tgt_skel = "".join(c for c in best_target if not c.isdigit())
            if src_skel == tgt_skel:
                skipped_ratio += 1
                continue
        canonical_original = norm_to_canonical[best_target]
        proposals[token] = canonical_original
    print(f"  skipped {skipped_short} short, {skipped_exact} already-canonical, "
          f"{skipped_ratio} by distance/length ratio")
    print(f"  skipped {skipped_exact} tokens that already normalize to canonical")
    return proposals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-freq", type=int, default=5,
                    help="minimum occurrences for a token to be considered")
    ap.add_argument("--max-distance", type=int, default=2,
                    help="max Levenshtein distance from canonical")
    ap.add_argument("--min-conf", type=float, default=0.55)
    args = ap.parse_args()

    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass

    print(f"[1/3] scanning {TRAJ_DIR}...", flush=True)
    counter = scan_trajectories(min_conf=args.min_conf)
    if not counter:
        print("no trajectory data", flush=True)
        return

    print(f"[2/3] proposing corrections (min_freq={args.min_freq}, "
          f"max_distance={args.max_distance})...", flush=True)
    mined = propose_corrections(counter, args.min_freq, args.max_distance)
    print(f"  {len(mined)} new mined corrections")

    # Overwrite mode: always replace existing JSON. The miner runs against
    # the full trajectory corpus each time, so its output is the authoritative
    # snapshot. Hand-curated additions live in ba_vocab.CORRECTIONS instead.
    merged: Dict[str, str] = dict(mined)

    # Drop any entry where key == value (no-op after normalization change).
    merged = {k: v for k, v in merged.items() if k != v}

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"[3/3] wrote {len(merged)} entries to {OUT_JSON}")
    print(f"       base curated (ba_vocab.CORRECTIONS): {len(BASE_CORRECTIONS)}")
    print(f"       newly mined                          : {len(mined)}")

    # Show a sample
    print("\nsample proposals (first 20):")
    for i, (k, v) in enumerate(sorted(mined.items())[:20]):
        print(f"  {k!r:>30} -> {v!r}  (freq={counter[k]})")


if __name__ == "__main__":
    main()
