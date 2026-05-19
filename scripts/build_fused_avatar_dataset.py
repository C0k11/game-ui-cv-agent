"""Build a YOLO detection dataset for the FUSED multi-class avatar detector.

Detector that simultaneously (a) finds avatar bboxes and (b) identifies which
character — like the emoticon model does for headpat bubbles, but with 250+
character classes instead of 1.

Two data tracks:
  1. MANUAL: user-labeled frames under data/raw_images/<run>/, filtered to
     character-only bboxes (UI classes like 房间区域 are skipped).  These are
     gold ground truth across multiple UI contexts (MomoTalk / cafe /
     schedule / student界面 / battle squad).
  2. SYNTHETIC: paste data/captures/角色头像_crop/ refs onto real schedule
     popup trajectory frames at the 3 slot positions inside each detected
     房间区域.  70% paste rate, 30% empty slots (teaches the detector to
     NOT fire on background).  Per-class cap prevents majority-class bias.

Output:
  data/yolo_datasets/fused_avatar_v1/
    images/train/*.jpg   (manual 80% + synthetic)
    images/val/*.jpg     (manual 20% — pure gold)
    labels/{train,val}/*.txt
    data.yaml            (nc: N, names: [char1, char2, ...])

Class index remapping: master class indices include both UI (143) and
character classes (~267 if user has added them).  We extract ONLY the
character classes and remap them to dense 0..N-1 for the fused dataset.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
RAW_IMAGES = REPO / "data" / "raw_images"
TRAJECTORIES = REPO / "data" / "trajectories"
CROP_DIR = REPO / "data" / "captures" / "角色头像_crop"
CROP_HARVESTED = REPO / "data" / "captures" / "角色头像_crop_harvested_named"
NAME_MAP_JSON = REPO / "data" / "student_name_map.json"
MASTER_FILE = RAW_IMAGES / "_classes.txt"
OUT_ROOT = Path(r"D:\Project\ml_cache\models\yolo\dataset\fused_avatar_v1")

STATIC_UI_MODEL = Path(r"D:\Project\ml_cache\models\yolo\runs\static_ui_v4_yolo26n\weights\best.pt")

# Master class layout (per 2026-05-18 trim):
#   indices  0..142 : UI classes from 5/18 trim (房间区域, 弹窗叉叉, etc.)
#   indices  143..  : characters added by user during fused-avatar labeling
# This boundary is fixed at trim time — UI never grows, character does.
MASTER_UI_BOUNDARY = 143

VAL_RATIO = 0.20
SEED = 42
PER_CLASS_CAP = 80     # cap synthetic samples per character (prevent bias)
SYNTH_PASTE_PROB = 0.70  # chance a slot gets a paste vs left empty
NEGATIVE_TARGET = 120  # auto-harvest this many no-avatar frames as hard negatives

# Skill+sub_state combos that reliably contain ZERO character avatars.
# These produce "explicit negative" training frames — model learns
# stars/hearts/icons in these contexts are NOT avatar candidates.
NEGATIVE_CONTEXTS = [
    ("Mail",          "enter"),       # mail list (icons + text, no avatars)
    ("Mail",          "claim_mail"),
    ("DailyTasks",    "claim_all"),   # task list (reward icons)
    ("DailyTasks",    "enter"),
    ("CampaignSweep", ""),            # stage select grid
    ("Craft",         "enter"),       # crafting list (item icons)
    ("Craft",         "claim"),
    ("Craft",         "quick_craft"),
    ("Bounty",        "enter"),       # bounty list
    ("Bounty",        "sweep"),
    ("Bounty",        "select_stage"),
    ("EventActivity", "shop"),        # event currency shop
    ("EventActivity", "farming"),     # event farming select
    ("EventActivity", "mission"),     # event mission list
    ("EventActivity", "enter"),       # event lobby
    ("Arena",         "enter"),       # arena lobby (before squad)
    ("Arena",         "check_tickets"),
    ("Arena",         "claim_rewards"),
    ("Schedule",      "exit"),        # transition back to lobby
    ("Cafe",          "earnings"),    # cafe earnings popup (currencies)
    ("Cafe",          "exit"),
    ("Club",          "enter"),       # club lobby
    ("Club",          "claim"),
    ("PassReward",    "enter"),       # battle pass progress
    ("PassReward",    "exit"),
]


# ── unicode-safe image IO ───────────────────────────────────────────────

def imread_u(p: Path) -> Optional[np.ndarray]:
    try:
        buf = np.fromfile(str(p), dtype=np.uint8)
        if buf.size == 0:
            return None
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite_u(p: Path, img: np.ndarray) -> bool:
    try:
        ext = p.suffix or ".jpg"
        ok, buf = cv2.imencode(ext, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            return False
        with open(p, "wb") as f:
            f.write(buf.tobytes())
        return True
    except Exception:
        return False


# ── class registry helpers ──────────────────────────────────────────────

def load_master() -> List[str]:
    if not MASTER_FILE.exists():
        return []
    return [
        c.strip()
        for c in MASTER_FILE.read_text(encoding="utf-8").splitlines()
        if c.strip()
    ]


def load_character_names() -> List[str]:
    """Character classes = master entries added after the UI baseline.

    Master layout fixed at 5/18 trim:
      indices  0..142  : UI classes (房间区域 / 弹窗叉叉 / 红点 / etc.)
      indices  143..   : user-added character classes (CN names like 若藻)

    UI never grows, characters grow as user labels.  This sidesteps the
    EN-vs-CN naming mismatch — user labels in CN, we just use what's there.
    """
    master = load_master()
    if len(master) <= MASTER_UI_BOUNDARY:
        return []
    return master[MASTER_UI_BOUNDARY:]


# ── manual label extraction ─────────────────────────────────────────────

def _parse_label_file(
    txt: Path,
    master: List[str],
    char_set: set,
    fused_idx_map: Dict[str, int],
) -> Optional[Tuple[Path, List[str]]]:
    """Parse one .txt label file → (jpg_path, [yolo_lines])."""
    jpg = txt.with_suffix(".jpg")
    if not jpg.exists():
        return None
    try:
        content = txt.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            content = txt.read_text(encoding="utf-16")
        except Exception:
            return None
    char_lines: List[str] = []
    for line in content.splitlines():
        parts = line.strip().split()
        if len(parts) < 5 or not parts[0].lstrip("-").isdigit():
            continue
        cls_idx = int(parts[0])
        if not (0 <= cls_idx < len(master)):
            continue
        cls_name = master[cls_idx]
        if cls_name not in char_set:
            continue
        new_idx = fused_idx_map[cls_name]
        char_lines.append(f"{new_idx} {parts[1]} {parts[2]} {parts[3]} {parts[4]}")
    return (jpg, char_lines) if char_lines else None


def extract_val_pool_samples(
    master: List[str],
    char_set: set,
    fused_idx_map: Dict[str, int],
) -> List[Tuple[Path, List[str]]]:
    """Read DEDICATED val frames from data/raw_images/_val_fused/frames/.

    These are held-out frames the user labeled specifically for validation —
    100% go to val, never to train.  The `_` prefix tells extract_manual_samples
    to skip this dir, while the `/frames/` subdir matches the dashboard's
    layout convention so the user can label it like any other dataset.
    Returns empty list if no such pool exists.
    """
    pool = RAW_IMAGES / "_val_fused" / "frames"
    if not pool.is_dir():
        return []
    samples: List[Tuple[Path, List[str]]] = []
    for txt in sorted(pool.glob("*.txt")):
        if txt.name == "classes.txt":
            continue
        res = _parse_label_file(txt, master, char_set, fused_idx_map)
        if res is not None:
            samples.append(res)
    return samples


def extract_manual_samples(
    master: List[str],
    char_set: set,
    fused_idx_map: Dict[str, int],
) -> List[Tuple[Path, List[str]]]:
    """Walk raw_images (excluding _-prefixed dirs like _val_pool, _negatives),
    return (image_path, [yolo_label_line, ...]) pairs.  All go to train when
    a dedicated val pool exists; otherwise stratified split is applied."""
    samples: List[Tuple[Path, List[str]]] = []
    for ds in sorted(RAW_IMAGES.iterdir()):
        if not ds.is_dir() or ds.name.startswith("_"):
            continue
        sub = ds / "frames" if (ds / "frames").is_dir() else ds
        for txt in sub.glob("*.txt"):
            if txt.name == "classes.txt":
                continue
            res = _parse_label_file(txt, master, char_set, fused_idx_map)
            if res is not None:
                samples.append(res)
    return samples


# ── synthetic compositing ───────────────────────────────────────────────

def find_negative_frames(limit: int) -> List[Path]:
    """Auto-harvest no-avatar frames from trajectories.

    Walks recent runs, picks ticks whose (skill, sub_state) appears in
    NEGATIVE_CONTEXTS, returns those .jpg paths.  Diversifies across runs.
    """
    target_set = set((s, ss) for s, ss in NEGATIVE_CONTEXTS)
    out: List[Path] = []
    if not TRAJECTORIES.is_dir():
        return out
    # Walk newest runs first (more relevant to current build), diversify per-context
    runs = sorted(
        [r for r in TRAJECTORIES.iterdir() if r.is_dir() and r.name.startswith("run_")],
        reverse=True,
    )
    per_context_cap = max(3, limit // max(1, len(target_set)))  # ~3-5 per context
    per_context_count: Dict[Tuple[str, str], int] = {}
    for run in runs:
        if len(out) >= limit:
            break
        for tj in sorted(run.glob("tick_*.json")):
            if len(out) >= limit:
                break
            try:
                d = json.loads(tj.read_text(encoding="utf-8"))
            except Exception:
                continue
            key = (d.get("skill") or "", d.get("sub_state") or "")
            if key not in target_set:
                continue
            if per_context_count.get(key, 0) >= per_context_cap:
                continue
            jpg = tj.with_suffix(".jpg")
            if not jpg.exists() or jpg.stat().st_size < 1000:
                continue
            out.append(jpg)
            per_context_count[key] = per_context_count.get(key, 0) + 1
    return out


def find_schedule_popup_bg_frames(limit: int) -> List[Path]:
    """Trajectory frames where schedule popup is open — used as backgrounds."""
    ROOM_NAMES = ("視聽室", "體育館", "圖書館", "教室", "實驗室", "射擊場", "載具庫")
    out: List[Path] = []
    if not TRAJECTORIES.is_dir():
        return out
    for run in sorted(TRAJECTORIES.iterdir(), reverse=True):
        if not run.is_dir() or not run.name.startswith("run_"):
            continue
        for tj in sorted(run.glob("tick_*.json")):
            try:
                d = json.loads(tj.read_text(encoding="utf-8"))
            except Exception:
                continue
            if d.get("skill") != "Schedule":
                continue
            has_header = False
            has_room = False
            for b in d.get("ocr_boxes", []) or []:
                t = b.get("text") or ""
                if "全體課程表" in t:
                    has_header = True
                if any(rn in t for rn in ROOM_NAMES):
                    has_room = True
                if has_header and has_room:
                    break
            if has_header and has_room:
                jpg = tj.with_suffix(".jpg")
                if jpg.exists() and jpg.stat().st_size > 1000:
                    out.append(jpg)
                    if len(out) >= limit:
                        return out
    return out


def synth_paste_into_room(
    bg_img: np.ndarray,
    ref_img: np.ndarray,
    room_xyxy: Tuple[int, int, int, int],
    slot_index: int,
) -> Optional[Tuple[int, int, int, int]]:
    """Paste ref into one of 3 slots within a room.  Returns paste bbox or None."""
    rx1, ry1, rx2, ry2 = room_xyxy
    rh = ry2 - ry1
    rw = rx2 - rx1
    # Strip is bottom 45% of card
    sy1 = ry1 + int(0.55 * rh)
    sy2 = ry2
    strip_h = sy2 - sy1
    if strip_h < 20:
        return None
    cell_w = rw / 3.0
    side = min(int(cell_w * 0.85), strip_h)  # 85% of cell width to leave gap
    if side < 20:
        return None
    cell_cx = rx1 + int((slot_index + 0.5) * cell_w)
    cell_cy = sy1 + strip_h // 2
    # Slight jitter for variety
    cell_cx += random.randint(-2, 2)
    cell_cy += random.randint(-2, 2)
    px1 = max(0, cell_cx - side // 2)
    py1 = max(0, cell_cy - side // 2)
    px2 = min(bg_img.shape[1], px1 + side)
    py2 = min(bg_img.shape[0], py1 + side)
    actual_w = px2 - px1
    actual_h = py2 - py1
    if actual_w < 16 or actual_h < 16:
        return None
    # Resize ref to target size with optional brightness jitter
    ref_resized = cv2.resize(ref_img, (actual_w, actual_h), interpolation=cv2.INTER_AREA)
    # Brightness ±5%
    bri = random.uniform(0.95, 1.05)
    ref_jit = np.clip(ref_resized.astype(np.float32) * bri, 0, 255).astype(np.uint8)
    bg_img[py1:py2, px1:px2] = ref_jit
    return (px1, py1, px2, py2)


def build_synthetic_samples(
    char_names: List[str],
    fused_idx_map: Dict[str, int],
    bg_frames: List[Path],
    refs: Dict[str, np.ndarray],
    target_per_class: int,
) -> List[Tuple[np.ndarray, List[str], str]]:
    """Generate synthetic composites.  Returns list of (img, yolo_lines, tag)."""
    from ultralytics import YOLO

    if not STATIC_UI_MODEL.is_file():
        print(f"[err] static_ui model missing: {STATIC_UI_MODEL}")
        return []
    print(f"[synth] loading static_ui model for room detection...")
    static_ui = YOLO(str(STATIC_UI_MODEL))
    room_class_idx = next(
        (k for k, v in static_ui.names.items() if v == "房间区域"), None
    )
    if room_class_idx is None:
        print(f"[err] static_ui has no '房间区域' class")
        return []

    class_counts: Counter = Counter()
    out: List[Tuple[np.ndarray, List[str], str]] = []

    available_chars = [c for c in char_names if c in refs]
    if not available_chars:
        print(f"[synth] no refs available, skipping synthesis")
        return []
    print(f"[synth] {len(available_chars)} characters with refs available")
    print(f"[synth] scanning {len(bg_frames)} background frames...")

    for bi, bg_path in enumerate(bg_frames):
        bg = imread_u(bg_path)
        if bg is None:
            continue
        h, w = bg.shape[:2]
        # Detect 房间区域
        det = static_ui(bg, conf=0.15, verbose=False)[0]
        raw_rooms = []
        for b in det.boxes:
            if int(b.cls[0]) == room_class_idx:
                raw_rooms.append((b.xyxy[0].tolist(), float(b.conf[0])))
        # IoU NMS
        raw_rooms.sort(key=lambda r: -r[1])
        rooms = []
        for (x1, y1, x2, y2), c in raw_rooms:
            keep = True
            for (kx1, ky1, kx2, ky2), _ in rooms:
                ix1, iy1 = max(x1, kx1), max(y1, ky1)
                ix2, iy2 = min(x2, kx2), min(y2, ky2)
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2-ix1) * (iy2-iy1)
                    a = (x2-x1) * (y2-y1)
                    b_area = (kx2-kx1) * (ky2-ky1)
                    if inter / max(1, a + b_area - inter) > 0.5:
                        keep = False; break
            if keep:
                rooms.append(((x1, y1, x2, y2), c))
        if not rooms:
            continue

        # Compose: paste refs into slots, record bboxes
        composite = bg.copy()
        bboxes: List[str] = []
        for (rx1, ry1, rx2, ry2), _ in rooms:
            for slot in range(3):
                if random.random() > SYNTH_PASTE_PROB:
                    continue
                # Pick a character not at cap yet
                tries = 0
                char_name = None
                while tries < 10:
                    candidate = random.choice(available_chars)
                    if class_counts[candidate] < target_per_class:
                        char_name = candidate
                        break
                    tries += 1
                if char_name is None:
                    # All chars at cap, skip
                    continue
                ref = refs[char_name]
                room_int = (int(rx1), int(ry1), int(rx2), int(ry2))
                paste_box = synth_paste_into_room(composite, ref, room_int, slot)
                if paste_box is None:
                    continue
                px1, py1, px2, py2 = paste_box
                cx = (px1+px2) / 2 / w
                cy = (py1+py2) / 2 / h
                bw = (px2-px1) / w
                bh = (py2-py1) / h
                cls_idx = fused_idx_map[char_name]
                bboxes.append(f"{cls_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                class_counts[char_name] += 1

        if bboxes:
            out.append((composite, bboxes, "synth"))

        if (bi + 1) % 50 == 0:
            print(f"  scanned {bi+1}/{len(bg_frames)} | composites: {len(out)} | "
                  f"unique chars used: {len([c for c, n in class_counts.items() if n > 0])}")

        # Stop early if we've hit target across all chars
        if all(class_counts[c] >= target_per_class for c in available_chars):
            print(f"  [stop] all characters reached cap {target_per_class}")
            break

    print(f"[synth] generated {len(out)} composites, total bboxes: "
          f"{sum(len(b) for _, b, _ in out)}")
    print(f"[synth] class distribution: top 10:")
    for c, n in class_counts.most_common(10):
        print(f"    {n}  {c}")
    return out


# ── main ────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth-bg-limit", type=int, default=500,
                    help="Max trajectory schedule frames to use as synthetic backgrounds")
    ap.add_argument("--per-class-cap", type=int, default=PER_CLASS_CAP)
    ap.add_argument("--skip-synth", action="store_true",
                    help="Manual labels only, no synthetic compositing (useful for sanity check)")
    ap.add_argument("--skip-negatives", action="store_true",
                    help="Skip auto-harvested negative frames (useful for ablation)")
    ap.add_argument("--negative-target", type=int, default=NEGATIVE_TARGET,
                    help="How many no-avatar frames to auto-harvest")
    args = ap.parse_args()

    master = load_master()
    if not master:
        print(f"[err] empty master at {MASTER_FILE}")
        return 1

    char_names = load_character_names()
    if not char_names:
        print(f"[err] master[{MASTER_UI_BOUNDARY}:] empty — user hasn't added "
              f"any character classes yet (master len={len(master)})")
        return 1
    print(f"[init] master has {len(master)} classes total — "
          f"{MASTER_UI_BOUNDARY} UI + {len(char_names)} characters")

    # All characters from master are already 'in master' by construction.
    # Build dense 0..N-1 remap sorted alphabetically for stable class order.
    fused_names = sorted(char_names)
    fused_idx_map = {name: i for i, name in enumerate(fused_names)}
    print(f"[init] fused detector will have {len(fused_names)} character classes")

    char_set = set(fused_names)

    # ── 1. Manual labels (train) + dedicated val pool ──
    manual_samples = extract_manual_samples(master, char_set, fused_idx_map)
    print(f"[manual] {len(manual_samples)} frames with character bboxes (→ train)")
    total_manual_boxes = sum(len(b) for _, b in manual_samples)
    print(f"[manual] total bboxes: {total_manual_boxes}")

    val_pool_samples = extract_val_pool_samples(master, char_set, fused_idx_map)
    if val_pool_samples:
        print(f"[val_pool] {len(val_pool_samples)} dedicated val frames found "
              f"(→ val, 100% held out from train)")
    else:
        print(f"[val_pool] none found at _val_fused/frames/ "
              f"— falling back to stratified split from train")

    # ── 2. Synthetic samples ──
    synth_samples_data: List[Tuple[np.ndarray, List[str], str]] = []
    if not args.skip_synth and len(fused_names) > 0:
        # Refs lookup strategy:
        #   1. CROP_HARVESTED uses CN names directly → primary source (matches master)
        #   2. CROP_DIR uses EN names → optional fallback via student_name_map
        # User labels are CN, so CROP_HARVESTED is the natural fit.
        refs: Dict[str, np.ndarray] = {}
        for c in fused_names:
            ref_path = CROP_HARVESTED / f"{c}.png"
            if ref_path.exists():
                im = imread_u(ref_path)
                if im is not None:
                    refs[c] = im
        print(f"[refs] {len(refs)} refs from CROP_HARVESTED (CN-named)")

        # Fallback: try EN-named via CN→EN map for chars not yet covered
        missing = [c for c in fused_names if c not in refs]
        if missing and NAME_MAP_JSON.exists():
            cn_to_en = json.loads(NAME_MAP_JSON.read_text(encoding="utf-8"))
            extra = 0
            for c in missing:
                en = cn_to_en.get(c)
                if not en:
                    continue
                ref_path = CROP_DIR / f"{en}.png"
                if ref_path.exists():
                    im = imread_u(ref_path)
                    if im is not None:
                        refs[c] = im
                        extra += 1
            print(f"[refs] +{extra} more via CN→EN name map (still missing {len(fused_names) - len(refs)})")

        bg_frames = find_schedule_popup_bg_frames(args.synth_bg_limit)
        print(f"[bg] found {len(bg_frames)} schedule popup backgrounds")

        synth_samples_data = build_synthetic_samples(
            fused_names, fused_idx_map, bg_frames, refs, args.per_class_cap
        )

    # ── 2b. Auto-harvested negative frames (no-avatar contexts) ──
    negative_frames: List[Path] = []
    if not args.skip_negatives:
        negative_frames = find_negative_frames(args.negative_target)
        print(f"[neg] auto-harvested {len(negative_frames)} negative frames "
              f"from trajectories ({len(NEGATIVE_CONTEXTS)} contexts)")

    # ── 3. Clean output, emit ──
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (OUT_ROOT / sub).mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)

    if val_pool_samples:
        # Dedicated val pool: 100% of manual → train, val_pool → val
        train_manual = manual_samples
        val_set = val_pool_samples
        n_classes_in_val = len({
            int(line.split()[0]) for s in val_set for line in s[1]
            if line.split() and line.split()[0].isdigit()
        })
        n_train_classes = len({
            int(line.split()[0]) for s in train_manual for line in s[1]
            if line.split() and line.split()[0].isdigit()
        })
        print(f"[split] dedicated val: train {len(train_manual)} (100% of manual) / "
              f"val {len(val_set)} (held-out pool); val covers "
              f"{n_classes_in_val}/{n_train_classes} classes")
    else:
        # Fallback: stratified split (rare classes pinned to train)
        class_to_frames: Dict[int, List[int]] = {}
        for i, (_, lines) in enumerate(manual_samples):
            seen = set()
            for line in lines:
                parts = line.split()
                if parts and parts[0].isdigit():
                    seen.add(int(parts[0]))
            for c in seen:
                class_to_frames.setdefault(c, []).append(i)

        must_train: set = set()
        for c, frame_idxs in sorted(class_to_frames.items(), key=lambda kv: len(kv[1])):
            if len(frame_idxs) <= 3:
                must_train.update(frame_idxs)
            else:
                must_train.update(frame_idxs[:2])

        remaining = [i for i in range(len(manual_samples)) if i not in must_train]
        rng.shuffle(remaining)
        n_val_target = max(1, int(len(manual_samples) * VAL_RATIO))
        n_val_actual = min(n_val_target, len(remaining))
        val_idx = set(remaining[:n_val_actual])
        train_idx = set(range(len(manual_samples))) - val_idx

        val_set = [manual_samples[i] for i in val_idx]
        train_manual = [manual_samples[i] for i in train_idx]
        n_classes_in_val = len({
            int(line.split()[0]) for s in val_set for line in s[1]
            if line.split() and line.split()[0].isdigit()
        })
        print(f"[split] stratified fallback: train {len(train_manual)} / "
              f"val {len(val_set)} (val covers "
              f"{n_classes_in_val}/{len(class_to_frames)} classes; "
              f"sparse classes pinned to train)")

    # Split negatives 80/20 train/val (val gets some for FP-rate measurement)
    rng.shuffle(negative_frames)
    n_neg_val = int(len(negative_frames) * VAL_RATIO)
    neg_val = negative_frames[:n_neg_val]
    neg_train = negative_frames[n_neg_val:]

    # Emit val (manual + a few negatives for FP measurement)
    n_val_boxes = 0
    for jpg, lines in val_set:
        stem = f"manual__{jpg.parent.name}__{jpg.stem}"
        shutil.copy2(jpg, OUT_ROOT / "images/val" / (stem + ".jpg"))
        (OUT_ROOT / "labels/val" / (stem + ".txt")).write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        n_val_boxes += len(lines)
    for jpg in neg_val:
        stem = f"neg__{jpg.parent.name}__{jpg.stem}"
        shutil.copy2(jpg, OUT_ROOT / "images/val" / (stem + ".jpg"))
        (OUT_ROOT / "labels/val" / (stem + ".txt")).write_text("", encoding="utf-8")
    print(f"[emit] val: {len(val_set) + len(neg_val)} files "
          f"({len(val_set)} manual gold + {len(neg_val)} negatives), "
          f"{n_val_boxes} boxes")

    # Emit train (manual 80% + synth + negatives)
    n_train_boxes = 0
    for jpg, lines in train_manual:
        stem = f"manual__{jpg.parent.name}__{jpg.stem}"
        shutil.copy2(jpg, OUT_ROOT / "images/train" / (stem + ".jpg"))
        (OUT_ROOT / "labels/train" / (stem + ".txt")).write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        n_train_boxes += len(lines)
    for si, (composite, lines, tag) in enumerate(synth_samples_data):
        stem = f"synth__{si:05d}"
        imwrite_u(OUT_ROOT / "images/train" / (stem + ".jpg"), composite)
        (OUT_ROOT / "labels/train" / (stem + ".txt")).write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        n_train_boxes += len(lines)
    for jpg in neg_train:
        stem = f"neg__{jpg.parent.name}__{jpg.stem}"
        shutil.copy2(jpg, OUT_ROOT / "images/train" / (stem + ".jpg"))
        (OUT_ROOT / "labels/train" / (stem + ".txt")).write_text("", encoding="utf-8")
    n_train_files = len(train_manual) + len(synth_samples_data) + len(neg_train)
    print(f"[emit] train: {n_train_files} files "
          f"({len(train_manual)} manual + {len(synth_samples_data)} synth + "
          f"{len(neg_train)} negatives), {n_train_boxes} positive boxes")

    # ── 4. data.yaml ──
    yaml_lines = [
        f"path: {OUT_ROOT.as_posix()}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(fused_names)}",
        "names:",
    ]
    for i, name in enumerate(fused_names):
        safe = name.replace("'", "\\'")
        yaml_lines.append(f"  {i}: '{safe}'")
    (OUT_ROOT / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    print(f"[yaml] data.yaml — {len(fused_names)} classes")
    print()
    print(f"[done] dataset → {OUT_ROOT}")
    print("Next: py scripts/train_yolo26.py fused_avatar_26m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
