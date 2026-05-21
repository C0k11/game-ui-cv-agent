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
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
RAW_IMAGES = REPO / "data" / "raw_images"
TRAJECTORIES = REPO / "data" / "trajectories"
CROP_DIR = REPO / "data" / "captures" / "角色头像_crop"                  # 54×59 EN-named
CROP_HARVESTED = REPO / "data" / "captures" / "角色头像_crop_harvested_named"  # 54×59 CN-named (from game OCR)
BIG_REF_DIR = REPO / "data" / "captures" / "角色头像"                    # 404×456 EN-named (large)
NAME_MAP_JSON = REPO / "data" / "student_name_map.json"
HARVEST_NAME_MAP_JSON = REPO / "data" / "captures" / "harvest_name_map.json"  # additional 205 CN→EN mappings
EXTENSION_NAME_MAP_JSON = REPO / "data" / "student_name_map_extension.json"  # manual BA char extensions
MASTER_FILE = RAW_IMAGES / "_classes.txt"
OUT_ROOT = Path(r"D:\Project\ml_cache\models\yolo\dataset\fused_avatar_v1")

SYNTH_TEMPLATES_DIR = REPO / "data" / "synth_templates"

STATIC_UI_MODEL = Path(r"D:\Project\ml_cache\models\yolo\runs\static_ui_v4_yolo26n\weights\best.pt")

# Known costume / skin suffixes that distinguish character variants.
# User's master uses 一花泳装, student_name_map uses 一花(泳装),
# EN files use Ichika_(Swimsuit).  This list lets us translate
# CN-without-parens → CN-with-parens → EN.
COSTUME_SUFFIXES = (
    "泳装", "正月", "体育服", "TERROR", "私服", "应援团", "乐团", "礼服",
    "睡衣", "魔女", "武装", "战斗", "万圣节", "骑士", "女仆", "兔女郎",
    "新年", "盛装", "婚纱", "圣诞", "运动", "和服", "侍女", "黑色",
    "幼女", "应援", "学校", "婴儿", "美少女", "假面舞会",
)

# Master class layout (per 2026-05-18 trim):
#   indices  0..142 : UI classes from 5/18 trim (房间区域, 弹窗叉叉, etc.)
#   indices  143..  : characters added by user during fused-avatar labeling
# This boundary is fixed at trim time — UI never grows, character does.
MASTER_UI_BOUNDARY = 143

VAL_RATIO = 0.20
SEED = 42
PER_CLASS_CAP = 200    # cap synthetic samples per character (raised from 80 → 200
                       # to allow more position/scale variants per ref)
SYNTH_PASTE_PROB = 0.70  # chance a slot gets a paste vs left empty
USE_LARGE_REF_PROB = 0.40  # when both small & large refs are available, use
                           # large (downsampled) this often — gives sharper
                           # synthesized faces than upsampling small refs.
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


def cn_to_paren_form(cn_name: str) -> str:
    """Convert master CN naming (一花泳装) to student_name_map key form (一花(泳装)).

    Strategy: if name ends in a known costume suffix, insert parens before it.
    Returns the original name if no suffix matches.
    """
    for sfx in COSTUME_SUFFIXES:
        if cn_name.endswith(sfx) and len(cn_name) > len(sfx):
            base = cn_name[:-len(sfx)]
            return f"{base}({sfx})"
    return cn_name


def _make_trad_to_simp():
    """Build a 繁→简 converter.  Prefers opencc if installed, else uses a
    minimal mapping covering BA character-name chars."""
    try:
        from opencc import OpenCC
        cc = OpenCC("t2s")
        return cc.convert
    except Exception:
        pass
    # Minimal fallback table — only covers chars commonly used in BA names
    table = str.maketrans(
        "亞愛陽麗靈葉節氣練結經顧讀東車馬見電華壽歲齋靜聲務態戰錢點舊親"
        "離禮選讓雙邊團綠紅藍黃歡樂畫麼個兒後業髮豐夢鬧願鳥魚鶴鸚鵡鳳"
        "黨貝鐵閉開韻響錄錯鄉鄰釋鑒鎖鍋釣銀銅錫鋼鎮醫斷證課註誌謝詢"
        "詳話認說請誤識覽覺觀間聞閣閱陣陰階雜難雞風飛餘養饋騎驚體寧"
        "豔靜變動連時間機億億兆勝點戀夢願節務態戰續總網絕辭",
        "亚爱阳丽灵叶节气练结经顾读东车马见电华寿岁斋静声务态战钱点旧亲"
        "离礼选让双边团绿红蓝黄欢乐画么个儿后业发丰梦闹愿鸟鱼鹤鹦鹉凤"
        "党贝铁闭开韵响录错乡邻释鉴锁锅钓银铜锡钢镇医断证课注志谢询"
        "详话认说请误识览觉观间闻阁阅阵阴阶杂难鸡风飞余养馈骑惊体宁"
        "艳静变动连时间机亿亿兆胜点恋梦愿节务态战续总网绝辞",
    )
    return lambda s: s.translate(table)


_TRAD_TO_SIMP = _make_trad_to_simp()


def build_cn_to_en_lookup() -> Dict[str, str]:
    """Build a CN-master-name → EN-file-name table.

    Combines THREE sources, with 繁→简 normalization so master 简体 labels
    can match harvest_name_map's 繁体 keys:
      1. student_name_map.json — 261 variant entries
      2. harvest_name_map.json:renamed — 205 entries (mostly 繁体)
      3. For each entry, generate variants:
         a. paren-removed (both ASCII () and fullwidth （） handled)
         b. simplified-Chinese form (亞伽里 → 亚伽里)
    """
    out: Dict[str, str] = {}

    def ingest(raw_map):
        for cn, en in raw_map.items():
            if not isinstance(en, str) or not en.strip():
                continue
            forms = {cn, _TRAD_TO_SIMP(cn)}
            extras = set()
            for f in forms:
                stripped = (f.replace("(", "").replace(")", "")
                             .replace("（", "").replace("）", ""))
                if stripped != f:
                    extras.add(stripped)
            forms |= extras
            for f in forms:
                out[f] = en

    if NAME_MAP_JSON.exists():
        try:
            ingest(json.loads(NAME_MAP_JSON.read_text(encoding="utf-8")))
        except Exception:
            pass
    if HARVEST_NAME_MAP_JSON.exists():
        try:
            harvest = json.loads(HARVEST_NAME_MAP_JSON.read_text(encoding="utf-8"))
            renamed = harvest.get("renamed") if isinstance(harvest, dict) else None
            if isinstance(renamed, dict):
                ingest(renamed)
        except Exception:
            pass

    # Manual extension map (covers BA chars missing from the auto-built maps —
    # 亚留, 爱丽丝, 紫, 千秋, 律, 律 etc. — built by hand from EN file inventory)
    if EXTENSION_NAME_MAP_JSON.exists():
        try:
            ingest(json.loads(EXTENSION_NAME_MAP_JSON.read_text(encoding="utf-8")))
        except Exception:
            pass

    return out


def load_refs_multi_source(fused_names: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
    """Load 2 ref images per character (small + large) — EN-named Yostar refs only.

    Sources (all clean official artwork, no game-screen harvested noise):
      * BIG_REF_DIR/<EN>.png  (404×456) — primary, sharp downsample-friendly
      * CROP_DIR/<EN>.png     (54×59)   — small variant, native game-icon size

    CN-named harvested refs (角色头像_crop_harvested_named/) are deliberately
    NOT used — they contain UI clutter (heart, star, lv text residue) and
    compression noise from being cut from real game screenshots.

    For each master CN char (一花泳装), name conversion tries 3 forms:
      1. Direct lookup in student_name_map (exact CN match)
      2. Paren form via heuristic (一花泳装 → 一花(泳装))
      3. Best-effort base-name match (strip suffixes one by one)

    Returns: { char_name: { "small": np.ndarray | None, "large": np.ndarray | None } }
    """
    cn_to_en = build_cn_to_en_lookup()

    def resolve_en(cn_name: str) -> Optional[str]:
        # Try direct, then paren form, then progressive base-name fallback
        if cn_name in cn_to_en:
            return cn_to_en[cn_name]
        paren = cn_to_paren_form(cn_name)
        if paren != cn_name and paren in cn_to_en:
            return cn_to_en[paren]
        # Last resort: if name has a suffix, try base name alone
        # (helps when master has a variant not in map but base char is)
        for sfx in COSTUME_SUFFIXES:
            if cn_name.endswith(sfx) and len(cn_name) > len(sfx):
                base = cn_name[:-len(sfx)]
                if base in cn_to_en:
                    return cn_to_en[base]
        return None

    out: Dict[str, Dict[str, np.ndarray]] = {}
    counts = {"small_en": 0, "large_en": 0, "unresolved_name": 0}
    for c in fused_names:
        en = resolve_en(c)
        if not en:
            counts["unresolved_name"] += 1
            continue
        entry: Dict[str, np.ndarray] = {}
        small_p = CROP_DIR / f"{en}.png"
        if small_p.exists():
            im = imread_u(small_p)
            if im is not None:
                entry["small"] = im
                counts["small_en"] += 1
        big_p = BIG_REF_DIR / f"{en}.png"
        if big_p.exists():
            im = imread_u(big_p)
            if im is not None:
                entry["large"] = im
                counts["large_en"] += 1
        if entry:
            out[c] = entry
    print(f"[refs] EN-only mode (CN-harvested disabled)")
    print(f"[refs] small (54×59):  {counts['small_en']}")
    print(f"[refs] large (404×456): {counts['large_en']}")
    print(f"[refs] name unresolved: {counts['unresolved_name']} / {len(fused_names)}")
    print(f"[refs] unique chars covered: {len(out)}/{len(fused_names)}")
    return out


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


def apply_ui_overlay_aug(ref_img: np.ndarray) -> np.ndarray:
    """Adversarial augmentation (Gemini Path B): overlay random UI elements.

    Trains the classifier to IGNORE Lv text / star / weapon icon / heart icon
    / alpha-dim overlay — visual noise that appears in real game contexts
    (cafe / schedule / student list / arena squad) but is NOT part of the
    character identity.  Without this, model overfits to "MomoTalk blue
    frame" or similar context-specific scaffolding.

    Each overlay applies with random probability.  Coordinates are relative
    to ref_img dimensions so it works for both small (54×59) and large
    (404×456) refs.
    """
    h, w = ref_img.shape[:2]
    out = ref_img.copy()

    # 50% — Lv text bottom-left or top-left  (real games show Lv1..Lv90 or MAX)
    if random.random() < 0.50:
        lv = random.randint(1, 90)
        text = f"Lv.{lv}" if random.random() < 0.75 else "MAX"
        font_scale = max(0.30, min(0.55, w / 100.0))
        x = random.choice([2, w - 35])
        y = random.choice([int(h * 0.28), h - 5])
        # White text with black outline for visibility
        cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_DUPLEX,
                    font_scale, (0, 0, 0), 2)
        cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_DUPLEX,
                    font_scale, (255, 255, 255), 1)

    # 35% — Star (top-left)
    if random.random() < 0.35:
        cx = max(6, min(w - 6, random.randint(5, 12)))
        cy = max(6, min(h - 6, random.randint(5, 12)))
        r = max(3, w // 12)
        cv2.circle(out, (cx, cy), r, (0, 220, 255), -1)  # yellow disc (simulated star)
        cv2.circle(out, (cx, cy), r, (0, 80, 120), 1)    # outline

    # 40% — Weapon-class icon (bottom-right corner)
    if random.random() < 0.40:
        size = max(8, w // 6)
        color = random.choice([
            (0, 0, 255),     # red
            (0, 165, 255),   # orange
            (255, 128, 0),   # blue (BGR)
            (255, 0, 255),   # magenta
            (255, 255, 0),   # cyan
        ])
        x1 = max(0, w - size - 1)
        y1 = max(0, h - size - 1)
        cv2.rectangle(out, (x1, y1), (w - 1, h - 1), color, -1)

    # 25% — Heart with number (bottom-right area, common in schedule popup)
    if random.random() < 0.25:
        # Pink heart blob
        size = max(4, w // 14)
        cx = w - size - 2
        cy = max(size + 2, h - size - 4)
        cv2.circle(out, (cx, cy), size, (180, 50, 230), -1)
        # Tiny number nearby
        num = str(random.randint(1, 99))
        cv2.putText(out, num, (max(0, cx - 8), min(h - 2, cy + size + 3)),
                    cv2.FONT_HERSHEY_DUPLEX, max(0.25, w / 220.0),
                    (255, 255, 255), 1)

    # 25% — Alpha-dim (simulates "not selected" or "cooling down" state)
    if random.random() < 0.25:
        alpha = random.uniform(0.55, 0.85)
        out = np.clip(out.astype(np.float32) * alpha, 0, 255).astype(np.uint8)

    return out


def apply_border_ablation(ref_img: np.ndarray) -> np.ndarray:
    """Random border crop/cover — forces classifier to use AVATAR pixels,
    not the UI-frame style (MomoTalk's blue ring, schedule's pink ring etc.)
    that becomes a 'cheat code' if model relies on it.
    """
    h, w = ref_img.shape[:2]
    if random.random() > 0.40:  # only 40% of refs get this
        return ref_img
    out = ref_img.copy()
    border = max(2, random.randint(2, max(3, w // 18)))
    sides = random.choice(["top", "bot", "left", "right", "all-thin", "all-thick"])
    color = (random.randint(0, 255),
             random.randint(0, 255),
             random.randint(0, 255))
    if sides == "all-thick":
        b = max(3, border + 2)
    elif sides == "all-thin":
        b = border
    else:
        b = border
    if sides in ("top", "all-thin", "all-thick"):
        out[:b, :] = color
    if sides in ("bot", "all-thin", "all-thick"):
        out[h - b:, :] = color
    if sides in ("left", "all-thin", "all-thick"):
        out[:, :b] = color
    if sides in ("right", "all-thin", "all-thick"):
        out[:, w - b:] = color
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



def classify_ctx_by_norm_boxes(boxes_norm):
    """Classify what UI context a labeled frame represents, by bbox layout.

    Used by cross-context synth to find non-schedule contexts (student list,
    momotalk, battle squad) to swap refs into.
    """
    n = len(boxes_norm)
    if n == 0:
        return "empty"
    sorted_y = sorted([b[1] for b in boxes_norm])
    y_unique = []
    for y in sorted_y:
        if not y_unique or abs(y - y_unique[-1]) > 0.05:
            y_unique.append(y)
    n_rows = len(y_unique)
    widths = sorted([b[2] for b in boxes_norm])
    median_w = widths[len(widths) // 2]
    if n_rows >= 5 and median_w < 0.05:
        return "momotalk"
    if median_w > 0.08 and 8 <= n <= 14 and n_rows <= 3:
        return "student_list"
    if median_w < 0.06 and n >= 6:
        return "schedule"
    if 2 <= n <= 6 and median_w > 0.06:
        return "battle_squad"
    return "other"


def build_cross_context_synth(
    manual_samples: List[Tuple[Path, List[str]]],
    ref_bundle: Dict[str, Dict[str, np.ndarray]],
    fused_idx_map: Dict[str, int],
    available_chars: List[str],
    target_per_class: int,
    class_counts: Counter,
) -> List[Tuple[np.ndarray, List[str], str]]:
    """Generate cross-context synthetic composites.

    Strategy: use USER'S OWN labeled frames as backgrounds.  For each existing
    bbox in a manual frame, with 55% probability REPLACE the avatar with a
    fresh random ref (with UI overlay + border ablation augmentation applied).
    The rest of the frame UI scaffolding stays intact (MomoTalk blue ring,
    student list card chrome, battle squad bar, etc.) — this teaches the
    model that character ID is *independent* of UI context.

    Per-class cap shared with the schedule-popup synth via class_counts
    Counter (passed in by reference).
    """
    # Pre-parse: group frames by detected context
    by_ctx: Dict[str, List] = defaultdict(list)
    for jpg, label_lines in manual_samples:
        boxes_pixel = []
        boxes_norm = []
        for line in label_lines:
            parts = line.split()
            if len(parts) < 5 or not parts[0].lstrip("-").isdigit():
                continue
            try:
                cls = int(parts[0])
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            except ValueError:
                continue
            boxes_norm.append((cx, cy, w, h))
            boxes_pixel.append((cls, cx, cy, w, h))
        if not boxes_pixel:
            continue
        ctx = classify_ctx_by_norm_boxes(boxes_norm)
        if ctx in ("schedule", "other", "empty"):
            continue  # schedule has its own dedicated synth pipeline
        by_ctx[ctx].append((jpg, boxes_pixel))

    print(f"[cross-ctx] bg frames by context: "
          f"{ {k: len(v) for k, v in by_ctx.items()} }")

    out: List[Tuple[np.ndarray, List[str], str]] = []
    N_VARIANTS = {
        "student_list": 5,
        "momotalk": 3,
        "battle_squad": 5,
        "cafe_invite": 4,
    }
    REPLACE_PROB = 0.55  # per-bbox replace chance

    for ctx, frames in by_ctx.items():
        n_var = N_VARIANTS.get(ctx, 3)
        ctx_count = 0
        for jpg, boxes_pixel in frames:
            bg = imread_u(jpg)
            if bg is None:
                continue
            H, W = bg.shape[:2]
            for vi in range(n_var):
                composite = bg.copy()
                new_label_strs: List[str] = []
                for orig_cls, cx, cy, w, h in boxes_pixel:
                    x1 = int((cx - w / 2) * W)
                    y1 = int((cy - h / 2) * H)
                    x2 = int((cx + w / 2) * W)
                    y2 = int((cy + h / 2) * H)
                    bw = x2 - x1
                    bh = y2 - y1
                    if bw < 16 or bh < 16:
                        new_label_strs.append(f"{orig_cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                        continue

                    if random.random() < REPLACE_PROB:
                        tries = 0
                        new_char = None
                        while tries < 12:
                            cand = random.choice(available_chars)
                            if class_counts[cand] < target_per_class:
                                new_char = cand
                                break
                            tries += 1
                        if new_char is not None:
                            bundle = ref_bundle[new_char]
                            if bundle.get("large") is not None and random.random() < USE_LARGE_REF_PROB:
                                ref = bundle["large"]
                            elif bundle.get("small") is not None:
                                ref = bundle["small"]
                            else:
                                ref = bundle["large"]
                            ref = ref.copy()
                            ref = apply_ui_overlay_aug(ref)
                            ref = apply_border_ablation(ref)
                            try:
                                ref_resized = cv2.resize(ref, (bw, bh), interpolation=cv2.INTER_AREA)
                            except Exception:
                                # fall back to keeping original
                                new_label_strs.append(f"{orig_cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                                continue
                            # Slight brightness jitter
                            bri = random.uniform(0.92, 1.08)
                            ref_jit = np.clip(ref_resized.astype(np.float32) * bri, 0, 255).astype(np.uint8)
                            composite[y1:y2, x1:x2] = ref_jit
                            new_cls = fused_idx_map[new_char]
                            new_label_strs.append(f"{new_cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                            class_counts[new_char] += 1
                            continue
                    # Keep original char (no paste needed, bg already has it)
                    new_label_strs.append(f"{orig_cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

                if new_label_strs:
                    out.append((composite, new_label_strs, f"synth_{ctx}"))
                    ctx_count += 1
        print(f"[cross-ctx] {ctx}: {ctx_count} composites generated")

    return out



def build_template_driven_synth(
    fused_idx_map,
    ref_bundle,
    available_chars,
):
    """Generate synth using user-configured dashboard templates.

    Each template at data/synth_templates/<context>.json specifies:
      - sample_image: bg frame (path relative to synth_templates/)
      - slot_rects_norm: where to paste refs (normalized 0-1 coords)
      - ref_transform: crop region of ref + shape mask + scale
      - augmentation: UI overlay probs, border ablation, brightness jitter
      - synth_count: how many composites to generate per template

    Replaces the static_ui-detected schedule synth AND the cross-context
    manual-frame-as-bg synth with a single user-controlled pipeline.
    """
    out = []
    if not SYNTH_TEMPLATES_DIR.is_dir():
        print(f"[tpl-synth] no templates dir at {SYNTH_TEMPLATES_DIR}, skipping")
        return out
    class_counts = Counter()
    tpl_files = sorted(SYNTH_TEMPLATES_DIR.glob("*.json"))
    if not tpl_files:
        print("[tpl-synth] no template JSON files found, skipping")
        return out

    for tpl_path in tpl_files:
        try:
            template = json.loads(tpl_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[tpl-synth] {tpl_path.name}: failed to read ({e})")
            continue
        ctx_name = template.get("context") or tpl_path.stem
        slots_norm = template.get("slot_rects_norm") or []
        if not slots_norm:
            print(f"[tpl-synth] {ctx_name}: no slots configured, skipping")
            continue
        sample_rel = template.get("sample_image") or ""
        sample_path = SYNTH_TEMPLATES_DIR / sample_rel
        if not sample_path.exists():
            print(f"[tpl-synth] {ctx_name}: sample image missing ({sample_path}), skipping")
            continue
        bg = imread_u(sample_path)
        if bg is None:
            print(f"[tpl-synth] {ctx_name}: bg decode failed, skipping")
            continue
        H, W = bg.shape[:2]

        rt = template.get("ref_transform") or {}
        crop_n = rt.get("crop_n") or {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0}
        cx1 = float(crop_n.get("x1", 0.0)); cy1 = float(crop_n.get("y1", 0.0))
        cx2 = float(crop_n.get("x2", 1.0)); cy2 = float(crop_n.get("y2", 1.0))
        shape_kind = rt.get("shape", "square")
        radius_px = int(rt.get("radius_px", 4) or 0)
        scale_factor = float(rt.get("scale", 1.0))

        aug = template.get("augmentation") or {}
        ui_overlay_prob = float(aug.get("ui_overlay_prob", 0.5))
        ui_comp = aug.get("ui_components") or {}
        border_prob = float(aug.get("border_ablation_prob", 0.4))
        bri_jit = aug.get("brightness_jitter") or [0.92, 1.08]

        target_count = int(template.get("synth_count", 200) or 200)
        bbox_mode = template.get("bbox_mode", "full")  # full | tight_face | both
        use_for = template.get("use_for", "train")  # train | val | both
        n_emitted = 0

        # ── Round-robin char picker (guarantees every char appears in this
        # context before any char repeats): refill a shuffled pool when empty.
        # This is per-context, so each context sees every char ≈ N times where
        # N = (target_count × slots) / len(chars).  For schedule_popup (17 slots
        # × 200 composites = 3400 picks / 252 chars) → 13 picks per char min.
        ctx_char_pool: List[str] = []
        ctx_per_char_count: Counter = Counter()
        def _pick_char_balanced():
            nonlocal ctx_char_pool
            # Refill pool if empty (also respects global PER_CLASS_CAP)
            for _ in range(3):  # at most 3 refills
                if not ctx_char_pool:
                    ctx_char_pool = [c for c in available_chars
                                     if class_counts[c] < PER_CLASS_CAP]
                    random.shuffle(ctx_char_pool)
                if not ctx_char_pool:
                    return None  # all chars hit global cap
                cand = ctx_char_pool.pop()
                if class_counts[cand] < PER_CLASS_CAP:
                    return cand
            return None

        for variant_i in range(target_count):
            composite = bg.copy()
            label_lines = []
            for slot in slots_norm:
                x1p = int(slot["x1"] * W); y1p = int(slot["y1"] * H)
                x2p = int(slot["x2"] * W); y2p = int(slot["y2"] * H)
                sw = x2p - x1p; sh = y2p - y1p
                if sw < 8 or sh < 8:
                    continue

                char_name = _pick_char_balanced()
                if char_name is None:
                    continue
                ctx_per_char_count[char_name] += 1

                bundle = ref_bundle[char_name]
                if bundle.get("large") is not None:
                    ref = bundle["large"]
                elif bundle.get("small") is not None:
                    ref = bundle["small"]
                else:
                    continue
                ref = ref.copy()
                if ref.ndim == 3 and ref.shape[2] == 4:
                    ref = ref[:, :, :3]

                rh, rw = ref.shape[:2]
                rx1 = int(cx1 * rw); ry1 = int(cy1 * rh)
                rx2 = int(cx2 * rw); ry2 = int(cy2 * rh)
                if rx2 - rx1 >= 4 and ry2 - ry1 >= 4:
                    ref = ref[ry1:ry2, rx1:rx2]

                # NOTE: aug (UI overlay + border ablation) applied AFTER resize
                # below — at slot pixel resolution so effects are visible at
                # the scale model actually sees during inference.

                # ── Compute slot AABB (rect: x1..x2; quad: polygon AABB) ──
                quad = slot.get("quad")
                quad_px = None
                if quad and len(quad) == 4:
                    try:
                        quad_px = np.array(
                            [[float(q["x"]) * W, float(q["y"]) * H] for q in quad],
                            dtype=np.float32,
                        )
                        aabb_x1 = max(0, int(quad_px[:, 0].min()))
                        aabb_y1 = max(0, int(quad_px[:, 1].min()))
                        aabb_x2 = min(W, int(quad_px[:, 0].max()))
                        aabb_y2 = min(H, int(quad_px[:, 1].max()))
                    except Exception:
                        quad_px = None
                if quad_px is None:
                    aabb_x1, aabb_y1 = x1p, y1p
                    aabb_x2, aabb_y2 = x2p, y2p
                aabb_w = aabb_x2 - aabb_x1; aabb_h = aabb_y2 - aabb_y1
                if aabb_w < 8 or aabb_h < 8:
                    continue

                # Preserve ref aspect ratio (COVER: fill AABB, overflow clipped)
                rh0, rw0 = ref.shape[:2]
                if rh0 <= 0 or rw0 <= 0:
                    continue
                ref_ar = rw0 / rh0
                max_w = max(4, int(aabb_w * scale_factor))
                max_h = max(4, int(aabb_h * scale_factor))
                slot_ar = max_w / max(max_h, 1)
                if ref_ar > slot_ar:
                    target_h = max_h; target_w = max(4, int(target_h * ref_ar))
                else:
                    target_w = max_w; target_h = max(4, int(target_w / ref_ar))
                try:
                    resized = cv2.resize(ref, (target_w, target_h), interpolation=cv2.INTER_AREA)
                except Exception:
                    continue
                bri = random.uniform(float(bri_jit[0]), float(bri_jit[1]))
                resized = np.clip(resized.astype(np.float32) * bri, 0, 255).astype(np.uint8)

                # ── Apply aug AFTER resize at slot pixel resolution ──
                aug_positions = rt.get("aug_positions") or {}
                if random.random() < ui_overlay_prob:
                    resized = _apply_ui_overlay_per_template(resized, ui_comp, aug_positions)
                if random.random() < border_prob:
                    resized = apply_border_ablation(resized)

                # Shape mask (circle/rounded_rect) on resized ref
                ref_mask = None
                if shape_kind == "circle":
                    ref_mask = np.zeros((target_h, target_w), dtype=np.uint8)
                    cv2.circle(ref_mask, (target_w // 2, target_h // 2),
                               min(target_w, target_h) // 2, 255, -1)
                elif shape_kind == "rounded_rect" and radius_px > 0:
                    ref_mask = np.zeros((target_h, target_w), dtype=np.uint8)
                    r = min(radius_px, target_h // 2, target_w // 2)
                    cv2.rectangle(ref_mask, (r, 0), (target_w - r, target_h), 255, -1)
                    cv2.rectangle(ref_mask, (0, r), (target_w, target_h - r), 255, -1)
                    cv2.circle(ref_mask, (r, r), r, 255, -1)
                    cv2.circle(ref_mask, (target_w - r, r), r, 255, -1)
                    cv2.circle(ref_mask, (r, target_h - r), r, 255, -1)
                    cv2.circle(ref_mask, (target_w - r, target_h - r), r, 255, -1)

                # Center ref in AABB; clip overflow to AABB ∩ image bounds
                ref_left = aabb_x1 + (aabb_w - target_w) // 2
                ref_top  = aabb_y1 + (aabb_h - target_h) // 2
                clip_x1 = max(0, aabb_x1, ref_left)
                clip_y1 = max(0, aabb_y1, ref_top)
                clip_x2 = min(W, aabb_x2, ref_left + target_w)
                clip_y2 = min(H, aabb_y2, ref_top + target_h)
                if clip_x2 - clip_x1 < 4 or clip_y2 - clip_y1 < 4:
                    continue
                sx1 = clip_x1 - ref_left; sy1 = clip_y1 - ref_top
                sx2 = sx1 + (clip_x2 - clip_x1); sy2 = sy1 + (clip_y2 - clip_y1)
                source_slice = resized[sy1:sy2, sx1:sx2]

                if ref_mask is not None:
                    mask_slice = ref_mask[sy1:sy2, sx1:sx2]
                    bg_patch = composite[clip_y1:clip_y2, clip_x1:clip_x2].copy()
                    for cc in range(3):
                        bg_patch[..., cc] = np.where(mask_slice > 0, source_slice[..., cc], bg_patch[..., cc])
                    composite[clip_y1:clip_y2, clip_x1:clip_x2] = bg_patch
                else:
                    composite[clip_y1:clip_y2, clip_x1:clip_x2] = source_slice

                if quad_px is not None:
                    poly_full = np.zeros((H, W), dtype=np.uint8)
                    cv2.fillPoly(poly_full, [quad_px.astype(np.int32)], 255)
                    bg_aabb = bg[aabb_y1:aabb_y2, aabb_x1:aabb_x2]
                    local_poly = poly_full[aabb_y1:aabb_y2, aabb_x1:aabb_x2]
                    patch = composite[aabb_y1:aabb_y2, aabb_x1:aabb_x2]
                    patch[local_poly == 0] = bg_aabb[local_poly == 0]
                    composite[aabb_y1:aabb_y2, aabb_x1:aabb_x2] = patch

                # YOLO label: choose bbox style per template's bbox_mode
                #   "full"       = clip rect (slot AABB ~ where ref is pasted)
                #   "tight_face" = inner region matching how user labels battle
                #                  scenes (just the head/face, not slot card)
                #   "both"       = 50/50 random pick
                actual_mode = bbox_mode
                if actual_mode == "both":
                    actual_mode = random.choice(["full", "tight_face"])
                if actual_mode == "tight_face":
                    # Tight: inset 10% left/right, top 5% to 55% (upper-half face)
                    cw_full = clip_x2 - clip_x1
                    ch_full = clip_y2 - clip_y1
                    tx1 = clip_x1 + int(cw_full * 0.10)
                    tx2 = clip_x2 - int(cw_full * 0.10)
                    ty1 = clip_y1 + int(ch_full * 0.05)
                    ty2 = clip_y1 + int(ch_full * 0.55)
                    if tx2 - tx1 < 8 or ty2 - ty1 < 8:
                        # Tight crop too small → fall back to full
                        tx1, ty1, tx2, ty2 = clip_x1, clip_y1, clip_x2, clip_y2
                else:
                    tx1, ty1, tx2, ty2 = clip_x1, clip_y1, clip_x2, clip_y2
                cx_n = (tx1 + tx2) / 2 / W
                cy_n = (ty1 + ty2) / 2 / H
                bw_n = (tx2 - tx1) / W
                bh_n = (ty2 - ty1) / H
                cls_idx = fused_idx_map[char_name]
                label_lines.append(f"{cls_idx} {cx_n:.6f} {cy_n:.6f} {bw_n:.6f} {bh_n:.6f}")
                class_counts[char_name] += 1

            if label_lines:
                # Tag with use_for so downstream split routes to train/val
                tag = f"synth_tpl_{ctx_name}"
                if use_for == "val":
                    tag = f"synth_val_tpl_{ctx_name}"
                elif use_for == "both":
                    tag = f"synth_both_tpl_{ctx_name}"
                out.append((composite, label_lines, tag))
                n_emitted += 1

        # Per-context coverage stats: every char should appear at least N times
        if ctx_per_char_count:
            counts = [ctx_per_char_count[c] for c in available_chars]
            n_zero = sum(1 for c in available_chars if ctx_per_char_count[c] == 0)
            n_covered = sum(1 for c in available_chars if ctx_per_char_count[c] > 0)
            print(f"[tpl-synth] {ctx_name}: emitted {n_emitted} composites "
                  f"(target {target_count}, {len(slots_norm)} slots/image)")
            print(f"[tpl-synth] {ctx_name} coverage: "
                  f"min={min(counts)} max={max(counts)} avg={sum(counts)/len(counts):.1f} "
                  f"· covered {n_covered}/{len(available_chars)} chars · {n_zero} zero")
        else:
            print(f"[tpl-synth] {ctx_name}: emitted {n_emitted} composites")

    print(f"[tpl-synth] total {len(out)} composites across {len(tpl_files)} templates")
    print(f"[tpl-synth] class coverage: {len([c for c in class_counts if class_counts[c] > 0])} unique chars")
    return out


def _apply_ui_overlay_per_template(ref_img, ui_components, aug_positions=None):
    """Apply UI overlay aug with user-configured anchor positions.
    `aug_positions` = {lv:{x,y}, star:{x,y}, weapon:{x,y}, heart:{x,y}}
    where x/y are normalized 0-1 within the slot.  ±5% jitter added.
    """
    h, w = ref_img.shape[:2]
    out = ref_img.copy()
    AP = aug_positions or {}
    def _jit(v): return max(0.0, min(1.0, v + random.uniform(-0.05, 0.05)))

    if random.random() < float(ui_components.get("lv_text", 0.5)):
        pos = AP.get("lv") or {"x": 0.05, "y": 0.15}
        nx, ny = _jit(float(pos.get("x", 0.05))), _jit(float(pos.get("y", 0.15)))
        lv = random.randint(1, 90)
        text = f"Lv.{lv}" if random.random() < 0.75 else "MAX"
        font_scale = max(0.35, min(0.65, w / 100.0))
        tw = int(35 * font_scale); th = int(20 * font_scale)
        x = max(2, min(w - tw - 2, int(nx * w) - tw // 2))
        y = max(th, min(h - 2, int(ny * h) + th // 2))
        cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 0, 0), 2)
        cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 255), 1)
    if random.random() < float(ui_components.get("star", 0.3)):
        pos = AP.get("star") or {"x": 0.10, "y": 0.10}
        nx, ny = _jit(float(pos.get("x", 0.10))), _jit(float(pos.get("y", 0.10)))
        r = max(4, w // 12)
        cx = max(r + 1, min(w - r - 1, int(nx * w)))
        cy = max(r + 1, min(h - r - 1, int(ny * h)))
        cv2.circle(out, (cx, cy), r, (0, 220, 255), -1)
        cv2.circle(out, (cx, cy), r, (0, 80, 120), 1)
    if random.random() < float(ui_components.get("weapon_icon", 0.4)):
        pos = AP.get("weapon") or {"x": 0.85, "y": 0.85}
        nx, ny = _jit(float(pos.get("x", 0.85))), _jit(float(pos.get("y", 0.85)))
        size = max(10, w // 6)
        color = random.choice([(0, 0, 255), (0, 165, 255), (255, 128, 0),
                               (255, 0, 255), (255, 255, 0)])
        cx = max(size // 2, min(w - size // 2, int(nx * w)))
        cy = max(size // 2, min(h - size // 2, int(ny * h)))
        cv2.rectangle(out, (cx - size // 2, cy - size // 2),
                      (cx + size // 2, cy + size // 2), color, -1)
    if random.random() < float(ui_components.get("heart", 0.2)):
        pos = AP.get("heart") or {"x": 0.85, "y": 0.85}
        nx, ny = _jit(float(pos.get("x", 0.85))), _jit(float(pos.get("y", 0.85)))
        size = max(6, w // 7)
        cx = max(size + 1, min(w - size - 1, int(nx * w)))
        cy = max(size + 1, min(h - size - 1, int(ny * h)))
        cv2.circle(out, (cx, cy), size, (147, 20, 255), -1)
        cv2.circle(out, (cx, cy), size, (255, 255, 255), 2)
        num = str(random.randint(1, 99))
        font_scale = max(0.35, w / 130.0)
        cv2.putText(out, num, (max(0, cx - size + 1), min(h - 2, cy + 4)),
                    cv2.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 255), 1)
    if random.random() < float(ui_components.get("alpha_dim", 0.25)):
        alpha = random.uniform(0.55, 0.85)
        out = np.clip(out.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
    return out


def build_synthetic_samples(
    char_names: List[str],
    fused_idx_map: Dict[str, int],
    bg_frames: List[Path],
    refs: Dict[str, np.ndarray],
    target_per_class: int,
    ref_bundle: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
) -> List[Tuple[np.ndarray, List[str], str]]:
    """Generate synthetic composites.  Returns list of (img, yolo_lines, tag).

    If ref_bundle is provided, each paste draws between bundle["small"] and
    bundle["large"] according to USE_LARGE_REF_PROB — large refs give sharper
    downsampled faces with less aliasing than 54×59 upscaled.
    """
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
                # Pick small or large ref per paste — large gives sharper
                # downsample at slot size ~60×60, small is original.
                if ref_bundle and char_name in ref_bundle:
                    bundle = ref_bundle[char_name]
                    if bundle.get("large") is not None and random.random() < USE_LARGE_REF_PROB:
                        ref = bundle["large"]
                    elif bundle.get("small") is not None:
                        ref = bundle["small"]
                    else:
                        ref = bundle.get("large") or refs[char_name]
                else:
                    ref = refs[char_name]

                # ── Gemini Path B: adversarial augmentation ──
                # Apply UI overlay (Lv text / star / weapon / heart / dim)
                # + border ablation BEFORE paste, so model sees realistic
                # noise on top of the ref.
                ref = ref.copy()
                ref = apply_ui_overlay_aug(ref)
                ref = apply_border_ablation(ref)

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
        # Multi-source ref loading: combines CN-named harvested mini-crops,
        # EN-named mini-crops via student_name_map bridge, and EN-named LARGE
        # 404×456 portraits.  Per-char each may have "small" and/or "large".
        ref_bundle = load_refs_multi_source(fused_names)

        # Collapse into flat refs dict for build_synthetic_samples().  Per
        # paste we pick small or large according to USE_LARGE_REF_PROB.  The
        # synth function downsamples to slot size, so large refs give sharper
        # antialiased results.
        refs: Dict[str, np.ndarray] = {}
        for c, bundle in ref_bundle.items():
            # Default to small if available, else large
            chosen = bundle.get("small") if bundle.get("small") is not None else bundle.get("large")
            if chosen is not None:
                refs[c] = chosen
        # Stash the bundle on the args namespace for downstream access
        args._ref_bundle = ref_bundle

        available_chars = [c for c in fused_names if c in ref_bundle]

        # ── Synth path selection ──
        # If dashboard templates exist at data/synth_templates/*.json, use the
        # user-configured template-driven pipeline (preferred).  Otherwise fall
        # back to legacy static_ui-detect + cross-context manual-frame swap.
        templates_dir = SYNTH_TEMPLATES_DIR
        template_files = list(templates_dir.glob("*.json")) if templates_dir.is_dir() else []
        # Only count a template as "usable" if it has slots configured + sample image
        usable_templates = []
        for tp in template_files:
            try:
                td = json.loads(tp.read_text(encoding="utf-8"))
                if td.get("slot_rects_norm") and td.get("sample_image"):
                    sp = templates_dir / td["sample_image"]
                    if sp.exists():
                        usable_templates.append(tp.stem)
            except Exception:
                pass

        if usable_templates:
            print(f"[synth-mode] using DASHBOARD TEMPLATES ({len(usable_templates)} usable): "
                  f"{usable_templates}")
            synth_samples_data = build_template_driven_synth(
                fused_idx_map=fused_idx_map,
                ref_bundle=ref_bundle,
                available_chars=available_chars,
            )
        else:
            print("[synth-mode] no dashboard templates → falling back to legacy "
                  "(static_ui rooms + cross-context manual-frame swap)")
            bg_frames = find_schedule_popup_bg_frames(args.synth_bg_limit)
            print(f"[bg] found {len(bg_frames)} schedule popup backgrounds")
            synth_samples_data = build_synthetic_samples(
                fused_names, fused_idx_map, bg_frames, refs, args.per_class_cap,
                ref_bundle=ref_bundle,
            )
            from collections import Counter as _Counter
            shared_counts = _Counter()
            for _, lines, _ in synth_samples_data:
                for line in lines:
                    p = line.split()
                    if p and p[0].isdigit():
                        idx = int(p[0])
                        if 0 <= idx < len(fused_names):
                            shared_counts[fused_names[idx]] += 1
            cross_ctx_samples = build_cross_context_synth(
                manual_samples=manual_samples,
                ref_bundle=ref_bundle,
                fused_idx_map=fused_idx_map,
                available_chars=available_chars,
                target_per_class=args.per_class_cap,
                class_counts=shared_counts,
            )
            print(f"[cross-ctx] generated {len(cross_ctx_samples)} additional composites")
            synth_samples_data.extend(cross_ctx_samples)

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
    # Route synth samples by their tag (use_for=val or both routes some to val/)
    n_synth_train = n_synth_val = 0
    n_val_boxes_synth = 0
    for si, (composite, lines, tag) in enumerate(synth_samples_data):
        # Tags: "synth_tpl_<ctx>" → train only
        #       "synth_val_tpl_<ctx>" → val only
        #       "synth_both_tpl_<ctx>" → both (write to train + duplicate to val)
        stem = f"synth__{si:05d}"
        if tag.startswith("synth_val_"):
            imwrite_u(OUT_ROOT / "images/val" / (stem + ".jpg"), composite)
            (OUT_ROOT / "labels/val" / (stem + ".txt")).write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )
            n_synth_val += 1
            n_val_boxes_synth += len(lines)
        elif tag.startswith("synth_both_"):
            imwrite_u(OUT_ROOT / "images/train" / (stem + ".jpg"), composite)
            (OUT_ROOT / "labels/train" / (stem + ".txt")).write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )
            n_train_boxes += len(lines)
            n_synth_train += 1
            # Also write to val (same image, will get different stem to avoid collision)
            val_stem = f"synth_v__{si:05d}"
            imwrite_u(OUT_ROOT / "images/val" / (val_stem + ".jpg"), composite)
            (OUT_ROOT / "labels/val" / (val_stem + ".txt")).write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )
            n_synth_val += 1
            n_val_boxes_synth += len(lines)
        else:
            # Default: train
            imwrite_u(OUT_ROOT / "images/train" / (stem + ".jpg"), composite)
            (OUT_ROOT / "labels/train" / (stem + ".txt")).write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )
            n_train_boxes += len(lines)
            n_synth_train += 1
    for jpg in neg_train:
        stem = f"neg__{jpg.parent.name}__{jpg.stem}"
        shutil.copy2(jpg, OUT_ROOT / "images/train" / (stem + ".jpg"))
        (OUT_ROOT / "labels/train" / (stem + ".txt")).write_text("", encoding="utf-8")
    n_train_files = len(train_manual) + n_synth_train + len(neg_train)
    print(f"[emit] train: {n_train_files} files "
          f"({len(train_manual)} manual + {n_synth_train} synth + "
          f"{len(neg_train)} negatives), {n_train_boxes} positive boxes")
    if n_synth_val:
        print(f"[emit] val + {n_synth_val} synth-val frames ({n_val_boxes_synth} boxes)")

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
