"""Multi-domain avatar synth for the UNIFIED 26x model.

Fixes the negative-label hazard WITHOUT teacher-patching or idx-remapping:
instead of taking fused_avatar_v1 (avatar-only labels) and bolting UI on, we
RE-SYNTHESIZE from frames that ALREADY carry BOTH ui + avatar master labels —
keep every UI/emoticon box's label intact, and rotate the AVATAR boxes through
all 252 characters. Output frames are multi-domain (ui+avatar) by construction,
master-idx, so:
  • no negative-label毒化 (UI boxes are labelled)
  • no fused_avatar局部→master remap (we emit master idx directly)
  • no teacher pass (background frame already carries the UI labels)
  • bonus: label a few dozen multi-domain momo/cafe frames → synth covers all 252
    characters by rotating refs into the avatar slots.

Background source = raw_images TRAIN frames (excl. _-dirs and VAL_SOURCES) that
have ≥1 avatar box (master 143-393) AND ≥1 UI box. Reuses ref loading + ref
augmentation from build_fused_avatar_dataset.

Output: data/raw_images/_synth_avatar_md/  → add to build_ui_v2 SYNTH_SOURCES.
Usage: py scripts/build_avatar_md_synth.py [--per-char 80] [--include-val(测逻辑)]
"""
from __future__ import annotations
import sys, random, json
from pathlib import Path
from collections import Counter

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
RAW = REPO / "data" / "raw_images"
OUT = RAW / "_synth_avatar_md"

from scripts.build_fused_avatar_dataset import (  # noqa: E402
    load_master, load_refs_multi_source, apply_ui_overlay_aug,
    apply_border_ablation, classify_ctx_by_norm_boxes, imread_u,
)
from scripts.build_ui_v2 import VAL_SOURCES  # noqa: E402

AVATAR_LO, AVATAR_HI = 143, 394   # master 角色段含柚子战斗(394, fused 第252类); 395+ 才是 UI-B
EMOTICON_IDX = 451
REPLACE_PROB = 0.55               # per avatar-box: chance to swap in a fresh char
USE_LARGE_REF_PROB = 0.5
N_VARIANTS = {"student_list": 5, "momotalk": 4, "battle_squad": 5,
              "cafe_invite": 4, "other": 3, "schedule": 2}

# 每个 context 的贴头像参数, 复用 fused 的 synth_templates (fused 用对了所以 0.966).
# circle: momotalk/cafe_invite ; square: schedule/student_list/battle_squad/tactical
_CTX2TMPL = {"momotalk": "momotalk", "cafe_invite": "cafe_invite", "student_list": "student_list",
             "battle_squad": "battle_squad", "schedule": "schedule_popup", "other": "momotalk"}
_TMPL_CACHE: dict = {}
_LARGE_FACE_CROP = (0.13, 0.02, 0.87, 0.52)   # large 半身立绘截脸(脸在上部中间); small 已是脸不用


def _ref_tf(ctx: str) -> dict:
    name = _CTX2TMPL.get(ctx, "momotalk")
    if name not in _TMPL_CACHE:
        p = REPO / "data" / "synth_templates" / f"{name}.json"
        rt = json.loads(p.read_text(encoding="utf-8")).get("ref_transform", {}) if p.exists() else {}
        _TMPL_CACHE[name] = rt
    return _TMPL_CACHE[name]


def _paste_face(comp, ref, x1, y1, x2, y2, shape_kind, crop_n, rng):
    """对齐 fused: crop截脸 + 保长宽比COVER(不压扁) + shape mask. 返回 True=贴成功."""
    if ref is None:
        return False
    if ref.ndim == 3 and ref.shape[2] == 4:
        ref = ref[:, :, :3]
    rh, rw = ref.shape[:2]
    cx1, cy1, cx2, cy2 = crop_n
    rx1, ry1, rx2, ry2 = int(cx1 * rw), int(cy1 * rh), int(cx2 * rw), int(cy2 * rh)
    if rx2 - rx1 >= 4 and ry2 - ry1 >= 4:
        ref = ref[ry1:ry2, rx1:rx2]
    rh, rw = ref.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    if rh < 4 or rw < 4 or bw < 8 or bh < 8:
        return False
    # 保长宽比 COVER: 填满 box, 多余裁掉 (不压扁半身像)
    ar = rw / rh; bar = bw / max(bh, 1)
    if ar > bar:
        th = bh; tw = max(4, int(round(th * ar)))
    else:
        tw = bw; th = max(4, int(round(tw / ar)))
    rr = cv2.resize(ref, (tw, th), interpolation=cv2.INTER_AREA)
    ox = max(0, (tw - bw) // 2); oy = max(0, (th - bh) // 2)
    rr = rr[oy:oy + bh, ox:ox + bw]
    if rr.shape[0] != bh or rr.shape[1] != bw:
        rr = cv2.resize(rr, (bw, bh), interpolation=cv2.INTER_AREA)
    bri = rng.uniform(0.92, 1.08)
    rr = np.clip(rr.astype(np.float32) * bri, 0, 255).astype(np.uint8)
    if shape_kind == "circle":
        mask = np.zeros((bh, bw), dtype=np.uint8)
        cv2.circle(mask, (bw // 2, bh // 2), min(bw, bh) // 2, 255, -1)
        patch = comp[y1:y2, x1:x2]
        for cc in range(3):
            patch[..., cc] = np.where(mask > 0, rr[..., cc], patch[..., cc])
    else:  # square: 截脸保比例后直接贴
        comp[y1:y2, x1:x2] = rr
    return True


def _is_avatar(c: int) -> bool:
    return AVATAR_LO <= c <= AVATAR_HI


def collect_md_bg(include_val: bool):
    """raw_images TRAIN frames with >=1 avatar box AND >=1 UI box (master idx)."""
    val = set() if include_val else set(VAL_SOURCES)
    bgs = []
    for ds in sorted(RAW.iterdir()):
        if not ds.is_dir() or ds.name.startswith("_") or ds.name in val:
            continue
        for txt in ds.glob("*.txt"):
            if txt.name == "classes.txt":
                continue
            boxes = []
            for ln in txt.read_text(encoding="utf-8", errors="ignore").splitlines():
                p = ln.split()
                if len(p) < 5:
                    continue
                try:
                    c = int(p[0]); cx, cy, w, h = (float(x) for x in p[1:5])
                except ValueError:
                    continue
                boxes.append((c, cx, cy, w, h))
            has_av = any(_is_avatar(b[0]) for b in boxes)
            has_ui = any((not _is_avatar(b[0])) and b[0] != EMOTICON_IDX for b in boxes)
            if has_av and has_ui:
                jpg = txt.with_suffix(".jpg")
                if jpg.exists():
                    bgs.append((jpg, boxes))
    return bgs


def main():
    argv = sys.argv[1:]
    per_char = 80
    include_val = False
    i = 0
    while i < len(argv):
        if argv[i] == "--per-char":
            per_char = int(argv[i + 1]); i += 2; continue
        if argv[i] == "--include-val":
            include_val = True; i += 1; continue
        i += 1

    master = load_master()
    avatar_names = master[AVATAR_LO:AVATAR_HI + 1]
    name2master = {n: AVATAR_LO + k for k, n in enumerate(avatar_names)}
    ref_bundle = load_refs_multi_source(avatar_names)
    avail = [n for n in avatar_names if n in ref_bundle]
    bgs = collect_md_bg(include_val)
    print(f"[md-synth] multi-domain bg frames: {len(bgs)}  |  chars with refs: {len(avail)}/{len(avatar_names)}"
          + ("  (include_val=测逻辑)" if include_val else "  (train only)"))
    if not bgs:
        print("  → 还没有 train 多域帧(需 UI+头像都标的 train 帧, 主要靠 v6weak 补头像那批)。等补全。")
        print("  → 想先验证逻辑可加 --include-val (拿 171121 等 val 帧测, 别用于正式产出)")
        return
    if not avail:
        print("  → 没加载到角色 ref (检查 data/captures/角色头像 大图)。")
        return

    if OUT.exists():
        import shutil as _sh; _sh.rmtree(OUT)  # 清旧帧防残留
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "classes.txt").write_text("\n".join(master) + "\n", encoding="utf-8")
    rng = random.Random(0)
    cls_count: Counter = Counter()
    written = 0
    for jpg, boxes in bgs:
        av_norm = [(b[1], b[2], b[3], b[4]) for b in boxes if _is_avatar(b[0])]
        ctx = classify_ctx_by_norm_boxes(av_norm)
        n_var = N_VARIANTS.get(ctx, 3)
        bg = imread_u(jpg)
        if bg is None:
            continue
        H, W = bg.shape[:2]
        for _vi in range(n_var):
            comp = bg.copy()
            labels = []
            for c, cx, cy, w, h in boxes:
                if not _is_avatar(c):
                    labels.append(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")  # UI/摸头框: 原样保留
                    continue
                x1, y1 = int((cx - w / 2) * W), int((cy - h / 2) * H)
                x2, y2 = int((cx + w / 2) * W), int((cy + h / 2) * H)
                bw, bh = x2 - x1, y2 - y1
                if bw < 16 or bh < 16 or rng.random() >= REPLACE_PROB:
                    labels.append(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")  # 保留原头像
                    continue
                new = None
                for _ in range(12):
                    cand = rng.choice(avail)
                    if cls_count[cand] < per_char:
                        new = cand; break
                if new is None:
                    labels.append(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                    continue
                b = ref_bundle[new]
                rt = _ref_tf(ctx)
                shape_kind = rt.get("shape", "circle")
                cn = rt.get("crop_n", {})
                small = b.get("small"); large = b.get("large")
                # 优先 small(脸特写, 用 template crop_n); 缺则 large(激进截脸到脸部)
                if small is not None:
                    ref = apply_border_ablation(apply_ui_overlay_aug(small.copy()))
                    crop_n = (cn.get("x1", 0.0), cn.get("y1", 0.0), cn.get("x2", 1.0), cn.get("y2", 1.0))
                elif large is not None:
                    ref = apply_border_ablation(apply_ui_overlay_aug(large.copy()))
                    crop_n = _LARGE_FACE_CROP
                else:
                    labels.append(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                    continue
                if not _paste_face(comp, ref, x1, y1, x2, y2, shape_kind, crop_n, rng):
                    labels.append(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                    continue
                labels.append(f"{name2master[new]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                cls_count[new] += 1
            stem = f"frame_{written:06d}"
            cv2.imwrite(str(OUT / f"{stem}.jpg"), comp, [cv2.IMWRITE_JPEG_QUALITY, 90])
            (OUT / f"{stem}.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
            written += 1

    print(f"[md-synth] wrote {written} frames → {OUT}")
    print(f"  char coverage: {len(cls_count)}/{len(avail)}  (avg {sum(cls_count.values())//max(1,len(cls_count))}/char)")
    print(f"  → add '_synth_avatar_md' to build_ui_v2 SYNTH_SOURCES")


if __name__ == "__main__":
    main()
