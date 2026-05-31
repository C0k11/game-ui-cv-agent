"""Synthesize UI-model training frames from a synth template.

破局 for rare UI cls (前往/进入羁绊剧情, 学生momotalk信息未读, 侧栏图标…):
a fixed UI element only shows up in a handful of real frames, and oversample-
duplicating those few frames just makes the model MEMORIZE them (overfit) —
it never learns the element independent of its background. This composites the
element onto MANY different backgrounds instead:

  • SLOTS: paste random character avatars (data/captures/角色头像/) into the
    template's slot rects → infinite BACKGROUND diversity (different students
    behind the same UI).
  • UI STAMPS: the FIXED UI elements (the unread badge, the bond-story button)
    stay at their template-defined rects and are LABELLED in every frame with
    their cls. → the model learns "this element = this cls, on any background".

Template schema additions (set in the dashboard synth editor):
  "target": "ui"                       # mark this template for UI-cls synth
  "ui_stamps": [                       # fixed boxes labelled every frame
     {"cls": 439, "x1":.., "y1":.., "x2":.., "y2":..},   # normalized
     ...
  ]
  (existing slot_rects_norm / ref_transform / augmentation reused for the
   avatar paste + bg diversity.)

Output: data/raw_images/_synth_<ctx>/ (frame_*.jpg + frame_*.txt + classes.txt)
so it can be added to build_ui_dataset TRAIN_SOURCES.

Usage:
  py scripts/build_ui_synth.py momotalk --count 300
  py scripts/build_ui_synth.py bond_story --count 300 --seed 7
"""
from __future__ import annotations
import sys, json, glob, math
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
TPL_DIR = REPO / "data" / "synth_templates"
AVATAR_DIR = REPO / "data" / "captures" / "角色头像"
MASTER_CLASSES = REPO / "data" / "raw_images" / "_classes.txt"
OUT_BASE = REPO / "data" / "raw_images"

# Deterministic PRNG (seedable) — avoids Math.random-style nondeterminism.
_rng = np.random.RandomState(0)


def _load_avatars(limit=400):
    paths = sorted(glob.glob(str(AVATAR_DIR / "*.jpg")) + glob.glob(str(AVATAR_DIR / "*.png")))
    imgs = []
    for p in paths[:limit]:
        a = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
        if a is not None:
            imgs.append(a)
    return imgs


def _paste_avatar(bg, avatar, rect_norm, shape="circle", jitter=0.05):
    """Paste a (jittered, resized, optionally circle-masked) avatar into rect."""
    H, W = bg.shape[:2]
    x1 = rect_norm["x1"] + (_rng.uniform(-jitter, jitter) * (rect_norm["x2"] - rect_norm["x1"]))
    y1 = rect_norm["y1"] + (_rng.uniform(-jitter, jitter) * (rect_norm["y2"] - rect_norm["y1"]))
    x2 = x1 + (rect_norm["x2"] - rect_norm["x1"])
    y2 = y1 + (rect_norm["y2"] - rect_norm["y1"])
    px1, py1, px2, py2 = int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)
    px1, py1 = max(0, px1), max(0, py1)
    px2, py2 = min(W, px2), min(H, py2)
    bw, bh = px2 - px1, py2 - py1
    if bw < 4 or bh < 4:
        return
    av = cv2.resize(avatar, (bw, bh), interpolation=cv2.INTER_AREA)
    if shape == "circle":
        mask = np.zeros((bh, bw), np.uint8)
        cv2.ellipse(mask, (bw // 2, bh // 2), (bw // 2, bh // 2), 0, 0, 360, 255, -1)
        roi = bg[py1:py2, px1:px2]
        m3 = mask[..., None].astype(np.float32) / 255.0
        bg[py1:py2, px1:px2] = (av * m3 + roi * (1 - m3)).astype(np.uint8)
    else:
        bg[py1:py2, px1:px2] = av


def synth_context(ctx: str, count: int = 300, seed: int = 0, val: bool = False):
    global _rng
    _rng = np.random.RandomState(seed)
    tpl_path = TPL_DIR / f"{ctx}.json"
    if not tpl_path.exists():
        print(f"template not found: {tpl_path}")
        return
    tpl = json.loads(tpl_path.read_text(encoding="utf-8"))
    stamps = tpl.get("ui_stamps") or []
    if not stamps:
        print(f"[WARN] template '{ctx}' has no ui_stamps — synth frames would have "
              f"NO labels. Place UI stamps in the dashboard synth editor first.")
        return
    sample_rel = tpl.get("sample_image") or f"samples/{ctx}.jpg"
    sample_path = TPL_DIR / sample_rel
    bg0 = cv2.imdecode(np.fromfile(str(sample_path), np.uint8), cv2.IMREAD_COLOR)
    if bg0 is None:
        print(f"sample image unreadable: {sample_path}")
        return
    slots = tpl.get("slot_rects_norm") or []
    shape = (tpl.get("ref_transform") or {}).get("shape", "circle")
    bj = (tpl.get("augmentation") or {}).get("brightness_jitter", [0.92, 1.08])
    avatars = _load_avatars()
    if not avatars and slots:
        print("[WARN] no avatars found — frames will all use the sample bg (low diversity)")

    out_dir = OUT_BASE / f"_synth_{ctx}{'_val' if val else ''}"
    out_dir.mkdir(parents=True, exist_ok=True)
    # seed classes.txt so it's a valid dataset source
    if MASTER_CLASSES.exists():
        (out_dir / "classes.txt").write_text(MASTER_CLASSES.read_text(encoding="utf-8"), encoding="utf-8")

    written = 0
    for i in range(count):
        bg = bg0.copy()
        # paste a random avatar into each slot → background diversity
        for s in slots:
            if avatars:
                av = avatars[_rng.randint(0, len(avatars))]
                _paste_avatar(bg, av, s, shape=shape)
        # brightness jitter (立绘 lighting varies)
        f = float(_rng.uniform(bj[0], bj[1]))
        bg = np.clip(bg.astype(np.float32) * f, 0, 255).astype(np.uint8)
        # write frame + label (the fixed UI stamps)
        stem = f"frame_{i:05d}"
        cv2.imwrite(str(out_dir / f"{stem}.jpg"), bg, [cv2.IMWRITE_JPEG_QUALITY, 90])
        lines = []
        for st in stamps:
            cls = int(st["cls"])
            cx = (st["x1"] + st["x2"]) / 2.0
            cy = (st["y1"] + st["y2"]) / 2.0
            w = st["x2"] - st["x1"]
            h = st["y2"] - st["y1"]
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        (out_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        written += 1
    print(f"synth '{ctx}': wrote {written} frames + labels to {out_dir}")
    print(f"  stamps: {[(int(s['cls'])) for s in stamps]}  slots: {len(slots)}  avatars: {len(avatars)}")
    print(f"  → add '_synth_{ctx}' to build_ui_dataset TRAIN_SOURCES")


def _crop_sprites(src_run: str, classes):
    """Crop every labelled instance of each cls from src_run → list of
    (cls, sprite_bgr, (cx,cy,w,h) norm). These are the real UI-element pixels."""
    d = OUT_BASE / src_run
    want = set(int(c) for c in classes)
    sprites = []
    for txt in sorted(glob.glob(str(d / "frame_*.txt"))):
        jpg = Path(txt).with_suffix(".jpg")
        if not jpg.exists():
            continue
        rows = []
        for ln in Path(txt).read_text(encoding="utf-8").strip().splitlines():
            p = ln.split()
            if len(p) < 5:
                continue
            try:
                c = int(p[0])
            except ValueError:
                continue
            if c in want:
                rows.append((c, *map(float, p[1:5])))
        if not rows:
            continue
        img = cv2.imdecode(np.fromfile(str(jpg), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        H, W = img.shape[:2]
        for c, cx, cy, w, h in rows:
            x1, y1 = int((cx - w / 2) * W), int((cy - h / 2) * H)
            x2, y2 = int((cx + w / 2) * W), int((cy + h / 2) * H)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 - x1 < 6 or y2 - y1 < 6:
                continue
            sprites.append((c, img[y1:y2, x1:x2].copy(), (cx, cy, w, h)))
    return sprites


def _bg_pool(limit=1500, exclude_run=None):
    """Diverse real backgrounds: sampled trajectory frames + lobby_diverse."""
    paths = sorted(glob.glob(str(REPO / "data" / "trajectories" / "run_*" / "*.jpg")))
    if paths:
        step = max(1, len(paths) // limit)
        paths = paths[::step][:limit]
    paths += glob.glob(str(REPO / "data" / "captures" / "lobby_diverse_*" / "*.jpg"))
    return paths


def synth_overlay(classes, src_run="run_20260529_123209", count=300,
                  out_name="overlay", seed=0, jitter=0.04):
    """OVERLAY synth (破局 for buttons/icons that are rare in real frames but
    fixed in appearance): crop the real UI-element sprite from src_run, paste it
    onto MANY diverse real backgrounds at its normalized position (size kept
    relative to bg, edges feathered, ±jitter), label it. Gives the model the
    element on hundreds of backgrounds — what oversample-dup / avatar-slot can't.
    Output: data/raw_images/_synth_<out_name>/."""
    global _rng
    _rng = np.random.RandomState(seed)
    sprites = _crop_sprites(src_run, classes)
    if not sprites:
        print(f"no sprites found for {classes} in {src_run}")
        return
    bgs = _bg_pool()
    if not bgs:
        print("no background frames found")
        return
    out_dir = OUT_BASE / f"_synth_{out_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if MASTER_CLASSES.exists():
        (out_dir / "classes.txt").write_text(MASTER_CLASSES.read_text(encoding="utf-8"), encoding="utf-8")
    written = 0
    for i in range(count):
        bgp = bgs[_rng.randint(0, len(bgs))]
        bg = cv2.imdecode(np.fromfile(bgp, np.uint8), cv2.IMREAD_COLOR)
        if bg is None:
            continue
        H, W = bg.shape[:2]
        cls, spr, (cx, cy, w, h) = sprites[_rng.randint(0, len(sprites))]
        # keep the sprite's normalized size relative to THIS bg
        tw, th = max(6, int(w * W)), max(6, int(h * H))
        spr_r = cv2.resize(spr, (tw, th), interpolation=cv2.INTER_AREA)
        jcx = cx + _rng.uniform(-jitter, jitter)
        jcy = cy + _rng.uniform(-jitter, jitter)
        px1 = int(jcx * W - tw / 2); py1 = int(jcy * H - th / 2)
        px1 = max(0, min(W - tw, px1)); py1 = max(0, min(H - th, py1))
        # feathered alpha to soften the rectangular seam
        mask = np.ones((th, tw), np.float32)
        fb = max(1, int(min(tw, th) * 0.06))
        mask[:fb, :] *= np.linspace(0, 1, fb)[:, None]
        mask[-fb:, :] *= np.linspace(1, 0, fb)[:, None]
        mask[:, :fb] *= np.linspace(0, 1, fb)[None, :]
        mask[:, -fb:] *= np.linspace(1, 0, fb)[None, :]
        roi = bg[py1:py1 + th, px1:px1 + tw].astype(np.float32)
        m3 = mask[..., None]
        bg[py1:py1 + th, px1:px1 + tw] = (spr_r * m3 + roi * (1 - m3)).astype(np.uint8)
        f = float(_rng.uniform(0.9, 1.1))
        bg = np.clip(bg.astype(np.float32) * f, 0, 255).astype(np.uint8)
        # label = the pasted sprite's actual position/size on this bg
        ncx = (px1 + tw / 2) / W; ncy = (py1 + th / 2) / H
        nw = tw / W; nh = th / H
        stem = f"frame_{i:05d}"
        cv2.imwrite(str(out_dir / f"{stem}.jpg"), bg, [cv2.IMWRITE_JPEG_QUALITY, 90])
        (out_dir / f"{stem}.txt").write_text(
            f"{cls} {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}\n", encoding="utf-8")
        written += 1
    from collections import Counter
    sc = Counter(c for c, _, _ in sprites)
    print(f"overlay synth '_synth_{out_name}': wrote {written} frames")
    print(f"  sprites: {dict(sc)}  bg pool: {len(bgs)}")
    print(f"  → add '_synth_{out_name}' to build_ui_dataset TRAIN_SOURCES")


def main():
    argv = sys.argv[1:]
    if not argv:
        print("usage: build_ui_synth.py <context> [--count 300] [--seed 0] [--val]")
        print("   or: build_ui_synth.py --overlay <cls1,cls2> --src <run> --out <name> [--count N]")
        return
    if argv[0] == "--overlay":
        classes = [int(x) for x in argv[1].split(",")]
        src = "run_20260529_123209"; out = "overlay"; count = 300; seed = 0
        i = 2
        while i < len(argv):
            if argv[i] == "--src": src = argv[i + 1]; i += 2; continue
            if argv[i] == "--out": out = argv[i + 1]; i += 2; continue
            if argv[i] == "--count": count = int(argv[i + 1]); i += 2; continue
            if argv[i] == "--seed": seed = int(argv[i + 1]); i += 2; continue
            i += 1
        synth_overlay(classes, src_run=src, count=count, out_name=out, seed=seed)
        return
    ctx = argv[0]; count = 300; seed = 0; val = False
    i = 1
    while i < len(argv):
        if argv[i] == "--count": count = int(argv[i + 1]); i += 2; continue
        if argv[i] == "--seed": seed = int(argv[i + 1]); i += 2; continue
        if argv[i] == "--val": val = True; i += 1; continue
        i += 1
    synth_context(ctx, count=count, seed=seed, val=val)


if __name__ == "__main__":
    main()
