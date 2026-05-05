"""Compare AvatarMatcher with old wiki templates vs new MomoTalk-derived
templates on real schedule trajectory frames. Measures accuracy + speed.

# -*- coding: utf-8 -*-

Usage: py scripts/test_avatar_matcher_swap.py [run_dir]

Picks ticks where schedule.py runs roster scan (sub_state=check_roster
during scanning roster avatars).
"""
import sys
import json
import time
import glob
import os
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vision.avatar_matcher import AvatarMatcher  # noqa


REGIONS = json.load(open(ROOT / "data" / "schedule_avatar_regions.json", "r", encoding="utf-8"))
STRIPS = REGIONS["strips"]
CELLS_PER_ROOM = REGIONS.get("cells_per_room", 4)

# Default favorites from app_config.json
APP_CONFIG = json.load(open(ROOT / "data" / "app_config.json", "r", encoding="utf-8"))
FAV_RAW = APP_CONFIG["profiles"]["default"]["target_favorites"]
FAVS = [n[:-4] if n.lower().endswith(".png") else n for n in FAV_RAW]


def load_tick_image(path: str) -> np.ndarray:
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)


def extract_cells(img: np.ndarray):
    """Yield (room_idx, slot, cell_bgr)."""
    h, w = img.shape[:2]
    for room_idx, s in enumerate(STRIPS):
        x1, y1, x2, y2 = (
            int(s["x1"] * w), int(s["y1"] * h),
            int(s["x2"] * w), int(s["y2"] * h),
        )
        strip = img[y1:y2, x1:x2]
        if strip.size == 0:
            continue
        sh, sw = strip.shape[:2]
        cell_w = max(1, sw // CELLS_PER_ROOM)
        cell_size = min(cell_w, sh)
        if cell_size < 16:
            continue
        sy = max(0, (sh - cell_size) // 2)
        for slot in range(CELLS_PER_ROOM):
            sx = slot * cell_w + max(0, (cell_w - cell_size) // 2)
            if sx + cell_size > sw:
                break
            cell = strip[sy:sy + cell_size, sx:sx + cell_size]
            if cell.size == 0:
                continue
            yield room_idx, slot, cell


def find_roster_ticks(run_dir: str, max_ticks: int = 6):
    """Pick ticks where schedule's roster is on screen."""
    files = sorted(glob.glob(os.path.join(run_dir, "tick_*.json")))
    out = []
    for f in files:
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        if d.get("skill") != "Schedule":
            continue
        if d.get("sub_state") != "check_roster":
            continue
        a = d.get("action") or {}
        if a.get("action") != "wait":
            continue
        if "scanning roster avatars" not in (a.get("reason") or ""):
            continue
        jpg = f.replace(".json", ".jpg")
        if os.path.exists(jpg):
            out.append((d["tick"], jpg))
        if len(out) >= max_ticks:
            break
    return out


_THRESHOLD = 0.35  # mirrors schedule.py::_AVATAR_MATCH_THRESHOLD


def benchmark(matcher: AvatarMatcher, label: str, frames):
    print(f"\n=== {label} ===")
    print(f"  templates: {len(matcher.all_template_names())}")
    fav_set = set(FAVS)
    all_names = matcher.all_template_names()
    # Warm-up: load all template histograms once (one-time cost not measured)
    t0 = time.perf_counter()
    for n in all_names:
        matcher._get_avatar_data(n)
    print(f"  warm-up template load: {(time.perf_counter()-t0)*1000:.1f} ms")

    total_cells = 0
    total_match_us = 0.0
    fav_hits = 0
    per_tick = []
    for tick, jpg in frames:
        img = load_tick_image(jpg)
        if img is None:
            continue
        tick_cells = 0
        tick_matches = []
        t_tick0 = time.perf_counter()
        for room, slot, cell in extract_cells(img):
            tick_cells += 1
            t0 = time.perf_counter()
            fav_m, fav_s, all_m, all_s = matcher.match_avatar_open_set(cell, fav_set, all_names)
            total_match_us += (time.perf_counter() - t0) * 1e6
            total_cells += 1
            accepted = (all_m in fav_set and all_m == fav_m
                        and fav_s > _THRESHOLD)
            if accepted:
                fav_hits += 1
            tick_matches.append((room, slot, all_m, all_s, fav_m, fav_s, accepted))
        per_tick.append((tick, time.perf_counter() - t_tick0, tick_cells, tick_matches))

    if total_cells:
        print(f"  cells scored: {total_cells}, fav-accepted: {fav_hits} ({fav_hits/total_cells*100:.1f}%)")
        print(f"  avg per cell: {total_match_us/total_cells:.0f} us")
        for tick, dur, n, _ in per_tick:
            print(f"  t{tick}: {dur*1000:.1f} ms total ({n} cells, {dur/n*1000 if n else 0:.1f} ms/cell)")
        # Print first frame's matches in detail
        if per_tick and per_tick[0][3]:
            tick, _, _, mts = per_tick[0]
            print(f"  --- t{tick} per-cell ---")
            for room, slot, all_m, all_s, fav_m, fav_s, ok in mts:
                flag = "★FAV" if ok else ""
                print(f"    r{room}s{slot}: top={all_m}/{all_s:.2f} fav={fav_m}/{fav_s:.2f} {flag}")
    return total_cells, fav_hits, total_match_us


def run():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else \
        str(ROOT / "data" / "trajectories" / "run_20260428_175656")
    print(f"run_dir: {run_dir}")
    frames = find_roster_ticks(run_dir, max_ticks=6)
    if not frames:
        print("No roster scan ticks found in run.")
        return
    print(f"Picked {len(frames)} roster ticks: {[t for t,_ in frames]}")

    avatars_dir = ROOT / "data" / "captures" / "角色头像"

    wiki_dir = str(avatars_dir.parent / "角色头像_crop")
    momo_dir = str(avatars_dir.parent / "角色头像_crop_from_momotalk_renamed")

    # Final config = current defaults (REF_SIZE=64, top_k=15)
    m = AvatarMatcher(str(avatars_dir), crop_dir=wiki_dir)
    benchmark(m, f"FINAL DEFAULTS (REF_SIZE={m._REF_SIZE}, top_k={m.STAGE2_TOP_K})", frames)


if __name__ == "__main__":
    run()
