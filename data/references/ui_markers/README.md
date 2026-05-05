# UI marker reference crops

Source-of-truth image crops for common BA HUD markers. Kept inside `data/`
so they ship with the packaged release.

| file | purpose | currently used by |
|---|---|---|
| `speed_1x_gray.png` | battle HUD speed button @ 1x (gray arrow) | HSV threshold only (not the image) |
| `speed_2x_blue.png` | battle HUD speed button @ 2x (blue arrow) | HSV threshold only |
| `speed_3x_yellow.png` | battle HUD speed button @ 3x (yellow arrow) | HSV threshold only |
| `auto_off_gray.png` | AUTO toggle OFF (gray background) | HSV threshold only |
| `auto_on_yellow.png` | AUTO toggle ON (yellow background) | HSV threshold only |
| `quest_cleared_book.png` | "story already read" book marker | **unused** — candidate for template match |
| `quest_cleared_star.png` | "quest cleared" yellow star badge | **unused** — OCR detects `★` / `★★★` on the same row today |

## Why the speed/auto PNGs exist even though HSV does the work

`brain/skills/event_activity.py::_classify_battle_btn` samples a ~5px patch
at the button position and classifies by HSV range:
- `gray`  : `S < 55`
- `yellow`: `15 ≤ H ≤ 35, S ≥ 120, V ≥ 160`
- `blue`  : `85 ≤ H ≤ 130, S ≥ 100`

These crops exist so a future maintainer can re-tune the thresholds
against ground-truth pixels without having to hunt for new captures.
If the game ever re-skins the HUD, re-crop + re-fit the thresholds here.

## Why `quest_cleared_star.png` isn't wired in yet

Trajectory inspection (run_20260424_024445 tick 95) shows OCR already
reads `★` and `★★★` boxes at `cx≈0.55` next to cleared node numbers,
so text-based detection covers the cleared-stage case. Template match is
a belt-and-suspenders fallback if OCR starts missing the unicode star.
