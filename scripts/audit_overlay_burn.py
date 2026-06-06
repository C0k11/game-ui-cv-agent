"""Audit trajectory runs for YoloOverlay-burned frames (top-HUD-band view).

Burn = a run captured with --game-overlay while the capture backend grabbed the
screen (so the topmost overlay window — vivid box outlines + dark RGB(30,30,30)
"classname conf" label bars — got composited into the screenshot). Useless /
poisonous as training data.

NO pixel heuristic separates burned from clean BA UI (the overlay lags one tick
so coords don't match; vivid edges, dark gray, and dark-on-light all occur in
clean UI / modal backdrops). The ONLY reliable tell is reading the labels — and
the highest-signal, most compact place to read them is the TOP HUD currency bar:
a burned frame draws "体力 0.xx / 信用点 0.xx / 青辉石 0.xx ..." on the (normally
clean) currency icons. This tool crops that band from one frame per run and
stacks them so you can eyeball many runs at once.

Burn is per-run (a manual launch flag), so one good frame per run classifies it
(the overlay is on for the whole run → every menu frame shows labels; one clean
top bar ⇒ clean run).

History: the only burned runs found were the 2026-05-28 overlay-testing session
(11 runs, deleted 2026-06-03). Everything else audited clean. Re-run this if you
ever capture new runs with --game-overlay.

Usage:
    py -3 scripts/audit_overlay_burn.py --alldates        # one run per date
    py -3 scripts/audit_overlay_burn.py --date 20260528   # every run of a date
    flags: --per N (runs per sheet) --pos F (frame pos in run)
Writes data/_burnaudit_<tag>_<n>.jpg (delete after reviewing).
"""
from __future__ import annotations
import argparse
import glob
import os
from collections import defaultdict
from pathlib import Path
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAJ = REPO_ROOT / "data" / "trajectories"
W = 1120
BAND = 0.12   # top fraction (HUD currency bar + the label bars drawn above it)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alldates", action="store_true")
    ap.add_argument("--date", default="")
    ap.add_argument("--pos", type=float, default=0.35, help="frame position within run")
    ap.add_argument("--per", type=int, default=8, help="bands per sheet")
    args = ap.parse_args()

    runs = sorted(d for d in glob.glob(str(TRAJ / "run_*")) if os.path.isdir(d))
    by_date = defaultdict(list)
    for rd in runs:
        by_date[os.path.basename(rd).replace("run_", "")[:8]].append(rd)

    def pick(rd):
        js = sorted(glob.glob(os.path.join(rd, "*.jpg")))
        return js[int(len(js) * args.pos)] if js else None

    items = []
    if args.alldates:
        for date in sorted(by_date):
            rds = by_date[date]
            j = pick(rds[len(rds) // 2])
            if j:
                items.append((f"{date}  ({len(rds)} runs)", j))
        tag = "alldates"
    elif args.date:
        for rd in by_date.get(args.date, []):
            j = pick(rd)
            if j:
                items.append((os.path.basename(rd), j))
        tag = args.date
    else:
        ap.error("pass --alldates or --date")

    bands = []
    for label, j in items:
        try:
            im = Image.open(j).convert("RGB")
        except Exception:
            continue
        w, h = im.size
        bands.append((label, im.crop((0, 0, w, int(h * BAND))).resize((W, int(W * BAND * h / w)))))

    lblh = 18
    for ci in range((len(bands) + args.per - 1) // args.per):
        chunk = bands[ci * args.per:(ci + 1) * args.per]
        totalh = sum(b.size[1] + lblh for _, b in chunk) + 8
        sheet = Image.new("RGB", (W, totalh), (15, 15, 15))
        dr = ImageDraw.Draw(sheet)
        y = 4
        for label, band in chunk:
            dr.text((4, y), label, fill=(0, 255, 0))
            sheet.paste(band, (0, y + lblh))
            y += band.size[1] + lblh
        suffix = f"_{ci}" if len(bands) > args.per else ""
        out = REPO_ROOT / "data" / f"_burnaudit_{tag}{suffix}.jpg"
        sheet.save(out, quality=84)
        print(f"chunk {ci}: {len(chunk)} bands -> {out} ({sheet.size})")


if __name__ == "__main__":
    main()
