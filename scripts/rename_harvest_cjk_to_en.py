"""Rename CJK-named harvest crops to canonical English names via Schale DB.

Input:  data/captures/角色头像_crop_harvested_named/<CJK_name>.png
Output: data/captures/角色头像_crop_from_momotalk_renamed/<EN_name>.png

Also writes data/captures/harvest_name_map.json for auditing.

Uses schaledb.com as the source of truth for EN↔CJK student name pairs.
Handles Trad (tw) + Simp (cn) + mixed. Applies post-rename aliases so
the output matches the user's existing favorite-file convention
(`Hanako_(Swimsuit).png`) rather than Schale's slightly different form.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from vision.io_utils import imread_any, imwrite_any  # noqa: E402
from vision.ocr_normalize import normalize as cjk_normalize  # noqa: E402

SRC = REPO / "data" / "captures" / "角色头像_crop_harvested_named"
DST = REPO / "data" / "captures" / "角色头像_crop_from_momotalk_renamed"
MAP_OUT = REPO / "data" / "captures" / "harvest_name_map.json"


def fetch_schale(lang: str) -> dict:
    url = f"https://schaledb.com/data/{lang}/students.min.json"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def strip_brackets(s: str) -> str:
    """Remove all CJK and ASCII brackets + their content. Returns base name."""
    out = []
    depth = 0
    for ch in s:
        if ch in "(（[【":
            depth += 1
            continue
        if ch in ")）]】":
            depth = max(0, depth - 1)
            continue
        if depth == 0:
            out.append(ch)
    return "".join(out).strip()


def normalize_variant_aliases(en_name: str) -> str:
    """Post-process Schale's en.Name to match the user's filename convention.

    - Schale uses `Neru (Bunny)`; user's favorite file is `Toki_(Bunny_Girl).png`
    - Schale uses `Aris`; user's favorite is `Arisu`
    Both conventions normalized here so renamed files are drop-in replacements.
    """
    name = en_name.strip()
    # Variant bracket aliases
    variant_aliases = {
        "(Bunny)": "(Bunny_Girl)",
        "(Hot Spring)": "(Hot_Spring)",
        "(New Year)": "(New_Year)",
        "(Cheer Squad)": "(Cheer_Squad)",
        "(Swim Team)": "(Swim_Team)",
    }
    for src, tgt in variant_aliases.items():
        if src in name:
            name = name.replace(src, tgt)
    # Student name spelling aliases (Schale ↔ user conventions)
    # Applied to the base segment BEFORE the space/paren
    name_aliases = {
        "Aris ": "Arisu ",
        "Aris(": "Arisu(",
    }
    if name.startswith("Aris"):
        if name == "Aris":
            name = "Arisu"
        else:
            for src, tgt in name_aliases.items():
                name = name.replace(src, tgt)
    return name.replace(" ", "_")


def build_map() -> dict[str, str]:
    """CJK (tw/cn/stripped) name → normalized EN filename-stem."""
    print("fetching Schale data (en/tw/cn)...", flush=True)
    en = fetch_schale("en")
    tw = fetch_schale("tw")
    cn = fetch_schale("cn")
    print(f"  en={len(en)} tw={len(tw)} cn={len(cn)} students", flush=True)
    cjk_to_en: dict[str, str] = {}
    base_en: dict[str, str] = {}  # base-name index (TW/CN → EN base)
    for sid, e in en.items():
        en_name = e.get("Name", "")
        if not en_name:
            continue
        en_norm = normalize_variant_aliases(en_name)
        tw_name = tw[sid]["Name"] if sid in tw else ""
        cn_name = cn[sid]["Name"] if sid in cn else ""
        # Primary mapping: exact TW / CN → EN
        if tw_name:
            cjk_to_en.setdefault(tw_name, en_norm)
        if cn_name:
            cjk_to_en.setdefault(cn_name, en_norm)
        # Also map the BASE (bracketless) form if this is a base student
        if "(" not in en_name and "（" not in en_name:
            if tw_name and "(" not in tw_name and "（" not in tw_name:
                base_en.setdefault(tw_name, en_norm)
            if cn_name and "(" not in cn_name and "（" not in cn_name:
                base_en.setdefault(cn_name, en_norm)
    # Also index by TC→SC folded form so an OCR output using different
    # Trad/Simp variants than Schale still resolves (e.g. OCR "遥香" vs
    # Schale TW "遙香" — 遥 vs 遙 are distinct chars).
    for k in list(cjk_to_en.keys()):
        norm = cjk_normalize(k).lower()
        cjk_to_en.setdefault(norm, cjk_to_en[k])
    for k in list(base_en.keys()):
        norm = cjk_normalize(k).lower()
        base_en.setdefault(norm, base_en[k])
    merged = dict(base_en)
    merged.update(cjk_to_en)
    return merged, cjk_to_en, base_en


def main() -> None:
    if not SRC.exists():
        print(f"source missing: {SRC}")
        sys.exit(1)
    DST.mkdir(parents=True, exist_ok=True)
    # Clean destination so repeat runs don't accumulate stale files
    for f in DST.iterdir():
        if f.suffix.lower() == ".png":
            f.unlink()

    merged, direct, base_only = build_map()
    print(f"  total CJK→EN keys: {len(merged)} (direct variant: {len(direct)})", flush=True)

    report = {"renamed": {}, "fragments_resolved": {}, "unmapped": []}
    for f in sorted(SRC.iterdir()):
        if f.suffix.lower() != ".png":
            continue
        stem = f.stem
        en = None
        stem_norm = cjk_normalize(stem).lower()
        # 1) exact CJK match (raw or Trad↔Simp folded)
        if stem in merged:
            en = merged[stem]
        elif stem_norm in merged:
            en = merged[stem_norm]
            report["fragments_resolved"][stem] = f"TC↔SC fold → {stem_norm}"
        # 1b) synthetic-close-bracket: OCR sometimes drops the trailing `)`,
        #     leaving `星野(泳装` which is unbalanced. Try appending `)` or `）`
        #     and look up again — recovers variant mappings that would
        #     otherwise fall through to base-only.
        if en is None:
            opens = sum(1 for c in stem if c in "(（")
            closes = sum(1 for c in stem if c in ")）")
            if opens == closes + 1:
                last_open = max(stem.rfind("("), stem.rfind("（"))
                # Match closer to opener type
                closer = "）" if last_open >= 0 and stem[last_open] == "（" else ")"
                synth = stem + closer
                synth_norm = cjk_normalize(synth).lower()
                if synth in merged:
                    en = merged[synth]
                    report["fragments_resolved"][stem] = f"synth-close → {synth}"
                elif synth_norm in merged:
                    en = merged[synth_norm]
                    report["fragments_resolved"][stem] = f"synth-close + TC↔SC → {synth_norm}"
        if en is None:
            # 2) fragment: strip stray leading/trailing brackets
            stripped = stem
            while stripped and stripped[-1] in "(（[【)）]】":
                stripped = stripped[:-1].rstrip()
            while stripped and stripped[0] in ")）]】(（[【":
                stripped = stripped[1:].lstrip()
            if stripped != stem and stripped in merged:
                en = merged[stripped]
                report["fragments_resolved"][stem] = f"bracket-strip → {stripped}"
            else:
                # 3) base-only match (strip all bracket content)
                base = strip_brackets(stem)
                if base and base in base_only:
                    en = base_only[base]
                    report["fragments_resolved"][stem] = f"all-brackets-strip → base:{base}"
                else:
                    # 4) substring match: stem appears as a SUBSTRING of a Schale
                    # name. Accept only if EXACTLY ONE base-form Schale name
                    # contains the stem (ambiguous multi-match rejected).
                    if stripped and len(stripped) >= 2:
                        hits = [
                            (cjk, en_val)
                            for cjk, en_val in merged.items()
                            if stripped in cjk and "(" not in en_val  # prefer base student
                        ]
                        # dedupe by en_val
                        uniq_en = {e for _, e in hits}
                        if len(uniq_en) == 1:
                            en = next(iter(uniq_en))
                            report["fragments_resolved"][stem] = f"substring → {stripped} in {hits[0][0]}"
                        elif len(uniq_en) > 1:
                            # Multi-match ambiguous — leave unmapped but record
                            report["unmapped"].append(f"{stem} (ambiguous: {sorted(uniq_en)[:3]})")
                            continue
            # 5) Edit-distance-1 fallback: OCR often reads Trad char as the
            # Japanese simplified equivalent (淚→涙, 櫻→桜). If stem has same
            # length as a Schale CJK name differing by exactly 1 char,
            # accept if unique. Only attempt for stems of length ≥ 3.
            if en is None and len(stem) >= 3 and len(stem) <= 8:
                def ed1(a, b):
                    """Quick edit-distance-1 check for same-length strings."""
                    if len(a) != len(b): return False
                    diffs = sum(1 for x, y in zip(a, b) if x != y)
                    return diffs == 1
                candidates = [(k, v) for k, v in merged.items() if ed1(stem, k)]
                uniq_en = {v for _, v in candidates}
                if len(uniq_en) == 1:
                    en = next(iter(uniq_en))
                    report["fragments_resolved"][stem] = f"edit-distance-1 → {candidates[0][0]}"
        if en is None:
            report["unmapped"].append(stem)
            continue
        # Write to DST with new name
        img = imread_any(str(f))
        if img is None:
            continue
        out_path = DST / f"{en}.png"
        imwrite_any(str(out_path), img)
        report["renamed"][stem] = en

    MAP_OUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print()
    print(f"renamed: {len(report['renamed'])}")
    print(f"fragments resolved: {len(report['fragments_resolved'])}")
    print(f"unmapped: {len(report['unmapped'])}")
    print(f"output: {DST}")
    print(f"audit:  {MAP_OUT}")
    if report["unmapped"]:
        print()
        print("unmapped (can't find in Schale — probably OCR noise):")
        for u in report["unmapped"]:
            print(f"  {u!r}")


if __name__ == "__main__":
    main()
