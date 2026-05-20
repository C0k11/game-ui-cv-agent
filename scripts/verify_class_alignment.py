"""Verify that every English-named ref in 角色头像/ + 角色头像_crop/ maps
to exactly one CN class in master[143:].

Outputs:
  _alignment_report.json  — machine-readable report
  _alignment_report.md    — human-readable (sorted by class index)
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    master = [l.strip() for l in (REPO / "data" / "raw_images" / "_classes.txt").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    chars_cn = master[143:]

    # Build CN→EN map from all sources
    cn_to_en: dict = {}
    for fname in ("student_name_map.json", "student_name_map_extension.json"):
        p = REPO / "data" / fname
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            for k, v in d.items():
                if isinstance(v, str):
                    cn_to_en[k] = v
    harv = REPO / "data" / "captures" / "harvest_name_map.json"
    if harv.exists():
        d = json.loads(harv.read_text(encoding="utf-8"))
        for k, v in (d.get("renamed") or {}).items():
            if isinstance(v, str):
                cn_to_en[k] = v

    # Reverse: EN → list of CN names
    en_to_cn = defaultdict(list)
    for cn, en in cn_to_en.items():
        en_to_cn[en].append(cn)

    # Find all PNG refs
    big_dir = REPO / "data" / "captures" / "角色头像"
    crop_dir = REPO / "data" / "captures" / "角色头像_crop"
    refs_big = {f.stem for f in big_dir.glob("*.png")} if big_dir.exists() else set()
    refs_crop = {f.stem for f in crop_dir.glob("*.png")} if crop_dir.exists() else set()
    all_refs = refs_big | refs_crop

    # Index master[143:] for fast lookup
    class_set = set(chars_cn)
    chars_cn_to_idx = {c: 143 + i for i, c in enumerate(chars_cn)}

    aligned = []           # ref → exactly 1 CN class
    missing_in_map = []    # ref has no CN mapping at all
    map_but_no_class = []  # ref maps to CN names not in master
    multiple_classes = []  # ref maps to >1 CN names that exist in master

    for en in sorted(all_refs):
        cn_candidates = en_to_cn.get(en, [])
        # Filter to candidates that exist in master
        cn_in_master = [c for c in cn_candidates if c in class_set]
        if not cn_candidates:
            missing_in_map.append({"en": en,
                                   "in_big": en in refs_big,
                                   "in_crop": en in refs_crop})
        elif not cn_in_master:
            map_but_no_class.append({"en": en, "cn_attempts": cn_candidates})
        elif len(cn_in_master) == 1:
            cn = cn_in_master[0]
            aligned.append({"en": en, "cn": cn, "idx": chars_cn_to_idx[cn]})
        else:
            multiple_classes.append({
                "en": en,
                "cn_in_master": cn_in_master,
                "indices": [chars_cn_to_idx[c] for c in cn_in_master],
            })

    # Also find CN classes with NO ref (orphan classes)
    aligned_cn = {a["cn"] for a in aligned}
    aligned_cn |= {c for entry in multiple_classes for c in entry["cn_in_master"]}
    orphan_cn = []
    for c in chars_cn:
        if c in aligned_cn:
            continue
        en_attempt = cn_to_en.get(c, "")
        orphan_cn.append({"cn": c, "idx": chars_cn_to_idx[c],
                          "en_attempt": en_attempt,
                          "en_attempt_has_ref": en_attempt in all_refs})

    report = {
        "master_total_classes": len(master),
        "char_classes_count": len(chars_cn),
        "refs_in_big": len(refs_big),
        "refs_in_crop": len(refs_crop),
        "refs_unique_total": len(all_refs),
        "aligned_count": len(aligned),
        "missing_in_map_count": len(missing_in_map),
        "map_but_no_class_count": len(map_but_no_class),
        "multiple_classes_count": len(multiple_classes),
        "orphan_cn_count": len(orphan_cn),
        "missing_in_map": missing_in_map,
        "map_but_no_class": map_but_no_class,
        "multiple_classes": multiple_classes,
        "orphan_cn": orphan_cn,
        "aligned_sample_first_20": aligned[:20],
    }

    (REPO / "_alignment_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # Build human-readable markdown
    md = []
    md.append("# Class Alignment Report")
    md.append("")
    md.append(f"- Master total: **{len(master)}** classes (0-142 UI + 143+ chars)")
    md.append(f"- Character classes: **{len(chars_cn)}**")
    md.append(f"- Refs found: **{len(all_refs)}** unique EN names")
    md.append(f"  (big: {len(refs_big)}, crop: {len(refs_crop)})")
    md.append(f"- Aligned 1:1: **{len(aligned)}**")
    md.append(f"- Orphan CN classes (no ref): **{len(orphan_cn)}**")
    md.append(f"- Refs missing from name map: **{len(missing_in_map)}**")
    md.append(f"- Refs map to non-master CN: **{len(map_but_no_class)}**")
    md.append(f"- Refs map to multiple master classes: **{len(multiple_classes)}**")
    md.append("")
    md.append("## All character classes (sorted by master index)")
    md.append("")
    md.append("| idx | CN class | EN ref | ref exists? |")
    md.append("|----:|----------|--------|:-----------:|")
    cn_to_en_lookup = {a["cn"]: a["en"] for a in aligned}
    for i, c in enumerate(chars_cn):
        idx = 143 + i
        en = cn_to_en_lookup.get(c, "")
        if not en:
            en = cn_to_en.get(c, "(no map)")
        has_ref = "✅" if en in all_refs else "❌"
        md.append(f"| {idx} | {c} | {en} | {has_ref} |")

    if orphan_cn:
        md.append("")
        md.append("## ⚠️ Orphan CN classes (in master but no usable ref)")
        md.append("")
        for o in orphan_cn:
            md.append(f"- `[{o['idx']}] {o['cn']}` — map says EN={o['en_attempt'] or 'NONE'}")
    if missing_in_map:
        md.append("")
        md.append("## ⚠️ Refs with NO entry in any CN→EN map")
        md.append("")
        for m in missing_in_map[:50]:
            md.append(f"- `{m['en']}` (big={m['in_big']}, crop={m['in_crop']})")
    if multiple_classes:
        md.append("")
        md.append("## ⚠️ Refs that map to multiple master CN classes (potential conflict)")
        md.append("")
        for m in multiple_classes:
            md.append(f"- `{m['en']}` → {m['cn_in_master']} (idx {m['indices']})")

    (REPO / "_alignment_report.md").write_text("\n".join(md), encoding="utf-8")

    # Console summary (encoding-safe)
    print(f"refs_total: {len(all_refs)}")
    print(f"aligned:    {len(aligned)}")
    print(f"orphans:    {len(orphan_cn)} CN classes with no ref")
    print(f"missing:    {len(missing_in_map)} refs not in any CN map")
    print(f"conflicts:  {len(multiple_classes)} refs map to >1 master CN")
    print()
    print("Reports written:")
    print(f"  _alignment_report.json")
    print(f"  _alignment_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
