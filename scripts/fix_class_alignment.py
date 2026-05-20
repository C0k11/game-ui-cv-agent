"""One-shot fixer for class registry alignment issues found by
verify_class_alignment.py.

Fixes:
  1. Delete duplicate `玲` (Gemini wrong-add, real Rei is 莉)
  2. Delete duplicate `蕾洁` (Gemini wrong-add, real Reijo is 令女)
  3. Delete UNUSED_NNN placeholder classes (compacts master)
  4. Add map entries for unmapped refs (Atori, Tomoe_(Cheongsam))
  5. Add `贵音 → Takane` mapping

Backs up _classes.txt + extension JSON before modifying.  Outputs a
diff log so you can review.

Usage:
    py scripts/fix_class_alignment.py --dry-run    # show what would change
    py scripts/fix_class_alignment.py              # apply
"""
from __future__ import annotations
import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MASTER = REPO / "data" / "raw_images" / "_classes.txt"
EXT = REPO / "data" / "student_name_map_extension.json"

# Names to drop from master (Gemini's wrong duplicates + placeholders + phantom skins
# + duplicates that crept back in after a prior fix)
DROP_FROM_MASTER = {
    "玲",                # Gemini wrong-add; real Rei is 莉
    "蕾洁",              # Gemini wrong-add; real Reijo is 令女
    "高岭",              # Fan-translation duplicate of 贵音 (both = Takane); 0 train usage
    "夏泳装",            # Phantom skin — BA's Natsu only has default + Band variant
    "琴里",              # Duplicate of 亚都梨 (both = Kotori); user uses 亚都梨
    "琴里(应援团)",      # Duplicate of 亚都梨应援团
    "芽留",              # Duplicate of 爱留 (both = Meru)
    "巴旗袍",            # Duplicate of 智惠旗袍 (巴 is alt translation for Tomoe)
    "晴奈运动服",        # Duplicate of 羽留奈(體育服) (Haruna)
    "昂",                # Orphan, no ref
    "宁",                # Orphan, no ref (ambiguous: Nene/Neru short form)
    "UNUSED_194",
    "UNUSED_389",
    "UNUSED_398",
    "UNUSED_399",
}

# Add map entries
ADD_TO_EXT = {
    "贵音":       "Takane",                # Takane.png exists, just no mapping
    "智惠旗袍":   "Tomoe_(Cheongsam)",     # English alias for Tomoe_(Qipao)
}

# Atori.png exists as ref but user doesn't have this character in roster —
# leave as orphan ref, harmless (just unused).  亚都梨 = Kotori (correct,
# already in map).
ADD_TO_MASTER: list = []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    master_lines = [l.strip() for l in MASTER.read_text(encoding="utf-8").splitlines() if l.strip()]
    ext = json.loads(EXT.read_text(encoding="utf-8")) if EXT.exists() else {}

    # Plan changes
    new_master = [c for c in master_lines if c not in DROP_FROM_MASTER]
    for new_cn in ADD_TO_MASTER:
        if new_cn not in new_master:
            new_master.append(new_cn)

    drops = [c for c in master_lines if c not in new_master]
    adds = [c for c in new_master if c not in master_lines]

    new_ext = dict(ext)
    ext_added = []
    for cn, en in ADD_TO_EXT.items():
        if cn not in new_ext:
            new_ext[cn] = en
            ext_added.append((cn, en))

    # Report
    print(f"=== Plan ===")
    print(f"Master before: {len(master_lines)} → after: {len(new_master)}")
    print(f"  Drop: {drops}")
    print(f"  Add:  {adds}")
    print(f"Extension entries to add: {len(ext_added)}")
    for cn, en in ext_added:
        print(f"  {cn} → {en}")

    if args.dry_run:
        print()
        print("[DRY RUN] no files modified")
        return 0

    # Backup
    bak_master = MASTER.with_suffix(".txt.bak_before_fix")
    bak_ext = EXT.with_suffix(".json.bak_before_fix")
    shutil.copy2(MASTER, bak_master)
    shutil.copy2(EXT, bak_ext)
    print(f"\nBackups: {bak_master.name}, {bak_ext.name}")

    # Apply
    MASTER.write_text("\n".join(new_master) + "\n", encoding="utf-8")
    EXT.write_text(json.dumps(new_ext, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Master + extension updated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
