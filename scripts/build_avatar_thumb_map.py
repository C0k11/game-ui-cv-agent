"""Build fused_avatar cls → EN portrait-filename map for the dashboard
schedule-target picker (so each option can show a face, not just a name).

Reuses build_fused_avatar_dataset's battle-tested CN→EN translation
(COSTUME_SUFFIXES paren-insertion + 繁→简 normalization + the merged name
maps). Output: data/avatar_thumb_map.json = {cls: en} for every cls that
actually has a portrait png in 角色头像/ or 角色头像_crop/.

Re-run after retraining the avatar model (cls set changes) or adding portraits.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.build_fused_avatar_dataset import (  # noqa: E402
    build_cn_to_en_lookup, cn_to_paren_form, _TRAD_TO_SIMP,
)

CLS_JSON = REPO / "data" / "fused_avatar_classes.json"
BIG = REPO / "data" / "captures" / "角色头像"
CROP = REPO / "data" / "captures" / "角色头像_crop"
OUT = REPO / "data" / "avatar_thumb_map.json"


def main() -> None:
    cls_list = json.loads(CLS_JSON.read_text(encoding="utf-8"))
    lookup = build_cn_to_en_lookup()  # CN (many variant forms) → EN file stem

    def find_en(cls: str):
        # Try the cls as-is, simplified, paren-inserted, and both combined.
        cand = [
            cls,
            _TRAD_TO_SIMP(cls),
            cn_to_paren_form(cls),
            _TRAD_TO_SIMP(cn_to_paren_form(cls)),
        ]
        for k in cand:
            en = lookup.get(k)
            if en:
                return en
        return None

    out = {}
    miss = []
    for c in cls_list:
        en = find_en(c)
        if en and ((BIG / f"{en}.png").exists() or (CROP / f"{en}.png").exists()):
            out[c] = en
        else:
            miss.append(c)

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"total {len(cls_list)} with_thumb {len(out)} missing {len(miss)}")
    (REPO / "data" / "_thumb_miss.json").write_text(
        json.dumps(miss, ensure_ascii=False, indent=0), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
