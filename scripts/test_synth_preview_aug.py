"""Quick sanity test: hit /api/v1/synth/preview with explicit aug probs and
check if the server applies augmentation.  Helps diagnose whether the
augmentation code is actually running.

Usage:
    py scripts/test_synth_preview_aug.py
"""
import base64
import json
from pathlib import Path
import urllib.request

REPO = Path(__file__).resolve().parents[1]
URL = "http://127.0.0.1:8000/api/v1/synth/preview/schedule_popup"


def main() -> int:
    tpl_path = REPO / "data" / "synth_templates" / "schedule_popup.json"
    if not tpl_path.exists():
        print(f"[err] template not found: {tpl_path}")
        return 1
    tpl = json.loads(tpl_path.read_text(encoding="utf-8"))

    # Force aug to MAX so every slot gets every overlay — guarantees visible aug
    tpl.setdefault("augmentation", {})
    tpl["augmentation"]["ui_overlay_prob"] = 1.0
    tpl["augmentation"]["border_ablation_prob"] = 1.0
    tpl["augmentation"]["ui_components"] = {
        "lv_text": 1.0, "star": 1.0, "weapon_icon": 1.0,
        "heart": 1.0, "alpha_dim": 1.0,
    }

    payload = {"template": tpl, "char_cn": ""}
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    print(f"POST {URL}")
    print(f"  template ui_overlay_prob = {tpl['augmentation']['ui_overlay_prob']}")
    print(f"  template border_prob     = {tpl['augmentation']['border_ablation_prob']}")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(f"[err] HTTP {e.code}: {e.read().decode('utf-8', 'replace')}")
        return 1
    except Exception as e:
        print(f"[err] request failed: {e}")
        return 1

    print("\n=== Response ===")
    print(f"image_size : {data.get('image_size')}")
    print(f"labels     : {len(data.get('labels', []))}")
    print(f"aug_stats  : {data.get('aug_stats')}")
    print(f"aug_config : {data.get('aug_config')}")

    img_b64 = data.get("image_b64")
    if img_b64:
        out = REPO / "synth_preview_aug_test.jpg"
        out.write_bytes(base64.b64decode(img_b64))
        print(f"\n  → saved preview to {out}")
        print(f"  open this file, you should see Lv/star/weapon/heart/dim/border on EVERY slot")

    stats = data.get("aug_stats") or {}
    if stats.get("ui_overlay", 0) == 0:
        print("\n[!] aug_stats.ui_overlay == 0 even with prob=1.0 — server is NOT")
        print("    running the aug code.  Most likely cause: uvicorn --reload didn't")
        print("    pick up server/app.py changes.  Restart the server:")
        print("        Ctrl+C, then py -m uvicorn server.app:app --host 127.0.0.1 --port 8000 --reload")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
