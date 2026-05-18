"""Generate a visual HTML report for avatar_cls model evaluation.

Shows side-by-side: actual image | ground truth | top-1 prediction | top-5
alternatives.  Groups by:
  - WRONG with HIGH conf (>=0.7): systematic errors (similar-looking chars)
  - WRONG with LOW conf (<0.5): uncertain — needs more data
  - CORRECT but LOW conf (<0.7): right but unsure — could be more confident
  - CORRECT (>=0.7): healthy

Output: data/yolo_datasets/avatar_cls_v2/REPORT.html
Open in browser, sort & scan visually.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np
from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[1]


def img_to_b64(img: np.ndarray, size: int = 96) -> str:
    h, w = img.shape[:2]
    if h != size or w != size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=r"D:\Project\ml_cache\models\yolo\runs\avatar_cls_v2_yolo26n\weights\best.pt")
    ap.add_argument("--val", default=str(REPO / "data" / "yolo_datasets" / "avatar_cls_v2" / "val"))
    ap.add_argument("--also-traj", action="store_true",
                    help="Also include data/yolo_datasets/avatar_cls/val (trajectory val)")
    ap.add_argument("--out", default=str(REPO / "data" / "yolo_datasets" / "avatar_cls_v2" / "REPORT.html"))
    args = ap.parse_args()

    model = YOLO(args.model)
    names = [model.names[i] for i in sorted(model.names.keys())]
    bilingual = json.loads((REPO / "data" / "avatar_cls_names_bilingual.json").read_text(encoding="utf-8"))
    en_to_cn = {v["en"]: v.get("cn") for v in bilingual.values()}

    def lbl(en: str) -> str:
        cn = en_to_cn.get(en)
        return f"{en}({cn})" if cn else en

    # Walk val + optionally trajectory val
    samples = []  # list of (img, truth, source)
    for cls_dir in sorted(Path(args.val).iterdir()):
        if not cls_dir.is_dir():
            continue
        for jpg in sorted(cls_dir.glob("*.jpg")):
            img = cv2.imdecode(np.fromfile(str(jpg), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                samples.append((img, cls_dir.name, "harvest", jpg.name))
    if args.also_traj:
        traj_val = REPO / "data" / "yolo_datasets" / "avatar_cls" / "val"
        for cls_dir in sorted(traj_val.iterdir()) if traj_val.is_dir() else []:
            if not cls_dir.is_dir():
                continue
            for jpg in sorted(cls_dir.glob("*.jpg")):
                img = cv2.imdecode(np.fromfile(str(jpg), dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    samples.append((img, cls_dir.name, "trajectory", jpg.name))

    print(f"Evaluating {len(samples)} samples...")
    rows = []
    for img, truth, src, fn in samples:
        r = model.predict(img, verbose=False, imgsz=224)[0]
        probs = r.probs.data.cpu().numpy()
        top5_idx = np.argsort(probs)[-5:][::-1]
        top5 = [(names[int(i)], float(probs[int(i)])) for i in top5_idx]
        top1_name, top1_conf = top5[0]
        ok = (top1_name == truth)
        # Tier
        if not ok and top1_conf >= 0.7:
            tier = "wrong_high"
        elif not ok:
            tier = "wrong_low"
        elif ok and top1_conf < 0.7:
            tier = "correct_low"
        else:
            tier = "correct_high"
        rows.append({
            "img": img,
            "truth": truth,
            "top5": top5,
            "ok": ok,
            "tier": tier,
            "src": src,
            "filename": fn,
        })

    # Tier counts
    from collections import Counter
    tier_counts = Counter(r["tier"] for r in rows)
    n_total = len(rows)
    n_correct = sum(1 for r in rows if r["ok"])
    print(f"Overall: {n_correct}/{n_total} = {100*n_correct/n_total:.2f}%")
    for t, n in tier_counts.items():
        print(f"  {t}: {n}")

    # Build HTML
    tier_order = ["wrong_high", "wrong_low", "correct_low", "correct_high"]
    tier_labels = {
        "wrong_high": "❌ WRONG with high conf (≥0.7) — systematic errors",
        "wrong_low":  "⚠️  WRONG with low conf (<0.7) — uncertain, fixable",
        "correct_low":  "🟡 CORRECT but low conf (<0.7) — right but unsure",
        "correct_high": "✅ CORRECT with high conf (≥0.7) — healthy",
    }
    tier_color = {
        "wrong_high":   "#fee2e2",
        "wrong_low":    "#fef3c7",
        "correct_low":  "#fef9c3",
        "correct_high": "#dcfce7",
    }

    html_parts = [f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>avatar_cls eval</title>
<style>
body {{ font-family: system-ui, sans-serif; background:#0f1115; color:#e0e0e0; padding:20px; max-width:1400px; margin:0 auto }}
h1 {{ color:#fff }}
h2 {{ margin-top:30px; padding:8px 12px; border-radius:6px }}
table {{ border-collapse: collapse; width:100%; margin:10px 0 }}
th, td {{ padding:8px 12px; border-bottom:1px solid #333; text-align:left; vertical-align:middle }}
th {{ background:#1a1d24; color:#aaa; text-transform:uppercase; font-size:11px }}
.img {{ display:block; width:96px; height:96px; image-rendering:pixelated; border-radius:4px }}
.label {{ font-family: monospace }}
.truth.cn {{ color:#94d8ff }}
.pred.right {{ color:#86efac }}
.pred.wrong {{ color:#fca5a5; font-weight:bold }}
.conf {{ font-family:monospace }}
.conf.high {{ color:#86efac }}
.conf.low {{ color:#fbbf24 }}
.conf.veryLow {{ color:#fca5a5 }}
.top5 {{ font-family:monospace; font-size:11px; color:#aaa; line-height:1.6 }}
.src {{ font-size:10px; color:#777; text-transform:uppercase }}
.summary {{ display:flex; gap:20px; flex-wrap:wrap; margin:15px 0 }}
.summary div {{ padding:10px 16px; background:#1a1d24; border-radius:6px; font-family:monospace }}
.summary .total {{ font-size:18px; font-weight:bold }}
</style></head><body>
<h1>avatar_cls v2 评估报告</h1>
<div class="summary">
<div class="total">总样本: {n_total}</div>
<div>准确率: {100*n_correct/n_total:.2f}% ({n_correct}/{n_total})</div>
<div>❌ 错+高 conf: {tier_counts.get('wrong_high',0)}</div>
<div>⚠️ 错+低 conf: {tier_counts.get('wrong_low',0)}</div>
<div>🟡 对+低 conf: {tier_counts.get('correct_low',0)}</div>
<div>✅ 对+高 conf: {tier_counts.get('correct_high',0)}</div>
</div>
"""]

    for tier in tier_order:
        tier_rows = [r for r in rows if r["tier"] == tier]
        if not tier_rows:
            continue
        # Sort: wrong_high by conf DESC (worst first), correct by conf ASC (uncertain first), correct_high by conf DESC
        if tier in ("wrong_high",):
            tier_rows.sort(key=lambda r: -r["top5"][0][1])
        elif tier in ("wrong_low", "correct_low"):
            tier_rows.sort(key=lambda r: r["top5"][0][1])
        else:
            tier_rows.sort(key=lambda r: -r["top5"][0][1])

        html_parts.append(f'<h2 style="background:{tier_color[tier]};color:#0a0a0a">{tier_labels[tier]} ({len(tier_rows)})</h2>')
        html_parts.append('<table><tr><th>image</th><th>truth (CN)</th><th>top-1 predicted</th><th>conf</th><th>top-5 alternatives</th><th>src</th></tr>')

        for r in tier_rows:
            top1_name, top1_conf = r["top5"][0]
            truth_lbl = lbl(r["truth"])
            pred_lbl = lbl(top1_name)
            conf_cls = "high" if top1_conf >= 0.7 else "low" if top1_conf >= 0.5 else "veryLow"
            pred_cls = "right" if r["ok"] else "wrong"
            top5_str = " | ".join(f'{lbl(n)}={p:.2f}' for n, p in r["top5"])
            b64 = img_to_b64(r["img"], size=96)
            html_parts.append(
                f'<tr>'
                f'<td><img class="img" src="data:image/jpeg;base64,{b64}"></td>'
                f'<td class="label truth cn">{truth_lbl}</td>'
                f'<td class="label pred {pred_cls}">{pred_lbl}</td>'
                f'<td class="conf {conf_cls}">{top1_conf:.3f}</td>'
                f'<td class="top5">{top5_str}</td>'
                f'<td class="src">{r["src"]}</td>'
                f'</tr>'
            )
        html_parts.append("</table>")

    html_parts.append("</body></html>")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(html_parts), encoding="utf-8")
    print(f"\nREPORT: {out}")
    print(f"Open in browser: file:///{out.as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
