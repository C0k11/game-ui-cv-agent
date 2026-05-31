"""Count per-class bbox + frame frequency in ui_v1 training set.

Scans the dataset that train_yolo26.py actually fed to ui_yolo26m_v1
(symlinks under D:/Project/ml_cache/models/yolo/dataset/ui_v1/labels/train).

Output:
  data/_ui_v1_class_freq.md   markdown table sorted by frame count asc
  data/_ui_v1_class_freq.json plain mapping cls_id -> {name, bbox_n, frame_n}

Used to pick under-represented classes for oversampling before
retraining ui_v1.
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

DS = Path("D:/Project/ml_cache/models/yolo/dataset/ui_v1")
REPO = Path(__file__).resolve().parents[1]

# Load class names
with open(DS / "data.yaml", "r", encoding="utf-8") as f:
    lines = f.read().splitlines()
names: dict[int, str] = {}
for ln in lines:
    s = ln.strip()
    if not s or s.startswith("path:") or s.startswith("train:") or s.startswith("val:") or s.startswith("nc:") or s == "names:":
        continue
    # `  47: '社团'` style
    if ":" in s:
        try:
            head, val = s.split(":", 1)
            cid = int(head.strip())
            v = val.strip().strip("'").strip('"')
            names[cid] = v
        except Exception:
            pass

bbox_count: dict[int, int] = defaultdict(int)
frame_count: dict[int, int] = defaultdict(int)
total_frames = 0
total_bboxes = 0
empty_frames = 0

lbl_train = DS / "labels" / "train"
for txt in sorted(lbl_train.glob("*.txt")):
    total_frames += 1
    seen_in_frame: set[int] = set()
    try:
        for ln in txt.read_text(encoding="utf-8").splitlines():
            s = ln.strip()
            if not s:
                continue
            parts = s.split()
            try:
                cid = int(parts[0])
            except Exception:
                continue
            bbox_count[cid] += 1
            seen_in_frame.add(cid)
            total_bboxes += 1
    except Exception as e:
        print(f"[!] failed reading {txt}: {e}")
        continue
    if not seen_in_frame:
        empty_frames += 1
    for cid in seen_in_frame:
        frame_count[cid] += 1

# Sort by frame_count asc (least learned first)
rows = []
for cid in sorted(names):
    rows.append((cid, names[cid], frame_count.get(cid, 0), bbox_count.get(cid, 0)))
rows.sort(key=lambda r: (r[2], r[3]))

out_md = REPO / "data" / "_ui_v1_class_freq.md"
out_json = REPO / "data" / "_ui_v1_class_freq.json"

md_lines = []
md_lines.append("# ui_v1 train set class frequency (sorted by frames asc)")
md_lines.append("")
md_lines.append(f"- train frames: {total_frames}  (empty: {empty_frames})")
md_lines.append(f"- total bboxes: {total_bboxes}")
md_lines.append(f"- classes used >=1 frame: {sum(1 for r in rows if r[2] > 0)} / {len(rows)}")
md_lines.append("")
md_lines.append("| cls_id | name | frames | bboxes |")
md_lines.append("|---|---|---:|---:|")
for cid, name, fn, bn in rows:
    md_lines.append(f"| {cid} | {name} | {fn} | {bn} |")
out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

json_map = {str(cid): {"name": name, "frames": fn, "bboxes": bn} for cid, name, fn, bn in rows}
out_json.write_text(json.dumps(json_map, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"[done] train frames={total_frames} (empty {empty_frames}); total bboxes={total_bboxes}")
print(f"       wrote {out_md}")
print(f"       wrote {out_json}")
print()
print("# Top 30 least-learned classes:")
for cid, name, fn, bn in rows[:30]:
    print(f"  cls {cid:3d} {name:30s}  frames={fn:3d}  bboxes={bn:4d}")
print()
print("# Key click classes — frame count:")
KEY_CLS = [4, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 23, 28, 31, 32,
           79, 89, 90, 106, 107, 108, 109, 118, 132, 138, 141, 142]
for cid in KEY_CLS:
    name = names.get(cid, f"<unknown cls {cid}>")
    fn = frame_count.get(cid, 0)
    bn = bbox_count.get(cid, 0)
    print(f"  cls {cid:3d} {name:30s}  frames={fn:3d}  bboxes={bn:4d}")
