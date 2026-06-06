# 复用 fused 旧 synth (rehearsal): fused_avatar_v1 帧 → raw_images/_fused_synth_remap,
# label 按名 remap (fused 局部 0-251 → master 143-394). teacher 补 UI 下一步用 yolo_prefill_run.
import re, shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FUSED = Path(r"D:\Project\ml_cache\models\yolo\dataset\fused_avatar_v1")
OUT = REPO / "data" / "raw_images" / "_fused_synth_remap"
MASTER = [c.strip() for c in (REPO / "data" / "raw_images" / "_classes.txt").read_text(encoding="utf-8").splitlines() if c.strip()]
master_idx = {n: i for i, n in enumerate(MASTER)}

# 读 fused data.yaml names ("  0: '一花'")
yaml_txt = (FUSED / "data.yaml").read_text(encoding="utf-8")
fused_names = {}
for m in re.finditer(r"^\s*(\d+):\s*'?([^'\n]+?)'?\s*$", yaml_txt, re.M):
    fused_names[int(m.group(1))] = m.group(2).strip()

local2master, unmapped = {}, []
for li, name in fused_names.items():
    mi = master_idx.get(name)
    if mi is not None and 143 <= mi <= 394:
        local2master[li] = mi
    else:
        unmapped.append((li, name))
print(f"fused {len(fused_names)} names → mapped {len(local2master)}, unmapped {len(unmapped)}")
if unmapped:
    print("UNMAPPED:", unmapped[:40])

# 复制帧 + remap label (drop 掉对不上的 box)
if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)
(OUT / "classes.txt").write_text("\n".join(MASTER) + "\n", encoding="utf-8")
n_f = n_b = n_drop = 0
for split in ["train", "val"]:
    lbl_dir, img_dir = FUSED / "labels" / split, FUSED / "images" / split
    if not lbl_dir.is_dir():
        continue
    for txt in lbl_dir.glob("*.txt"):
        alts = list(img_dir.glob(txt.stem + ".*"))
        if not alts:
            continue
        lines = []
        for ln in txt.read_text(encoding="utf-8").splitlines():
            p = ln.split()
            if len(p) < 5:
                continue
            mi = local2master.get(int(float(p[0])))
            if mi is None:
                n_drop += 1
                continue
            lines.append(f"{mi} {p[1]} {p[2]} {p[3]} {p[4]}")
            n_b += 1
        if not lines:
            continue
        stem = f"fs_{split}_{txt.stem}"
        shutil.copy(alts[0], OUT / f"{stem}.jpg")
        (OUT / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        n_f += 1
print(f"DONE: wrote {n_f} frames, {n_b} avatar boxes remapped, dropped {n_drop} unmapped boxes → {OUT}")
print("NEXT: teacher 补 UI → py scripts/yolo_prefill_run.py _fused_synth_remap --model ui --mode merge")
