"""Manual eval of ui detector weights on REAL game screenshots → xlsx.

val mAP is unreliable (51-frame set + noise; val-best epoch repeatedly
turned out to be the WORST on real frames). This tests the cls we
actually need to click, on real per-page frames, at production imgsz,
and writes a comparison xlsx so multiple weights can be审核 side-by-side.

Usage:
  # single weight
  py scripts/eval_ui_real.py D:\\...\\best_real.pt
  # compare several (one column each in the xlsx)
  py scripts/eval_ui_real.py v1=D:\\...\\ui_v1\\weights\\best.pt v2=D:\\...\\ui_v2\\weights\\best_real.pt v3=D:\\...\\ui_v3\\weights\\best_real.pt
  # options: --imgsz 960 --conf 0.15 --device cpu --out D:\\...\\_ui_eval.xlsx

Output xlsx: one sheet per page (LOBBY/MAIL/...), rows = KEY cls,
columns = each weight's max conf (空白 = 未检出).
Default out: D:\\Project\\ai game secretary\\data\\_ui_eval.xlsx
"""
from __future__ import annotations
import sys, os
from ultralytics import YOLO

# cls we MUST detect to navigate / act, grouped by which page they live on
PAGE_CLS = {
    "LOBBY": ["咖啡厅入口", "课程表入口", "学生入口", "社交入口", "制造入口",
              "商店入口", "招募入口", "任务大厅入口", "MomoTalk", "每日领奖",
              "邮件箱", "红点", "黄点"],
    "MAIL": ["一次领取黄色", "全部领取_黄", "领取_黄", "领取蓝色", "红点", "弹窗叉叉"],
    "CAFE": ["咖啡厅收益", "咖啡厅邀请卷", "邀请键", "确认键", "移动至2号点", "弹窗叉叉"],
    "SCHEDULE": ["全体课程表", "课程表票", "课程表开始", "入场键", "确认键"],
    "CRAFT": ["快速制造", "开始制造", "领取_黄", "确认键"],
    "STORY": ["进入章节", "入场键", "跳过故事键", "关卡得星_0", "new", "剧情图标未完成"],
}

# representative real frame per page (hardcoded — pick clean ones, not
# transitional). Multiple candidates per page; first existing wins.
BASE = "D:/Project/ai game secretary/data/trajectories/"
PAGE_FRAMES = {
    "LOBBY": ["run_20260528_184135/tick_0010.jpg", "run_20260528_024602/tick_0010.jpg"],
    "MAIL":  ["run_20260528_024602/tick_0012.jpg", "run_20260528_023726/tick_0012.jpg"],
    "CAFE":  [],   # TODO: 挑一张干净 cafe 主界面帧
    "SCHEDULE": [],  # TODO: 挑一张课程表帧
    "CRAFT": [],     # TODO
    "STORY": [],     # TODO
}


def pick_frame(cands):
    for p in cands:
        full = BASE + p
        if os.path.exists(full):
            return full
    return None


def parse_args(argv):
    weights = {}   # label -> path
    imgsz, conf, device = 960, 0.15, "cpu"
    out = "D:/Project/ai game secretary/data/_ui_eval.xlsx"
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--imgsz": imgsz = int(argv[i+1]); i += 2; continue
        if a == "--conf": conf = float(argv[i+1]); i += 2; continue
        if a == "--device": device = argv[i+1]; i += 2; continue
        if a == "--out": out = argv[i+1]; i += 2; continue
        if "=" in a:
            label, path = a.split("=", 1); weights[label] = path
        else:
            weights[f"w{len(weights)+1}"] = a
        i += 1
    return weights, imgsz, conf, device, out


def eval_weight(path, imgsz, conf, device):
    """Return {page: {cls: max_conf}} for one weight."""
    m = YOLO(path)
    out = {}
    for page, cands in PAGE_FRAMES.items():
        img = pick_frame(cands)
        if not img:
            out[page] = None  # no frame
            continue
        r = m(img, conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
        best = {}
        for b in r.boxes:
            nm = m.names.get(int(b.cls[0]), str(int(b.cls[0])))
            c = float(b.conf[0])
            if nm not in best or c > best[nm]:
                best[nm] = c
        out[page] = best
    return out


def main():
    weights, imgsz, conf, device, out = parse_args(sys.argv[1:])
    if not weights:
        weights = {"v3": "D:/Project/ml_cache/models/yolo/runs/ui_yolo26m_v3/weights/best_real.pt"}
    print(f"weights: {weights}")
    print(f"imgsz={imgsz} conf={conf} device={device}")

    # results[label] = {page: {cls: conf}}
    results = {lbl: eval_weight(p, imgsz, conf, device) for lbl, p in weights.items()}

    # also print to terminal
    for page in PAGE_CLS:
        img = pick_frame(PAGE_FRAMES[page])
        tag = os.path.basename(img) if img else "无帧"
        print(f"\n[{page}] ({tag})")
        for cls in PAGE_CLS[page]:
            cells = []
            for lbl in weights:
                d = results[lbl].get(page)
                v = d.get(cls) if d else None
                cells.append(f"{lbl}={v:.2f}" if v is not None else f"{lbl}=-")
            print(f"  {cls:14s} {'  '.join(cells)}")

    # write xlsx
    try:
        from openpyxl import Workbook
        wb = Workbook(); wb.remove(wb.active)
        for page in PAGE_CLS:
            ws = wb.create_sheet(page[:31])
            ws.append(["cls"] + list(weights.keys()))
            for cls in PAGE_CLS[page]:
                row = [cls]
                for lbl in weights:
                    d = results[lbl].get(page)
                    v = d.get(cls) if d else None
                    row.append(round(v, 3) if v is not None else "")
                ws.append(row)
        wb.save(out)
        print(f"\n[xlsx] {out}")
    except ImportError:
        # fallback: csv
        import csv
        csv_out = out.replace(".xlsx", ".csv")
        with open(csv_out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            for page in PAGE_CLS:
                w.writerow([f"== {page} =="] + list(weights.keys()))
                for cls in PAGE_CLS[page]:
                    row = [cls]
                    for lbl in weights:
                        d = results[lbl].get(page)
                        v = d.get(cls) if d else None
                        row.append(round(v, 3) if v is not None else "")
                    w.writerow(row)
        print(f"\n[csv (openpyxl 没装)] {csv_out}")


if __name__ == "__main__":
    main()
