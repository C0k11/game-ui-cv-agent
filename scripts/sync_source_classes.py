# -*- coding: utf-8 -*-
"""把各源目录的 `classes.txt` 副本同步到 master —— **只同步纯改名，绝不掩盖真 drift**。

背景（2026-08-11 发现）：`build_ui_v2.py` 的 schema drift 校验从**废案改名那次
   之后就一直红着**，`return 1` 直接退出  自那以后 build 一次都没跑成过。
   源目录里的 `classes.txt` 是**打标那一刻的快照**，master 后来把 18 个废案类
   改成 `_废弃N_原名_原因`（类表按行号索引，废案只能改名不能删行），副本没跟。

为什么不能简单地"全部覆盖了事"：
   那道校验的价值是抓**顺序错乱**（源目录按另一套 idx 打的标 = 整批标注全错位，
   是会毁掉训练集的那种错）。无差别覆盖等于把这道闸拆了。
    这里**逐条分类**：
     · benign  —— master 是 `_废弃{i}_{原名}_...` 而副本正好是 `{原名}`  纯改名
     · benign  —— 副本比 master 短（老类表，前缀一致） 只是没跟上新增类
     · DRIFT —— 其他任何不一致  **打印出来并拒绝同步**，交给人判断

用法:
    python scripts/sync_source_classes.py            # 只报告，不写盘
    python scripts/sync_source_classes.py --write    # 确认没问题后再写
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# 中文 Windows 控制台默认 GBK，编不出 / 这类符号会直接 UnicodeEncodeError
#   把脚本打断（不是显示成问号，是**崩**）。errors="replace" 保证只丢字形不丢命。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw_images"


def _norm(s: str) -> str:
    """去掉点/空格/下划线等分隔符 —— 改名时顺带做过规范化
    （`D.U. 白鸟区`  `DU白鸟区`），不归一化就会把纯改名误判成 drift。"""
    return re.sub(r"[\s._·・\-]+", "", s)


def is_benign(idx: int, src_name: str, master_name: str) -> bool:
    """副本 idx 处是 src_name，master 是 master_name —— 这是纯改名吗？

    只认一种形态：master 变成了 `_废弃{行号}_{原名}_{理由}`。
    **行号必须等于它自己的 idx** —— 这一条是关键：真发生顺序错乱时，
    `_废弃N_` 里的 N 对不上所在行，照样会被判成 drift。
    """
    if src_name == master_name:
        return True
    m = re.match(r"^_废弃(\d+)_(.*)$", master_name)
    if not m:
        return False
    if int(m.group(1)) != idx:          # 废案编号必须就是它自己的行号
        return False
    body, src = _norm(m.group(2)), _norm(src_name)
    # `_废弃2_清辉石_是30青辉石的错别字重复类`  原名是第一段 `清辉石`
    return bool(src) and body.startswith(src)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default="_classes.txt")
    ap.add_argument("--write", action="store_true", help="确认后真的写盘")
    a = ap.parse_args()

    mpath = RAW / a.master
    if not mpath.is_file():
        print(f"[!] 找不到 master: {mpath}")
        return 2
    master = [l for l in mpath.read_text(encoding="utf-8").splitlines()]
    print(f"[master] {a.master}  {len(master)} 行")

    benign, drift, ok, missing = [], [], 0, 0
    for d in sorted(p for p in RAW.iterdir() if p.is_dir()):
        cf = d / "classes.txt"
        if not cf.exists():
            missing += 1
            continue
        sc = cf.read_text(encoding="utf-8").splitlines()
        if sc == master[:len(sc)]:
            ok += 1
            continue
        bad = [(i, sc[i], master[i]) for i in range(min(len(sc), len(master)))
               if sc[i] != master[i]]
        if all(is_benign(i, s, m) for i, s, m in bad):
            benign.append((d.name, bad))
        else:
            drift.append((d.name, [(i, s, m) for i, s, m in bad
                                   if not is_benign(i, s, m)]))

    print(f"[扫描] 已对齐 {ok} / 纯改名待同步 {len(benign)} / "
          f"真 drift {len(drift)} / 没有 classes.txt {missing}")

    if drift:
        print("\n以下目录不是改名，是真 drift —— **不同步**，请人工判断:")
        for name, bad in drift[:20]:
            print(f"  {name}")
            for i, s, m in bad[:4]:
                print(f"     idx {i}: 副本={s!r}  master={m!r}")
        print("    这类目录若真按另一套 idx 打过标，整批标注是错位的，"
              "**不能靠改 classes.txt 修**，要重映射 label 或剔出源列表。")

    if benign:
        names = sorted({m for _n, bad in benign for _i, _s, m in bad})
        print(f"\n[纯改名] 涉及 {len(names)} 个类名，例如:")
        for n in names[:6]:
            print(f"     {n}")
        print(f"[纯改名] {len(benign)} 个目录待同步"
              f"{'（--write 已开，写盘）' if a.write else '（只报告；加 --write 才写）'}")
        if a.write:
            for name, _bad in benign:
                cf = RAW / name / "classes.txt"
                old = cf.read_text(encoding="utf-8").splitlines()
                cf.write_text("\n".join(master[:len(old)]) + "\n", encoding="utf-8")
            print(f"[纯改名] 已同步 {len(benign)} 个 classes.txt（保持各自原有行数）")

    return 1 if drift else 0


if __name__ == "__main__":
    sys.exit(main())
