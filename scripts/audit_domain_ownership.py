# -*- coding: utf-8 -*-
"""master idx 域归属常设自检(2026-08-13 猎杀静默噪音的交付物)。

抓的病(全是实锤过的):
  1. master 表本身: 空行(会让剔空行的读法整体错位)/重名。
  2. 域归属对账: build 脚本 REMAP vs cls_domains 域表, 两头都查 --
     REMAP 里混进非 battle 域 id = 会把 UI/头像教进战斗模型;
     battle 身份段的 id 不在 REMAP = 它的标注在静默蒸发(41% 丢框那个病)。
  3. 写路径边界对账: yolo_prefill_run._BATTLE_ID 必须与 cls_domains.BATTLE_ID
     一致, 否则同一个 idx 被两条预填写路径同时认领(overwrite 互相冲框)。
  4. 硬编码边界扫描: 域边界比较只许出现在 cls_domains / 白名单文件里,
     别处再写 `i >= 476` / `143 <= i <= 394` 直接 fail -- 这是复发的根(同一个
     病 2026-07-14 / 2026-08-13 犯了两次)。

用法: py -X utf8 scripts/audit_domain_ownership.py    (rc=0 干净 / rc=1 有病)
对齐后挂进 test_offline 的架构不变量段。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import cls_domains as D  # noqa: E402

FAILS: list = []
WARNS: list = []


def fail(msg: str) -> None:
    FAILS.append(msg)
    print("FAIL " + msg)


def warn(msg: str) -> None:
    WARNS.append(msg)
    print("WARN " + msg)


def ok(msg: str) -> None:
    print("  ok " + msg)


def check_master() -> list:
    master = D.load_master()
    empties = [i for i, n in enumerate(master) if not n]
    trailing = 0
    for n in reversed(master):
        if n:
            break
        trailing += 1
    mid_empty = [i for i in empties if i < len(master) - trailing]
    if mid_empty:
        fail(f"master 表中段有空行 idx={mid_empty[:5]} -- 剔空行的读法会整体错位")
    else:
        ok(f"master {len(master)} 行, 无中段空行")
    names = [n for n in master if n]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        fail(f"master 重名: {sorted(dup)[:5]}")
    else:
        ok("无重名")
    return master


def _extract_dict_ints(src: str, varname: str) -> dict:
    m = re.search(varname + r"\s*=\s*\{([\s\S]*?)\}", src)
    if not m:
        return {}
    out = {}
    for k, v in re.findall(r"(\d+)\s*:\s*(\d+)", m.group(1)):
        out[int(k)] = int(v)
    return out


def check_battle_remap(master: list) -> None:
    p = ROOT / "scripts" / "build_battle_v11.py"
    remap = _extract_dict_ints(p.read_text(encoding="utf-8"), "REMAP")
    if not remap:
        fail("解析不到 build_battle_v11.REMAP")
        return
    bad_in = [mi for mi in remap if D.domain(mi) != "battle"]
    if bad_in:
        fail(f"REMAP 混入非 battle 域 id: "
             f"{[(mi, master[mi]) for mi in bad_in][:5]} -- 会教进战斗模型")
    else:
        ok(f"REMAP {len(remap)} 个 id 全在 battle 域内")
    missing = [i for i in range(D.BATTLE_ID[0], D.BATTLE_ID[1] + 1)
               if i not in remap and not D.is_discard(master[i])]
    if missing:
        fail(f"battle 身份段 id 不在 REMAP: "
             f"{[(i, master[i]) for i in missing]} -- 它们的标注在静默蒸发")
    else:
        ok("battle 身份段 全部有 REMAP 承接")


def check_prefill_span() -> None:
    # 2026-08-13 改判据: 原来是正则扫 yolo_prefill_run 里的本地边界常量再对账。
    #   但"对账两份副本"本身就是这轮要根治的病 -- 正解是那边**根本不该有副本**。
    #   prefill 现已直接 `from scripts.cls_domains import domain`, 所以这里改成
    #   验行为: 逐 idx 比对 _ui_span 与 cls_domains.domain, 不一致就 fail。
    #   (顺带治了旧写法的假阳: 正则会匹配到注释里描述性的赋值字面量。)
    try:
        sys.path.insert(0, str(ROOT))
        from scripts.yolo_prefill_run import _ui_span   # noqa: PLC0415
    except Exception as e:                              # noqa: BLE001
        fail(f"导入 yolo_prefill_run._ui_span 失败: {type(e).__name__}: {e}")
        return
    names = D.load_master()
    bad = [i for i in range(len(names))
           if _ui_span(i) != (D.domain(i) == "ui" or i == D.EMOTICON)]
    if bad:
        fail(f"prefill _ui_span 与域表不一致 {len(bad)} 个 idx: {bad[:12]} "
             f"-- 这些 idx 会被 ui 和别的域两条写路径同时认领")
    else:
        ok(f"prefill _ui_span 与域表逐 idx 一致 ({len(names)} 个)")


# 域边界比较只许出现在这些文件里
_BOUNDARY_WHITELIST = {
    "scripts/cls_domains.py",
    "scripts/audit_domain_ownership.py",
    "scripts/_repro_silent_noise.py",     # 取证脚本, 故意复刻旧写法做对比
    # 锁定文件(另一个 agent 在改), 它内部已是等价语义, 对齐后从白名单摘掉
    "scripts/yolo_prefill_run.py",
    "scripts/build_ui_v2.py",
}
# 历史一次性脚本: 只 WARN 不 FAIL(改了也没人再跑, 但新代码不许长这样)
_LEGACY_PAT = re.compile(
    r"^(_cmp_|_eval_)|scripts/(v6b_|verify_v6b|acceptance_|fix_shirokuro"
    r"|fix_axis|split_stacked|prepare_battle_val|make_battle_val"
    r"|eval_ui_version_cmp|_analyze_flywheel)")
_BOUNDARY_RE = re.compile(
    r"(?<![\d.])((?:[<>]=?\s*476)|(?:476\s*<=)|(?:[<>]=?\s*394)"
    r"|(?:394\s*<=)|(?:143\s*<=)|(?:<=\s*143)|(?:>=\s*48[45]))")


def check_hardcoded_boundaries() -> None:
    hits_fail, hits_warn = [], []
    for py in ROOT.rglob("*.py"):
        rel = py.relative_to(ROOT).as_posix()
        if (rel in _BOUNDARY_WHITELIST or rel.startswith(".venv")
                or "__pycache__" in rel or rel.startswith("data/")):
            continue
        try:
            src = py.read_text(encoding="utf-8")
        except Exception:
            continue
        for ln_no, line in enumerate(src.splitlines(), 1):
            code = line.split("#", 1)[0]
            if _BOUNDARY_RE.search(code):
                item = f"{rel}:{ln_no}: {line.strip()[:70]}"
                (hits_warn if _LEGACY_PAT.search(rel) else hits_fail).append(item)
    for h in hits_fail:
        fail("硬编码域边界(该走 cls_domains): " + h)
    for h in hits_warn:
        warn("历史脚本硬编码边界(不再跑就不管): " + h)
    if not hits_fail:
        ok("活跃代码无硬编码域边界")


def main() -> int:
    master = check_master()
    check_battle_remap(master)
    check_prefill_span()
    check_hardcoded_boundaries()
    print()
    if FAILS:
        print(f"{len(FAILS)} 项不一致(上方 FAIL), {len(WARNS)} 项历史遗留")
        return 1
    print(f"全部一致 ({len(WARNS)} 项历史遗留 WARN)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
