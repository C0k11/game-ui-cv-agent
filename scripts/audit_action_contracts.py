# -*- coding: utf-8 -*-
"""动作契约审计 —— 扫描每个 flow 的每一发动作，看它「看没看到下一步 cls」。

⭐用户 2026-08-12 要求:「从头到尾每个环节都要排查**按键逻辑以及打架**，
   还有就是**看没看到下一步 cls 的逻辑**」。
   一晚上抓到 5 处同形 bug（cafe 开邀请券 / craft 快速製造 / shop 切大赛 tab /
   战术商店選擇購買 / 战术商店 arena_done 自取消），全都是
   **点了"会开面板/弹框"的键，却没约定"下一步该看到什么"**。
   ⇒ 与其等 live 一个个撞出来，不如静态列出来逐个过。

判定分档:
  DIALOG   目标在 `gate._EXPECT_DIALOG_AFTER` 里 → 闸自动给严格契约，安全
  EXPLICIT 调用点写了 expect= / expect_gone=                → 安全
  DEFAULT  两者都没有 → 走默认契约 `expect_gone=(自己,)`
           ⚠**开面板/切页类动作用默认契约必然误判**（面板一弹出就把按钮盖住，
             契约立刻"兑现"放行，而页面身份还要 confirm_frames 帧才切过去）
  SWIPE    滑动 → 闸给 settle 契约（按住 8 帧），保证「滑一次→扫一次」

用法: python scripts/audit_action_contracts.py [--all]
      默认只列**需要人看**的（DEFAULT 且疑似开面板/切页）。
"""
import ast
import io
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

FLOW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "routing_v2", "flow")

# 「这一发会让屏幕换个样子」的措辞 —— 命中就该有显式契约
_OPENS = re.compile(r"开|開|打开|進入|进入|切到|切回|进 |去 |打开|展开|列表|面板|"
                    r"弹|彈|快速制造|邀请卷|邀請券|tab|入口|页|頁")


def contracts_of(call: ast.Call):
    got = set()
    for kw in call.keywords:
        if kw.arg in ("expect", "expect_gone") and not (
                isinstance(kw.value, ast.Tuple) and not kw.value.elts):
            got.add(kw.arg)
    return got


def reason_of(call: ast.Call):
    for a in call.args:
        if isinstance(a, ast.Constant) and isinstance(a.value, str):
            return a.value
        if isinstance(a, ast.JoinedStr):
            return "".join(v.value for v in a.values
                           if isinstance(v, ast.Constant)) + " …"
    return ""


def main():
    show_all = "--all" in sys.argv
    try:
        from routing_v2.act.gate import _EXPECT_DIALOG_AFTER as DLG
    except Exception:
        DLG = frozenset()
    rows, n_tap, n_swipe = [], 0, 0
    for fn in sorted(os.listdir(FLOW_DIR)):
        if not fn.endswith(".py") or fn.startswith("_"):
            continue
        path = os.path.join(FLOW_DIR, fn)
        src = open(path, encoding="utf-8").read()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            f = node.func.id
            if f == "swipe":
                n_swipe += 1
                rows.append((fn, node.lineno, "SWIPE", reason_of(node), ""))
                continue
            if f not in ("tap_box", "tap_at"):
                continue
            n_tap += 1
            r = reason_of(node)
            got = contracts_of(node)
            # 目标 cls：tap_box 第一个参数常是变量，取不到就留空
            tgt = ""
            if node.args and isinstance(node.args[0], ast.Attribute):
                tgt = node.args[0].attr
            kind = ("EXPLICIT" if got else
                    "DIALOG" if tgt and tgt in {c for c in DLG} else "DEFAULT")
            risky = kind == "DEFAULT" and bool(_OPENS.search(r))
            if show_all or risky:
                rows.append((fn, node.lineno, kind + ("  ⚠疑似开面板/切页" if risky else ""),
                             r[:52], "/".join(sorted(got))))
    print(f"扫描 {n_tap} 个点击点 + {n_swipe} 个滑动点\n")
    print("%-18s %6s  %-26s %s" % ("文件", "行", "契约", "reason"))
    print("-" * 104)
    for fn, ln, kind, r, got in rows:
        print("%-18s %6d  %-26s %s%s" % (fn, ln, kind, r, f"  [{got}]" if got else ""))
    risky = [x for x in rows if "⚠" in x[2]]
    print("-" * 104)
    print(f"⚠需要人看的（默认契约 + 疑似开面板/切页）: {len(risky)} 处")
    print("⭐判断标准：这一发点下去，**屏幕会不会变成另一个样子**（弹面板/换页/开列表）？")
    print("  会 → 必须写显式 expect=（等新东西出现），默认契约在这里必然误判。")


if __name__ == "__main__":
    main()
