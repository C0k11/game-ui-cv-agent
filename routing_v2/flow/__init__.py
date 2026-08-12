# -*- coding: utf-8 -*-
"""流程层：每个玩法一个 Flow，只描述「看到 X 就做 Y」。

Flow 里**不许**出现：金钱判断、落地复验、连发冷却、sleep、写死坐标。
那些各自都只有一个实现处（act/gate.py、act/money.py、act/action.py），
在这里重写就是在复现老代码"同一个 bug 修 N 次"的病（§A2）。

不 eager import 子模块（registry 会拉起全部 flow，容易和上层绕成循环）。
"""
__all__ = ["base", "nav", "interrupt", "battle", "registry"]
