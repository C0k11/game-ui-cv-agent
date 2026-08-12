# -*- coding: utf-8 -*-
"""动作层：Action + 三道闸（金钱 / 落地复验 / 连发）+ 余额台账。

⚠这里**故意不 eager import 子模块**：`state/pages.py` 要用 `act/money.py` 的
   金钱判据（判据只能有一份），而 `act/gate.py` 又要用 `state/vocab.py` ——
   包级 __init__ 一旦 eager import 就会绕成循环。子模块请写全路径 import。
"""
__all__ = ["action", "gate", "ledger", "money"]
