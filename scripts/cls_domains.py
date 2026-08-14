# -*- coding: utf-8 -*-
"""master 表域归属的唯一权威(单一定义, 全仓引用这里)。

2026-08-13 猎杀「静默噪音」的产物: 同一套域边界在 6+ 个文件里各写各的,
数据从尾部增长就静默错分 --
  - yolo_prefill_run 旧 `i < 476` 把 484-527 共 44 个新 UI 类静默丢了
    (387 帧上 2,049 框丢 1,436, 41%), 现象被读成"模型很废";
  - audit_cls_usage / l3_health_report 的 `i >= 476 -> battle` 把同一批
    44 槽(41 个活类)错分成 battle, 常设审计自己在撒谎;
  - build_battle_v11 REMAP 收 484(战斗失败), 而 (476,483) 圈定漏了它 --
    两条预填写路径同时认领 484。
修法 = 本模块 + scripts/audit_domain_ownership.py 常设自检(边界再漂移就 fail)。

域布局(0-based master idx):
  [143, 394]        avatar   fused_avatar 模型(学生名 251 + 柚子战斗394)
  {128..136, 412}   battle   HUD 状态类(battle 模型词表的单帧直写段)
  [476, 484]        battle   战场身份/结果类(我方/敌方/塞特的愤怒/Boss/主教/
                             球/黑白/大蛇/战斗失败 -- 484 以 build_battle_v11
                             REMAP 为准, 它在训)
  451               ui + emoticon 唯一有意重叠(摸头, ui v6+ 兼职, teacher 补标)
  其余(含尾部新增)  ui       **排除法**: 新类默认归 ui, 不会再静默漏

铁律: 新增 battle 身份类(新 boss 等)必须同步改 BATTLE_ID 上界 + build 脚本
REMAP; audit_domain_ownership 会对账 REMAP 与本表, 漏一头就 fail。
"""
from __future__ import annotations

import os

AVATAR = (143, 394)
BATTLE_ID = (476, 484)
BATTLE_HUD = frozenset(range(128, 137)) | {412}
EMOTICON = 451          # 与 ui 有意重叠(唯一豁免)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASTER_PATH = os.path.join(_ROOT, "data", "raw_images", "_classes.txt")


def domain(i: int) -> str:
    """这个 master idx 归哪个模型域。ui 是排除法兜底(尾部新增默认 ui)。"""
    if AVATAR[0] <= i <= AVATAR[1]:
        return "avatar"
    if BATTLE_ID[0] <= i <= BATTLE_ID[1] or i in BATTLE_HUD:
        return "battle"
    return "ui"


def load_master(path: str = None) -> list:
    """按行号原样读 master 表, **绝不剔除空行**。

    audit_cls_usage 旧版 `[ln for ln in f if ln.strip()]` 在中段出现空行时
    会让后面所有 idx 整体错位(当前表 528 行 0 空行, 是潜伏雷不是哑弹)。
    """
    with open(path or MASTER_PATH, encoding="utf-8") as f:
        return [ln.rstrip("\n").strip() for ln in f]


def is_discard(name: str) -> bool:
    return name.startswith("_废弃")
