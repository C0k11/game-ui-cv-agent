# -*- coding: utf-8 -*-
"""从**实际构建出来的 ui_v2 数据集**重算 `vocab.HEALTH`。

⛔为什么要有这个脚本：`HEALTH` 是 2026-08-08 手工敲进去的，而数据集每次
   `build_ui_v2` 都会变。手工表一旦过时，`vocab.require()` 就会:
     · 把**已经学会**的类继续当死类 ⇒ 好判据被降级成"可选成员"，
       白白丢掉一条本可以当唯一信号的证据（`战斗失败` 就是活例子：
       表里 (0,0)，实际 train=39/val=28）
     · 反过来，把**已经没样本**的类当健康 ⇒ 死判据静默上线
       （[[cls_ownership_audit]] 那族「守卫悄悄死掉」）
⛔⛔**什么时候才能把输出贴回去 —— 只有一个时机：新模型训完并 active 切过去之后。**
   `HEALTH` 描述的是「**当前上线的模型**学到了什么」，不是「下一版数据集里有什么」。
   数据集重建完、模型还没训（或训了没上线）时贴回去 = **高估现役模型的能力**：
   把它其实没学过的类从 DEAD 提成 WEAK/ok，`require(sole_signal=True)` 就不再拦，
   一条死判据于是静默上线 —— 正是这张表本来要防的事，方向还反了。
   ⇒ 脚本会自己比对「数据集构建时间」和「active 权重的时间」，前者更新就拒绝 --emit。

用法:
    python scripts/sync_vocab_health.py            # 只对账，列出差异（随时可跑）
    python scripts/sync_vocab_health.py --emit     # 打印新表（仅在模型已上线后）
"""
import io
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from routing_v2.state import vocab as V           # noqa: E402

RAW = r"D:\Project\ai game secretary\data\raw_images"
DS = r"D:\Project\ml_cache\models\yolo\dataset\ui_v2"


def counts():
    """(train, val) per cls name，直接数**构建产物**而不是 raw_images。

    ⭐这个区别是致命的：raw_images 里有标注 ≠ 进了训练集
      （build 有 DROP_UI / 头像段过滤 / 空标签帧剔除 / 源列表白名单）。
      [[v16_dataset_integration]] 那条「接了源≠进了集」说的就是这个。
    """
    names = [l.strip() for l in open(os.path.join(RAW, "_classes.txt"),
                                     encoding="utf-8") if l.strip()]
    out = defaultdict(lambda: [0, 0])
    for i, split in enumerate(("train", "val")):
        d = os.path.join(DS, "labels", split)
        if not os.path.isdir(d):
            print(f"⛔ 没有 {d} —— 先跑 build_ui_v2.py")
            sys.exit(2)
        for f in os.listdir(d):
            if not f.endswith(".txt"):
                continue
            for line in open(os.path.join(d, f), encoding="utf-8"):
                p = line.split()
                if p:
                    c = int(p[0])
                    if c < len(names):
                        out[names[c]][i] += 1
    return out


def main():
    cur = counts()
    rows, diff = [], []
    for cls, (t0, v0) in V.HEALTH.items():
        t1, v1 = cur.get(cls, [0, 0])
        rows.append((cls, t1, v1))
        if (t0, v0) != (t1, v1):
            was = "DEAD" if t0 == 0 else ("WEAK" if t0 < 100 else "ok")
            now = "DEAD" if t1 == 0 else ("WEAK" if t1 < 100 else "ok")
            diff.append((cls, t0, v0, t1, v1, was, now, was != now))

    print(f"登记 {len(V.HEALTH)} 个 cls，其中 {len(diff)} 个数字变了\n")
    print("%-26s %14s %14s   %s" % ("类名", "表里(train/val)", "实际(train/val)", "等级"))
    print("-" * 82)
    for cls, t0, v0, t1, v1, was, now, moved in sorted(diff, key=lambda r: -r[7]):
        flag = f"  ⛔{was}→{now}" if moved else ""
        print("%-26s %14s %14s%s" % (cls, f"{t0}/{v0}", f"{t1}/{v1}", flag))
    moved = [d for d in diff if d[7]]
    print("-" * 82)
    print(f"⛔**等级变了**的有 {len(moved)} 个 —— 这些会真的改变 require() 的行为:")
    for cls, t0, v0, t1, v1, was, now, _ in moved:
        print(f"   {cls:<24} {was} → {now}   ({t0}/{v0} → {t1}/{v1})")

    # ⛔时机闸：数据集比现役权重新 = 这份数据还没变成模型，贴回去就是高估它。
    import json
    import time
    ds_t = os.path.getmtime(os.path.join(DS, "data.yaml"))
    # ⛔权重路径从 registry 读，别硬写：active 会换、版本目录名也不统一。
    w, tag = "", "?"
    try:
        reg = json.load(open(os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "data", "model_registry.json"),
            encoding="utf-8"))["ui"]
        tag = reg["active"]
        v = reg["versions"][tag]
        w = v.get("weights") or v.get("path") or v.get("pt") or ""
    except Exception as e:
        print(f"⚠读不到 registry 的 active 权重（{e}）— 时机闸退化为只警告")
    w_t = os.path.getmtime(w) if w and os.path.exists(w) else 0.0
    stale = ds_t > w_t
    print()
    print("数据集构建 %s   现役权重 %s %s"
          % (time.strftime("%m-%d %H:%M", time.localtime(ds_t)), tag,
             time.strftime("%m-%d %H:%M", time.localtime(w_t)) if w_t else "(找不到)"))
    if stale:
        print("⛔**数据集比现役权重新** —— 这些数字属于**还没训出来的模型**。")
        print("   现在贴回 vocab.py 会让代码以为现役模型学过它没学过的类。")
        print("   ⇒ 上面的对账可以看，但 --emit 被拒绝。等 v16 训完 + active 切过去再跑。")

    if "--emit" in sys.argv:
        if stale:
            sys.exit(3)
        print("\n\n# ══ 粘回 vocab.py（数据集实测）══")
        print("HEALTH: Dict[str, Tuple[int, int]] = {")
        for cls, t, v in rows:
            const = next((k for k, val in vars(V).items()
                          if isinstance(val, str) and val == cls and k.isupper()), None)
            print(f"    {const or repr(cls)}: ({t}, {v}),")
        print("}")


if __name__ == "__main__":
    main()
