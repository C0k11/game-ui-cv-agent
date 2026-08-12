# -*- coding: utf-8 -*-
"""把 workflow 审查的 journal 整理成清单（一次性工具）。"""
import json
from pathlib import Path

J = Path(r"C:\Users\shien\.claude\projects\D--Project-ai-game-secretary"
         r"\aee7b7b9-5fd6-48bd-b57b-73079d21f0ae\subagents\workflows"
         r"\wf_cd274705-985\journal.jsonl")

finds, verdicts = [], []
for ln in J.read_text(encoding="utf-8", errors="replace").splitlines():
    try:
        rec = json.loads(ln)
    except Exception:
        continue
    if rec.get("type") != "result":
        continue
    val = rec.get("value") or rec.get("result") or {}
    if isinstance(val, str):
        try:
            val = json.loads(val)
        except Exception:
            continue
    if isinstance(val, dict) and "findings" in val:
        finds.extend(val["findings"])
    elif isinstance(val, dict) and "real" in val:
        verdicts.append(val)

out = ["原始发现 %d 条，验证判决 %d 条" % (len(finds), len(verdicts)), ""]
for i, f in enumerate(finds, 1):
    out.append("%2d [%s] %s" % (i, f.get("severity", "?"), f.get("title", "")[:80]))
    out.append("     %s:%s" % (str(f.get("file", "")).split("/")[-1], f.get("line", "-")))
    out.append("     %s" % str(f.get("why_breaks", ""))[:150])
    out.append("")
Path("routing_v2/_audit_list.txt").write_text("\n".join(out), encoding="utf-8")
print("\n".join(out))
