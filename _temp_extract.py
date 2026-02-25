import json, sys

path = sys.argv[1] if len(sys.argv) > 1 else r"D:\Project\ai game secretary\data\trajectories\run_20260224_220023\trajectory.jsonl"
with open(path, encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        act = d.get("action", {})
        if not act: continue
        reason = act.get("reason", "")
        if not reason: continue
        phase = act.get("_pipeline_phase", "")
        tick = str(act.get("_pipeline_tick", ""))
        step = d.get("step", "?")
        a = act.get("action", "")
        print(f"{step:>3} {phase:<20} t={tick:<3} {a:<6} {reason[:120]}")
