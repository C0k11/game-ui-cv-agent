import json

path = r"D:\Project\ai game secretary\data\trajectories\run_20260222_222827\trajectory.jsonl"
with open(path, encoding="utf-8") as f:
    for i, line in enumerate(f):
        d = json.loads(line)
        act = d.get("action", {})
        if act is None:
            continue
        reason = act.get("reason", "")
        if reason:
            phase = act.get("_pipeline_phase", "")
            tick = act.get("_pipeline_tick", "")
            step = d.get("step", "?")
            tgt = act.get("target", "")
            a = act.get("action", "")
            fr = act.get("from", "")
            to = act.get("to", "")
            extra = ""
            if fr:
                extra = f" from={fr} to={to}"
            elif tgt:
                extra = f" target={tgt}"
            print(f"step={step:>3} phase={phase:<20} tick={tick:<3} {a:<6}{extra}  {reason[:150]}")
