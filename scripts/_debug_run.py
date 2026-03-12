"""Quick trajectory debug dumper. Run with: python scripts/_debug_run.py <run_dir>"""
import json, os, sys

run = sys.argv[1] if len(sys.argv) > 1 else r"D:\Project\ai game secretary\data\trajectories\run_20260312_054412"
files = sorted([f for f in os.listdir(run) if f.endswith('.json')])
print(f"Total: {len(files)} ticks\n")
for f in files:
    t = int(f.replace('tick_', '').replace('.json', ''))
    with open(os.path.join(run, f), 'r', encoding='utf-8') as fh:
        d = json.load(fh)
    skill = d.get('skill', '')
    sub = d.get('sub_state', '')
    act = d.get('action', {}).get('action', '')
    reason = d.get('action', {}).get('reason', '')[:65]
    yolo = d.get('yolo_boxes', [])
    hp = len([b for b in yolo if 'headpat' in b.get('cls', '').lower()])
    hp_s = f' HP:{hp}' if hp else ''
    print(f"t{t:04d} {skill:12s} {sub:18s} {act:6s} {reason}{hp_s}")
