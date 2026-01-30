import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

def parse_log_line(line: str) -> Optional[Dict]:
    # Parse log lines to extract timestamp and action json
    # Format: [2026-01-29T21:33:38] step=1 dry_run=0 action={...}
    
    m = re.match(r"^\[(.*?)\] step=(\d+) .*? action=(.*)$", line)
    if not m:
        return None
    
    ts_str, step, action_str = m.groups()
    try:
        dt = datetime.fromisoformat(ts_str)
        ts = dt.timestamp()
    except:
        return None
        
    try:
        action = json.loads(action_str)
    except:
        # fallback for truncated or messy json in logs
        return None
        
    return {
        "timestamp": ts,
        "step": int(step),
        "action": action
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True, help="Path to session directory (containing meta.jsonl and images)")
    parser.add_argument("--log", required=True, help="Path to agent.out.log")
    parser.add_argument("--out", required=True, help="Output jsonl path")
    parser.add_argument("--window_ms", type=int, default=2000, help="Max time difference (ms) to match log to image")
    
    args = parser.parse_args()
    
    session_dir = Path(args.session)
    meta_path = session_dir / "meta.jsonl"
    
    if not meta_path.exists():
        print(f"Error: {meta_path} not found")
        return

    # 1. Load Session Frames
    frames = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                frames.append(rec)
            except:
                pass
    
    print(f"Loaded {len(frames)} frames from session.")
    if not frames:
        return

    # 2. Load Log Actions
    actions = []
    log_path = Path(args.log)
    if log_path.exists():
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                parsed = parse_log_line(line.strip())
                if parsed:
                    actions.append(parsed)
    
    print(f"Loaded {len(actions)} actions from log.")
    if not actions:
        return

    # 3. Correlate
    # For each action, find the closest frame strictly BEFORE the action timestamp (perception happens before action)
    # Actually, the log timestamp is when the action is DECIDED/EXECUTED. 
    # The VLM perception happened slightly earlier.
    # The log contains: "step=X stage=perception_end elapsed_s=..." and "step=X stage=policy_end"
    # The action line "[...] step=X ... action={...}" is usually logged at 'execute' stage or result of 'decide'.
    
    # Simple heuristic: Match action at time T to frame at time T-delta where delta is small.
    # Frame timestamp is when it was captured.
    # Action timestamp is when it was logged.
    # We want the image that the agent SAW.
    # The log action dict usually has "_perception": {"raw": ...} but not the image path.
    # However, if we collected data *while* the agent was running, we can match by timestamp.
    
    # We need to be careful: 
    # If the agent is running at 0.5 FPS, and we capture at 2 FPS, we have multiple frames per action.
    # We want the frame closest to (ActionTime - ProcessingTime).
    # Ideally, we want the frame that was most likely used.
    # But since we are collecting separately, we just find the closest frame in a reasonable window.
    
    pairs = []
    
    # Sort both by timestamp
    frames.sort(key=lambda x: x["timestamp"])
    actions.sort(key=lambda x: x["timestamp"])
    
    # For each action, find best frame
    for act in actions:
        ats = act["timestamp"]
        
        # Find frame closest to ats, but preferably before it (since perception precedes action)
        # Filter frames within window_ms
        candidates = [f for f in frames if 0 <= (ats - f["timestamp"]) <= (args.window_ms / 1000.0)]
        
        if not candidates:
            # Try looking slightly ahead if clock sync is weird, but generally look behind
            candidates = [f for f in frames if abs(ats - f["timestamp"]) <= (args.window_ms / 1000.0)]
            
        if not candidates:
            continue
            
        # Pick closest
        best_frame = min(candidates, key=lambda x: abs(x["timestamp"] - ats))
        
        # Construct Training Example
        # Prompt: The system prompt + user goal + "Image size..."
        # Response: The JSON action
        
        # We can extract the actual prompt used from the log if available: action["_prompt"]
        prompt = act["action"].get("_prompt")
        if not prompt:
            # Reconstruct or use simplified
            prompt = "You are an autonomous agent. Goal: Keep the game running safely. Return JSON action."
            
        # Clean up action for training (remove debug fields)
        clean_action = dict(act["action"])
        for k in list(clean_action.keys()):
            if k.startswith("_"):
                del clean_action[k]
        
        # Also remove 'raw' if present in root
        clean_action.pop("raw", None)
        
        response_json = json.dumps(clean_action, ensure_ascii=False)
        
        img_rel = best_frame["image"]
        # Convert relative to absolute if needed, or keep relative to session root
        # train script expects path.
        # If image path in meta is relative to session root, we need to join with session_dir
        # Actually meta.jsonl usually stores relative path to the --out root?
        # In collect_window_dataset.py: "image": str(img_path.relative_to(root))
        # So it's relative to the parent of session_dir.
        
        # We need to reconstruct full path for training script
        # session_dir is "data/captures/session_..."
        # meta image is "session_.../frame_..."
        # So if we are at project root, we can just resolve it.
        # But args.session might be absolute.
        
        # Let's try to resolve absolute path to image
        # Assuming args.session is "data/captures/session_X"
        # and meta "image" is "session_X/frame_Y"
        # Then parent of args.session joined with meta image is the path.
        
        root_dir = session_dir.parent
        full_img_path = (root_dir / img_rel).resolve()
        
        if not full_img_path.exists():
            # Try relative to session dir directly?
            # If meta says "session_X/frame.png" and we are in "session_X", maybe it's just "frame.png"?
            # collect_window_dataset: img_path = out_dir / name; relative_to(root) -> session/frame
            pass

        pairs.append({
            "image": str(full_img_path),
            "prompt": prompt,
            "response": response_json,
            "timestamp": ats,
            "frame_timestamp": best_frame["timestamp"]
        })

    print(f"Matched {len(pairs)} pairs.")
    
    with open(args.out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    
    print(f"Written to {args.out}")

if __name__ == "__main__":
    main()
