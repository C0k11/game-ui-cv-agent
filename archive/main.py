import argparse
import os

from agent.loop import build_default_agent
from config import ADB_SERIAL, HF_CACHE_DIR, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-base-url", default=LLM_BASE_URL)
    parser.add_argument("--llm-model", default=LLM_MODEL)
    parser.add_argument("--adb-serial", default=ADB_SERIAL)
    parser.add_argument("--hf-cache-dir", default=HF_CACHE_DIR)
    parser.add_argument("--od", action="append", default=[])
    parser.add_argument("--steps", type=int, default=0)
    args = parser.parse_args()

    api_key = os.environ.get("LLM_API_KEY", LLM_API_KEY)
    agent = build_default_agent(
        llm_base_url=args.llm_base_url,
        llm_model=args.llm_model,
        llm_api_key=api_key,
        adb_serial=args.adb_serial,
        hf_cache_dir=args.hf_cache_dir,
        od_queries=args.od or None,
    )

    n = 0
    while True:
        action = agent.step()
        print(action)
        n += 1
        if args.steps and n >= args.steps:
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
