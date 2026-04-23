import os
import subprocess
import sys
from pathlib import Path
import argparse

import uvicorn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    os.environ.setdefault("PYTHONUTF8", "1")
    try:
        os.environ.pop("TRANSFORMERS_CACHE", None)
    except Exception:
        pass

    # Pre-import YOLO in the MAIN thread to avoid a Python 3.13
    # threaded circular import in torchvision when the pipeline
    # worker thread lazy-imports ultralytics later.  Swallow any
    # error here — if ultralytics cannot be imported the pipeline
    # will fall back to OCR-only detection.
    try:
        from ultralytics import YOLO  # noqa: F401
    except Exception as exc:
        print(f"[run_backend] ultralytics pre-import failed: {exc}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()

    if args.detach:
        logs_dir = REPO_ROOT / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        out_log = logs_dir / "backend.out.log"
        err_log = logs_dir / "backend.err.log"

        env = os.environ.copy()
        env.setdefault("PYTHONUTF8", "1")
        try:
            env.pop("TRANSFORMERS_CACHE", None)
        except Exception:
            pass

        creationflags = 0
        if hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
            creationflags |= subprocess.CREATE_NEW_PROCESS_GROUP
        if hasattr(subprocess, "CREATE_NO_WINDOW"):
            creationflags |= subprocess.CREATE_NO_WINDOW

        with open(out_log, "ab", buffering=0) as out_f, open(err_log, "ab", buffering=0) as err_f:
            p = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "server.app:app",
                    "--host",
                    str(args.host),
                    "--port",
                    str(args.port),
                    "--log-level",
                    "info",
                ],
                cwd=str(REPO_ROOT),
                env=env,
                stdout=out_f,
                stderr=err_f,
                creationflags=creationflags,
            )

        print(f"backend started: pid={p.pid} url=http://{args.host}:{args.port}")
        return

    uvicorn.run("server.app:app", host=str(args.host), port=int(args.port), log_level="info")


if __name__ == "__main__":
    main()
