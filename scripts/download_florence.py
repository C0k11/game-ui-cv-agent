import argparse
import os
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="microsoft/Florence-2-large-ft")
    ap.add_argument("--cache-dir", default=r"D:\Project\ml_cache\models")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_dir))
    try:
        os.environ.pop("TRANSFORMERS_CACHE", None)
    except Exception:
        pass

    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    dtype = torch.float16 if str(args.device).lower().startswith("cuda") and torch.cuda.is_available() else torch.float32
    print(f"Downloading {args.model_id} to {cache_dir} (dtype={dtype}, device={args.device})")

    AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True, cache_dir=str(cache_dir))

    load_attempts = [
        {
            "trust_remote_code": True,
            "dtype": dtype,
            "attn_implementation": "eager",
            "cache_dir": str(cache_dir),
        },
        {
            "trust_remote_code": True,
            "torch_dtype": dtype,
            "attn_implementation": "eager",
            "cache_dir": str(cache_dir),
        },
        {
            "trust_remote_code": True,
            "torch_dtype": dtype,
            "cache_dir": str(cache_dir),
        },
    ]
    last_err = None
    for kwargs in load_attempts:
        try:
            model = AutoModelForCausalLM.from_pretrained(args.model_id, **kwargs)
            del model
            last_err = None
            break
        except TypeError as e:
            last_err = e
    if last_err is not None:
        raise last_err

    print("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
