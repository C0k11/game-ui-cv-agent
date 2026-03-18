"""Step 3: Fine-tune PaddleOCR PP-OCRv4 recognition model on Blue Archive data.

This script:
1. Downloads PP-OCRv4 Chinese rec pretrained model if not present
2. Splits labels.txt into train/val sets
3. Builds a PaddleOCR training config matching PP-OCRv4 architecture
4. Runs fine-tuning via the cloned PaddleOCR repo (data/ocr_training/PaddleOCR/)

Prerequisites:
    - Python 3.11 venv at data/ocr_training/ppocr_venv/
    - PaddlePaddle GPU + deps installed in that venv
    - PaddleOCR repo cloned to data/ocr_training/PaddleOCR/

Usage (from repo root):
    data\\ocr_training\\ppocr_venv\\Scripts\\python.exe scripts/ocr_training/03_train.py
"""
import argparse
import os
import random
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "data" / "ocr_training"
PPOCR_REPO = DATA_DIR / "PaddleOCR"
WORK_DIR = DATA_DIR / "ppocr_train"
MODEL_DIR = WORK_DIR / "pretrained"
OUTPUT_DIR = WORK_DIR / "output"

# PP-OCRv4 Chinese rec pretrained (train weights, not inference-only)
PRETRAINED_URL = "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_train.tar"
PRETRAINED_TAR = MODEL_DIR / "ch_PP-OCRv4_rec_train.tar"
PRETRAINED_DIR = MODEL_DIR / "ch_PP-OCRv4_rec_train"

# Character dict from cloned repo
KEYS_PATH = PPOCR_REPO / "ppocr" / "utils" / "ppocr_keys_v1.txt"


def download_pretrained():
    """Download PP-OCRv4 pretrained model if not present."""
    if PRETRAINED_DIR.exists() and any(PRETRAINED_DIR.glob("*.pdparams")):
        print(f"Pretrained model already at {PRETRAINED_DIR}")
        return

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not PRETRAINED_TAR.exists():
        print(f"Downloading PP-OCRv4 rec pretrained model...")
        print(f"  URL: {PRETRAINED_URL}")
        urllib.request.urlretrieve(PRETRAINED_URL, str(PRETRAINED_TAR))
        print(f"  Saved: {PRETRAINED_TAR} ({PRETRAINED_TAR.stat().st_size / 1e6:.1f} MB)")

    print("Extracting...")
    with tarfile.open(str(PRETRAINED_TAR), "r") as tar:
        tar.extractall(str(MODEL_DIR))
    print(f"  Extracted to {PRETRAINED_DIR}")


def split_labels(train_ratio: float = 0.9):
    """Split labels.txt into train and val sets."""
    labels_path = DATA_DIR / "labels.txt"
    if not labels_path.exists():
        print(f"[ERROR] {labels_path} not found.")
        print(f"  Run 01_extract_crops.py and 02_generate_synthetic.py first.")
        sys.exit(1)

    lines = [l.strip() for l in labels_path.read_text("utf-8").splitlines() if l.strip()]
    random.shuffle(lines)

    split_idx = int(len(lines) * train_ratio)
    train_lines = lines[:split_idx]
    val_lines = lines[split_idx:]

    train_path = DATA_DIR / "train_labels.txt"
    val_path = DATA_DIR / "val_labels.txt"
    train_path.write_text("\n".join(train_lines), encoding="utf-8")
    val_path.write_text("\n".join(val_lines), encoding="utf-8")

    print(f"Split: {len(train_lines)} train, {len(val_lines)} val")
    return train_path, val_path


def build_config(args, train_path: Path, val_path: Path) -> Path:
    """Build PaddleOCR training config YAML matching PP-OCRv4 mobile rec architecture."""

    # Use forward-slash paths for YAML (works cross-platform in PaddleOCR)
    def fwd(p: Path) -> str:
        return str(p).replace("\\", "/")

    max_text_length = 25

    config = {
        "Global": {
            "debug": False,
            "use_gpu": True,
            "epoch_num": args.epochs,
            "log_smooth_window": 20,
            "print_batch_step": 50,
            "save_model_dir": fwd(OUTPUT_DIR),
            "save_epoch_step": 10,
            "eval_batch_step": [0, 1000],
            "cal_metric_during_train": True,
            "pretrained_model": fwd(PRETRAINED_DIR / "student"),
            "checkpoints": None,
            "save_inference_dir": fwd(OUTPUT_DIR / "inference"),
            "use_visualdl": False,
            "infer_img": None,
            "character_dict_path": fwd(KEYS_PATH),
            "max_text_length": max_text_length,
            "infer_mode": False,
            "use_space_char": True,
            "distributed": False,
            "save_res_path": fwd(OUTPUT_DIR / "rec_results.txt"),
            "d2s_train_image_shape": [3, 48, 320],
        },
        "Optimizer": {
            "name": "Adam",
            "beta1": 0.9,
            "beta2": 0.999,
            "lr": {
                "name": "Cosine",
                "learning_rate": args.lr,
                "warmup_epoch": 5,
            },
            "regularizer": {
                "name": "L2",
                "factor": 3.0e-05,
            },
        },
        "Architecture": {
            "model_type": "rec",
            "algorithm": "SVTR_LCNet",
            "Transform": None,
            "Backbone": {
                "name": "PPLCNetV3",
                "scale": 0.95,
            },
            "Head": {
                "name": "MultiHead",
                "head_list": [
                    {
                        "CTCHead": {
                            "Neck": {
                                "name": "svtr",
                                "dims": 120,
                                "depth": 2,
                                "hidden_dims": 120,
                                "kernel_size": [1, 3],
                                "use_guide": True,
                            },
                            "Head": {
                                "fc_decay": 0.00001,
                            },
                        },
                    },
                    {
                        "NRTRHead": {
                            "nrtr_dim": 384,
                            "max_text_length": max_text_length,
                        },
                    },
                ],
            },
        },
        "Loss": {
            "name": "MultiLoss",
            "loss_config_list": [
                {"CTCLoss": None},
                {"NRTRLoss": None},
            ],
        },
        "PostProcess": {
            "name": "CTCLabelDecode",
        },
        "Metric": {
            "name": "RecMetric",
            "main_indicator": "acc",
        },
        "Train": {
            "dataset": {
                "name": "SimpleDataSet",
                "data_dir": fwd(DATA_DIR),
                "ext_op_transform_idx": 1,
                "label_file_list": [fwd(train_path)],
                "transforms": [
                    {"DecodeImage": {"img_mode": "BGR", "channel_first": False}},
                    {"RecConAug": {
                        "prob": 0.5,
                        "ext_data_num": 2,
                        "image_shape": [48, 320, 3],
                        "max_text_length": max_text_length,
                    }},
                    {"RecAug": None},
                    {"MultiLabelEncode": {"gtc_encode": "NRTRLabelEncode"}},
                    {"RecResizeImg": {"image_shape": [3, 48, 320]}},
                    {"KeepKeys": {"keep_keys": [
                        "image", "label_ctc", "label_gtc", "length", "valid_ratio",
                    ]}},
                ],
            },
            "loader": {
                "shuffle": True,
                "batch_size_per_card": args.batch_size,
                "drop_last": True,
                "num_workers": 4,
            },
        },
        "Eval": {
            "dataset": {
                "name": "SimpleDataSet",
                "data_dir": fwd(DATA_DIR),
                "label_file_list": [fwd(val_path)],
                "transforms": [
                    {"DecodeImage": {"img_mode": "BGR", "channel_first": False}},
                    {"MultiLabelEncode": {"gtc_encode": "NRTRLabelEncode"}},
                    {"RecResizeImg": {"image_shape": [3, 48, 320]}},
                    {"KeepKeys": {"keep_keys": [
                        "image", "label_ctc", "label_gtc", "length", "valid_ratio",
                    ]}},
                ],
            },
            "loader": {
                "shuffle": False,
                "drop_last": False,
                "batch_size_per_card": args.batch_size,
                "num_workers": 2,
            },
        },
    }

    config_path = WORK_DIR / "ba_rec_train.yml"
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    print(f"Training config: {config_path}")
    return config_path


def run_training(config_path: Path):
    """Launch PaddleOCR training using the cloned repo."""
    train_script = PPOCR_REPO / "tools" / "train.py"
    if not train_script.exists():
        print(f"[ERROR] {train_script} not found.")
        print(f"  Clone PaddleOCR: git clone --depth 1 https://github.com/PaddlePaddle/PaddleOCR.git data/ocr_training/PaddleOCR")
        sys.exit(1)

    cmd = [
        sys.executable,
        str(train_script),
        "-c", str(config_path),
    ]

    print(f"\n{'='*60}")
    print(f"Starting training...")
    print(f"  Python:  {sys.executable}")
    print(f"  Script:  {train_script}")
    print(f"  Config:  {config_path}")
    print(f"  Output:  {OUTPUT_DIR}")
    print(f"{'='*60}\n")

    # Run from the PaddleOCR repo root so imports resolve
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PPOCR_REPO) + os.pathsep + env.get("PYTHONPATH", "")

    # Add nvidia pip-installed DLL paths (cudnn, cublas) to PATH
    try:
        import nvidia.cudnn, nvidia.cublas
        for mod in (nvidia.cudnn, nvidia.cublas):
            dll_dir = os.path.join(os.path.dirname(mod.__file__), "bin")
            if os.path.isdir(dll_dir):
                env["PATH"] = dll_dir + os.pathsep + env.get("PATH", "")
                print(f"  Added to PATH: {dll_dir}")
    except ImportError:
        pass

    result = subprocess.run(cmd, cwd=str(PPOCR_REPO), env=env)
    if result.returncode != 0:
        print(f"\n[ERROR] Training failed (exit code {result.returncode})")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Training complete! Best model saved to: {OUTPUT_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune PaddleOCR rec model for Blue Archive")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs (default: 100)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size per GPU (default: 32, safe for 24GB VRAM)")
    parser.add_argument("--lr", type=float, default=0.0005,
                        help="Learning rate (default: 0.0005, lower than base 0.001 for fine-tuning)")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Train/val split ratio")
    parser.add_argument("--skip-download", action="store_true", help="Skip pretrained model download")
    args = parser.parse_args()

    print(f"Blue Archive OCR Fine-tuning (PP-OCRv4)")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LR:         {args.lr}")
    print()

    # Verify PaddleOCR repo
    if not PPOCR_REPO.exists():
        print(f"[ERROR] PaddleOCR repo not found at {PPOCR_REPO}")
        print(f"  Run: git clone --depth 1 https://github.com/PaddlePaddle/PaddleOCR.git {PPOCR_REPO}")
        sys.exit(1)

    # Verify character dict
    if not KEYS_PATH.exists():
        print(f"[ERROR] Character dict not found at {KEYS_PATH}")
        sys.exit(1)

    # 1. Download pretrained model
    if not args.skip_download:
        download_pretrained()

    # 2. Split data
    train_path, val_path = split_labels(args.train_ratio)

    # 3. Build config
    config_path = build_config(args, train_path, val_path)

    # 4. Run training
    run_training(config_path)


if __name__ == "__main__":
    main()
