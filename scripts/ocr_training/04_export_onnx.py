"""Step 4: Export trained PaddleOCR model to ONNX for use with RapidOCR.

Converts the fine-tuned PP-OCRv4 rec model from PaddlePaddle format to ONNX,
then copies it to data/ocr_model/ba_rec.onnx for pipeline integration.

Usage (from repo root):
    data\\ocr_training\\ppocr_venv\\Scripts\\python.exe scripts/ocr_training/04_export_onnx.py
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "data" / "ocr_training"
PPOCR_REPO = DATA_DIR / "PaddleOCR"
WORK_DIR = DATA_DIR / "ppocr_train"
OUTPUT_DIR = WORK_DIR / "output"
FINAL_MODEL_DIR = REPO / "data" / "ocr_model"


def find_best_model() -> Path:
    """Find the best trained model in the output directory."""
    best = OUTPUT_DIR / "best_accuracy.pdparams"
    if best.exists():
        return OUTPUT_DIR / "best_accuracy"

    # Check for latest checkpoint
    checkpoints = sorted(OUTPUT_DIR.glob("iter_epoch_*"), reverse=True)
    if checkpoints:
        return checkpoints[0]

    return None


def export_to_inference(model_path: Path, config_path: Path) -> Path:
    """Export PaddlePaddle model to inference format."""
    inference_dir = OUTPUT_DIR / "inference"
    export_script = PPOCR_REPO / "tools" / "export_model.py"

    if not export_script.exists():
        print(f"[ERROR] {export_script} not found")
        sys.exit(1)

    # Write a small wrapper that forces CPU before PaddleOCR export_model runs,
    # because paddlepaddle-gpu tries to load cuDNN even with use_gpu=false.
    wrapper = OUTPUT_DIR / "_cpu_export.py"
    wrapper_code = f'''
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["FLAGS_use_gpu"] = "0"
import paddle
paddle.set_device("cpu")
sys.path.insert(0, r"{PPOCR_REPO}")
from tools.program import load_config, merge_config, ArgsParser
from ppocr.utils.export_model import export
FLAGS = ArgsParser().parse_args()
config = load_config(FLAGS.config)
config = merge_config(config, FLAGS.opt)
export(config)
'''
    wrapper.write_text(wrapper_code, encoding="utf-8")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PPOCR_REPO) + os.pathsep + env.get("PYTHONPATH", "")

    # Add nvidia cuDNN from pip package to PATH (paddlepaddle-gpu needs it)
    cudnn_bin = Path(sys.executable).parent.parent / "Lib" / "site-packages" / "nvidia" / "cudnn" / "bin"
    if cudnn_bin.exists():
        env["PATH"] = str(cudnn_bin) + os.pathsep + env.get("PATH", "")

    cmd = [
        sys.executable, str(wrapper),
        "-c", str(config_path),
        "-o", f"Global.pretrained_model={str(model_path).replace(chr(92), '/')}",
        "-o", f"Global.save_inference_dir={str(inference_dir).replace(chr(92), '/')}",
        "-o", "Global.use_gpu=false",
        "-o", "Global.export_with_pir=false",
    ]

    print(f"Exporting to inference format (CPU mode)...")
    print(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PPOCR_REPO), env=env)
    if result.returncode != 0:
        print(f"[ERROR] Export to inference format failed")
        sys.exit(1)

    return inference_dir


def convert_to_onnx(inference_dir: Path) -> Path:
    """Convert PaddlePaddle inference model to ONNX."""
    onnx_path = OUTPUT_DIR / "ba_rec.onnx"

    # paddle2onnx 1.x uses Python API; 2.x uses CLI (__main__).
    # Write a small script that works with both.
    conv_script = OUTPUT_DIR / "_p2o_convert.py"
    conv_code = f'''
import paddle2onnx
model_file = r"{inference_dir / "inference.pdmodel"}"
params_file = r"{inference_dir / "inference.pdiparams"}"
save_file = r"{onnx_path}"
paddle2onnx.export(model_file, params_file, save_file, opset_version=14, enable_onnx_checker=True)
print("ONNX export OK:", save_file)
'''
    conv_script.write_text(conv_code, encoding="utf-8")

    print(f"Converting to ONNX...")
    result = subprocess.run([sys.executable, str(conv_script)], cwd=str(REPO))
    if result.returncode != 0:
        print(f"[ERROR] ONNX conversion failed. Install: pip install paddle2onnx")
        sys.exit(1)

    return onnx_path


def main():
    parser = argparse.ArgumentParser(description="Export trained model to ONNX")
    parser.add_argument("--model-dir", type=str, default="auto",
                        help="Path to trained model (default: auto-detect best)")
    args = parser.parse_args()

    # Find config
    config_path = WORK_DIR / "ba_rec_train.yml"
    if not config_path.exists():
        print(f"[ERROR] Training config not found at {config_path}")
        print(f"  Run 03_train.py first.")
        sys.exit(1)

    # Find model
    if args.model_dir == "auto":
        model_path = find_best_model()
        if model_path is None:
            print(f"[ERROR] No trained model found in {OUTPUT_DIR}")
            print(f"  Run 03_train.py first.")
            sys.exit(1)
    else:
        model_path = Path(args.model_dir)

    print(f"Model: {model_path}")

    # Step 1: Export to inference format
    inference_dir = OUTPUT_DIR / "inference"
    if not (inference_dir / "inference.pdmodel").exists():
        inference_dir = export_to_inference(model_path, config_path)
    else:
        print(f"Inference model already exists at {inference_dir}")

    # Step 2: Convert to ONNX
    onnx_path = convert_to_onnx(inference_dir)
    print(f"ONNX model: {onnx_path}")

    # Step 3: Embed character dict into ONNX metadata (RapidOCR looks for 'character' key)
    char_dict_path = PPOCR_REPO / "ppocr" / "utils" / "ppocr_keys_v1.txt"
    if char_dict_path.exists():
        import onnx
        model = onnx.load(str(onnx_path))
        chars = char_dict_path.read_text(encoding="utf-8")
        meta = model.metadata_props.add()
        meta.key = "character"
        meta.value = chars
        onnx.save(model, str(onnx_path))
        print(f"Embedded {len(chars.splitlines())} chars into ONNX metadata")
    else:
        print(f"[WARN] Character dict not found at {char_dict_path}")

    # Step 4: Copy to final location
    FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    final_path = FINAL_MODEL_DIR / "ba_rec.onnx"
    shutil.copy2(str(onnx_path), str(final_path))

    # Also copy character dict for reference
    if char_dict_path.exists():
        shutil.copy2(str(char_dict_path), str(FINAL_MODEL_DIR / "ppocr_keys_v1.txt"))

    print(f"\n{'='*60}")
    print(f"Export complete!")
    print(f"  ONNX model: {final_path}")
    print(f"\nThe pipeline will automatically load this model on next start.")
    print(f"  (Configured in brain/pipeline.py, server/app.py, vision/engine.py)")


if __name__ == "__main__":
    main()
