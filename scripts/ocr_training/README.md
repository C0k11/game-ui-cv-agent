# Blue Archive OCR Fine-tuning Pipeline

Fine-tune PaddleOCR recognition model on Blue Archive game text for improved
Traditional/Simplified Chinese recognition with stylized game fonts.

## Pipeline Steps

```
01_extract_crops.py   → Extract text crops from trajectory screenshots
02_generate_synthetic.py → Generate synthetic training data with CJK fonts
03_train.py           → Fine-tune PP-OCRv4 rec model with PaddlePaddle
04_export_onnx.py     → Export trained model to ONNX format
05_evaluate.py        → Evaluate model accuracy on trajectory data
```

## Quick Start

```powershell
# 1. Install dependencies
pip install -r scripts/ocr_training/requirements.txt

# 2. Extract training data from trajectories (~30k screenshots)
py -3 scripts/ocr_training/01_extract_crops.py

# 3. Generate synthetic training data (needs fonts in data/fonts/)
py -3 scripts/ocr_training/02_generate_synthetic.py

# 4. Fine-tune (requires GPU, ~2-4 hours on RTX 4090)
py -3 scripts/ocr_training/03_train.py

# 5. Export to ONNX
py -3 scripts/ocr_training/04_export_onnx.py

# 6. Evaluate
py -3 scripts/ocr_training/05_evaluate.py
```

## Output
- `data/ocr_model/ba_rec.onnx` — Fine-tuned recognition model
- Pipeline automatically loads custom model when present

## Data Sources
- **Trajectory screenshots**: 30k+ real game screenshots with OCR bounding boxes
- **Synthetic text**: Generated with CJK fonts + Blue Archive styling
- **Corrections dictionary**: Known OCR misread patterns from pipeline code
