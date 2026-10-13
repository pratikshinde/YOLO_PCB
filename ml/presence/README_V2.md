# Antigravity Presence Detection Pipeline (v2)

This pipeline trains, validates, and deploys the end-to-end component presence detection model for the **LG Indicator** red PCB (116 x 83 mm, 86 SMT part slots) using **YOLO11n-cls**.

---

## 1. Directory Structure

```text
D:\Projects\YOLO_PCB_Componet\
├── data\presence\
│   ├── golden\red\grf.jpeg            # Canonical golden reference image
│   ├── raw\red\
│   │   ├── good\                      # Good boards: G01..G06 (R_01..R_43), P1..P15
│   │   └── removed\m_01..m_10\        # 10 removal boards + *.txt.txt label files
│   └── test\red\                      # 11 held-out test photos (blind evaluation)
├── ml\presence\
│   ├── make_dataset_v2.py             # Crop extraction and dataset generator
│   ├── train_v2.py                    # 60-epoch YOLO11n-cls training script
│   ├── evaluate_v2.py                 # Cross-validation & tolerance evaluator
│   ├── test_blind_v2.py               # Blind test runner on held-out photos
│   ├── export_v2.py                   # ONNX exporter & OpenCV DNN numerical verifier
│   ├── crops_v2\                      # 15,394 real crops + index.csv
│   ├── datasets_v2\                   # 5 CV fold datasets + final dataset
│   ├── runs_v2\                       # Training checkpoints (last.pt) and metrics
│   ├── models_v2\                     # presence_cls.onnx and presence_cls.json
│   └── report_v2\                     # DATA_INVENTORY, DATA_PROBLEMS, REPORT_V2.md
└── archive\2026-10-10_old_checks\     # Obsolete difference map checks & simulation
```

---

## 2. Execution Sequence (Step-by-Step)

Activate the Python virtual environment:
```powershell
.\ml\presence\.venv\Scripts\Activate.ps1
```

### Step 1: Data Preprocessing & Crop Extraction
Extracts 128x128 canonical crops from aligned whole-board photos, generates oversampled folds with synthetic window-shift and tolerance jitter crops, and performs split audit:
```powershell
python ml\presence\make_dataset_v2.py
```
Outputs:
- Real crops in `ml\presence\crops_v2\all\`
- Crop index in `ml\presence\crops_v2\index.csv`
- Contact sheets in `ml\presence\report_v2\sheets\`
- Datasets for `fold_1`..`fold_5` and `final` in `ml\presence\datasets_v2\`
- Split isolation report in `ml\presence\report_v2\split_audit.md`

### Step 2: Model Training
Trains YOLO11n-cls models across the 5 cross-validation folds and the full final dataset (60 epochs each, cosine LR schedule, no early stopping, workers=0):
```powershell
# Train all 5 folds and final model sequentially
python -u ml\presence\train_v2.py --fold all --epochs 60 --patience 60

# Or train a specific fold individually:
python -u ml\presence\train_v2.py --fold fold_1 --epochs 60 --patience 60
python -u ml\presence\train_v2.py --fold final --epochs 60 --patience 60
```
Outputs:
- Trained model checkpoints in `ml\presence\runs_v2\<fold>\weights\last.pt`
- Training metrics and logs in `ml\presence\runs_v2\train_log.txt`

### Step 3: Validation & Metric Evaluation
Evaluates all 5 cross-validation folds using held-out board photos and each fold's `last.pt`. Selects and freezes the optimal threshold, measures false alarm rates, and checks window-shift tolerance:
```powershell
python ml\presence\evaluate_v2.py
```
Outputs:
- Comprehensive report in `ml\presence\report_v2\REPORT_V2.md`
- Visual sheets for false alarms in `ml\presence\report_v2\sheets\`

### Step 4: Blind Evaluation on Held-Out Test Set
Scrambles filenames of photos in `data\presence\test\red\` into a scratch directory, evaluates them blindly using `predict_photo.py` with the frozen final model (`runs_v2\final\weights\last.pt`), and scores them post-hoc:
```powershell
python ml\presence\test_blind_v2.py --threshold 0.50
```
Outputs:
- Blind test evaluation logs in `ml\presence\report_v2\blind_test_results.json`

### Step 5: ONNX Model Export & Parity Verification
Exports `runs_v2\final\weights\last.pt` to ONNX (imgsz 128, batch 1, opset 12), generates the model card `presence_cls.json`, and verifies numerical parity against PyTorch on 500 crops using OpenCV 4.10 DNN:
```powershell
python ml\presence\export_v2.py --threshold 0.50
```
Outputs:
- Production model: `ml\presence\models_v2\presence_cls.onnx`
- Metadata & Model Card: `ml\presence\models_v2\presence_cls.json`
- Verification: `Max diff < 1e-4`
