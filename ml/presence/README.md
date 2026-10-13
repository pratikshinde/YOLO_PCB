# Presence model: is every SMT part on the red LG Indicator board?

**Status 2026-10-12: the v3 model is trained, evaluated and inside the Android app** (`../../android_app`). Everything about it, all experiments and the limits: **`v3/REPORT_V3.md`**. The pipeline is in `v3/` (build_cache, build_datasets, train_eval, decision_rules, export_app_model, label_overrides.json). The v1 / v2 parts below are history (they were trained on partly wrong labels, see the report).

Current model: YOLO11n-cls, exposure-normalised crops, median over 9 windows per part, threshold 0.35. On whole boards it had never seen: 0 of 291 removed parts missed, 1 of 15,103 good parts called missing. **Only resistors are proven**; R2K2 and the ICs have no real missing example.

## Check a photo on the PC (same model and steps as the app)

```
D:\Projects\YOLO_PCB_Componet\ml\presence\.venv\Scripts\python.exe D:\Projects\YOLO_PCB_Componet\ml\presence\predict_photo.py "C:\path\photo.jpg"
```

A folder or a pattern also works (`...\data\presence\test\red`). For every photo it prints **PASS**, **FAIL** (with the missing part names) or **RETAKE** (board not found / cut off / too many parts flagged = not a plausible board photo), and writes `predictions\<photo>_result.jpg` (aligned board, **yellow** box = missing, cyan = doubtful) and `predictions\<photo>_scores.csv` (all 86 scores).
Options: `--threshold 0.35`, `--doubt 0.10`, `--windows 9|1` (thorough or fast), `--out`, `--model`, `--card`, `--no-norm` (for the older v1 / v2 models), `--golden`. Defaults come from the model card `v3\final\presence_model.json`.

**Do not use the photos in `data\presence\raw` as a test:** the final model was trained on them. Test with new photos (`data\presence\test\red`: 11 photos, labels not yet known).

### v1 / v2 history (below this line)

## What is where

| Path | What |
|---|---|
| `predict_photo.py` | the checker above (golden alignment -> 86 crops -> model) |
| `models\presence_cls.onnx` + `.json` | the final network for the phone (5.9 MB, input 1x3x128x128 RGB/255, output [p(missing), p(present)]); the json is the model card with the exact crop rule. Tested with OpenCV 4.10 DNN (the OpenCV inside the app): identical to PyTorch |
| `runs\final\weights\best.pt` | the final trained network (trained on all 4 removed boards + all good photos) |
| `runs\fold_1..4` | cross-validation models: fold k never saw the removed board it is judged on |
| `report\REPORT.md` | evaluation written by Antigravity (cross-validation) |
| `report\LEAKAGE_CHECK.md`, `checks\`, `board_groups.json` | leakage check (text files, split, same-board photos), the scripts that reproduce it, and which photos are the same physical board |
| `make_dataset.py`, `train.py`, `evaluate.py` | Antigravity's scripts (crops, YOLO11n-cls training, evaluation) |
| `crops\`, `datasets\` | the 128x128 crops and the folds built from them |

## How good is it (measured)

* Antigravity's cross-validation, 50 usable photos: all 102 removed-part crops found, 0 of 4198 present crops called missing at 0.5.
* My independent check on **all 71 photos** (including the 21 that the training script had skipped), each judged by a model that never saw that board: 134 of 134 removed-part crops found, 1 of 5972 present crops called missing at 0.5 (C100N_8, score 0.60), none at 0.6 or above; all 28 photos of removed boards FAIL, none of the 43 good photos FAIL (but those 43 photos are only **6 boards**, see "Is there data leakage?"). The weakest correct detections: R39 0.84, Q2 0.93, all others 0.99 or more.
* Threshold: anything from 0.5 to 0.8 separates the classes on this data. The 0.35 recommended by `evaluate.py` is lower than necessary (2 false calls in my check). `predict_photo.py` uses 0.5.
* Position tolerance (crop window moved against the part, present parts only): 0.5 mm -> 0.7 % called missing, 1.0 mm -> 23 %, 1.5 mm -> 55 %, 2 mm -> 75 %. So a part that sits 0.5 mm off is fine; at 0.9-1.0 mm some are already called missing (more along the short side of the part). The training had no real shifted parts.

## Limits (be honest about these)

* Only **resistors (R)** and the transistors **Q2, Q9** were ever removed (4 boards, 19 part instances). Capacitors (C, C100N), diodes (D, D4148) and the ICs (U1, U2) have **no real missing example**: for them the model is untested.
* **10 physical boards in total**: the 43 good photos are only 6 boards (6-9 photos each), plus the 4 removal boards. No independent test set yet.
* Weakest present slot: C100N_8 (right edge of the board) scores 0.2-0.6 in some photos.
* Red board only. A blue or green board will be flagged wrongly (many parts) and gives RETAKE.

## Is there data leakage? (checked 2026-10-09; details and scripts: `report\LEAKAGE_CHECK.md`, `checks\`)

* **The text files never reach the prediction.** `predict_photo.py` only opens the photo, the golden image, `regions.json` and the model. The text files are used for the training labels and for scoring afterwards. Blind test (scrambled file names, no text files, each photo judged by a model that never saw its board): 28/28 removed-board photos with exactly the right part list, 26/26 good photos PASS.
* **Split by photo is clean:** no photo or crop is in both train and validation of any fold; the slots missing on the held-out board were never missing on a training board.
* **One real leak:** the 43 good photos are only **6 physical boards** (`board_groups.json`) and the fold blocks cut through 3 of them (11 of 28 validation photos had their board in training). Re-measured with only unseen boards the result is the same (highest false score 0.078 on clean good boards; all 134 removed-part crops still found), so nothing was inflated - but the evidence rests on 10 boards, not 47.
* The runs of `predict_photo.py` on photos from `data\presence` with the default (final) model are not a test: it was trained on them.

## Train again with more data

`make_dataset.py` has the 4 removal boards (`m_01..m_04`), the good photo names (`R_*.jpg`) and the fold blocks hard-coded, and it uses the strict alignment (it skipped 21 photos that `predict_photo.py` aligns fine: `min_inlier_ratio` 0.25 -> 0.10 plus a grey margin). To add boards it has to be extended first. Order after that: `make_dataset.py` -> `train.py --fold all` (about 30 min on the RTX 5050) -> `evaluate.py`.
Split by **board**, not by photo number (`board_groups.json`), and name new good photos by board (`G07_01.jpg`, `G07_02.jpg`: the part before the underscore is the board). A new board counts as one board however many photos it has.
Most useful new data: **more different boards** (good ones too), boards with **capacitors, diodes and ICs removed** (each part type at least 3 times), then a separate set of new boards photographed last that no training step ever sees.
