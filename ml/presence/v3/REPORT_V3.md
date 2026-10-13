# Fine-tuning and evaluation report - missing-part check (2026-10-12)

Goal (business requirement): find every missing SMT component of the red LG Indicator board from one phone photo, with as few false alarms as possible, offline in the app.
Everything below was measured on **whole boards the model had never seen** (5-fold leave-boards-out cross-validation, last epoch, no epoch picking on validation data).

## 1. What was wrong before, and what was fixed

| Problem found | Fix |
|---|---|
| The v2 training that was reported as done was **not finished**: only fold 1 (5 of 60 epochs) and fold 2 (12 epochs), both interrupted; no folds 3-5, no final model, no ONNX. | Rebuilt the pipeline in `ml/presence/v3/` (5 folds in about 20 minutes each experiment, final model, export). |
| **Labels in the text files were wrong** in places: parts that were missing but listed as present (continued boards were not listed cumulatively), one part listed on the wrong board, unreadable tokens (`C8`, `R20`, `D'4`, `D'8`). The model was taught that empty pads are fine. | `v3/label_overrides.json` (your text files are untouched): m_01 +C100N_8; m_05 +R59; m_07 +R23, C100N_8; m_08 +R28, R38, D4148_4, D4148_8; m_09 +R40, R48; m_10 -R48. Every entry checked by eye on the crops. |
| The 10 removal folders are really **6 physical boards** (m_01/07, m_02/08, m_03/09, m_04/06 are the same board continued; m_05 and m_10 are separate). | Continued boards stay together in one fold. |

## 2. Data used

* 21 good boards (G01-G06 = the first 43 photos, P1-P15 = 6 photos each), 6 physical removal boards, 179 usable photos (7 retake photos skipped).
* Held-out crops in the 5 folds: **291 missing** and 15,103 present part crops. Real missing instances per type: R 45 (6 boards), Q 6 (4), D 3 (2), D4148 2 (1), C100N 2 (1), C 1 (1), R2K2 0, U 0.

## 3. Experiments (each = 5 folds; threshold 0.5 unless stated)

| Experiment | What changed | Missed of 291 | False calls of 15,103 | Good photos wrongly failed (of 128) | Exact missing list (of 51 removal photos) |
|---|---|---|---|---|---|
| E0 baseline | corrected labels, yolo11n-cls, 20 epochs | 4 | 1 | 1 | 47 |
| E3 strong augmentation | hsv_s 0.8, hsv_v 0.7, rotation 10, scale 0.25 | 3 | 0 | 0 | 48 |
| E6 degraded copies | 30 % of the training crops copied with blur / exposure / noise / JPEG | 5 | 0 | 0 | 46 |
| E7 combination | tolerance 0.9 mm + E6 + E3 | 12 | 0 | 0 | 39 |
| E8 colour normalisation | per-channel gain per photo | 11 | 0 | 0 | 40 |
| **E10 exposure normalisation** | one exposure gain per photo before cutting crops | **1** | 1 | 1 | **50** |
| E11 E10 + E3 | exposure normalisation + strong augmentation | 6 | 0 | 0 | 45 |

Findings: recall is what matters, and **exposure normalisation (E10) is the only change that clearly improves it**: it removes the failure on over-dark / over-saturated photos (the one miss of the baseline family that is photo-related). Colour normalisation per channel (E8) and the wide-tolerance combination (E7) lose recall. Strong augmentation (E3) is a good alternative with the fewest false alarms but misses 3.
Not run (queued, scripts ready): margin 2.5 mm (E2), bigger network (E1), no shifted-window negatives (E4), window tolerance 0.9 mm alone (E5), and the part-type transfer tests T0-T7 (`python run_queue.py ...`).

## 4. Decision rule on top of the chosen model (E10)

Looking at every part through 9 slightly shifted windows (centre, 4 x +-0.4 mm, 4 x +-0.8 mm) and taking the **median** score removes the dependence on small alignment errors:

| Rule | Missed of 291 | False calls | Good photos failing | Lowest score on a missing part | Highest score on a present part |
|---|---|---|---|---|---|
| centre window only | 1 | 1 | 1 / 128 | 0.05 | 0.57 |
| median of 5 windows | 0 | 1 | 1 / 128 | 0.60 | 0.57 |
| **median of 9 windows (thorough mode)** | **0** | 1 | 1 / 128 | **0.93** | 0.59 |
| lowest of the windows ("any window sees the part") | 11 | 0 | 0 | - | - |
| highest of the windows | 0 | 6 | 2 / 128 | - | - |

The median of 9 gives all 291 missing parts a score of at least 0.93 (centre only: some at 0.05). The single false call is a good part whose window slipped sideways by the local alignment (P15 R37); with the missing threshold at 0.7 it also disappears, at the price of a smaller safety margin for part types that have no real missing examples. Default threshold: **0.35** (High 0.25 / Normal 0.35 / Low 0.5 in Settings).

Combining several photos of one board (on the same held-out boards): 3 photos with a majority vote gave 0 missed, 0 false calls, 0 of 494 good combinations failing (173 of 173 exact). Not built into the app (it needs three photos per board); it is the option if even the single false call per ~130 good photos is too many.

## 5. The model that is in the app

* YOLO11n-cls (3 MB weights, ONNX 5.9 MB), trained on all boards, 20 epochs, **exposure-normalised crops**, 128 x 128 px at 20 px/mm, part box + 1.5 mm.
* Thorough mode (default): median over 9 windows; fast mode: one window. Missing threshold 0.35, doubtful from 0.10 (listed, does not fail the board); more than 20 flagged parts = RETAKE.
* Cross-validation result of exactly this setting: **0 of 291 missing parts missed, 1 of 15,103 good parts called missing, 1 of 128 good photos failed, 51 of 51 removal photos with exactly the right list.**
* Checked: ONNX under OpenCV 4.10 DNN equals PyTorch (difference 0.000000); the Kotlin code equals the Python reference on 11 photos (same decisions, same parts, scores within 0.025, 0.8 % of the local shifts differ by more than 0.05 mm); runs on the Redmi phone (fast mode 8.9 s for a photo).
* The 10 held-out **test photos** (`data/presence/test/red`, never used for training or tuning, unlabelled): 6 PASS, 3 FAIL, 1 RETAKE (board cut off). The FAIL photos name R48 + D1 + Q2; R37 + R42 + C14 + C100N_4; R1 + R34 + C2 + D4148_4 + D4148_8 + Q10. By eye on the result images the boxes sit on empty slots (capacitors and diodes included), but **the labels of these photos are not known to me: please compare with your notes.**

## 6. Limits (honest)

1. Only **resistors are proven** (45 real removals on 6 boards). Q (6 instances), D (3), D4148 (2 on one board), C100N (2), C (1) were found in cross-validation but have too few examples to claim anything; R2K2 and the ICs U1 / U2 have **no real missing example at all**.
2. 6 physical removal boards and 21 good boards: the false-alarm rate on new boards is only roughly known (1 in 128 good photos here).
3. A part moved 0.9 mm in a window is still called present in 95 % of the cases (centre rule); at 1.2 mm it is called missing in about half.
4. Stress tests (held-out crops, degraded): blur, darkness, noise and heavy JPEG each cost a few percent of recall for the centre window; thorough mode makes this much less sensitive.
5. Red board only; a board of another colour should give RETAKE.

## 7. What would improve it further

1. Real removals of the part types that have none (U1, R2K2 x8, more C100N / D4148 / C / D / Q) on about 8 more boards, plus a labelled test set. This is the biggest gain; no tuning can replace it.
2. Fill in the labels of the 11 test photos, then they become a real test.
3. Optional experiments: `python run_queue.py E2_margin25 E1_small E4_nosyn E5_tol09 T0_base T7_combo` (about 20 minutes each).

## 8. Reproduce

```
cd ml\presence\v3
..\.venv\Scripts\python.exe build_cache.py                  # align every photo once
..\.venv\Scripts\python.exe build_datasets.py E10_expnorm --final
..\.venv\Scripts\python.exe train_eval.py E10_expnorm --workers 4 --final     # 5 folds + evaluation + final model
..\.venv\Scripts\python.exe decision_rules.py E10_expnorm
..\.venv\Scripts\python.exe export_app_model.py E10_expnorm                    # ONNX + model card -> android_app\assets\model
```
Run one training at a time (two at once crash Windows with error 1455). Check a photo: `ml\presence\.venv\Scripts\python.exe ml\presence\predict_photo.py PHOTO`.
