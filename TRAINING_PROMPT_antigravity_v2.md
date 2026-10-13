# TASK FOR ANTIGRAVITY: retrain the PCB "is the component present?" model on the new, larger data set, end to end

You are working in `D:\Projects\YOLO_PCB_Componet` (Windows 11, PowerShell, RTX 5050 8 GB, 16 GB RAM).
A first model already exists in `ml\presence\` (YOLO11n-cls, trained on 10 boards). The user has now added much more data. Retrain carefully and honestly.
Read this whole file before doing anything. Work step by step. If something in the data is unclear or wrong, STOP that step, write it into `ml\presence\report\DATA_PROBLEMS.md` and continue with what is certain. NEVER guess a label and NEVER invent results.

## 0. The one goal
For the red "LG Indicator" board (116 x 83 mm, SMT stage), decide for each of the **86 SMT parts**: **present** or **missing**. Nothing else (no misplaced / wrong part / solder defect classes).
A part up to about 0.9 mm off its place still counts as present. Method: whole-board phone photo -> align to the golden reference -> cut one 128x128 crop per part slot -> YOLO11n-cls says missing/present -> board FAILS if any part is missing.
The golden reference is `data\presence\golden\red\grf.jpeg` (SHA-256 starts BA859CA72C6B742F). It is used for alignment and for cutting crops. NEVER replace it with another image.

## 1. Hard rules
1. Never modify, move, rename or delete anything under `data\`. Read only.
2. Write only under `ml\presence\` (new things), plus the archive move in step 2. Do not edit `scripts\golden_compare.py`, `android_app\`, `predict_photo.py`, or the existing `runs\final`, `runs\fold_*`, `models\`, `report\REPORT.md` (keep them as the old baseline; write new outputs next to them with a `v2` suffix: `runs_v2\`, `datasets_v2\`, `crops_v2\`, `report_v2\`, `models_v2\`).
3. No git commands.
4. The `*_missing.txt` label files are used ONLY (a) to label training crops of removal boards and (b) to score predictions afterwards. They must never be an input to any prediction.
5. Never split train / validation / test by photo. Photos of one physical board are almost identical. **Split by board only.**
6. The test boards (`data\presence\test\...`, if present) are used ONCE, at the very end, after the model and threshold are frozen. Never train, tune the threshold, or pick epochs on them.
7. Report real numbers only. If something fails, say so with the output.

## 2. Clean-up of the old project parts (ARCHIVE, do not delete)
The old colour-check / misplaced / displaced / "difference" logic is obsolete (it was proven not to find really removed parts). Move (do not delete) these to `archive\2026-10-10_old_checks\` keeping their relative paths, and write `archive\2026-10-10_old_checks\MANIFEST.md` listing every moved file:
- `scripts\test_missing_simulation.py`, `scripts\golden_acceptance.py`, any other script or doc whose only purpose is the old exposure / MISPLACED / DIFFERENT / DIFFERENCE-map check or its simulation.
- Anything under `outputs\`, `data\...` old products ALREADY in `archive\` stay where they are.
Do NOT move `scripts\golden_compare.py` (alignment code the new pipeline needs), `scripts\test_golden_compare.py`, anything in `android_app\`, `ml\`, `data\`, `planner.md`, `README`s. In `MANIFEST.md` also list (do not touch) the old code that remains inside `golden_compare.py` and `android_app\` so the user can plan removing it from the app later.

## 3. Inspect the new data (read only) and write `report_v2\DATA_INVENTORY.md`
Follow `data\presence\HOW_TO_ADD_PHOTOS.txt` and `data\presence\README.md` for the intended layout, but check what is really there:
1. Walk `data\presence\` and list every folder with the number of photos and of text files.
2. Golden: confirm `golden\red\grf.jpeg` has the SHA-256 prefix above.
3. Good boards: photos named by board (e.g. `G07_01.jpg`: the part before the underscore is the board). The OLD photos `R_01..R_43` have no board ID: they are 6 boards, see `ml\presence\board_groups.json` (G01..G06). Use that file for them.
4. Removal boards: one folder per board (or per step); `<board>_missing.txt` with comma separated part names (case insensitive). Parse strictly against the 86 valid part names from `android_app\assets\golden\regions.json` (entries with `kind == "part"`; names like R12, Q9, C14, D13, U1, U2, R2K2_1..8, C100N_1..8, D4148_1..8). Windows may hide `.txt`, files can be called `x.txt.txt`: accept `*.txt*`. An unknown name, a never-fitted part (R50, R32, Q11, D14 must never be listed), or a missing file = a DATA PROBLEM: list it, do not guess.
   Step-by-step boards (`...\stepN\`): the missing list belongs to the photos of THAT step only.
5. Verify the labels with your own eyes on a sample: for 3 removal photos, cut the crop of each listed part and confirm the pads are empty (view the crops). Also confirm a few unlisted parts are present.
6. Count: number of physical boards (good / removal / test / bare), photos per board, removed part instances **per part type** (R, R2K2, C, C100N, D, D4148, Q, U). Give the table. State clearly which part types still have fewer than 15 real missing examples or come from fewer than 5 boards: those are NOT proven.
7. Board identity check: the old photos were found to be 6 boards by comparing solder texture. Re-use `ml\presence\checks\identity_all.py` logic (high-pass crop correlation of the same slot; same board >= 0.87, different boards <= 0.71, calibrate again on the removal boards) on ALL photos to confirm that the board IDs in the file names are right: no two IDs are the same physical board, no ID mixes two boards, no good board is also a removal / test board. Report every conflict in `DATA_PROBLEMS.md` and resolve by merging groups for splitting purposes (when unsure put both IDs in the same fold).

## 4. Build the crops (`ml\presence\make_dataset_v2.py`, new file; do not edit make_dataset.py)
Re-use the product pipeline from `ml\presence\predict_photo.py` (import `align`, `square_crop`, `make_params`; it uses `scripts\golden_compare.py`): 20 px/mm canonical frame, `min_inlier_ratio` 0.10, grey-margin retry when the board touches the photo edge (this rescues photos the first script skipped), local optical-flow shift per part, square crop of side `round(max(w,h) + 2*1.5*20)` px, BORDER_REFLECT_101 outside the frame, resize 128x128 INTER_AREA.
- Skip a photo only if it cannot be aligned even then; list each skipped photo and the reason in `report_v2\skipped_photos.json`. Skipped photos are never used anywhere.
- Save crops as PNG `crops_v2\all\<board>__<photo>__<part>.png` and `crops_v2\index.csv` (board, photo, part, part_type, label, step, dx, dy, residual_mm).
- Label = `missing` only if the part is in the missing list of that board / step, else `present`. Never-fitted parts are not part slots (the 86 only).
- Alignment quality gate: write the alignment residual per photo; list photos with residual > 0.5 mm and exclude them from training and validation (still list them).
- Synthetic "window-shift" crops (a part displaced out of its window, label `missing`): crop window moved 1.3-2.5 mm in a random direction from PRESENT parts, only from training boards, at most as many as there are real missing crops. Never in validation or test.
- Additionally add "tolerance" jitter for `present`: windows moved 0.0-0.6 mm in a random direction (label `present`), training boards only, about 20 % of the present training crops. Do not use shifts between 0.6 and 1.3 mm.
- Make contact sheets (one per board, a few photos) to eyeball: `report_v2\sheets\`.

## 5. Splits (BY BOARD)
- Folds = leave-boards-out cross-validation. Put every removal board in a fold so that each fold validates on roughly equal numbers of removal boards, and give every fold at least 2 good boards. All photos of a board are in the same fold. Use up to 5 folds; with fewer than 15 removal boards use as many folds as removal boards (leave-one-removal-board-out) if that is not more than 8 folds.
- Train set of a fold never contains any photo, crop or (by the identity check) any board of its validation set. Write `report_v2\split_audit.md` proving it (0 shared photos, 0 byte-identical crops, 0 shared boards, held-out missing slots listed). Re-use `ml\presence\checks\audit_split.py` ideas.
- Train: oversample real missing crops to about 1 : 4 missing : present (copies named `__repN`, training only).
- Validation: real crops only, no copies, no synthetic.
- Final model: trained on ALL boards except the test boards.
- Test boards (`data\presence\test\red\...`): never in any training or validation set.

## 6. Training (`ml\presence\train_v2.py`)
- Environment: the existing venv `ml\presence\.venv` (Python 3.11, torch 2.11.0+cu128, ultralytics 8.4.174). Use it, do not recreate it. Check `torch.cuda.is_available()` first.
- Start every fold AND the final model from the pretrained `ml\presence\yolo11n-cls.pt` (NOT from the old `runs\final`, which has seen the old boards and would leak into the folds).
- Settings: imgsz 128, epochs 60, **patience 60** (the validation accuracy saturates at 100 % in the first epochs, so the default early stop ends training too early and the learning rate is never reduced), batch 64, `cos_lr=True`, fliplr 0.5, flipud 0, hsv_h 0.015, hsv_s 0.5, hsv_v 0.4, degrees 5, translate 0.04, scale 0.15, erasing 0, workers 0, seed 0, `project=runs_v2`, `exist_ok=True`.
- Keep `last.pt` (end of the cosine schedule) as the model to evaluate and ship; do not pick the "best epoch" on validation data.
- Print epoch time estimates first; run the folds one after the other; log to `runs_v2\train_log.txt`. If CUDA fails, say so and use `--device cpu` only for a quick smoke test, not for the real run.

## 7. Evaluation (`ml\presence\evaluate_v2.py`) -> `report_v2\REPORT_V2.md`
Per held-out board, with each fold's `last.pt` on its held-out photos (all photos of those boards that aligned):
1. Part level: found / missed, false calls, at thresholds 0.3, 0.5, 0.7. Also the lowest score among real missing parts and the highest among present parts.
2. **By part type** (R, R2K2, C, C100N, D, D4148, Q, U): present count, false calls, real missing count, recall, and the number of distinct boards the missing examples come from. Mark a type "NOT PROVEN" if it has fewer than 15 real missing examples or fewer than 5 boards.
3. Board level: removal photos that FAIL, good photos that PASS, and whether the missing list is exactly right.
4. Window-shift tolerance: present parts with the window moved 0.5 / 0.9 / 1.0 / 1.5 mm in 4 directions: share called missing at the chosen threshold. Goal: at 0.9 mm below 5 %.
5. Worst 30 false calls and every missed part, with crop paths, and contact sheets.
6. Threshold choice: the smallest threshold with 0 missed real parts and the fewest false calls on the VALIDATION folds; the threshold must be a round value in 0.4-0.8 with margin on both sides; explain the margin. Freeze it before touching the test set.
7. Honest limitations section: which part types are proven, how many boards, what was never seen.
8. Leakage section: confirm no test / validation board in training, and that no prediction step reads label files.

## 8. Final model, test set, export
1. Train the final model (all non-test boards, same settings). Write `runs_v2\final\weights\last.pt`.
2. If `data\presence\test\red\` has boards: run the frozen final model + threshold ONCE through `ml\presence\predict_photo.py --model <final> --threshold <frozen>` on scrambled copies of the test photos (copy to a scratch folder as `img_xxxx.jpg`, no text files; use the same blind approach as `ml\presence\checks\blind_test.py`), then score against the labels AFTERWARDS. Report: removed parts found X of Y by part type, good boards falsely failed X of Y, doubtful parts. Do not retrain after seeing the test result; if it is bad, report it as bad.
3. Export ONNX: `YOLO(last.pt).export(format="onnx", imgsz=128, batch=1, dynamic=False, simplify=False, opset=12)` -> `models_v2\presence_cls.onnx` + `presence_cls.json` model card (copy the format of `models\presence_cls.json`; add SHA-256, training boards count, the frozen threshold, the part types that are NOT proven).
4. Check the ONNX against PyTorch on 500 crops with OpenCV DNN (install `opencv-python-headless==4.10.0.84` with `pip install --target <scratch folder>`, not into the venv): max difference must be < 1e-4.

## 9. Deliverables (all in `ml\presence\`)
`make_dataset_v2.py`, `train_v2.py`, `evaluate_v2.py`, `crops_v2\`, `datasets_v2\`, `runs_v2\`, `models_v2\presence_cls.onnx` + `.json`, `report_v2\REPORT_V2.md`, `report_v2\DATA_INVENTORY.md`, `report_v2\DATA_PROBLEMS.md`, `report_v2\split_audit.md`, `report_v2\skipped_photos.json`, `archive\2026-10-10_old_checks\MANIFEST.md`, and `ml\presence\README_V2.md` (how to rerun everything in the right order, with commands).
At the end print a summary: boards by type, crops, per-type recall and false calls on the validation folds, the frozen threshold, the test-set result (if any), the NOT-PROVEN part types, and every problem you found.

## 10. Things that went wrong last time (do not repeat)
- Splitting good photos by photo number cut through physical boards (leak). Split by board.
- Early stopping at 100 % validation accuracy ended training after ~26 epochs: use patience 60 and `last.pt`.
- Strict alignment (inlier ratio 0.25) skipped 21 usable photos: use 0.10 plus the grey-margin retry.
- The "recommended threshold" was tuned on the same data it was evaluated on: choose it on validation folds, freeze it, then test.
- Only R resistors and Q2, Q9 had ever been removed: report per part type and say which types are unproven.
