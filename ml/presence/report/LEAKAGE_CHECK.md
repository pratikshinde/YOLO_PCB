# Leakage check (2026-10-09)

Question: does any label (the `*_missing` text files) reach the prediction, and does anything of a validation board sit in the training set?
Scripts that reproduce every number: `ml/presence/checks/` (`audit_split.py`, `identity_all.py`, `blind_test.py`).

## 1. Do the text files reach the prediction? No

* `predict_photo.py` opens only: the photo, the golden image, `regions.json` (positions of the 86 parts, nothing about missing parts) and the model weights. It never opens a `.txt` file (searched). `golden_compare.py` reads only `regions.json`.
* The text files are used in exactly two places: `make_dataset.py` (to label the crops of the removed boards as `missing` for training - that is the supervision) and the scoring (`evaluate.py`, my checks) after the predictions exist.
* Blind end-to-end test (`checks/blind_test.py`): photos copied to a scratch folder under scrambled names (`img_1234.jpg`), no text files, no folder hints; every photo judged by the fold model that never saw its board; text files read only afterwards.
  Result: **28 of 28 photos of removed boards: right FAIL and exactly the right part list; 26 of 26 good photos from boards the judging model never saw: PASS.**
* Not a test: runs of `predict_photo.py` with the default (final) model on photos from `data/presence` - the final model was trained on all of them (that is why it prints 1.00 for every removed part).

## 2. Training / validation split by photo (`checks/audit_split.py`)

For all 4 folds: 0 photos in both train and val, 0 byte-identical crop files in both, 0 synthetic or repeated crops in val, synthetic (shifted-window) crops only from training photos.
The parts that are missing on the held-out removal board were **never missing on any training board** (every removed slot is missing on exactly one board), so the model finds slots it never saw empty: it did not memorise slots.

## 3. Real leakage found: the good photos are only 6 boards (`checks/identity_all.py`)

Two photos of the same physical board are easy to tell from two different boards: the solder fillets and flux marks of one slot correlate >= 0.87 between photos of one board and <= 0.71 between different boards (calibrated on the removal boards: 4 boards, 6-8 photos each).

* The 43 good photos `R_01..R_43` are **6 boards, 6-9 photos each**: R_01-07, R_08-14, R_15-21, R_22-28, R_29-37, R_38-43 (`board_groups.json`). No good photo is one of the 4 removal boards (highest similarity 0.76).
* Together with the removal boards: **10 physical boards** in the whole data set.
* `make_dataset.py` split the good photos by photo number blocks (R_01-11 / 12-22 / 23-33 / 34-43), which cut through three boards. In the cross-validation, **11 of the 28 usable good validation photos had their own board in the training set** (R_09-14, R_29, R_30, R_32, R_33, R_37). The 15 skipped good photos are not in any training set, but 6 of them (R_08, 22, 31, 34, 35, 36) also have a same-board photo in the training set of the model that judged them.

## 4. Effect on the numbers (my check, held-out fold models, 71 photos)

| photos | count | present crops | highest p(missing) on a present part | present parts >= 0.5 |
|---|---|---|---|---|
| removal boards (4 boards, never seen) | 28 | 2274 | 0.598 (C100N_8, board m_01) | 1 |
| good photos whose board is NOT in training | 26 | 2236 | 0.078 | 0 |
| good photos whose board is ALSO in training | 17 | 1462 | 0.220 | 0 |

All 134 removed-part crops (all on never-seen boards) are found at 0.5, lowest score 0.838. **The leakage did not inflate the result**: the leak-free groups look as good as the leaky one. But "0 false alarms on 43 good boards" really means **6 good boards (+ 4 removal boards)**: the false-alarm rate on new boards is much less certain than 43 boards would suggest.

## 5. Smaller points

* `best.pt` / early stopping used the held-out board's validation accuracy. On every fold it was 1.00 from epoch 1-6 on, so the selection changed nothing; strictly the fold numbers are validation, not test, numbers.
* The recommended threshold (0.35 in `REPORT.md`) was picked on the same predictions; any value from 0.5 to 0.8 works on this data, `predict_photo.py` uses 0.5.
* Only R resistors and Q2 / Q9 were ever removed; C, C100N, D, D4148 and U have no real missing example (not a leak, a gap).

## 6. What to change for the next training

1. Split by **board**, not by photo number: use `board_groups.json`; validate on whole boards (leave-one-board-out over all boards, good ones included).
2. New photos: name them by board (`G07_01.jpg`, `G07_02.jpg`, ...): the part before the underscore is the board. One more board counts as ONE board, however many photos it has; seven more boards are worth far more than seventy more photos of the boards we have.
3. Keep a **test set of new boards** that no training or tuning step ever sees.
