# Data Problems and Irregularities Log (v2)

Date: 2026-10-10  
Author: Antigravity Automated Dataset Audit

---

## 1. Unrecognized / Ambiguous Part Names in Missing Files

Per project rules (hard rule 4 and section 3.4), missing part files are parsed strictly against the 86 valid part names defined in `android_app/assets/golden/regions.json`. Any token not matching one of the 86 parts must never be guessed as a label.

### A. Board `m_07` (`raw/red/removed/m_07/m_07.txt.txt`)
- **Offending Token:** `C8`
- **Missing File Content:** `C16,R42,Q2,R10,R26,R22,R36,C8`
- **Issue:** `C8` is not an SMT part name in `regions.json`. The valid SMT capacitors are `C2`, `C14`, `C16`, and `C100N_1` through `C100N_8`.
- **Resolution:** `C8` is flagged as an invalid token and excluded from the missing list. Only the 7 strictly verified parts (`C16`, `R42`, `Q2`, `R10`, `R26`, `R22`, `R36`) are treated as missing; all other slots on `m_07` are treated as present.

### B. Board `m_08` (`raw/red/removed/m_08/m_08.txt.txt`)
- **Offending Tokens:**
  - `R20`
  - `D'4(not mentioned the label in the board, due to less space)`
  - `D'8(not mentioned the label in the board, due to less space)`
- **Missing File Content:**
  `R12,R20,R19,R54,R34,D'4(not mentioned the label in the board, due to less space),D'8(not mentioned the label in the board, due to less space)`
- **Issue:**
  1. `R20` does not exist on the board or in `regions.json` (the valid resistors are `R1`..`R4`, `R9`..`R13`, `R18`..`R19`, `R21`..`R31`, `R34`..`R40`, `R42`, `R45`, `R48`..`R49`, `R51`..`R60`, and `R2K2_1`..`8`).
  2. `D'4` and `D'8` are informal notations with embedded comments intended to represent the 1F4 diodes (`D4148_4` and `D4148_8`).
- **Resolution:** Per hard rule, `R20`, `D'4`, and `D'8` are flagged as invalid tokens and omitted from the missing label set to avoid unverified labels. The verified missing parts for `m_08` are `R12`, `R19`, `R34`, `R54`.

---

## 2. Windows File Name Extension Quirk (`*.txt.txt`)
All 10 removal board label files in `data/presence/raw/red/removed/m_01` .. `m_10` were saved with double extensions (e.g. `m_01.txt.txt`), caused by Windows hiding extensions for known file types.
- **Resolution:** Ingestion accepts `*.txt*` patterns to handle these transparently.

---

## 3. Physical Board Overlaps / Solder Texture Similarity (Removal Boards)

Pairwise high-pass solder texture correlation revealed high similarity and shared removal sets between several pairs of removal boards:
1. **`m_03` and `m_09`**: Identical missing list (`Q2, R27, R39, R45`), pairwise solder texture correlation = 0.702.
2. **`m_01` and `m_07`**: Shared missing parts (`R10, R22, R26, R36`), correlation = 0.688.
3. **`m_02` and `m_08`**: Shared missing parts (`R12, R19, R34, R54`), correlation = 0.676.
4. **`m_04` and `m_06`**: Shared missing parts (`Q9, R21, R29, R51`), correlation = 0.625.
5. **`m_05` and `m_10`**: Photographed in the same batch with similar HDR light profile.

- **Resolution:** To prevent any data leakage across cross-validation folds, each of these linked pairs is assigned to the **same fold**:
  - Fold 1: `m_01`, `m_07`
  - Fold 2: `m_02`, `m_08`
  - Fold 3: `m_03`, `m_09`
  - Fold 4: `m_04`, `m_06`
  - Fold 5: `m_05`, `m_10`

---

## 4. Skipped Photos (Quality / Visibility Gate Failures)

7 photos out of 196 could not be aligned or failed quality checks:
1. `data/presence/raw/red/good/P12/IMG_20261010_121651_048_HDR.jpg`: Blur failure (sharpness 21.0 < 25.0 min threshold).
2. `data/presence/raw/red/good/P14/IMG_20261010_122446_604_HDR.jpg`: Board not completely visible (coverage < 0.97).
3. `data/presence/raw/red/good/P3/IMG_20261020_122647_377_HDR.jpg`: Aspect ratio distorted (1.83 vs 1.40 expected).
4. `data/presence/raw/red/good/P7/IMG_20261010_122613_618_HDR.jpg`: Board not completely visible (coverage < 0.97).
5. `data/presence/raw/red/good/P8/IMG_20261010_122606_619_HDR.jpg`: Board not completely visible (coverage < 0.97).
6. `data/presence/raw/red/removed/m_05/IMG_20261010_122906_744_HDR.jpg`: Aspect ratio distorted (1.77 vs 1.40 expected).
7. `data/presence/test/red/IMG_20261010_130118_086_HDR.jpg`: Aspect ratio distorted (1.63 vs 1.40 expected).

All 7 photos are logged in `report_v2/skipped_photos.json` and excluded from training, validation, and evaluation.
