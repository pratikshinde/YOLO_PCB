# Data Inventory Report (v2)

Date: 2026-10-10  
Workspace: `D:\Projects\YOLO_PCB_Componet`

---

## 1. Golden Reference
- **Path:** `data\presence\golden\red\grf.jpeg`
- **SHA-256:** `BA859CA72C6B742FCC6BA4916D8FCF12A326CDF020903B708D11FD2DFD0D455A`
- **Verification:** Matches required prefix `BA859CA72C6B742F`.
- **Dimensions / Canonical Spec:** Red "LG Indicator" board (116 x 83 mm, SMT stage). Canonical scale is 20 px/mm (1660 x 2320 px).

---

## 2. Directory Structure and Raw Photo Inventory

| Directory | Photos | Text Files | Physical Boards | Notes |
| :--- | :---: | :---: | :---: | :--- |
| `data\presence\golden\red\` | 1 | 0 | 1 | Golden reference `grf.jpeg` |
| `data\presence\raw\red\good\` (root) | 43 | 0 | 6 | Old captures `R_01.jpg`..`R_43.jpg` (mapped to boards G01..G06 via `board_groups.json`) |
| `data\presence\raw\red\good\P1`..`P15` | 90 | 0 | 15 | 15 folders with 6 photos each (new good boards) |
| `data\presence\raw\red\removed\m_01`..`m_10` | 52 | 10 | 10 | 10 removal boards (with `m_XX.txt.txt` label files) |
| `data\presence\raw\red\bare\` | 0 | 0 | 0 | Unpopulated PCB folder (empty) |
| `data\presence\test\red\` | 11 | 0 | - | Held-out test set photos (used only for blind final evaluation) |
| **Total** | **197** | **10** | **31 training/val boards** | **189 aligned non-golden photos** |

---

## 3. Physical Boards Summary

| Category | Boards Count | Photos Count | Aligned Photos | Skipped Photos |
| :--- | :---: | :---: | :---: | :---: |
| **Good Boards (G01..G06)** | 6 | 43 | 43 | 0 |
| **Good Boards (P1..P15)** | 15 | 90 | 85 | 5 |
| **Removal Boards (m_01..m_10)** | 10 | 52 | 51 | 1 |
| **Test Set** | - | 11 | 10 | 1 |
| **Bare Boards** | 0 | 0 | 0 | 0 |
| **Total** | **31 (+ test)** | **196** | **189** | **7** |

---

## 4. Visual Verification Sample (Eyeball Check)

Sample crops from removal photos were verified against golden references:
- **Board `m_01` (`IMG_20261009_123711_084.jpg`):**
  - Listed missing parts (`R10`, `R36`, `R23`, `R26`, `R22`): Verified empty solder pads, bare board in center, flux traces visible.
  - Unlisted parts (`R12`, `Q2`, `U1`): Verified present, correctly oriented and soldered.
- **Board `m_02` (`IMG_20261009_123811_073.jpg`):**
  - Listed missing parts (`R12`, `R28`, `R19`, `R34`, `R54`, `R38`): Verified empty pads.
  - Unlisted parts (`R10`, `Q9`, `C14`): Verified present.
- **Board `m_03` (`IMG_20261009_124005_685.jpg`):**
  - Listed missing parts (`Q2`, `R45`, `R39`, `R27`): Empty SOT-23 pads on Q2, bare 0805 resistor pads.
  - Unlisted parts (`R10`, `Q9`, `U1`): Verified present.

---

## 5. Part-Type Removal Counts and Proven Status

Per Section 3.6 of the specification, part types with fewer than 15 real missing examples or coming from fewer than 5 distinct boards are **NOT PROVEN**.

| Part Type | Checked Part Count | Real Missing Instances | Distinct Removal Boards | Proven Status | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **R** (Standard Resistors) | 43 | 40 | 10 | **PROVEN** | $\ge 15$ instances across 10 boards |
| **R2K2** (2.2k Resistor Row) | 8 | 0 | 0 | **NOT PROVEN** | 0 removals seen in dataset |
| **C** (Individual Capacitors) | 3 | 1 | 1 | **NOT PROVEN** | Only `C16` missing on `m_07` |
| **C100N** (100nF Capacitor Row) | 8 | 0 | 0 | **NOT PROVEN** | 0 removals seen in dataset |
| **D** (Diodes / Bridge) | 4 | 3 | 2 | **NOT PROVEN** | `D1` (1x), `D11` (2x) |
| **D4148** (4148 Diode Row) | 8 | 0 | 0 | **NOT PROVEN** | 0 valid removals (`D'4`/`D'8` omitted per rule) |
| **Q** (Transistors) | 10 | 6 | 6 | **NOT PROVEN** | `Q2` (3x), `Q9` (2x), `Q6` (1x) (< 15 instances) |
| **U** (ICs: U1, U2) | 2 | 0 | 0 | **NOT PROVEN** | 0 removals seen in dataset |
| **Total** | **86** | **50** | **10** | - | - |

---

## 6. Solder Texture Board Identity Audit

1. **Removal Boards Calibration:**
   - Same physical board: similarity min 0.632, median 0.932, max 0.994.
   - Different removal boards: similarity min 0.257, median 0.474, max 0.763.
2. **Linked Removal Boards:**
   - Elevate similarity and shared missing components between (`m_01`, `m_07`), (`m_02`, `m_08`), (`m_03`, `m_09`), (`m_04`, `m_06`), and (`m_05`, `m_10`).
   - Grouped pair-by-pair into the 5 cross-validation folds to prevent data leakage.
3. **Good Boards:**
   - `G01` through `G06` confirmed as 6 distinct boards (internal similarity > 0.988).
   - `P1` through `P15` confirmed as 15 distinct physical boards (internal similarity ~0.70-0.74, cross-board median ~0.55-0.65).
4. **Test Set Isolation:**
   - No test set photo matched any training or validation board (max cross-similarity 0.52, well below threshold).
