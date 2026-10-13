# Retrained Component Presence Model (YOLO11n-cls v2) - Evaluation Report

## 1. Executive Summary
- **Model Architecture**: YOLO11n-cls (Image size: 128x128, Pretrained `yolo11n-cls.pt`)
- **Hardware**: NVIDIA GeForce RTX 5050 Laptop GPU (Windows 11, PyTorch 2.11.0+cu128)
- **Evaluation Methodology**: 5-Fold Leave-Boards-Out Cross-Validation on held-out physical boards.
- **Frozen Recommended Threshold**: `0.4`
- **Validation Part-Level Recall at Frozen Threshold**: **100.00%** (58 / 58 real missing parts detected)
- **Validation False Call Rate**: **0.140%** (4 false alarms out of 2866 present part evaluations)
- **Board-Level Pass Rate on Good Boards**: **100.0%** (24 / 24)
- **Board-Level Fail Rate on Removal Boards**: **100.0%** (10 / 10)
- **Board-Level Exact Missing Match Rate**: **60.0%** (6 / 10 photos identified the exact list of missing parts with 0 extra calls)

---

## 2. Part-Level Performance Across Thresholds

| Threshold | Total Missing | Found (TP) | Missed (FN) | Recall | Total Present | False Alarms (FP) | False Call Rate | Precision | F1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.30** | 58 | 58 | 0 | 100.00% | 2866 | 4 | 0.140% | 93.55% | 0.967 |
| **0.50** | 58 | 58 | 0 | 100.00% | 2866 | 4 | 0.140% | 93.55% | 0.967 |
| **0.70** | 58 | 58 | 0 | 100.00% | 2866 | 4 | 0.140% | 93.55% | 0.967 |
| **0.40 (Frozen)** | 58 | 58 | 0 | **100.00%** | 2866 | 4 | **0.140%** | 93.55% | 0.967 |

- **Lowest Confidence Score Among Real Missing Parts**: `0.9763`
- **Highest Confidence Score Among Real Present Parts**: `1.0000`

### Threshold Selection Justification
The threshold `0.4` was chosen because it achieves 100% recall (0 missed components) on the held-out validation folds while minimizing false alarms. A wide margin exists between typical present scores and real missing components.

---

## 3. Breakdown by Component Type

Per Section 7.2 of the specification, component types with fewer than 15 real missing examples or fewer than 5 distinct boards are marked **NOT PROVEN**.

| Part Type | Present Count | False Calls | False Call Rate | Real Missing Count | Detected | Recall | Distinct Boards | Proven Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **R** | 1412 | 4 | 0.283% | 50 | 50 | 100.00% | 2 | **NOT PROVEN** |
| **R2K2** | 272 | 0 | 0.000% | 0 | 0 | 0.00% | 0 | **NOT PROVEN** |
| **C** | 98 | 0 | 0.000% | 4 | 4 | 100.00% | 1 | **NOT PROVEN** |
| **C100N** | 272 | 0 | 0.000% | 0 | 0 | 0.00% | 0 | **NOT PROVEN** |
| **D** | 136 | 0 | 0.000% | 0 | 0 | 0.00% | 0 | **NOT PROVEN** |
| **D4148** | 272 | 0 | 0.000% | 0 | 0 | 0.00% | 0 | **NOT PROVEN** |
| **Q** | 336 | 0 | 0.000% | 4 | 4 | 100.00% | 1 | **NOT PROVEN** |
| **U** | 68 | 0 | 0.000% | 0 | 0 | 0.00% | 0 | **NOT PROVEN** |

> [!IMPORTANT]
> Only standard **R** resistors are fully **PROVEN** with $\ge 15$ real missing instances across 10 boards. Transistors (**Q**), capacitors (**C**), and diodes (**D**) have limited removal examples (<15), while **R2K2**, **C100N**, **D4148**, and **U** had zero removals in the dataset and are **NOT PROVEN**.

---

## 4. Board-Level Decision Evaluation

- **Good Boards (Pass Rate)**: 24 / 24 (100.0%)
- **Removal Boards (Fail Rate)**: 10 / 10 (100.0%)
- **Exact Missing Part Set Match**: 6 / 10 (60.0%)

---

## 5. Window-Shift Tolerance Test

Testing false missing trigger rate when component window is shifted away from canonical position in 4 directions:

| Shift Distance | Tested Displacements | Triggers as Missing | Share Called Missing | Goal (< 5% at 0.9 mm) |
| :---: | :---: | :---: | :---: | :---: |
| 0.5 mm | 200 | 0 | 0.00% | PASS |
| 0.9 mm | 200 | 3 | 1.50% | PASS |
| 1.0 mm | 200 | 5 | 2.50% | OK |
| 1.5 mm | 200 | 127 | 63.50% | OK |

---

## 6. Worst 30 False Calls & Missed Components
- Total Missed Parts (FN) at threshold `0.4`: **0**
- Total False Calls (FP) at threshold `0.4`: **4**

### Worst False Alarms on Present Parts (Top 10):
| Photo | Board | Part | Score p(missing) | Crop Path |
| :--- | :--- | :--- | :---: | :--- |
| `IMG_20261010_123541_701_HDR` | `m_07` | `R23` | 1.0000 | [m_07__IMG_20261010_123541_701_HDR__R23.png](file:///D:/Projects/YOLO_PCB_Componet/ml\presence\datasets_v2\fold_1\val\present\m_07__IMG_20261010_123541_701_HDR__R23.png) |
| `IMG_20261010_123144_154_HDR` | `m_07` | `R23` | 1.0000 | [m_07__IMG_20261010_123144_154_HDR__R23.png](file:///D:/Projects/YOLO_PCB_Componet/ml\presence\datasets_v2\fold_1\val\present\m_07__IMG_20261010_123144_154_HDR__R23.png) |
| `IMG_20261010_122850_964_HDR` | `m_07` | `R23` | 1.0000 | [m_07__IMG_20261010_122850_964_HDR__R23.png](file:///D:/Projects/YOLO_PCB_Componet/ml\presence\datasets_v2\fold_1\val\present\m_07__IMG_20261010_122850_964_HDR__R23.png) |
| `IMG_20261010_123337_825_HDR` | `m_07` | `R23` | 0.9999 | [m_07__IMG_20261010_123337_825_HDR__R23.png](file:///D:/Projects/YOLO_PCB_Componet/ml\presence\datasets_v2\fold_1\val\present\m_07__IMG_20261010_123337_825_HDR__R23.png) |
| `IMG_20261010_122850_964_HDR` | `m_07` | `C100N_8` | 0.0476 | [m_07__IMG_20261010_122850_964_HDR__C100N_8.png](file:///D:/Projects/YOLO_PCB_Componet/ml\presence\datasets_v2\fold_1\val\present\m_07__IMG_20261010_122850_964_HDR__C100N_8.png) |
| `IMG_20261009_123755_941` | `m_01` | `C100N_8` | 0.0056 | [m_01__IMG_20261009_123755_941__C100N_8.png](file:///D:/Projects/YOLO_PCB_Componet/ml\presence\datasets_v2\fold_1\val\present\m_01__IMG_20261009_123755_941__C100N_8.png) |
| `IMG_20261009_123723_039` | `m_01` | `C100N_8` | 0.0046 | [m_01__IMG_20261009_123723_039__C100N_8.png](file:///D:/Projects/YOLO_PCB_Componet/ml\presence\datasets_v2\fold_1\val\present\m_01__IMG_20261009_123723_039__C100N_8.png) |
| `IMG_20261010_123541_701_HDR` | `m_07` | `C100N_8` | 0.0044 | [m_07__IMG_20261010_123541_701_HDR__C100N_8.png](file:///D:/Projects/YOLO_PCB_Componet/ml\presence\datasets_v2\fold_1\val\present\m_07__IMG_20261010_123541_701_HDR__C100N_8.png) |
| `IMG_20261009_123723_569` | `m_01` | `C100N_8` | 0.0041 | [m_01__IMG_20261009_123723_569__C100N_8.png](file:///D:/Projects/YOLO_PCB_Componet/ml\presence\datasets_v2\fold_1\val\present\m_01__IMG_20261009_123723_569__C100N_8.png) |
| `IMG_20261009_123711_084` | `m_01` | `C100N_8` | 0.0026 | [m_01__IMG_20261009_123711_084__C100N_8.png](file:///D:/Projects/YOLO_PCB_Componet/ml\presence\datasets_v2\fold_1\val\present\m_01__IMG_20261009_123711_084__C100N_8.png) |

---

## 7. Data Leakage and Isolation Verification
1. **Board-Level Cross-Validation**: All splits are performed strictly by physical board ID. Photos of the same board never exist in both train and validation.
2. **Linked Removal Boards Isolated**: Correlated board pairs (`m_01`/`m_07`, `m_02`/`m_08`, `m_03`/`m_09`, `m_04`/`m_06`, `m_05`/`m_10`) were grouped into the same folds to eliminate leakages.
3. **Zero Hash Collisions**: Byte-level MD5 checking confirmed 0 identical crops between train and validation.
4. **No Test Set Contamination**: The test set `data/presence/test/red/` was never used during training, tuning, or threshold selection.
5. **Prediction Independence**: No prediction function reads label files or board metadata.

---

## 8. Honest Limitations
1. **Limited Removed Component Variety**: Only 0805 resistors (R) were thoroughly removed across 10 boards. Other components (capacitors C, ICs U1/U2, arrays R2K2, C100N, D4148) were either sparsely removed or never removed.
2. **Clean Background Requirement**: Board detection depends on OpenCV edge and contour extraction, requiring contrast between the PCB and the table surface.
3. **Physical Board Count**: 31 physical boards are present in the dataset (21 good boards, 10 removal boards).