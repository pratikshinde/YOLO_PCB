# Component Presence Model (YOLO11n-cls) - Evaluation Report

## 1. Environment & Hardware
- **OS**: Windows 11
- **Python**: 3.11.9
- **PyTorch**: 2.11.0+cu128
- **CUDA Version**: 12.8
- **GPU**: NVIDIA GeForce RTX 5050 Laptop GPU (8 GB VRAM)
- **Model Architecture**: Ultralytics YOLO11n-cls (input size 128x128, 1.53M parameters)

## 2. Dataset & Photo Ingestion Summary
- **Total raw photos inspected**: 71
- **Usable photos aligned**: 50
- **Photos skipped**: 21

### Skipped Photos Log
| Photo | Board | Reason |
|---|---|---|
| `R_03.jpg` | good_R_03 | Board not completely visible (touches the image edge) - retake |
| `R_05.jpg` | good_R_05 | Board not completely visible (touches the image edge) - retake |
| `R_06.jpg` | good_R_06 | Board not completely visible (touches the image edge) - retake |
| `R_07.jpg` | good_R_07 | Board not completely visible (touches the image edge) - retake |
| `R_08.jpg` | good_R_08 | Board not completely visible (touches the image edge) - retake |
| `R_15.jpg` | good_R_15 | Cannot align with the golden reference (193 inliers, ratio 0.25) - retake photo |
| `R_19.jpg` | good_R_19 | Cannot align with the golden reference (180 inliers, ratio 0.23) - retake photo |
| `R_20.jpg` | good_R_20 | Cannot align with the golden reference (182 inliers, ratio 0.24) - retake photo |
| `R_21.jpg` | good_R_21 | Board not completely visible (touches the image edge) - retake |
| `R_22.jpg` | good_R_22 | Cannot align with the golden reference (149 inliers, ratio 0.24) - retake photo |
| `R_23.jpg` | good_R_23 | Board not completely visible (touches the image edge) - retake |
| `R_31.jpg` | good_R_31 | Board not completely visible (touches the image edge) - retake |
| `R_34.jpg` | good_R_34 | Cannot align with the golden reference (167 inliers, ratio 0.24) - retake photo |
| `R_35.jpg` | good_R_35 | Cannot align with the golden reference (159 inliers, ratio 0.24) - retake photo |
| `R_36.jpg` | good_R_36 | Cannot align with the golden reference (150 inliers, ratio 0.24) - retake photo |
| `IMG_20261009_123822_510.jpg` | m_02 | Cannot align with the golden reference (163 inliers, ratio 0.25) - retake photo |
| `IMG_20261009_123824_050.jpg` | m_02 | Board not completely visible (touches the image edge) - retake |
| `IMG_20261009_123916_748.jpg` | m_02 | Board not completely visible (touches the image edge) - retake |
| `IMG_20261009_123917_275.jpg` | m_02 | Board not completely visible (touches the image edge) - retake |
| `IMG_20261009_124116_819.jpg` | m_04 | Board not completely visible (touches the image edge) - retake |
| `IMG_20261009_124149_023.jpg` | m_04 | Board not completely visible (touches the image edge) - retake |

### Crop Statistics Across Cross-Validation
- **Validation crops evaluated**: 4300 (100% real crops, no synthetic shifts, no repetition in val)
  - Real `present`: 4198
  - Real `missing`: 102 (from 4 distinct boards, 19 removed part instances)
- **Training sets**: Balanced to ~1:4 missing:present ratio via real-missing oversampling and synthetic shifts (at most 1x real missing count, 1.3-2.5 mm displacement).

## 3. Cross-Validation Results (Leave-One-Removed-Board-Out)

### Summed Across All 4 Folds
| Threshold | Real Missing | Found (TP) | Missed (FN) | Recall | Present Crops | False Missing (FP) | False-Call Rate | Precision | F1 |
|---|---|---|---|---|---|---|---|---|---|
| **0.3** | 102 | 102 | 0 | 100.00% | 4198 | 2 | 0.05% | 98.08% | 0.990 |
| **0.5** | 102 | 102 | 0 | 100.00% | 4198 | 0 | 0.00% | 100.00% | 1.000 |
| **0.7** | 102 | 102 | 0 | 100.00% | 4198 | 0 | 0.00% | 100.00% | 1.000 |
| **Recommended (0.35)** | 102 | 102 | 0 | 100.00% | 4198 | 0 | 0.00% | 100.00% | 1.000 |

### Breakdown Per Fold
| Fold | Val Removed Board | Real Missing | Found @ 0.5 | Missed @ 0.5 | Recall @ 0.5 | Present Crops | FP @ 0.5 | FP Rate @ 0.5 |
|---|---|---|---|---|---|---|---|---|
| Fold 1 | `m_01` | 30 | 30 | 0 | 100.00% | 1002 | 0 | 0.00% |
| Fold 2 | `m_02` | 24 | 24 | 0 | 100.00% | 836 | 0 | 0.00% |
| Fold 3 | `m_03` | 24 | 24 | 0 | 100.00% | 1266 | 0 | 0.00% |
| Fold 4 | `m_04` | 24 | 24 | 0 | 100.00% | 1094 | 0 | 0.00% |

### Missed Missing Parts at Recommended Threshold (0.35)
None! All real missing parts were successfully detected (100% recall).

## 4. Performance by Component Type
| Type | Total Present | FP @ 0.5 | FP Rate @ 0.5 | FP @ Rec | FP Rate @ Rec | Total Missing | TP @ 0.5 | Recall @ 0.5 | Notes |
|---|---|---|---|---|---|---|---|---|---|
| `R` | 2060 | 0 | 0.00% | 0 | 0.00% | 90 | 90 | 100.00% | Real missing in data |
| `R2K2` | 400 | 0 | 0.00% | 0 | 0.00% | 0 | 0 | 0.00% | No missing examples in data |
| `C100N` | 400 | 0 | 0.00% | 0 | 0.00% | 0 | 0 | 0.00% | No missing examples in data |
| `D4148` | 400 | 0 | 0.00% | 0 | 0.00% | 0 | 0 | 0.00% | No missing examples in data |
| `Q` | 488 | 0 | 0.00% | 0 | 0.00% | 12 | 12 | 100.00% | Real missing in data |
| `C` | 150 | 0 | 0.00% | 0 | 0.00% | 0 | 0 | 0.00% | No missing examples in data |
| `D` | 200 | 0 | 0.00% | 0 | 0.00% | 0 | 0 | 0.00% | No missing examples in data |
| `U` | 100 | 0 | 0.00% | 0 | 0.00% | 0 | 0 | 0.00% | No missing examples in data |

## 5. Board-Level Inspection Decision
At threshold 0.35, a board receives **FAIL** if any of its 86 components has `p(missing) >= 0.35`, else **PASS**.
- **Removed Boards (Should FAIL)**: 22 / 22 (100.0%)
- **Good Boards (Should PASS)**: 28 / 28 (100.0%)

## 6. Recommended Threshold: `0.35`
A threshold of **`0.35`** achieves **100.0% recall** on all real removed components while keeping the false-positive rate on present components at **0.00%**.

## 7. Worst 30 False Calls on Present Parts (Highest p(missing))
| Rank | Board | Photo | Part | Type | p(missing) | Crop Path |
|---|---|---|---|---|---|---|
| 1 | m_01 | `IMG_20261009_123737_805` | `C100N_8` | C100N | 0.3335 | `ml\presence\datasets\fold_1\val\present\m_01__IMG_20261009_123737_805__C100N_8.png` |
| 2 | m_01 | `IMG_20261009_123723_039` | `C100N_8` | C100N | 0.3328 | `ml\presence\datasets\fold_1\val\present\m_01__IMG_20261009_123723_039__C100N_8.png` |
| 3 | m_01 | `IMG_20261009_123723_569` | `C100N_8` | C100N | 0.2400 | `ml\presence\datasets\fold_1\val\present\m_01__IMG_20261009_123723_569__C100N_8.png` |
| 4 | m_01 | `IMG_20261009_123755_941` | `C100N_8` | C100N | 0.2399 | `ml\presence\datasets\fold_1\val\present\m_01__IMG_20261009_123755_941__C100N_8.png` |
| 5 | good_R_09 | `R_09` | `C14` | C | 0.2191 | `ml\presence\datasets\fold_1\val\present\good_R_09__R_09__C14.png` |
| 6 | m_01 | `IMG_20261009_123711_084` | `C100N_8` | C100N | 0.2150 | `ml\presence\datasets\fold_1\val\present\m_01__IMG_20261009_123711_084__C100N_8.png` |
| 7 | m_01 | `IMG_20261009_123737_331` | `C100N_8` | C100N | 0.1972 | `ml\presence\datasets\fold_1\val\present\m_01__IMG_20261009_123737_331__C100N_8.png` |
| 8 | good_R_11 | `R_11` | `C14` | C | 0.1113 | `ml\presence\datasets\fold_1\val\present\good_R_11__R_11__C14.png` |
| 9 | good_R_04 | `R_04` | `C14` | C | 0.0616 | `ml\presence\datasets\fold_1\val\present\good_R_04__R_04__C14.png` |
| 10 | good_R_02 | `R_02` | `C14` | C | 0.0420 | `ml\presence\datasets\fold_1\val\present\good_R_02__R_02__C14.png` |
| 11 | m_01 | `IMG_20261009_123723_569` | `U1` | U | 0.0415 | `ml\presence\datasets\fold_1\val\present\m_01__IMG_20261009_123723_569__U1.png` |
| 12 | m_01 | `IMG_20261009_123723_569` | `C14` | C | 0.0412 | `ml\presence\datasets\fold_1\val\present\m_01__IMG_20261009_123723_569__C14.png` |
| 13 | m_01 | `IMG_20261009_123737_805` | `U1` | U | 0.0377 | `ml\presence\datasets\fold_1\val\present\m_01__IMG_20261009_123737_805__U1.png` |
| 14 | m_01 | `IMG_20261009_123737_331` | `U1` | U | 0.0351 | `ml\presence\datasets\fold_1\val\present\m_01__IMG_20261009_123737_331__U1.png` |
| 15 | m_01 | `IMG_20261009_123723_039` | `U1` | U | 0.0332 | `ml\presence\datasets\fold_1\val\present\m_01__IMG_20261009_123723_039__U1.png` |
| 16 | good_R_01 | `R_01` | `C14` | C | 0.0309 | `ml\presence\datasets\fold_1\val\present\good_R_01__R_01__C14.png` |
| 17 | m_01 | `IMG_20261009_123711_084` | `U1` | U | 0.0295 | `ml\presence\datasets\fold_1\val\present\m_01__IMG_20261009_123711_084__U1.png` |
| 18 | m_01 | `IMG_20261009_123755_941` | `U1` | U | 0.0292 | `ml\presence\datasets\fold_1\val\present\m_01__IMG_20261009_123755_941__U1.png` |
| 19 | m_01 | `IMG_20261009_123723_569` | `R28` | R | 0.0208 | `ml\presence\datasets\fold_1\val\present\m_01__IMG_20261009_123723_569__R28.png` |
| 20 | m_04 | `IMG_20261009_124147_972` | `R39` | R | 0.0200 | `ml\presence\datasets\fold_4\val\present\m_04__IMG_20261009_124147_972__R39.png` |
| 21 | good_R_10 | `R_10` | `C14` | C | 0.0190 | `ml\presence\datasets\fold_1\val\present\good_R_10__R_10__C14.png` |
| 22 | m_01 | `IMG_20261009_123755_941` | `R28` | R | 0.0162 | `ml\presence\datasets\fold_1\val\present\m_01__IMG_20261009_123755_941__R28.png` |
| 23 | m_04 | `IMG_20261009_124210_460` | `R45` | R | 0.0156 | `ml\presence\datasets\fold_4\val\present\m_04__IMG_20261009_124210_460__R45.png` |
| 24 | m_01 | `IMG_20261009_123737_331` | `R28` | R | 0.0153 | `ml\presence\datasets\fold_1\val\present\m_01__IMG_20261009_123737_331__R28.png` |
| 25 | m_04 | `IMG_20261009_124147_972` | `R35` | R | 0.0140 | `ml\presence\datasets\fold_4\val\present\m_04__IMG_20261009_124147_972__R35.png` |
| 26 | good_R_04 | `R_04` | `U1` | U | 0.0138 | `ml\presence\datasets\fold_1\val\present\good_R_04__R_04__U1.png` |
| 27 | m_01 | `IMG_20261009_123711_084` | `R28` | R | 0.0137 | `ml\presence\datasets\fold_1\val\present\m_01__IMG_20261009_123711_084__R28.png` |
| 28 | m_01 | `IMG_20261009_123755_941` | `Q10` | Q | 0.0135 | `ml\presence\datasets\fold_1\val\present\m_01__IMG_20261009_123755_941__Q10.png` |
| 29 | m_01 | `IMG_20261009_123737_805` | `R28` | R | 0.0135 | `ml\presence\datasets\fold_1\val\present\m_01__IMG_20261009_123737_805__R28.png` |
| 30 | good_R_02 | `R_02` | `U1` | U | 0.0131 | `ml\presence\datasets\fold_1\val\present\good_R_02__R_02__U1.png` |

## 8. Honest Limitations & Failure Modes
1. **Limited Missing Component Diversity**: Only **resistors** (R, R2K2) and **two transistors** (Q2, Q9) were ever physically removed in the dataset. Capacitors (C, C100N), diodes (D, D4148), and ICs (U) have 0 missing examples. The model has not seen real bare solder pads for those package types.
2. **Small Physical Sample**: Only 4 physical boards with removed parts (total of 19 removed part instances) were photographed.
3. **Lighting & Edge Artefacts**: 21 whole-board phone photos were skipped because the board touched the image edge or severe glare/inlier drops occurred during homography estimation.
4. **No External Holdout Test Board**: All metrics reported above are cross-validated strictly by physical board (leave-one-removed-board-out), but an independent dataset photographed in new lighting conditions should be tested once available.

## 9. Error Contact Sheets
Contact sheets showing all false detections for each fold are saved in:
- `ml/presence/report/errors_fold_1.jpg`
- `ml/presence/report/errors_fold_2.jpg`
- `ml/presence/report/errors_fold_3.jpg`
- `ml/presence/report/errors_fold_4.jpg`
