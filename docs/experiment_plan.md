# Experiment Plan

| | |
|---|---|
| Status | Planned. **No experiment has been run**, and no dataset exists yet. |
| Date | 2026-10-02 |
| Related | [model_strategy.md](model_strategy.md) · [data_collection_plan.md](data_collection_plan.md) · [requirements.md](requirements.md) · [`experiments/TEMPLATE.yaml`](../experiments/TEMPLATE.yaml) |

---

## 1. Preconditions

No experiment starts until:

1. **Dataset v1 exists**, collected per [data_collection_plan.md](data_collection_plan.md) (good boards, plus defective boards for evaluation), and `scripts/validate_dataset.py` reports no issues.
2. **The board-level split is frozen** (`src/pcb_inspection/dataset/splitting.py`) and stored with the dataset version.
3. **Test-set defects are annotated**, as far as localization is to be evaluated.
4. **Provisional targets are agreed** with production (escape rate, false-call rate, latency), or explicitly recorded as unknown.
5. **Shared evaluation code is written and tested**, so that every approach is scored by exactly the same code. *Not implemented yet.*

## 2. Common protocol

1. Every experiment uses a named, immutable dataset version and its split file.
2. **Tuning happens on validation only.** The test set is evaluated once per final configuration. If test results lead to changes, the next evaluation is reported as "test reused", or uses a new test set.
3. Seeds are fixed. *Proposal:* stochastic training is repeated with at least 3 seeds, and the spread is reported.
4. Hardware is recorded, and latency is measured on hardware representative of the station.
5. Each experiment gets one record, `experiments/EXP-NNN_<short-name>/record.yaml`, copied from [`experiments/TEMPLATE.yaml`](../experiments/TEMPLATE.yaml).
6. Failure cases are saved and categorised (section 3.7).
7. No result is reported without its counts (number of boards, number of defect instances).

## 3. Metrics

### 3.1 Board level

| Ground truth \ Verdict | PASS | REVIEW | FAIL |
|---|---|---|---|
| Good board | correct | review load | **false call** |
| Defective board | **escape** | caught (by operator) | correct |

- Escape rate = defective→PASS / all defective boards. **This is the primary metric.**
- False-call rate = good→FAIL / all good boards.
- Review rate = →REVIEW / all boards.

REVIEW counts as "caught" for the escape rate, but it costs operator time. Always report both.

### 3.2 Defect level
- **Matching a prediction to a ground-truth defect:**
  - **Component level:** same refdes and same defect type.
  - **Box level:** IoU ≥ 0.5 and same defect type. The 0.5 threshold is a proposal; also report other thresholds.
  - **Mask level:** region overlap above an agreed threshold.
- **Counts:** TP, FP, FN.
- **Metrics:** precision = TP / (TP + FP), recall = TP / (TP + FN), F1 = 2·P·R / (P + R).
- **Reported:** per defect type, per severity where relevant, plus the macro average.

### 3.3 Localization quality
- **Detection:** mAP@0.5 and mAP@0.5:0.95 (COCO-style).
- **Segmentation:** IoU and Dice per defect.
- **Anomaly maps:** pixel-AUROC and AUPRO. AUPRO is per-region overlap, so it is not dominated by large defects.
- **Misalignment measurement:** position error (mm) and rotation error (°), compared with the inspector's measurement or the CAD position.

### 3.4 Image level (anomaly detection)
- Image AUROC for a threshold-free comparison.
- The section 3.1 and 3.2 metrics at the chosen operating point. **The operating point is chosen on validation.**

### 3.5 Performance
- **Latency:** end-to-end and per stage, with and without capture time. Report p50, p95 and max over the test set after warm-up, and cold-start time separately.
- **CPU:** utilisation, mean and peak.
- **GPU:** utilisation, mean and peak; GPU memory, peak.
- **RAM:** peak.
- **Tools:** e.g. `psutil`, NVML / `nvidia-smi`.
- **Recorded with:** the exact CPU, GPU, RAM and OS.

### 3.6 Statistical reporting
- 95 % confidence intervals (Wilson) for every rate.
- Rule of three: zero misses on *n* independent defects gives a 95 % upper bound on the miss rate of ≈ 3/*n*.
- A per-lot and per-session breakdown, to expose drift.
- A **per-colour** and **per-inspection-stage** breakdown of every metric. Board colour varies between blue / red / green on the same PCB; the stages are after SMT/reflow and after THT. Where data allow, add a leave-one-colour-out test ([model_strategy.md](model_strategy.md), section 9).

### 3.7 Failure-case categories
Small component · low contrast · registration error · lighting / glare · parallax · label / sticker · alternate part · rework artefact · annotation error · unknown.

## 4. Experiments

Experiments 1 to 6 are those requested in the project brief. **Experiment 0 is a recommended addition**: it checks that the camera can see the defects at all, before any model is compared.

### E0: Imaging feasibility and repeatability (no ML)

The full protocol for the LG Indicator is in [experiment_0_camera_visibility.md](experiment_0_camera_visibility.md). The first run uses a **mobile phone** as the prototype camera (mentor decision, 2026-10-02). It is ready to run, with a bench checklist and an image checker (`scripts/check_e0_images.py`). Phone results are not production imaging; E0 is repeated later with the production-candidate hardware. The table below describes the general E0 design.

| | |
|---|---|
| **Goal** | Confirm that the imaging setup makes the in-scope defects visible, and quantify normal variation. |
| **Method** | • Capture good boards repeatedly (with re-seating) across sessions and days.<br>• Capture real defective boards under each candidate lighting setup.<br>• Measure: mm/px, distortion, sharpness, position repeatability (px and mm), brightness and colour drift, and the pixel size of the smallest component.<br>• A qualified inspector rates each real defect as visible / not visible per setup. |
| **Output** | • A frozen station configuration: settings in the profile's `imaging` section, hardware in `station` in `configs/project.yaml`.<br>• Image-quality thresholds.<br>• A list of defect types confirmed visible or invisible. |
| **Decision** | If required defects are not visible, change the imaging (resolution, lighting, angle, 3D) **before** any model work. |

### E1: Reference comparison

**Status (2026-10-02):** a first baseline combining E1 and E2 (registration + reference comparison) was run on the first real phone photos. See [experiments/experiment_001/README.md](../experiments/experiment_001/README.md). Registration ≈0.05 mm residual; false candidates are dominated by capture conditions (parallax of tall parts, gloss, automatic phone exposure). Controlled capture is needed next: **Experiment 1B** ([experiments/experiment_001b/README.md](../experiments/experiment_001b/README.md)). Its tooling is ready; it is waiting for the controlled photo session.

| | |
|---|---|
| **Hypothesis** | Fixture repeatability alone is good enough for pixel-level comparison. |
| **Method** | • Reference: one golden board, then the mean of N train boards.<br>• Difference measures: absolute difference (grey / colour / Lab), SSIM.<br>• Then pre-blur, threshold, and filter blobs by minimum area. |
| **Variables** | Reference type, colour space, filter size, threshold, minimum blob area. |
| **Metrics** | Board level, defect level, localization (blob vs annotation overlap), latency. |
| **What it tells us** | The baseline false-call rate caused by normal variation. |

### E2: Reference comparison + alignment

| | |
|---|---|
| **Hypothesis** | Registration reduces false calls compared with E1, at equal recall. |
| **Method** | • Registration: fiducial-based, feature-based (ORB, AKAZE, SIFT + RANSAC), ECC.<br>• Global alignment, then per-region refinement.<br>• Per-region comparison using CAD regions (if available), against a statistical reference built from the train boards. |
| **Variables** | Registration method, transform model (similarity / affine / homography), region size, reference statistics. |
| **Extra metrics** | Registration residual error (px and mm) at fiducials or landmarks; registration failure rate. |

### E3: YOLO / object detection

| | |
|---|---|
| **Preconditions** | Box annotations. A licence review of the chosen implementation (NFR-12). |
| **Sub-experiments** | • **E3a:** component detection + expected-list matching (missing / extra / wrong / polarity / offset).<br>• **E3b:** direct defect detection, only for defect types with enough examples. |
| **Variables** | Model family and size, input resolution, tiling, augmentation. Trained on train boards only. |
| **Metrics** | mAP, defect level, board level after the decision engine, latency on both GPU and CPU. |

### E4: Visual anomaly detection

| | |
|---|---|
| **Method** | Train on good *train* boards only. Compare at least 2 method families (e.g. PatchCore, PaDiM, EfficientAD). |
| **Variables** | Input (whole image / tiles / registered per-region crops), backbone, resolution, tile size, method-specific parameters. |
| **Metrics** | Image AUROC, pixel-AUROC, AUPRO, defect level at the operating point, board level, latency, memory. |
| **Analysis** | Trace each false call to a source of normal variation (lot, alternate part, label, lighting). |

### E5: Segmentation

| | |
|---|---|
| **Preconditions** | Enough defective examples with masks for D6 to D8. **If not available, the experiment is documented as deferred** rather than run on inadequate data. |
| **Variables** | Architecture, resolution / tiling, loss function for class imbalance. |
| **Metrics** | IoU, Dice, defect-level precision and recall, latency. |

### E6: Hybrid architecture

| | |
|---|---|
| **Method** | Combine the best module per defect type from E1 to E5 into a full pipeline, with the decision-engine rules tuned on validation. |
| **Ablation** | Remove each module in turn and measure the defect coverage lost. |
| **Metrics** | All board-level and defect-level metrics, end-to-end latency, resources, and an estimate of the effort to onboard a new variant. |

### Order and dependencies

| Order | Experiment | Needs |
|---|---|---|
| 1 | E0 | Good boards, some real defective boards, candidate lighting |
| 2 | E1 | Good boards (reference); annotated defective test boards |
| 3 | E2 | As E1, plus fiducials / CAD if available |
| 4 | E4 | Good train boards; annotated defective validation and test boards |
| 5 | E3 | Box annotations; licence review |
| 6 | E5 | Mask annotations for enough defects |
| 7 | E6 | Results of the above |

## 5. Experiment record

Every experiment fills in [`experiments/TEMPLATE.yaml`](../experiments/TEMPLATE.yaml). The template includes every mandatory field from the project brief:
- dataset version
- approach and configuration
- precision, recall, F1, false positives, false negatives
- localization quality
- inference latency
- CPU, GPU and memory use
- failure cases

It also records the hypothesis, hardware, software version, reference set and conclusion.

## 6. Model selection
The selection uses the multi-criteria list in [model_strategy.md](model_strategy.md), section 7, with the escape rate first, and never accuracy alone. The decision will be documented with the IDs of the experiments it is based on. **It is not made in this phase.**
