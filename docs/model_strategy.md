# Model Strategy: Candidate Approaches

| | |
|---|---|
| Status | Comparison only. **No approach is selected.** Selection happens after experiments on real data, using the criteria in section 7. |
| Date | 2026-10-02 |
| Related | [requirements.md](requirements.md) · [experiment_plan.md](experiment_plan.md) · [data_collection_plan.md](data_collection_plan.md) |

All strengths, weaknesses and ratings below describe the **expected** behaviour of each approach in general. None of them has been measured on our boards.

---

## 1. Where the model sits

```
capture → image quality check → registration → [ inspection module(s) ] → findings → decision engine → PASS / FAIL / REVIEW
```

- **Image quality check, registration and the decision engine are needed whatever inspection method is chosen.** The candidates below compete only for the *inspection module(s)* slot.
- **Findings contract.** Every inspection module, whatever its internals, reports *findings*: defect type, confidence, location, component refdes if known, and source. Modules are therefore interchangeable, and the decision engine stays model-agnostic. The contract is implemented as `pcb_inspection.decision_engine.Finding`.

## 2. Approaches

### 2.1 Reference-image comparison (fixture alignment only)

| | |
|---|---|
| **How it works** | Capture one or more golden boards. Compare each new image directly with the reference (absolute difference, SSIM, colour distance), threshold the result, and filter the remaining blobs. |
| **Needs** | Verified good boards. A very repeatable fixture and lighting. No training labels. |
| **Strengths** | Simple and fast; no training. Explainable (difference image). Reacts to *any* change, including unseen defect types. A new variant only needs golden captures. |
| **Weaknesses** | Extremely sensitive to shifts of a few pixels, lighting drift, lot colour variation, placement tolerance, alternate parts and labels. Tall components cause parallax. Cannot name the defect type. A single golden board encodes its own quirks. |
| **Role** | Baseline. Shows how much normal variation exists. |

### 2.2 Image registration + reference comparison

| | |
|---|---|
| **How it works** | Align each image to the reference before comparing. |
| **Registration options** | • **Fiducial-based:** most accurate, if the board has fiducials (PCB-17).<br>• **Feature-based:** ORB / AKAZE / SIFT keypoints + RANSAC. SIFT has been patent-free since 2020 and is in OpenCV.<br>• **Intensity-based:** ECC maximisation, phase correlation.<br>• **Per-region refinement:** absorbs the residual error from board warpage and parallax. |
| **Comparison options** | • **Per-region** instead of whole-image. Component regions come from the centroid file, so each finding carries a component identity.<br>• **Statistical reference** from many good boards (per-pixel or per-region mean and spread) instead of one golden board, so tolerance is learned from real variation.<br>• **CAD-rendered reference** (Gerber copper / silkscreen) for traces. |
| **Needs** | Several good boards; fiducials or stable features. Ideally the BOM, centroid file and Gerber. |
| **Strengths** | Much more robust to position. Localizes the defect and names the component (with CAD). Explainable (expected vs actual crop). Low data requirement. A new variant needs captures plus CAD. |
| **Weaknesses** | A homography assumes a flat scene, so tall components and warpage leave residual misalignment. Still sensitive to lighting and appearance variation. Naming the defect type needs extra logic. Thresholds must be tuned per region type. |
| **Role** | Strong baseline, and a candidate backbone for a hybrid. |

### 2.3 YOLO / object detection

| | |
|---|---|
| **How it works** | A detector trained on labelled boxes predicts objects with boxes, classes and confidences. |
| **Possible uses** | (a) **Component detection** and type classification, then a comparison of the detected set with the expected list (missing / extra / wrong / misplaced / polarity).<br>(b) **Direct defect detection**, with defect types as classes. |
| **Needs** | Many labelled boxes. For (b), many examples *per defect class*. Rare defects are the core difficulty, together with class imbalance. |
| **Strengths** | Robust to position and appearance variation that is covered by the training data. Fast inference on GPU. Mature tooling. Oriented-box variants handle rotation. |
| **Weaknesses** | • "Missing" is an *absence*: a detector cannot report it without an expected list.<br>• Small components on large boards need high input resolution or tiling, which costs latency.<br>• Unseen defect types are not detected.<br>• Each new variant may need relabelling and retraining.<br>• Confidence scores are not calibrated probabilities. |
| **Licensing** | Ultralytics YOLO (YOLOv5 / v8 / 11) is AGPL-3.0, with a separate commercial licence. Permissively licensed detectors also exist (e.g. torchvision detection models, BSD-3-Clause). Review before use (NFR-12). |
| **Role** | Candidate for component-level checks, if the labelled data and the number of variants make it worthwhile. **Not assumed.** |

### 2.4 Image segmentation

| | |
|---|---|
| **How it works** | Per-pixel classification. Semantic segmentation (e.g. U-Net, DeepLab families) or instance segmentation (e.g. Mask R-CNN family). |
| **Possible uses** | Traces / copper, damaged regions, scratches, solder regions. Or segmenting the board into material classes (copper, mask, silkscreen, component, solder) to support rule-based checks. |
| **Needs** | Pixel masks: the most expensive annotation. Many examples of the defect. |
| **Strengths** | Precise extent and shape. Suited to thin structures (traces, scratches) and irregular damage. |
| **Weaknesses** | Annotation cost. Class imbalance (defect pixels are a tiny fraction). Thin structures need high resolution. |
| **Role** | Candidate for D6 to D8 if the data allow. It may not be feasible with few defect examples. |

### 2.5 Visual anomaly detection

| | |
|---|---|
| **How it works** | Learn the appearance of **good boards only**. At inference, score deviation per pixel or patch, producing an anomaly map and an image score. |
| **Method families** | • Memory-bank feature methods (PatchCore)<br>• Distribution modelling (PaDiM)<br>• Student–teacher (EfficientAD)<br>• Normalizing flows (FastFlow, CFlow)<br>Open-source implementations exist, e.g. Anomalib (Apache-2.0). |
| **Needs** | Good images only for training, which matches our first dataset. Defective images with masks for evaluation and threshold setting. |
| **Strengths** | Detects unseen and rare defects. No defect labels needed for training. The heat map gives a localization. |
| **Weaknesses** | • No defect type: only "anomalous".<br>• Normal variation must be well represented, or false calls rise.<br>• Small defects on large boards need tiling or registration-aligned patches.<br>• Thresholds need some defective validation data.<br>• Memory and latency depend on the method. |
| **Role** | Candidate for "unknown defect" coverage. A natural early experiment, because it trains on good boards. |

### 2.6 Hybrid architecture
Combine modules so that each covers what the others cannot. Two examples are given as **hypotheses to test, not decisions**:

- **H1:** registration → CAD-driven per-component checks (presence, polarity, offset; classical methods or small classifiers) → whole-board anomaly detection for everything else → decision engine.
- **H2:** registration → object detector for components, matched against the expected list → segmentation for PCB damage and traces → anomaly-detection fallback → decision engine.

Every added module must justify itself by the defect coverage it adds (ablation in E6), weighed against its latency and maintenance cost.

## 3. Comparison (expected properties, to be verified)

| Criterion | Reference | Registration + reference | Object detection | Segmentation | Anomaly detection | Hybrid |
|---|---|---|---|---|---|---|
| Training labels needed | None | None | Boxes | Masks | None (good images only) | Depends on modules |
| Defective images needed for training | No | No | Yes, many per class | Yes | No | Depends |
| Defective images needed for evaluation | Yes | Yes | Yes | Yes | Yes | Yes |
| Names the defect type | No | Partly (region / CAD rules) | Yes | Yes | No | Yes |
| Names the component (refdes) | No | Yes, with CAD | With expected-list matching | No | With CAD regions | Yes |
| Detects unseen defect types | Yes (any change) | Yes | No | No | Yes | If anomaly detection is included |
| Sensitivity to position variation | Very high | Low–medium | Low | Low | Medium (lower with registration) | Low |
| Sensitivity to lighting / lot variation | High | High | Medium (if trained on it) | Medium | Medium–high | Medium |
| New-variant onboarding | Golden captures | Captures + CAD | Relabel + retrain | Relabel + retrain | Good captures + retrain | Sum of modules |
| Explanation shown to the operator | Difference image | Expected vs actual crop | Box + class | Mask | Heat map | Combined |
| Expected compute | Low | Low–medium | Medium–high (GPU helps) | High | Medium–high | Highest |
| Main risk | False calls from normal variation | Residual misalignment, lighting | Too few defect examples | Annotation cost | False calls; no defect type | Complexity |

## 4. Defect × approach suitability (expected)

Legend:
- ● primary candidate
- ◐ possible / partial
- ○ unlikely

The last column states limits that **no model** can overcome with 2D RGB.

| Defect | Reference | Registration + reference | Detection | Segmentation | Anomaly | Limit beyond any model |
|---|---|---|---|---|---|---|
| D1 Missing component | ◐ | ● | ● (with expected list) | ○ | ◐ | Parts hidden under others |
| D2 Damaged component | ◐ | ◐ | ◐ (needs examples) | ◐ | ● | Internal cracks |
| D3 Misaligned component | ○ | ● (measurement) | ● (oriented boxes) | ○ | ◐ | Height-related cases need 3D |
| D4 Wrong component | ○ | ◐ | ● (classification) | ○ | ◐ | Unmarked look-alikes are not distinguishable |
| D4b Reversed polarity | ○ | ● | ● | ○ | ◐ | Parts without a polarity mark |
| D5 Missing / damaged lead | ○ | ● | ◐ | ◐ | ◐ | Hidden terminations need X-ray; lifted leads need 3D |
| D6 Broken trace | ○ | ◐ (Gerber reference) | ○ | ◐ | ◐ | Inner layers; contrast under solder mask |
| D7 Missing trace | ○ | ● (Gerber reference) | ○ | ◐ | ◐ | Inner layers |
| D8 Damaged PCB | ◐ | ◐ | ◐ | ● | ● | Internal delamination |
| D9 Solder defect | ○ | ◐ | ◐ | ◐ | ◐ | Volume needs 3D or multi-angle light; hidden joints need X-ray |
| D10 Foreign object | ◐ | ● | ◐ | ◐ | ● | Below resolution |
| D11 Other anomaly | ◐ | ◐ | ○ | ○ | ● | — |

## 5. Decision engine (independent of the model)

Findings and stage results are turned into PASS / FAIL / REVIEW by configurable rules, defined per inspection profile (section `decision_engine` of [`configs/profiles/lg_indicator/variant_1.yaml`](../configs/profiles/lg_indicator/variant_1.yaml)). There is no single global confidence threshold.

**Implemented now** (`src/pcb_inspection/decision_engine/`, unit-tested):

**Rule matching**
- Rules are matched per finding, by defect type and component criticality (`critical` / `non_critical` / `any`).
- The first matching rule applies.
- Each rule has its own `fail_threshold` and `review_threshold`.

**Fail-safe behaviour**
- Any required stage that did not run or failed gives REVIEW.
- No required stages configured gives REVIEW.
- A finding without a matching rule gives REVIEW.
- A rule with unset thresholds gives REVIEW.

**Board verdict**
- The verdict is the most severe outcome (FAIL > REVIEW > PASS).
- Every outcome carries a human-readable reason.

**With the shipped configuration** (all thresholds null, no required stages), **every board gets REVIEW**. This is intended: the system cannot pass a board until production-approved rules exist.

**Still open, to define with production:**
- Which defect types make a board FAIL outright.
- How critical components are treated (PCB-14).
- Thresholds per defect type and per source, tuned on validation data. Scores from different methods are not comparable and must be calibrated per source.
- The relative cost of an escape, a false call and a review (PRD-23).

**Example rules from the project brief** (illustrative, *not configured*):
- Confirmed missing critical component → FAIL.
- Confirmed broken trace → FAIL.
- High-confidence visible defect → FAIL.
- No finding, and all stages completed → PASS.
- Uncertain result → REVIEW.

## 6. What we will not do
- Assume YOLO, or any other single model, is the answer before the experiments.
- Use public PCB datasets as a substitute for our own data. Public benchmarks may be cited for literature comparison, and used for pre-training only with explicit approval.
- Train on synthetic defects and report the result as real-world performance.
- Choose a model by accuracy or mAP alone.

## 7. Selection criteria (how the final choice will be made)

In proposed priority order, to be confirmed with production:

1. **Escape rate on held-out boards** (with confidence interval), at an acceptable false-call rate and review rate.
2. **Coverage** of the defect types production requires ([requirements.md](requirements.md), section 5).
3. **Latency** within the inspection time budget, on hardware that can be deployed.
4. **Robustness** across production lots, capture days, **board colours** and **inspection stages**.
5. **Explainability** of FAIL and REVIEW results to operators.
6. **New-variant onboarding effort** and long-term maintenance.
7. **Hardware cost** and compute requirements.
8. **Licensing** compatible with the intended use.

## 8. Open strategic questions
- **Existing inspection (PRD-07).** The current inspection is **manual** (2026-10-02). What manual inspection covers, and why an automated system is wanted, is still UNKNOWN. Commercial inline AOI systems are mature for SMT inspection, so a comparison with commercial options is part of a responsible build-or-buy decision.
- **CAD availability (PCB-11, PCB-12, PCB-22).** Still UNKNOWN. This decides whether CAD-driven regions, component identities and Gerber references are possible at all. Without CAD, component regions would have to be drawn once on a reference image.
- **Number of variants (PRD-02).** The current scope is one variant of one board (LG Indicator). Onboarding effort therefore matters less today, but the inspection-profile structure keeps future variants possible.

## 9. Board colour and inspection stage: one model or several?

The LG Indicator is the same PCB in different colours (blue / red / green reported, depending on availability; PCB-18). It is inspected at two stages: after SMT/reflow and after THT (PRD-06). **No colour- or stage-specific models are created in advance.** Both are recorded as data (`colour`, `inspection_stage`) so that the question can be answered with evidence.

| Option | When it would be justified | How it is tested |
|---|---|---|
| **One model / reference for all colours** | Performance on each colour is similar to the overall performance | Report every metric per colour; leave-one-colour-out test (train without one colour, test on it) |
| **Colour normalisation or augmentation** inside one model | A single model works, but the colour shifts cause false calls | Compare with and without normalisation, per colour |
| **Colour-specific references or settings** (e.g. a golden reference per colour for reference comparison, or a per-colour exposure setting) | Reference comparison or anomaly detection mistakes colour for a defect | Per-colour false-call rate on good boards |
| **Per-stage references / component maps** | The board looks different after SMT/reflow and after THT (if THT parts are fitted later) | Per-stage metrics; whether a single reference set raises false calls at one stage |

Which option is chosen is a result of experiments E1 to E6 ([experiment_plan.md](experiment_plan.md)), not a design assumption.
