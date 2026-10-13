# Requirements: PCB Pre-Test Visual Inspection System

| | |
|---|---|
| Status | Draft 0.3, Phase 1 (foundation). Updated with two rounds of verified production answers (Production/PCB mentor, 2026-10-02). Nothing here has been validated on real boards. |
| Date | 2026-10-02 |
| Current scope | LG Indicator, top side: [pcb_inspection_spec.md](pcb_inspection_spec.md). Answer status: [production_profile.md](production_profile.md). |
| Related | [production_questions.md](production_questions.md) · [data_collection_plan.md](data_collection_plan.md) · [model_strategy.md](model_strategy.md) · [experiment_plan.md](experiment_plan.md) · [`configs/project.yaml`](../configs/project.yaml) · [`configs/profiles/`](../configs/profiles) |

**Conventions.**
- **UNKNOWN / TBC** marks information that only the production/PCB team can supply, or that must be measured on the real setup. Question IDs such as `PCB-10` refer to [production_questions.md](production_questions.md).
- No value here has been invented. A number is either a formula, a statistical fact, or explicitly labelled *illustrative* or *proposal*.
- Requirement IDs (`FR-`, `NFR-`, `DR-`, `ER-`) are stable, so they can be referenced from code, tests and experiment records.

---

## 1. Business problem

Assembled PCBs currently go straight to electrical/functional test. Some assembly and board defects are physically visible before that point. The proposed system photographs each assembled board and inspects it visually, so that visibly defective boards are stopped or flagged *before* electrical test.

The expected benefits below are **hypotheses to be confirmed with production data**, not established facts:

| # | Hypothesis | Confirmed by |
|---|---|---|
| B1 | Electrical test time and fixture wear are spent on boards that are visibly defective. | Defect Pareto and test failure logs (PCB-24, PRD-03) |
| B2 | Some visible defects pass electrical test (e.g. a missing decoupling capacitor, mechanical or cosmetic damage). | Escape records, customer returns (PCB-26) |
| B3 | Gross visible defects (e.g. solder bridges on supply rails) can damage the board or the test fixture during powered test. | Test engineering input |
| B4 | A localized defect report (type, position, reference designator) shortens repair time. | Repair station feedback |
| B5 | A per-board image record improves traceability. | Traceability requirements (PRD-21) |

**Still unknown:**
- **The measurable business target**, for example a required reduction in electrical-test failures or a maximum escape rate. Until it exists, the acceptance criteria in section 10 cannot be finalised.
- **The role relative to existing inspection.** The current inspection is manual (PRD-07). What manual inspection covers, and why an automated system is wanted, is UNKNOWN.

### 1.1 Scope

| In scope, this phase | Out of scope, this phase |
|---|---|
| Requirements, data-collection protocol, model strategy, experiment plan | Model training and model selection |
| Project configuration with explicit unknowns | Hardware integration (camera SDK, PLC, MES) |
| Dataset metadata validation and board-level splitting code | Operator UI |
| Decision-engine contract (rules, fail-safe behaviour) | Production deployment |

### 1.2 Physical limits of a 2D camera system

Whatever model is used, a 2D colour camera **cannot** inspect:
- **Hidden solder joints** (BGA, LGA, QFN thermal pads, parts under shields). These need X-ray.
- **Inner-layer copper and internal board defects** (inner-layer opens, internal delamination). These are caught by bare-board testing at the PCB fabricator, by electrical test, or by X-ray.
- **Values of parts without visible marking** (e.g. chip capacitors normally carry no marking). Only electrical test catches these.
- **Electrical function** in general.

## 2. Target workflow

```
ASSEMBLED PCB → CAMERA → IMAGE CAPTURE → IMAGE QUALITY CHECK → ALIGNMENT / REGISTRATION
→ VISUAL INSPECTION → DEFECT DETECTION → DEFECT LOCALIZATION → DECISION ENGINE
→ PASS / FAIL / REVIEW → ELECTRICAL / FUNCTIONAL TEST
```

### 2.1 Two operating modes

The system is **camera-first**: the camera image is the inspection input. The data-management layer exists to support development; it is not part of the operator's workflow.

| | **Inspection mode** (the product) | **Development / dataset mode** (now) |
|---|---|---|
| Who | Line operator | Developer / engineer, with the operator for ground truth |
| Workflow | 1. Place the PCB.<br>2. The camera captures it.<br>3. The system loads the PCB's inspection profile.<br>4. AI inspection.<br>5. **PASS / FAIL / REVIEW**.<br>On FAIL, it shows defect type, location, confidence and an annotated image. | Capture images, collect metadata and ground truth, run experiments |
| Typed per inspection | **Nothing** | Nothing per image. Optional session context (stage, distance, lighting) once per batch; ground truth only for defective boards. |
| Inspection profile | Loaded automatically (station configuration; later barcode / MES) | Recorded |
| Inspection stage | From recipe / station configuration / PLC-MES / operator selection (source not decided) | Recorded per batch |
| Board colour | **Not an input.** The inspection must tolerate it; experiments decide how. | Estimated automatically from the image |
| Board identity | Automatic inspection ID + timestamp; barcode / serial later | Optional board ID (folder name) for repeat photos and ground truth |
| Image ID, file names, capture time, camera | Generated / read automatically | Generated / read automatically (`scripts/e0_ingest.py`) |

The inspection mode is **not implemented yet**. It needs a trained or configured inspection method, which needs real images first. Configuration: `inspection_mode` in [`configs/project.yaml`](../configs/project.yaml).

## 3. Functional requirements

| ID | Requirement | Open points |
|---|---|---|
| FR-01 | **Image capture.** Capture every side to be inspected with fixed, recorded camera, lens and lighting settings. Automatic exposure, gain and white balance stay off. | Current scope: top side only (PCB-07) |
| FR-02 | **Board identification.** Link each inspection to the board's unique ID / serial number *once board identification exists*. Until then, key each result by inspection ID and timestamp. | Current scope: no barcode / DataMatrix / serial identification; future consideration (PCB-21, PRD-19). Per-board traceability in production is not possible until it is added. |
| FR-03 | **Image quality check.** Before inspection, verify at least: sharpness (focus), exposure (clipped fraction, mean brightness relative to reference), board completely inside the field of view, no motion blur, no foreign object (hand, tool) in view. A failed check means recapture or REVIEW, **never PASS**. | Basic checks are implemented for the Experiment 0 phone images (`src/pcb_inspection/image_quality/`, `scripts/check_e0_images.py`; several thresholds PROVISIONAL). Production thresholds: TBC from real captures. |
| FR-04 | **Registration.** Align each image to the reference coordinate frame of its variant, and report the residual alignment error. A failed registration, or an error above tolerance, gives REVIEW. | Fiducials / tooling holes: NEEDS CLARIFICATION (PCB-17). Tolerance: TBC |
| FR-05 | **Inspection.** Produce defect candidates ("findings") for each defect type in scope (section 5). | All listed categories are considered relevant at this stage (PCB-24 "all"). Actual frequency: UNKNOWN. Targets: UNKNOWN (PRD-22). |
| FR-06 | **Localization.** Report each finding's location in image pixels. Where CAD data are available, also report board millimetres and the reference designator. | CAD availability: UNKNOWN (PCB-11, PCB-12, PCB-22) |
| FR-07 | **Decision engine.** A separate, configurable rule layer turns findings and stage results into PASS / FAIL / REVIEW. A single global confidence threshold is **not** acceptable. | Rules: TBC with production (PCB-14, PRD-22, PRD-23) |
| FR-08 | **Fail-safe.** PASS is issued only when every required stage succeeded and no finding reached its review threshold. Unknown defect types, unconfigured rules, and failed or skipped stages give at least REVIEW. | Implemented and unit-tested in `src/pcb_inspection/decision_engine/`. Not yet connected to any real inspection stage. |
| FR-09 | **Explainable output.** As specified in section 7. | — |
| FR-10 | **Record keeping.** Store per inspection: input image(s), result record, and the software, configuration, model and reference-set versions. | Storage is required (PRD-20); local storage is allowed (PRD-25). Location and retention: UNKNOWN. |
| FR-11 | **Result signalling.** Deliver the verdict to the line. | Mechanism: UNKNOWN (PRD-15) |
| FR-12 | **REVIEW workflow.** An operator can confirm or override a result. Overrides are recorded with operator ID and reason, and feed back into the dataset. | Operator is available (PRD-17). Maximum review rate: UNKNOWN. |
| FR-13 | **Variants.** Support every variant and revision on the line. The active variant is selected automatically or by the operator. | Current scope: 1 model (LG Indicator), 1 variant, 1 revision (PCB-02, PCB-03, PRD-02).<br>• Each model / variant is an inspection profile, `configs/profiles/<model>/<variant>.yaml`.<br>• Board colour (blue / red / green, PCB-18) is a visual variation of the same variant, not a variant.<br>• Inspection stages: after SMT/reflow and after THT (PRD-06).<br>See [pcb_inspection_spec.md](pcb_inspection_spec.md), section 2. |
| FR-14 | **Reference management.** Golden images, region maps and CAD-derived references are versioned per inspection profile (model / variant / revision). | — |
| FR-15 | **No manual metadata in inspection mode.** The operator only places the PCB. Profile, stage and identity are resolved automatically (section 2.1), and colour is never an operator input. | Stage source not decided (`inspection_mode.stage_source`) |
| FR-16 | **Automatic capture metadata in dataset mode.** IDs, file naming, capture time, camera, image size and checksum are generated or read automatically. Manual entry is limited to ground truth and optional per-batch context. | Implemented for Experiment 0 (`scripts/e0_ingest.py`) |

## 4. Non-functional requirements

| ID | Requirement | Target |
|---|---|---|
| NFR-01 | **Inspection time** per board, end-to-end including capture, within the line's budget. | UNKNOWN (PRD-04, PRD-05) |
| NFR-02 | **Escape rate** (defective board → PASS). This is the primary quality metric. | UNKNOWN (PRD-22) |
| NFR-03 | **False-call rate** (good board → FAIL). | UNKNOWN (PRD-22) |
| NFR-04 | **REVIEW rate.** | Maximum UNKNOWN (PRD-17) |
| NFR-05 | **Determinism.** The same image with the same versions produces the same result. | Required |
| NFR-06 | **Robustness to normal variation.** Board position within fixture tolerance; lot-to-lot solder-mask and silkscreen shade; approved alternate parts; placement within tolerance; labels; lighting drift within limits. | Magnitudes UNKNOWN (PCB-16, PCB-18, PCB-23, PCB-28) |
| NFR-07 | **Deployment environment.** On-premises / offline operation. | Local storage of images and results is allowed; cloud requirements are not defined (PRD-25). Compute: UNKNOWN (PRD-24). |
| NFR-08 | **Compute hardware.** CPU-only versus GPU is an open decision, made from measured latency and resource use. | Prototype compute: NVIDIA RTX 5050 GPU system (PRD-24; other specifications not provided). Production compute: TBC. |
| NFR-09 | **Maintainability.** The effort to onboard a new variant or revision (captures, labelling, tuning) is measured in experiments, and is a model-selection criterion. | Measured |
| NFR-10 | **Traceability.** Every result is traceable to the images and all versions, and to the board ID once board identification exists. | No board identification in current scope (PRD-19). Details: UNKNOWN (PRD-21). |
| NFR-11 | **Data confidentiality.** Board images can reveal proprietary design. | No confidentiality restrictions reported (DATA-08, PRD-25). |
| NFR-12 | **Licensing.** Third-party software and model licences must permit the intended use. For example, Ultralytics YOLO is AGPL-3.0, with a separate commercial licence; this must be reviewed before any production use. | Review before adoption |
| NFR-13 | **ESD.** Data collection and the production station follow the site's ESD rules. | Mentor: "No rules." Recorded as: no special restrictions reported by the mentor. A formal company ESD policy is UNKNOWN unless documented (PRD-27, DATA-09). |
| NFR-14 | **Availability.** Uptime target, and a fallback procedure when the station is down. | UNKNOWN (PRD-29) |

## 5. Defect requirements

The feasibility ratings below are an **engineering assessment for a top-down 2D colour camera. They have not been validated.** The experiments in [experiment_plan.md](experiment_plan.md) decide.

Ratings:
- **High**: normally visible, given adequate resolution and lighting.
- **Partial**: only some sub-cases are visible.
- **Low**: rarely visible.
- **No**: not physically observable with 2D RGB.

| ID | Config key | Defect | 2D RGB feasibility | Main limiting factor |
|---|---|---|---|---|
| D1 | `missing_component` | Missing component | High | Resolution relative to the smallest package |
| D2 | `damaged_component` | Broken / damaged component | Partial | Fine and internal cracks are invisible |
| D3 | `misaligned_component` | Misaligned component | High (in-plane) / Low (height) | Calibration, registration; height needs 3D |
| D4 | `wrong_component` | Wrong component | High → No, depending on part | Unmarked look-alike parts cannot be distinguished |
| D4b | `wrong_polarity` | Reversed polarity / orientation | High if a polarity mark is visible | Parts without a visible polarity feature |
| D5 | `missing_or_damaged_lead` | Missing / damaged component pin | Partial | Hidden terminations need X-ray; lifted leads need 3D or angled views |
| D6 | `broken_trace` | Broken PCB trace | Low | Solder mask contrast, resolution, inner layers invisible |
| D7 | `missing_trace` | Missing PCB trace | Low → Partial | Same as D6 |
| D8 | `damaged_pcb` | Damaged PCB region | Partial → High (surface) | Internal damage invisible |
| D9 | `solder_defect` | Solder bridges, insufficient / excess solder, solder balls | Partial | Fillet shape is 3D; hidden joints invisible |
| D10 | `foreign_object` | Debris, loose parts, wire clippings, solder splashes | High (above resolution) | Size and contrast |
| D11 | `other_anomaly` | Any other visible deviation from known-good appearance | Partial | Depends on deviation |

### D1 Missing component
- **Visible when** the empty pads are resolved clearly and the component is not hidden under another part or a shield.
- **Imaging:** resolution derived from the smallest package (section 8.1); diffuse lighting to suppress glare from bare pads and solder.
- **Candidate approaches:**
  - Per-region comparison against a registered reference or CAD region.
  - Presence classification per expected component region.
  - Object detection plus a comparison against the expected component list. A detector finds what is present, so "missing" is inferred from what the list expects but the detector did not find.
  - Anomaly detection as a fallback.
- **Needs** the BOM and centroid file (PCB-11, PCB-12) to report *which* component is missing.

### D2 Broken / damaged component
- **Sub-cases:** cracked or chipped body, crushed or burnt package, broken leads, damaged connector housing.
- **Visible:** gross damage. **Often not visible:** hairline cracks. **Never visible:** internal cracks (e.g. flex cracks inside chip capacitors).
- **Imaging:** high resolution. Low-angle (dark-field) lighting emphasises edges and surface cracks; diffuse lighting shows colour changes such as burns.
- **Candidate approaches:** anomaly detection (damage appearance is open-ended); reference comparison. Supervised detection only if there are enough examples per damage type.

### D3 Misaligned component
- **Sub-cases:** X/Y shift; rotation (skew); tombstoning (one end lifted, part standing up); billboarding (part on its side); floating or lifted part.
- **Feasibility:**
  - In-plane shift and rotation beyond tolerance: High. Requires calibrated mm/px and accurate registration.
  - Tombstoning: Partial. The top-down appearance changes, but not always clearly.
  - Height-related cases (lifted, floating): Low. These need 3D measurement or angled views.
- **Imaging:** geometric calibration. A telecentric lens if the parallax of tall parts disturbs measurement. The tolerance comes from the acceptance standard (PCB-23).
- **Candidate approaches:** registration plus measured offset/rotation against the CAD centroid; oriented-box detection; reference comparison.

### D4 Wrong component
- **Feasibility by sub-case:**
  - Different package, size or colour: High.
  - Different marking on a marked part: Partial. The marking must be legible, and laser-etched markings are low-contrast and lighting-sensitive.
  - Same package, no marking, different value: **No.** Chip capacitors normally carry no marking, and small chip resistors are often unmarked.
- **Imaging:** reading markings needs higher local resolution and lighting tuned to the marking. ICs may need a dedicated close-up view.
- **Candidate approaches:** per-region classification against the expected part from the BOM; marking verification (OCR); reference comparison.
- **Must know:** look-alike positions (PCB-15) and approved alternates (PCB-16). A correct alternate part looks different from the golden board.

### D4b Reversed polarity / orientation
- Covers diodes, electrolytic and tantalum capacitors, IC pin 1, keyed connectors.
- **High** when a polarity feature is visible (band, notch, pin-1 dot, bevel). **No** when the part has none.
- **Candidate approaches:** per-region orientation classification; template comparison with the reference.

### D5 Missing component / pin, where visually detectable
- **Sub-cases:** missing, bent or broken leads (gull-wing ICs, connectors); lifted leads; missing or short pin protrusion of through-hole parts.
- **Feasibility:**
  - Visible leads (SOIC/QFP gull-wing; THT pins on the solder side): High to Partial at sufficient resolution.
  - Lifted leads: Low from top-down, because the gap is vertical. Needs angled views or 3D.
  - Hidden terminations (BGA, LGA, QFN): **No.** Needs X-ray.
- **Imaging:** resolution matched to the finest lead pitch; a bottom-side view for THT parts; angled cameras or 3D for coplanarity.
- **Candidate approaches:** per-lead region analysis in the registered image; reference comparison; detection or segmentation of leads.

### D6 Broken PCB trace
- **Low.** Only outer-layer traces are candidates, only where components do not hide them, and only for breaks large enough to resolve through the solder mask. Inner layers: **No.**
- **Why it is hard:**
  - Solder mask covers traces and reduces contrast.
  - Trace widths and break gaps can be very small.
  - Fabrication breaks are normally caught by bare-board AOI or electrical test at the PCB fabricator.
  - On an assembled board, electrical test is the authoritative check.
- **Imaging:**
  - Very high resolution relative to trace width (PCB-20).
  - Wavelength and lighting chosen to maximise copper-to-laminate contrast under the mask (to evaluate: monochrome camera with selected LED wavelengths).
  - Low-angle lighting for mechanical scratches that cut traces.
- **Candidate approaches:** registration plus comparison with a reference image or with a rendering of the Gerber copper layer; copper segmentation; anomaly detection.

### D7 Missing PCB trace
- A trace is absent where the design expects one (fabrication defect), or has been removed by damage.
- **Low to Partial.** Outer layer only. A whole missing segment of a visible trace is more detectable than a hairline break. Otherwise the same constraints as D6 apply.
- **Candidate approach:** reference or Gerber comparison is the natural method, because an absent structure can only be defined relative to an expectation.

### D8 Damaged PCB region
- **Sub-cases:** scratches (through the solder mask, exposing copper); burns or heat discolouration; cracks; chipped or broken board edges; blistering or delamination; measling; stains.
- **Visible:** surface damage of sufficient size (Partial to High). **Rarely visible:** internal delamination or measling that does not show at the surface (Low).
- **Imaging:** low-angle lighting for scratches and relief; diffuse lighting for discolouration. The field of view must include the board edges.
- **Candidate approaches:** segmentation (if labelled masks are available); anomaly detection; reference comparison.

### D9 Solder defects (part of "other visible defects")
- **Sub-cases:** bridges, insufficient or excess solder, solder balls, unsoldered joints, all on *visible* joints.
- **Feasibility:**
  - Bridges between visible leads: High to Partial at sufficient resolution.
  - Solder volume and fillet shape: Low with a single diffuse top light. Commercial AOI uses multi-angle (often colour-coded) lighting or 3D measurement, because fillet shape is a 3D property.
  - Hidden joints: **No.**
- **Candidate approaches:** per-joint region classification; anomaly detection; specialised lighting.

### D10 Foreign object, and D11 other anomaly
- **Foreign object:** High for objects whose size and contrast are above the resolution limit. Approaches: reference comparison, anomaly detection.
- **Other anomaly:** any unclassified deviation from known-good appearance. Approach: anomaly detection. The result is REVIEW unless rules say otherwise.

### Conformal coating (if applicable)
If boards are coated before the inspection point (PCB-19), their appearance and glare change. Coatings with a UV tracer allow coating-coverage inspection under UV light. **LG Indicator: no conformal coating (PCB-19), so this does not apply in the current scope.**

## 6. Input requirements

**Per inspection:**
- Image(s) of each inspected side: lossless PNG or TIFF, no in-camera processing, bit depth recorded.
- Board ID / serial number, variant, revision, side.
- Station and capture-settings identifier.

**Per variant (reference data):**
- From production:
  - BOM: refdes → part number, package, description (PCB-11).
  - Centroid / pick-and-place file: refdes, X, Y, rotation, side (PCB-12).
  - Gerber or ODB++: outer copper, solder mask, silkscreen, outline, fiducials (PCB-22).
  - Assembly drawing.
  - List of critical and polarized components (PCB-13, PCB-14).
  - Acceptance criteria (PCB-23).
- Captured by us: reference images from **several** verified-good boards, not a single golden board.

## 7. Output requirements (explainable result)

Each inspection produces one result record:

| Field | Content |
|---|---|
| `inspection_id`, `timestamp` | Unique ID and time |
| `board_id`, `serial_number` | Physical board identity |
| `pcb_variant`, `pcb_revision`, `sides` | What was inspected |
| `verdict` | `PASS` / `FAIL` / `REVIEW` |
| `reasons[]` | Ordered, human-readable reasons, each with the outcome it produced. Implemented as `Decision.reasons`. |
| `findings[]` | Per finding:<br>• defect type, confidence, source stage/method<br>• location: bounding box in pixels, mask, and board mm if CAD is available<br>• component refdes, criticality<br>• reference comparison info (difference score; expected vs observed position or orientation)<br>• evidence crops: actual and reference |
| `stage_status[]` | Image-quality metrics, registration residual error, which inspection stages ran and whether they succeeded |
| `annotated_image` | The capture with findings overlaid (box/mask, defect type, refdes, confidence), with reference crops side by side |
| `processing_time_ms` | Total and per stage |
| `versions` | Software, configuration, model(s), reference set |
| `operator_review` | Operator ID, final decision, comment (FAIL / REVIEW boards) |

**Operator explanation requirement.** Every FAIL or REVIEW must carry at least one reason that names the defect type, the location (refdes when known) and the confidence. It must be shown with an image crop next to the corresponding reference crop, so the operator can see *why* without consulting an engineer.

## 8. Hardware requirements (provisional)

Every value depends on the answers in [production_questions.md](production_questions.md). This section defines **how** the values will be derived, not the values themselves.

### 8.1 Spatial resolution
```
mm_per_px        = FOV_mm / sensor_px               (per axis)
target_mm_per_px = smallest_feature_mm / px_per_feature
sensor_px       >= FOV_mm / target_mm_per_px
```
**Planning guideline, to be validated in E0:**
- The smallest feature to be *detected* should span at least about 3 px.
- Features to be *classified or measured* (component presence, polarity marks, offsets) should span **≥ 10 px** across the smallest relevant dimension.

*Illustrative only, not our board:* FOV 120 mm, smallest feature 0.5 mm (the width of an 0402 chip), 10 px per feature → 0.05 mm/px → at least 2 400 px on that axis, before margin.

If one camera cannot reach the required mm/px over the whole board, the options are multiple cameras, tiled capture on a motion stage, or a smaller field of view per view. The lens must also resolve the sensor's pixel pitch: pixel count alone does not guarantee resolution.

### 8.2 Camera
- An industrial machine-vision camera with full manual control of exposure, gain and white balance.
- A standard interface (GigE Vision / USB3 Vision, GenICam) with an SDK usable from Python.
- Global shutter if the board moves during exposure (PRD-10).
- Colour versus monochrome is decided in E0. Colour helps distinguish components. Monochrome gives better sharpness and sensitivity per pixel, and allows wavelength-selective lighting.

### 8.3 Lens
- Fixed focal length, low distortion, lockable focus and aperture.
- A telecentric lens if the parallax of tall components, or precise offset measurement, requires it. Its field of view is limited by the front lens diameter, and cost rises with it.
- **Focal length** (thin-lens approximation; confirm with the vendor): `f ≈ WD · m / (1 + m)`, where `m = sensor_width / FOV_width`.
- **Depth of field** (approximation): `DOF ≈ 2 · N · c · (1 + m) / m²`, where N is the f-number and c the acceptable blur (≈ 1–2 pixel pitches).
  - Stopping down increases DOF, but diffraction blur grows: Airy disk diameter ≈ `2.44 · λ · N · (1 + m)`. Choose the aperture by measurement, not by formula alone.
  - The DOF must cover the range from the board surface to the tops of the relevant components (PCB-06).

### 8.4 Lighting
- An enclosure that blocks ambient light, and stabilised LED illumination.
- Baseline: diffuse illumination (dome or flat diffuse) to suppress specular glare from solder and metal.
- **Options to evaluate in E0:**
  - Coaxial: flat reflective surfaces.
  - Low-angle ring / dark-field: scratches, edges, relief.
  - Multi-angle or colour-coded: solder fillets.
  - Selected wavelengths: copper contrast under solder mask.
  - UV: conformal coating, if applicable.

### 8.5 Mounting and fixture
- A rigid, vibration-free camera mount, with the optical axis perpendicular to the board.
- Repeatable board positioning (stops, pins or tooling holes) and a matte, uniform background.
- Two-sided inspection needs a flip fixture or a second camera (PRD-14).

### 8.6 Compute
TBC from the latency and resource measurements in [experiment_plan.md](experiment_plan.md), section 3.5.

### 8.7 Beyond 2D
- **3D** (structured light / laser) for component heights and solder volume.
- **Angled cameras** for lifted leads and tombstones.
- **X-ray** for hidden joints.

These are added only if the required defect scope demands them and the budget allows. The decision follows the production answers (PCB-10, PCB-24, PRD-26).

## 9. Data requirements

| ID | Requirement |
|---|---|
| DR-01 | The first dataset consists of **known-good, fully assembled boards**, with many distinct physical boards per variant. Repeated images of one board are not a substitute. |
| DR-02 | Each board is captured several times with re-seating, in more than one session, to measure normal variation. |
| DR-03 | Defective data come preferably from real rejected or failed boards. Controlled defects require written authorization. Synthetic defects only where justified, and never in validation or test. |
| DR-04 | Every capture carries the metadata defined in [data_collection_plan.md](data_collection_plan.md), section 6. It is validated with `scripts/validate_dataset.py`. |
| DR-05 | Splits are made by **physical board** (`board_id`). Test boards are never used for training, reference building or tuning. |
| DR-06 | Raw images are lossless, unmodified and read-only after capture. All processing writes elsewhere. |
| DR-07 | Each dataset version is immutable, with its own split file and file hashes. |
| DR-08 | Permission to store and process the images exists (DATA-08). |
| DR-09 | Production artifacts are available: BOM, centroid file, Gerber/ODB++, defect history (section D of production_questions.md). |
| DR-10 | Every label traces back to a verification by a named, qualified person (DATA-06). |

## 10. Evaluation requirements

| ID | Requirement |
|---|---|
| ER-01 | **Board-level** confusion matrix of ground truth {good, defective} against verdict {PASS, REVIEW, FAIL}.<br>• Escape rate = defective→PASS / all defective.<br>• False-call rate = good→FAIL / all good.<br>• Review rate. |
| ER-02 | **Defect-level** results per defect type: TP, FP, FN, precision, recall, F1. Matching criteria are defined per annotation type ([experiment_plan.md](experiment_plan.md), section 3.2). |
| ER-03 | **Localization quality** with a metric suited to each approach (IoU / mAP, Dice, pixel-AUROC / AUPRO, position error in mm). |
| ER-04 | **Performance.** Latency p50 / p95 / max per stage and end-to-end; CPU, GPU and memory use; hardware recorded. |
| ER-05 | Evaluate on **held-out physical boards only**. Tune thresholds on validation; use test only for final reporting. |
| ER-06 | **Statistical honesty.** Report counts as well as percentages, with 95 % confidence intervals (e.g. Wilson).<br>• With zero misses on *n* independent defect instances, the 95 % upper bound on the miss rate is ≈ 3/*n* (the "rule of three").<br>• So demonstrating a miss rate ≤ 1 % for one defect type needs ≈ 300 independent instances with no misses.<br>• Small defect sets cannot prove low escape rates, and claims must match the sample size. |
| ER-07 | **Generalisation.** Evaluate across production lots and capture days, not only across boards. |
| ER-08 | **No selection by accuracy alone.** Use the criteria in [model_strategy.md](model_strategy.md), section 7. |
| ER-09 | **Acceptance criteria:** UNKNOWN. To be agreed with production (PRD-22, PRD-23, PRD-28). |
